#!/usr/bin/env python3
"""
    dspybright

    Query -> Expand -> Retrieve -> Rerank -> (Double Rerank)
    
    dspy implementation of the xrr2 retrieval pipeline.
"""

import os
import re
import random
import sys
import json
import argparse
import numpy as np
import pydantic
from rich import print
from typing import List, Dict, Any, Optional, Callable, Union
import pandas as pd
import dspy
from dspy.evaluate import Evaluate

from utils import load_bright, compute_metrics
from retrievers import BM25S

np.random.seed(55)

# -
# Data Containers

class Query(pydantic.BaseModel):
    id: str
    query: str
    query_expanded: Optional[str] = None
    excluded_ids: List[str] = []
    _gt_ids: List[str] = []

class Document(pydantic.BaseModel):
    id: str
    doc: str

class RetrievalResult(pydantic.BaseModel):
    id: str
    score: float
    rank: Optional[int] = None

class QueryResults(pydantic.BaseModel):
    query: Query
    raw_results: List[RetrievalResult]
    qe_results: List[RetrievalResult] 
    reranked_results: List[RetrievalResult]
    double_reranked_results: Optional[List[RetrievalResult]] = None

# -
# Signatures (i/o operations)

class QueryExpansionSignature(dspy.Signature):
    """Provide an extensive elaboration on the user's inquiry, covering the problem or question itself, and the surrounding context and potential avenues for addressing it.
    
    1. Analyze the Inquiry: Break down the user's question into its fundamental components. What are they really asking?
    2. Contextualize: What background information, related concepts, or common scenarios are relevant to understanding this inquiry fully?
    3. Explore Potential Responses/Solutions/Information: Describe various ways one might address the inquiry, or different facets of information that would be pertinent. For each, mention specific terms, ideas, or steps involved.
    4. Synthesize into a Detailed Discourse: Weave all of this into a coherent and detailed piece of writing. The aim is to generate a text that is dense with relevant information and terminology related to the original inquiry."""
    
    query: str = dspy.InputField(desc="The original user's inquiry")
    expanded_query: str = dspy.OutputField(desc="Extensive elaboration covering the problem, context, and potential avenues for addressing it")

class RerankingSignature(dspy.Signature):
    """First identify the essential problem in the query. Think step by step to reason about why each document is relevant or irrelevant. Rank these passages based on their relevance to the query. Output the ranking result as a list of document IDs, where the first element is the most relevant passage."""
    
    query: str = dspy.InputField(desc="The user's query")
    documents: List[Document] = dspy.InputField(desc="List of candidate documents to rank")
    topk: int = dspy.InputField(desc="Number of top documents to return in ranking")
    reasoning: str = dspy.OutputField(desc="Step by step reasoning about document relevance")
    ranking: List[RetrievalResult] = dspy.OutputField(desc="List of documents ranked by relevance with scores, most relevant first")

# -
# Modules

class QueryExpander(dspy.Module):
    def __init__(self, model="gpt-4o"):
        super().__init__()
        self.expand = dspy.ChainOfThought(QueryExpansionSignature)
        self.model  = model
    
    def forward(self, query: str) -> str:
        with dspy.context(lm=dspy.LM(f'{self.model}'), api_key=os.getenv("OPENAI_API_KEY" if "openai" in self.model else "GEMINI_API_KEY")):
            result = self.expand(query=query)
            return result.expanded_query

class DocumentReranker(dspy.Module):
    def __init__(self, model="gpt-4o"):
        super().__init__()
        self.rerank = dspy.ChainOfThought(RerankingSignature)
        self.model = model
    
    def forward(self, query: str, documents: List[Dict], topk: int = 10, knowledge_base=None) -> List[int]:
        # Convert dict documents to Document pydantic models
        doc_objects = []
        for doc in documents:
            if 'doc' in doc:
                # Document already has text
                doc_text = doc['doc']
            elif knowledge_base:
                # Look up document text from knowledge base
                doc_text = knowledge_base.get_document(doc['id'])
            else:
                # Fallback - use ID as text (not ideal)
                doc_text = doc['id']
            
            doc_objects.append(Document(id=doc['id'], doc=doc_text))
        
        with dspy.context(lm=dspy.LM(f'{self.model}'), api_key=os.getenv("OPENAI_API_KEY" if "openai" in self.model else "GEMINI_API_KEY")):
            result = self.rerank(
                query=query,
                documents=doc_objects,
                topk=topk
            )
            
            # Extract indices from RetrievalResult objects
            try:
                doc_id_to_idx = {doc['id']: i for i, doc in enumerate(documents)}
                ranked_indices = [doc_id_to_idx[rr.id] for rr in result.ranking if rr.id in doc_id_to_idx]
                return ranked_indices[:topk]
            except Exception as e:
                print(f"Reranking failed: {e}, using original order")
                return list(range(min(topk, len(documents))))

class SingleQueryPipeline(dspy.Module):
    """End-to-end retrieval pipeline for a single query"""
    
    def __init__(self, 
                 knowledge_base,  # Pass the KB object instead of functions
                 qe_model: str = "openai/gpt-4o",
                 rr_model: str = "openai/gpt-4o",
                 topk0: int = 100,
                 topk1: int = 10,
                 double_rr: int = 10):
        super().__init__()
        
        # Initialize components
        self.query_expander = QueryExpander(model=qe_model)
        self.reranker       = DocumentReranker(model=rr_model)
        
        # Store the knowledge base object instead of function references
        self.knowledge_base = knowledge_base
        
        # Parameters
        self.topk0     = topk0
        self.topk1     = topk1
        self.double_rr = double_rr
    
    def _rerank_single(self, query: Query, hits: List[Dict], seed: Optional[int] = None) -> List[RetrievalResult]:
        """Rerank documents for a single query"""
        
        # Skip if no relevant documents in hits
        if not any(hit['id'] in query._gt_ids for hit in hits):
            return [RetrievalResult(id=hit['id'], score=self.topk1 - i) for i, hit in enumerate(hits[:self.topk1])]
        
        # Permutation for robustness
        rng             = np.random.default_rng(123 + seed)
        perm            = rng.permutation(len(hits))
        hits_for_rerank = [hits[i] for i in perm]
        original_order  = perm
        
        try:
            # Get reranked indices
            ranked_indices = self.reranker(
                query       = query.query,
                documents   = hits_for_rerank,
                topk        = self.topk1,
                knowledge_base=self.knowledge_base
            )
            
            # Convert back to RetrievalResult format with scores
            reranked_results = []
            for i, idx in enumerate(ranked_indices):
                if idx < len(hits_for_rerank):
                    # If we permuted, map back to original indices
                    original_idx = original_order[idx] if original_order is not None else idx
                    result = RetrievalResult(
                        id=hits[original_idx]['id'],
                        score=self.topk1 - i,  # Higher score for higher rank
                        rank=i + 1
                    )
                    reranked_results.append(result)
            return reranked_results
            
        except Exception as e:
            print(f"Reranking failed for query {query.id}: {e}")
            return [RetrievalResult(id=hit['id'], score=self.topk1 - i) for i, hit in enumerate(hits[:self.topk1])]

    def _double_rerank_single(self, query: Query, reranked_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Perform double reranking for a single query"""
        
        # Convert back to hits format for reranking
        hits_for_rerank = [{'id': result.id, '_score': result.score} for result in reranked_results]
        
        # Multiple reranking rounds
        all_rerank_results = []
        for i in range(self.double_rr):
            round_results = self._rerank_single(query, hits_for_rerank, seed=i)
            all_rerank_results.append(round_results)
        
        # Aggregate results by document ID
        score_accumulator = {}
        for round_results in all_rerank_results:
            for result in round_results:
                if result.id not in score_accumulator:
                    score_accumulator[result.id] = 0
                score_accumulator[result.id] += result.score
        
        # Convert back to sorted list
        aggregated_results = [RetrievalResult(id=doc_id, score=total_score) for doc_id, total_score in score_accumulator.items()]
        
        # Sort by aggregated score
        aggregated_results.sort(key=lambda x: x.score, reverse=True)
        
        # Add ranks
        for i, result in enumerate(aggregated_results):
            result.rank = i + 1
        
        return aggregated_results
    
    def forward(self, query_text, query_id, excluded_ids=None, gt_ids=None) -> QueryResults:
        """Complete end-to-end retrieval for a single query"""
        
        # Handle both Example objects and direct arguments
        if hasattr(query_text, 'query_text'):  # It's a dspy.Example
            example = query_text
            query_text = example.query_text
            query_id = example.query_id
            excluded_ids = getattr(example, 'excluded_ids', [])
            gt_ids = getattr(example, 'gt_ids', [])
        
        # Convert inputs to Query object
        query = Query(
            id=str(query_id),
            query=query_text,
            excluded_ids=excluded_ids or [],
            _gt_ids=gt_ids or []
        )
                
        # Step 1: Raw retrieval using knowledge base
        raw_hits    = self.knowledge_base.retrieve(query.query, self.topk0, query.excluded_ids)
        raw_results = [RetrievalResult(id=hit['id'], score=hit['_score']) for hit in raw_hits]
        
        # Step 2: Query expansion
        query.query_expanded = self.query_expander(query.query)
                
        # Step 3: Retrieval with expanded query
        qe_hits    = self.knowledge_base.retrieve(query.query_expanded, self.topk0, query.excluded_ids)
        qe_results = [RetrievalResult(id=hit['id'], score=hit['_score']) for hit in qe_hits]
                
        # Step 4: Reranking
        reranked_results = self._rerank_single(query, qe_hits)
                
        # Step 5: Optional double reranking
        double_reranked_results = None
        if self.double_rr > 0:
            double_reranked_results = self._double_rerank_single(query, reranked_results)
    
        return QueryResults(
            query                   = query,
            raw_results             = raw_results,
            qe_results              = qe_results,
            reranked_results        = reranked_results,
            double_reranked_results = double_reranked_results
        )

# ---------- Knowledge Base Interface ----------

class KnowledgeBase:
    """Encapsulates knowledge base data and provides access functions"""
    
    def __init__(self, kb_data: List[Dict]):
        self.retriever = BM25S(data=kb_data, doc_key='doc')
        self.doc_lookup = {doc['id']: doc['doc'] for doc in kb_data}
    
    def retrieve(self, query: str, topk: int, excluded_ids: List[str]) -> List[Dict]:
        """Retrieve documents using BM25"""
        return self.retriever.run(
            query=query,
            topk=topk,
            excluded_ids=excluded_ids
        )
    
    def get_document(self, doc_id: str) -> str:
        """Get document text by ID"""
        return self.doc_lookup.get(doc_id, "")

def extract_hits(results: List[QueryResults], stage: str) -> List[List[Dict]]:
    hits = []
    for result in results:
        if stage == 'raw':
            stage_results = result.raw_results
        elif stage == 'qe':
            stage_results = result.qe_results
        elif stage == 'reranked':
            stage_results = result.reranked_results
        elif stage == 'double_reranked':
            stage_results = result.double_reranked_results or []
        
        # Convert back to the expected format
        stage_hits = [{'id': r.id, '_score': r.score} for r in stage_results]
        hits.append(stage_hits)
    return hits

def retrieval_metric(example, pred, trace=None):
    """
    DSPy-compatible metric function for retrieval evaluation
    
    Args:
        example: dspy.Example with gt_ids field
        pred: QueryResults object from pipeline
        trace: Optional trace (unused)
    
    Returns:
        float: recall@10 score
    """
    try:
        # Extract ground truth IDs
        gt_ids = example.gt_ids
        predicted_results = pred.double_reranked_results
        
        # Convert to the format expected by compute_metrics
        predicted_hits = [{'id': r.id, '_score': r.score} for r in predicted_results]
        
        # Compute metrics for this single example
        metrics = compute_metrics(results=[predicted_hits], gt=[gt_ids])
        return metrics['recall_10']
        
    except Exception as e:
        print(f"Error in metric computation: {e}")
        return 0.0

# ---------- Main Script ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='psychology')
    parser.add_argument('--qe_model', type=str, default='openai/gpt-4o')
    parser.add_argument('--rr_model', type=str, default='gemini/gemini-2.5-pro-preview-03-25')
    parser.add_argument('--topk0', type=int, default=100)
    parser.add_argument('--topk1', type=int, default=10)
    parser.add_argument('--double_rr', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--outdir', type=str, default='./results')
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    return args

def main():
    args = parse_args()
    
    # Configure DSPy
    lm = dspy.LM(f'{args.qe_model}', 
                 api_key=os.getenv("OPENAI_API_KEY" if "openai" in args.qe_model else "GEMINI_API_KEY"),
                 temperature=0.8,
                 top_p=0.8,
                 max_retries=3)
    dspy.configure(lm=lm)
        
    # Load data
    print(f'[green]========== Loading {args.task}', file=sys.stderr)
    dataset = load_bright(args.task, pre_reasoning=None, long_context=False)   
    
    random.shuffle(dataset['queries'])
    
    queries_train = dataset['queries'][:args.n_train]
    queries_test  = dataset['queries'][args.n_train:]    
        
    # Initialize knowledge base
    kb = KnowledgeBase(dataset['kb'])
    
    # Initialize pipeline with function interfaces
    pipeline = SingleQueryPipeline(
        knowledge_base = kb,
        qe_model       = args.qe_model,
        rr_model       = args.rr_model,
        topk0          = args.topk0,
        topk1          = args.topk1,
        double_rr      = args.double_rr
    )
    
    # Create train and test datasets
    ds_trn = []
    for qq in queries_train:
        ds_trn.append(
            dspy.Example(
                query_text      = qq['query'], 
                query_id        = str(qq['id']),
                excluded_ids    = qq.get('excluded_ids', []),
                gt_ids          = qq['_gt_ids']
            ).with_inputs("query_text", "query_id", "excluded_ids", "gt_ids")
        )
    
    ds_tst = []
    for qq in queries_test:
        ds_tst.append(
            dspy.Example(
                query_text      = qq['query'],
                query_id        = str(qq['id']),
                excluded_ids    = qq.get('excluded_ids', []),
                gt_ids          = qq['_gt_ids']
            ).with_inputs("query_text", "query_id", "excluded_ids", "gt_ids")
        )

    # Create test set evaluator
    test_evaluator = Evaluate(
        devset=ds_tst, 
        num_threads=5, 
        display_progress=True, 
        display_table=len(ds_tst), 
        provide_traceback=True
    )
    
    train_evaluator = Evaluate(
        devset=ds_trn, 
        num_threads=5, 
        display_progress=True, 
        display_table=len(ds_trn), 
        provide_traceback=True
    )
    
    # 1. Evaluate unoptimized pipeline on test set
    print("\n=== Evaluating Unoptimized Pipeline on Test Set ===")
    unoptimized_score = test_evaluator(pipeline, metric=retrieval_metric)
    print(f"Unoptimized Test Set Recall@10: {unoptimized_score}")
    
    # 2. Evaluate unoptimized pipeline on training set
    print("\n=== Evaluating Unoptimized Pipeline on Training Set ===")
    unoptimized_score = train_evaluator(pipeline, metric=retrieval_metric)
    print(f"Unoptimized Training Set Recall@10: {unoptimized_score}")
    
    # 2. Train/Optimize pipeline on training set
    print("\n=== Training Pipeline on Training Set ===")
    optimizer = dspy.MIPROv2(metric=retrieval_metric, auto="medium", num_threads=5)
    optimized_pipeline = optimizer.compile(
        pipeline, 
        trainset=ds_trn, 
        requires_permission_to_run=False
    )
    print(f"Optimized Pipeline: {optimized_pipeline}")
    
    
    # 3. Evaluate optimized pipeline on test set
    print("\n=== Evaluating Optimized Pipeline on Test Set ===")
    optimized_score = test_evaluator(optimized_pipeline, metric=retrieval_metric)
    print(f"Optimized Test Set Recall@10: {optimized_score}")
    
    # Print final comparison
    print("\n=== Final Results ===")
    print(f"Unoptimized Test Set Recall@10: {unoptimized_score}")
    print(f"Optimized Test Set Recall@10: {optimized_score}")
    print(f"Improvement: {optimized_score - unoptimized_score:.4f}")

if __name__ == "__main__":
    main()