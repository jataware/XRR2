#!/usr/bin/env python3
"""
    ezbright.__main__

    Query -> Expand -> Retrieve -> Rerank -> (Double Rerank)
"""

import os
import re
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich import print as rprint

from ezprompt import EZPrompt, run_batch
from ezprompt.cache import disk_cache

from .utils import load_bright, compute_metrics
from .retrievers import BM25S
from .rerank import rerank, rerank_multiple

EZPrompt.set_default_logdir('./ezlogs')
np.random.seed(123)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='psychology')
    parser.add_argument('--qe_model', type=str, default='gpt-4o')
    parser.add_argument('--rr_model', type=str, default='gpt-4o')

    parser.add_argument('--qe_method', type=str, default='v2')
    parser.add_argument('--rr_method', type=str, default='v2')
    parser.add_argument('--topk0', type=int, default=40)
    parser.add_argument('--topk1', type=int, default=10)

    parser.add_argument('--double_rr', type=int, default=0)

    parser.add_argument('--outdir', type=str, default='./results')

    args = parser.parse_args()

    args.qe_prompt = open(f'./ezbright/prompts/{args.qe_method}_query_expander.md').read()
    args.rr_prompt = open(f'./ezbright/prompts/{args.rr_method}_reranker.md').read()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    return args

args     = parse_args()
task     = args.task

# --
# IO

rprint(f'[green]========== Loading {task}', file=sys.stderr)
dataset  = load_bright(task, pre_reasoning=None, long_context=False)
queries  = dataset['queries']
kb       = dataset['kb']
kb_dict  = {xx['id'] : xx['doc'] for xx in kb}
gt       = [query['_gt_ids'] for query in queries]

# --
# Helpers

# BM25 Retriever
@disk_cache(cache_dir='./ezlogs/retrieve_orig', verbose=False, ignore_fields=['rt'])
def do_retrieval(rt, queries, topk, query_key='query'):
     return [
        rt.run(
            query        = query[query_key],
            topk         = topk,
            excluded_ids = query['excluded_ids']
        ) for query in tqdm(queries, desc=f'Retrieving ({query_key})')
    ]

rt = BM25S(data=kb, doc_key='doc')

# [TODO] what's sensitivity to these parameters?
#        are these even good parameters?
llm_kwargs = {
    "temperature" : 0.8, 
    "top_p"       : 0.8,
    "max_retries" : 3
}

# Query Expander
query_expander = EZPrompt(
    name='query_expander',
    template=args.qe_prompt,
    before=lambda query: {
        'QUERY' : query,
    },
    llm_kwargs={**llm_kwargs, "model" : args.qe_model, "max_tokens" : 2048},
    cache_dir='./ezlogs/cache'
)

# Reranker
def _reranker_after(output_str, __TOPK):
    if output_str is None: return None
    
    idxs = re.findall(r"(?:```json\s*)(.+)(?:```)", output_str, re.DOTALL)
    if len(idxs) > 0:
        return {"idxs" : [int(xx) for xx in json.loads(idxs[-1].strip())]}
    else:
        raise Exception(f'No idxs found in {output_str}')


reranker = EZPrompt(
    name='reranker',
    template=args.rr_prompt,
    before=lambda query, docs, topk: {
        'QUERY'     : query,
        'DOC_STR'   : ''.join(["[{}]. {}\n\n".format(i + 1, re.sub('\n+', ' ', doc)) for i, doc in enumerate(docs)]),
        'TOPK'      : topk,
        '__TOPK'    : topk,
    },
    after=_reranker_after,
    llm_kwargs={**llm_kwargs, "model" : args.rr_model},
    cache_dir='./ezlogs/cache'
)



# Metrics w/ raw queries
rprint('[green]========== Raw Retrieval', file=sys.stderr)
all_hits0   = do_retrieval(rt, queries, args.topk0, query_key='query')
metrics0    = compute_metrics(results=all_hits0, gt=gt)
metrics0_ub = compute_metrics(results=all_hits0, gt=gt, ub=True)

# --
# Step 1: Query Expansion

rprint('[green]========== Query Expansion', file=sys.stderr)
ltasks  = {qid:query_expander.larun(query=query['query']) for qid, query in enumerate(queries)}
results = run_batch(ltasks, max_calls=128)
for qid in results.keys():
    queries[qid]['query_expanded'] = results[qid]['output_str']


# --
# Step 2: Retrieval

rprint('[green]========== QE Retrieval', file=sys.stderr)
all_hits_qe   = do_retrieval(rt, queries, args.topk0, query_key='query_expanded')
metrics_qe    = compute_metrics(results=all_hits_qe, gt=gt)
metrics_qe_ub = compute_metrics(results=all_hits_qe, gt=gt, ub=True)

# --
# Rerank

rprint('[green]========== Rerank', file=sys.stderr)
all_hits_rr   = rerank(reranker, kb_dict, queries, all_hits_qe, gt, topk1=args.topk1)
metrics_rr    = compute_metrics(results=all_hits_rr, gt=gt)
metrics_rr_ub = compute_metrics(results=all_hits_rr, gt=gt, ub=True)

# --
# Double Rerank

if args.double_rr > 0:
    rprint('[green]========== Double Rerank', file=sys.stderr)
    # Rerank again (listwise)
    # [TODO] Try pairwise + Elo score

    all_hits_rr2   = rerank_multiple(reranker, kb_dict, queries, all_hits_rr, gt, topk1=args.topk1, n_reranks=args.double_rr)
    metrics_rr2    = compute_metrics(results=all_hits_rr2, gt=gt)
    metrics_rr2_ub = compute_metrics(results=all_hits_rr2, gt=gt, ub=True)

# --
# Format results

def format_results(all_hits):
    return 

# --
# Print

table = Table(title="Retrieval Results")
table.add_column("Task",         justify="right")
table.add_column("Metric",       justify="right")
table.add_column("Q",            justify="right")
table.add_column("Q+E",          justify="right")
table.add_column("Q+E+RR",       justify="right")
table.add_column("Q+E+RR2",      justify="right")
table.add_column("Q[UB]",        justify="right")
table.add_column("Q+E[UB]",      justify="right")
table.add_column("Q+E+RR[UB]",   justify="right")
table.add_column("Q+E+RR2[UB]",  justify="right")

for metric in metrics0.keys():
    table.add_row(
        task,
        metric, 
        f"{metrics0[metric]:.5f}", 
        f"{metrics_qe[metric]:.5f}", 
        f"{metrics_rr[metric]:.5f}",
        f"{metrics_rr2[metric]:.5f}" if args.double_rr else '',
        f"{metrics0_ub[metric]:.5f}", 
        f"{metrics_qe_ub[metric]:.5f}", 
        f"{metrics_rr_ub[metric]:.5f}",
        f"{metrics_rr2_ub[metric]:.5f}" if args.double_rr else '',
        style='white' if metric == 'ndcg_cut_10' else 'yellow'
    )

console = Console()
console.print(table)

# --
# Save predictions

def _format_results(queries, all_hits):
    return {
        str(query['id']) : {
            xx['id'] : xx['_score'] for xx in sorted(hit, key=lambda x: x['_score'], reverse=True)
        } for query, hit in zip(queries, all_hits)
    }

with open(os.path.join(args.outdir, f'{args.task}__qe_results.json'), 'w') as f:
    json.dump(_format_results(queries, all_hits_qe), f, indent=2)

with open(os.path.join(args.outdir, f'{args.task}__rr_results.json'), 'w') as f:
    json.dump(_format_results(queries, all_hits_rr), f, indent=2)

if args.double_rr > 0:
    with open(os.path.join(args.outdir, f'{args.task}__rr2_results.json'), 'w') as f:
        json.dump(_format_results(queries, all_hits_rr2), f, indent=2)