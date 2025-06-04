#!/usr/bin/env python3
"""
Enhanced DSPy training with optimizer sweep and multi-dataset support.
"""

import os
import random
import sys
import json
import argparse
import numpy as np
from rich import print
from typing import List, Dict
import dspy
from dspy.evaluate import Evaluate
from collections import defaultdict
import time
from datetime import datetime

from utils import load_bright_multi, compute_metrics
from main_train import (
    SingleQueryPipeline, KnowledgeBase, retrieval_metric
)

np.random.seed(55)
random.seed(55)

def load_combined_data(tasks: List[str], train_ratio=0.7, val_ratio=0.15):
    """Load and combine multiple datasets with stratified splits"""
    print(f"[green]Loading tasks: {', '.join(tasks)}")
    
    all_data = load_bright_multi(tasks)
    
    # Combine all queries and KB with task prefixes
    combined_queries, combined_kb = [], []
    
    for task in tasks:
        for query in all_data[task]['queries']:
            query['task'] = task
            query['excluded_ids'] = [f"{task}_{eid}" for eid in query['excluded_ids']]
            query['_gt_ids'] = [f"{task}_{gid}" for gid in query['_gt_ids']]
            combined_queries.append(query)
        
        for doc in all_data[task]['kb']:
            combined_kb.append({
                'id': f"{task}_{doc['id']}",
                'doc': doc['doc'],
                'task': task
            })
    
    # Stratified split by task
    train_queries, val_queries, test_queries = [], [], []
    
    for task in tasks:
        task_queries = [q for q in combined_queries if q['task'] == task]
        random.shuffle(task_queries)
        
        n = len(task_queries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_queries.extend(task_queries[:n_train])
        val_queries.extend(task_queries[n_train:n_train + n_val])
        test_queries.extend(task_queries[n_train + n_val:])
    
    # Final shuffle
    for split in [train_queries, val_queries, test_queries]:
        random.shuffle(split)
    
    print(f"[green]Split sizes - Train: {len(train_queries)}, Val: {len(val_queries)}, Test: {len(test_queries)}")
    return train_queries, val_queries, test_queries, combined_kb

def to_dspy_examples(queries: List[Dict]) -> List[dspy.Example]:
    """Convert queries to DSPy examples"""
    return [
        dspy.Example(
            query_text=q['query'], 
            query_id=str(q['id']),
            excluded_ids=q.get('excluded_ids', []),
            gt_ids=q['_gt_ids'],
            task=q.get('task', 'unknown')
        ).with_inputs("query_text", "query_id", "excluded_ids", "gt_ids")
        for q in queries
    ]

def get_optimizers():
    """Return optimizer configurations"""
    return {
        'labeled_fewshot': (dspy.LabeledFewShot, {'k': 3}),
        'bootstrap_fewshot': (dspy.BootstrapFewShot, {
            'metric': retrieval_metric,
            'max_labeled_demos': 8,
            'max_bootstrapped_demos': 4
        }),
        'bootstrap_fewshot_rs': (dspy.BootstrapFewShotWithRandomSearch, {
            'metric': retrieval_metric,
            'max_labeled_demos': 8,
            'max_bootstrapped_demos': 4,
            'num_candidate_programs': 8
        }),
        'copro': (dspy.COPRO, {
            'metric': retrieval_metric,
            'depth': 3,
            'breadth': 10
        }),
        'miprov2_few': (dspy.MIPROv2, {
            'metric': retrieval_metric,
            'auto': "medium",
            'num_threads': 10
        }),
    }

def print_results(results):
    """Print results summary"""
    print(f"\n[green]{'='*80}")
    print(f"[green]RESULTS SUMMARY")
    print(f"[green]{'='*80}")
    
    successful = {k: v for k, v in results.items() if v['success']}
    if not successful:
        print("[red]No successful runs!")
        return
    
    # Sort by test score
    sorted_results = sorted(successful.items(), key=lambda x: x[1]['test_score'], reverse=True)
    
    print(f"{'Optimizer':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Time':<8}")
    print("-" * 70)
    
    for opt_name, result in sorted_results:
        print(f"{opt_name:<20} {result['train_score']:<8.4f} {result['val_score']:<8.4f} {result['test_score']:<8.4f} {result['opt_time']:<8.1f}")
    
    best = sorted_results[0]
    print(f"\n[green]🏆 Best: {best[0]} (Test: {best[1]['test_score']:.4f})")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',      nargs='+', default=['psychology', 'earth_science', 'robotics', 'sustainable_living', 'biology', 'economics'])
    parser.add_argument('--optimizers', nargs='+', default=['labeled_fewshot', 'bootstrap_fewshot', 'bootstrap_fewshot_rs', 'copro', 'miprov2_few'])
    parser.add_argument('--qe_model', type=str, default='openai/gpt-4o-mini')
    parser.add_argument('--rr_model', type=str, default='openai/gpt-4o-mini')
    parser.add_argument('--topk0', type=int, default=100)
    parser.add_argument('--topk1', type=int, default=10)
    parser.add_argument('--double_rr', type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--outdir', type=str, default='./results')
    parser.add_argument('--run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"run_{timestamp}"
    
    return args

def main():
    args = parse_args()
    
    print(f"[green]Enhanced DSPy Optimization")
    print(f"[green]Tasks: {', '.join(args.tasks)}")
    print(f"[green]Optimizers: {', '.join(args.optimizers)}")
    
    # Configure DSPy
    lm = dspy.LM(
        f'{args.qe_model}', 
        api_key=os.getenv("OPENAI_API_KEY" if "openai" in args.qe_model else "GEMINI_API_KEY"),
        temperature=0.7,
        max_retries=3
    )
    dspy.configure(lm=lm)
    
    # Load data
    train_queries, val_queries, test_queries, combined_kb = load_combined_data(
        args.tasks, args.train_ratio, args.val_ratio
    )
    print(f"[green]Train queries: {len(train_queries)}")
    print(f"[green]Val queries:   {len(val_queries)}")
    print(f"[green]Test queries:  {len(test_queries)}")
    
    # Convert to DSPy examples
    train_examples = to_dspy_examples(train_queries)
    val_examples   = to_dspy_examples(val_queries)
    test_examples  = to_dspy_examples(test_queries)
    
    # Create pipeline
    kb = KnowledgeBase(combined_kb)
    pipeline = SingleQueryPipeline(
        knowledge_base=kb,
        qe_model=args.qe_model,
        rr_model=args.rr_model,
        topk0=args.topk0,
        topk1=args.topk1,
        double_rr=args.double_rr
    )
    
    # Establish baselines on all splits
    print(f"\n[cyan]{'='*50}")
    print(f"[cyan]ESTABLISHING BASELINES")
    print(f"[cyan]{'='*50}")
    
    train_eval = Evaluate(devset=train_examples, num_threads=10, display_progress=True, display_table=0)
    val_eval   = Evaluate(devset=val_examples, num_threads=10, display_progress=True, display_table=0)
    test_eval  = Evaluate(devset=test_examples, num_threads=10, display_progress=True, display_table=0)
    
    print(f"[cyan]Evaluating baseline pipeline on train set...")
    baseline_train_score = train_eval(pipeline, metric=retrieval_metric)
    
    print(f"[cyan]Evaluating baseline pipeline on validation set...")
    baseline_val_score = val_eval(pipeline, metric=retrieval_metric)
    
    print(f"[cyan]Evaluating baseline pipeline on test set...")
    baseline_test_score = test_eval(pipeline, metric=retrieval_metric)
    
    print(f"\n[yellow]BASELINE SCORES:")
    print(f"[yellow]Train: {baseline_train_score:.4f}")
    print(f"[yellow]Val:   {baseline_val_score:.4f}")
    print(f"[yellow]Test:  {baseline_test_score:.4f}")
    
    # Run optimizer sweep - MAIN LOGIC
    print(f"\n[blue]{'='*50}")
    print(f"[blue]RUNNING OPTIMIZER SWEEP")
    print(f"[blue]{'='*50}")
    
    optimizers = get_optimizers()
    results = {}
    
    for opt_name in args.optimizers:
        if opt_name not in optimizers:
            print(f"[red]Unknown optimizer: {opt_name}")
            continue
            
        optimizer_class, params = optimizers[opt_name]
        print(f"\n[blue]Running: {opt_name}")
        
        start_time = time.time()
        
        try:
            # Optimize pipeline
            if optimizer_class is None:
                optimized_pipeline = pipeline
                opt_time = 0
            else:
                opt_start = time.time()
                optimizer = optimizer_class(**params)
                optimized_pipeline = optimizer.compile(
                    pipeline, 
                    trainset=train_examples
                )
                opt_time = time.time() - opt_start
            
            # Evaluate on all splits
            train_score = train_eval(optimized_pipeline, metric=retrieval_metric)
            val_score   = val_eval(optimized_pipeline, metric=retrieval_metric)
            test_score  = test_eval(optimized_pipeline, metric=retrieval_metric)
            
            results[opt_name] = {
                'name': opt_name,
                'train_score': train_score,
                'val_score': val_score,
                'test_score': test_score,
                'opt_time': opt_time,
                'total_time': time.time() - start_time,
                'success': True
            }
            
            print(f"[green]✓ {opt_name}: Train={train_score:.4f}, Val={val_score:.4f}, Test={test_score:.4f}")
            
        except Exception as e:
            print(f"[red]✗ {opt_name} failed: {str(e)}")
            results[opt_name] = {
                'name': opt_name,
                'train_score': 0.0,
                'val_score': 0.0,
                'test_score': 0.0,
                'opt_time': 0,
                'total_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    # Print and save results
    print_results(results)
    
    output_file = os.path.join(args.outdir, f"{args.run_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'baselines': {
                'train': baseline_train_score,
                'val': baseline_val_score,
                'test': baseline_test_score
            },
            'data_sizes': {
                'train': len(train_examples),
                'val': len(val_examples), 
                'test': len(test_examples),
                'kb': len(combined_kb)
            },
            'results': results
        }, f, indent=2)
    
    print(f"[green]Results saved to: {output_file}")

if __name__ == "__main__":
    main() 