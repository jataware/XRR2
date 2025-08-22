"""
    xrr2.utils

    Utility functions for xrr2.
"""

import pytrec_eval
import numpy as np
from datasets import load_dataset as hf_load_dataset

# --
# Data

TASK_CHOICES = [
    'biology',
    'earth_science',
    'economics',
    'psychology',
    'robotics',
    'stackoverflow',
    'sustainable_living',
    'pony',
    'aops',
    'leetcode',
    'theoremqa_theorems',
    'theoremqa_questions'
]

def load_bright_multi(tasks, pre_reasoning=None, long_context=False, cache_dir='.cache'):
    for task in tasks:
        assert task in TASK_CHOICES, f"Task {task} not in {TASK_CHOICES}"
    
    # Queries
    all_queries = hf_load_dataset(
        'xlangai/bright', 
        f"{pre_reasoning}_reason" if (pre_reasoning is not None) else 'examples', 
        cache_dir=cache_dir,
    )

    # Knowledge Base
    all_kbs = hf_load_dataset(
        'xlangai/bright', 
        'long_documents' if long_context else 'documents', 
        cache_dir=cache_dir,
    )

    # Format
    out = {}
    for task in tasks:
        queries = [{
            'id'           : xx['id'],
            'query'        : xx['query'],
            'excluded_ids' : sorted(set([eid for eid in xx['excluded_ids'] if eid != 'N/A'])), # Why isn't this always unique (aops)?
            '_gt_ids'      : xx['gold_ids_long' if long_context else 'gold_ids'],
            '_gt_ans'      : xx.get('gold_answer', None)
        } for xx in all_queries[task]]
        
        kb = [{
            'id'  : xx['id'],
            'doc' : xx['content']
        } for xx in all_kbs[task]]
        
        out[task] = {
            'queries' : queries,
            'kb'      : kb
        }

    return out


def load_bright(task, pre_reasoning=None, long_context=False, cache_dir='.cache'):
    return load_bright_multi([task], pre_reasoning, long_context, cache_dir)[task]

# --
# Metrics

def compute_metrics(results, gt, k_values=[1, 5, 10, 25, 50, 100], ub=False):
    # results: list[list[dict]] :=  {"id" : , "_score" : }
    # gt: list[list[str]] := list of list of ids
    
    if ub:
        results = [
            [
                {'id' : xx['id'], '_score' : xx['_score'] + (xx['id'] in _gt) * 1000000} 
                for xx in results
            ] for results, _gt in zip(results, gt)
        ]
    
    
    results = {
        f'{idx:04d}' : {
            xx['id'] : xx['_score'] for xx in results[idx]
        }
        for idx in range(len(results))
    }
    
    gt = {
        f'{idx:04d}' : {
            gt_id : 1 for gt_id in gt[idx]
        }
        for idx in range(len(gt))
    }
    
    # Run evaluation
    k_str     = ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        gt, 
        {
            # f"map_cut.{k_str}",
            f"ndcg_cut.{k_str}",
            f"recall.{k_str}",
            # f"P.{k_str}",
            # "recip_rank"
        }
    )
    scores = evaluator.evaluate(results)
    
    # Average over all queries
    scores_avg = {
        metric: float(round(np.mean([scores[qid][metric] for qid in scores]), 5))
        for metric in scores[next(iter(scores))]
    }
    return scores_avg
