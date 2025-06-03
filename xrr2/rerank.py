"""
    xrr2.rerank
    
    Reranking functions for xrr2.
"""

import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pandas as pd

from ezprompt import run_batch

def rerank(reranker, kb_dict, queries, all_hits, gt, topk1=10, seed=None):
    """ rerank w/ LLM """

    do_perm = seed is not None
    if do_perm:
        rng = np.random.default_rng(123 + seed)

    # queue jobs
    ltasks = {}
    cands  = {}
    for qid, (query, _hits, _gt) in tqdm(enumerate(zip(queries, all_hits, gt)), desc='Reranking'):
        
        # shortcut for efficiency - we don't bother reranking if we don't have any hits.
        if not any([xx['id'] in _gt for xx in _hits]):
            continue
        
        docs = [kb_dict[xx['id']] for xx in _hits]
        
        if do_perm:
            perm       = rng.permutation(len(docs))
            docs       = [docs[i] for i in perm]
            cands[qid] = [_hits[i] for i in perm]
        else:
            cands[qid] = _hits

        ltasks[qid] = reranker.larun(query=query['query'], docs=docs, topk=topk1, _cache_idx=seed)

    results = run_batch(ltasks, max_calls=64)
    
    for k, v in results.items():
        # fix 1-based indexing
        results[k]['idxs'] = [i - 1 for i in v['idxs']]

    all_hits_ = deepcopy(all_hits)
    for qid in results.keys():
        try:
            all_hits_[qid] = [{'id' : cands[qid][idx]['id'], '_score' : topk1 - i} for i, idx in enumerate(results[qid]['idxs'])]
        except:
            # fallback
            all_hits_[qid] = sorted(all_hits[qid], key=lambda x: x['_score'], reverse=True)[:topk1]
        
    return all_hits_

def rerank_multiple(reranker, kb_dict, queries, all_hits, gt, topk1=10, n_reranks=3):
    """ rerank multiple times and aggregate the results """
    # [QUESTION] What is the variability here?  Seems potentially high
    
    all_hits_ = [
        rerank(reranker, kb_dict, queries, all_hits, gt, topk1=topk1, seed=i) for i in range(n_reranks)
    ]

    return [
        (
            pd.DataFrame(sum(x, []))
                .groupby(['id'])
                .sum()
                .reset_index()
                .to_dict(orient='records')
        )
        for x in zip(*all_hits_)
    ]