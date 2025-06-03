# XRR2
_Expand -> Retrieve -> Rerank -> Rerank - simple method with strong results on [BRIGHT](https://brightbenchmark.github.io/) benchmark_

## Overview

XRR2 (eXpand -> Retrieve -> Rerank -> Rerank) is a conceptually simple pipeline, similar to pipelines described in the original BRIGHT [paper](https://arxiv.org/pdf/2407.12883).

For each query:
  1) __Expand:__ Use query expansion LLM (`openai/gpt-4o`) to expand the query using [this prompt](./xrr2/prompts/v2_query_expander.md)
  2) __Retrieve:__ topk0=100 results using (modified) [BM25s](https://github.com/jataware/bm25s)
    - Standard BM25 assumes short queries, and thus weights document vectors but does not weight the query vectors.  Since our queries are the relatively lengthy output of the LLM query expansion, we want to weight the query vectors as well.  (This is done in the original [BRIGHT bm25 implementation](https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py#L196)
  3) __Rerank:__ Pass all topk0=100 documents from the previous step to reranking LLM (`gemini/gemini-2.5-flash-preview-04-17`).  Ask for the topk1=10 most relevant documents using [this prompt](./xrr2/prompts/v2_reranker.md)
  4) __Rerank (again):__ Pass the topk1=10 documents from the previous step to the reranking LLM _again_.  Repeat this N=5 times and average the results.
     - This step boosts ndcg@10, but at the time of writing we still get SOTA results even if it is omitted.

## Results

### Methods
- `rr` - Steps 1-3 above
- `rr2`- Steps 1-4 above

```
┏ ━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃                Task ┃ NDCG@10 - rr ┃  NDCG@10 - rr2 ┃
┡ ━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│             biology │      0.62310 │        0.63137 │
│       earth_science │      0.55802 │        0.55440 │
│           economics │      0.37408 │        0.38486 │
│          psychology │      0.54178 │        0.52883 │
│            robotics │      0.35234 │        0.37096 │
│       stackoverflow │      0.36901 │        0.38242 │
│  sustainable_living │      0.43194 │        0.44636 │
│                   - │            - │              - │
│            leetcode │      0.21329 │        0.21861 │
│                pony │      0.33238 │        0.35037 │
│                   - │            - │              - │
│                aops │      0.16690 │        0.15691 │
│ theoremqa_questions │      0.34057 │        0.34403 │
│  theoremqa_theorems │      0.45734 │        0.46188 │
│                   - │            - │              - │
│             __AVG__ │      0.39673 │        0.40258 │
└ ────────────────────┴──────────────┴────────────────┘
```

## Other Thoughts

### Open Questions / Future Work

_Ranking w/ LLMs:_ What is the "right" way to do this?  Pointwise, pairwise or listwise?  Tournaments, sliding window, divide-and-conquer?  Do those methods give consistent results?  How do rank most efficiently?

_Prompt Optimization:_ We re-wrote the query expansion prompt from the original BRIGHT repo, but we didn't touch the reranking prompts.  Could that help?

_Stability:_ Rate limits & structured outputs are annoying, and we're not handling those errors perfectly at the moment.  To successfully run this code, you might have call `xrr2/__main__.py` multiple times.  Previously successful results are cached to disk, so it runs fast, but it is definitely annoying and could be improved w/ better error handling / retrys.

### BRIGHT2.0?

_Choice of Metrics:_ The primary metric for BRIGHT is nDGC@10.  This is a sensible metric if we're retrieving content that will be read by humans - lots of people might only read the first couple of items.  However, if we're retrieving content that will be further processed by an LLM-based system (e.g. in RAG), the _order_ of the top-k don't necessarily matter.  With that in mind, we suggest that BRIGHT should also keep track of best known results as measured by recall-at-10.

_Document Length Bias:_ In some of the BRIGHT datasets, positive documents tend to be substantially longer than distractor documents.  AFAICT, this is an artifact of how the dataset was collected.  Ideally, this could be fixed in a `BRIGHT2.0`.  At a minimum, practitioners should be aware of this feature of the dataset.
 - [TODO] More detailed explanation of this  ...

_Train/Validation Splits:_ We would love to see official train / validation splits for BRIGHT.  Without them - as time goes on - we're likely going to see some (accidental) overfitting.  We would suggest a validation split consisting of 1/3 of the records from 2/3rds of the tasks + _all_ of the records from the remaining 1/3rd of the tasks.  This lets us measure generalization both within and between tasks.
