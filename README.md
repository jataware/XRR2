# XRR2
_Expand -> Retrieve -> Rerank -> Rerank - simple method with strong results on [BRIGHT](https://brightbenchmark.github.io/) benchmark_

## Overview

XRR2 (eXpand -> Retrieve -> Rerank -> Rerank) is a conceptually simple pipeline, similar to pipelines described in the original BRIGHT [paper](https://arxiv.org/pdf/2407.12883).

For each query:
  - Use an LLM (gpt-4o) to __expand__ the query using [this prompt](./ezbright/prompts/v2_query_expander.md)
  - 

## Results

### Methods
- `rr` - Expand -> Retrieve -> Rerank Top100
- `rr2` - Expand -> Retrieve -> Rerank Top100 -> Rerank Top10 averaged over 5 runs

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

### BRIGHT2.0?

_Choice of Metrics:_ The primary metric for BRIGHT is nDGC@10.  This is a sensible metric if we're retrieving content that will be read by humans - lots of people might only read the first couple of items.  However, if we're retrieving content that will be further processed by an LLM-based system (e.g. in RAG), the _order_ of the top-k don't necessarily matter.  With that in mind, we suggest that BRIGHT should also keep track of best known results as measured by recall-at-10.

_Document Length Bias:_ In some of the BRIGHT datasets, positive documents tend to be substantially longer than distractor documents.  AFAICT, this is an artifact of how the dataset was collected.  Ideally, this could be fixed in a `BRIGHT2.0`.  At a minimum, practitioners should be aware of this feature of the dataset.
 - [TODO] More detailed explanation of this  ...

_Train/Validation Splits:_ We would love to see official train / validation splits for BRIGHT.  Without them - as time goes on - we're likely going to see some (accidental) overfitting.  We would suggest a validation split consisting of 1/3 of the records from 2/3rds of the tasks + _all_ of the records from the remaining 1/3rd of the tasks.  This lets us measure generalization both within and between tasks.
