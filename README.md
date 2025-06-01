# ezbright
_Expand -> Retrieve -> Rerank - simple method with strong results on BRIGHT benchmark_

---

## Methods
- `rr` - Expand -> Retrieve -> Rerank Top100
- `rr2` - Expand -> Retrieve -> Rerank Top100 -> Rerank Top10 averaged over 5 runs

Details coming soon ...

## Results
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
│ theoremqa_questions │      0.34057 │                │
│  theoremqa_theorems │      0.45734 │        0.46188 │
│                   - │            - │              - │
│             __AVG__ │      0.39673 │        0.40791 │
└ ────────────────────┴──────────────┴────────────────┘
```

