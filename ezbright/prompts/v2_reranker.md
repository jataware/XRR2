The following passages are related to query: {QUERY}

{DOC_STR}First identify the essential problem in the query.
Think step by step to reason about why each document is relevant or irrelevant.
Rank these passages based on their relevance to the query.
Please output the ranking result of passages as a list, where the first element is the id of the most relevant passage, the second element is the id of the second most element, etc.
Please strictly follow the format to output a list of {TOPK} ids corresponding to the most relevant {TOPK} passages, sorted from the most to least relevant passage. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format like

```json
[... integer ids here ...]
```