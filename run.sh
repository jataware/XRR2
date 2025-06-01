#!/bin/bash

# run.sh

# --
# Install

# pixi install
# pixi shell

# --
# Run all tasks

export HF_DATASETS_OFFLINE=1
function run_task() {
    TASK=$1
    QE_MODEL=gpt-4o
    RR_MODEL=gemini/gemini-2.5-flash-preview-04-17
    python -m ezbright \
        --topk0     100        \
        --qe_model  $QE_MODEL  \
        --rr_model  $RR_MODEL  \
        --double_rr 5          \
        --task      $TASK
}

run_task biology
run_task earth_science
run_task economics
run_task psychology
run_task robotics
run_task stackoverflow
run_task sustainable_living

run_task leetcode
run_task pony

run_task aops
run_task theoremqa_questions
run_task theoremqa_theorems


# --
# Final scoring

python -m ezbright.official

# [TODO] error handling, so it runs solidly w/o restart 

# Ideas
# - repeated pairwise comparison of top 10
# - prompt optimization (w/ dspy ... need to run on a small sample of prompts?)
#  + did poor mans version of this - asked gemini-2.5-pro for good prompts and tested 
#    them with recal@100 on biology.
# - for the math tasks, may want to do query expansion differently ...
# - try reversing the order of the docs in the reranker.  or repeating them?

# - theoremqa_questions has some issue w/ query expansion ... the upper bound is below other systems scores
#   - bm25s tokenizer?  

# What does 1 point gain in NDCG mean?  Is this stupid?