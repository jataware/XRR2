#!/bin/bash

# run.sh

# --
# Install

# pixi install
# pixi shell

# --
# Run all tasks

# Good to turn this on after the first time, so that we don't keep on 
# hitting the internet.
# export HF_DATASETS_OFFLINE=1
function run_task() {
    TASK=$1
    QE_MODEL=gpt-4o
    RR_MODEL=gemini/gemini-2.5-flash-preview-04-17
    python -m xrr2 \
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

python -m xrr2.official
