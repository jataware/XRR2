"""
    xrr2.official

    Official evaluation script for xrr2.
"""

import json
import pytrec_eval
import pandas as pd
from glob import glob
from datasets import load_dataset as hf_load_dataset
from rich.table import Table
from rich.console import Console

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    # print(output)
    return output

if __name__ == "__main__":
    metric_names = ["NDCG@10"]
    key          = 'gold_ids'
    all_examples = hf_load_dataset('xlangai/bright', 'examples',cache_dir='.cache')

    all_metrics = []
    for result_path in glob("./results/*_results.json"):
        try:
            task    = result_path.split("/")[-1].split('__')[0]
            setting = result_path.split("/")[-1].split('__')[1].replace('_results.json', '')
            
            scores   = json.load(open(result_path))
            examples = all_examples[task]

            ground_truth = {}
            for e in examples:
                ground_truth[e['id']] = {}
                for gid in e[key]:
                    ground_truth[e['id']][gid] = 1
                for did in e['excluded_ids']:
                    assert not did in scores[e['id']]
                    assert not did in ground_truth[e['id']]
            
            metrics = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
            metrics['_task']    = task
            metrics['_setting'] = setting

            all_metrics.append(metrics)
        except:
            print(f"ERROR @ {result_path}")
    

    # add averages
    avg = pd.DataFrame(all_metrics).drop(columns=['_task']).groupby('_setting').mean().reset_index()
    avg['_task'] = '__AVG__'
    all_metrics += avg.to_dict(orient='records')

    # --
    # Print

    console = Console()
    table = Table(border_style="green", width=console.width)
    table.add_column("Task", justify="right")
    for metric_name in metric_names:
        for setting in ['qe', 'rr', 'rr2']:
            table.add_column(f"{metric_name} - {setting}", justify="right")

    # Group metrics by task
    task_metrics = {}
    for metrics in all_metrics:
        task = metrics['_task']
        if task not in task_metrics:
            task_metrics[task] = {}
        task_metrics[task][metrics['_setting']] = metrics

    TASK_ORDER = [
        "biology",
        "earth_science",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "leetcode",
        "pony",
        "aops",
        "theoremqa_questions",
        "theoremqa_theorems",
        "__AVG__",
    ]

    # Add rows for each task
    for task in TASK_ORDER:
        row = [task]
        for metric_name in metric_names:
            for setting in ['qe', 'rr', 'rr2']:
                if setting in task_metrics[task]:
                    row.append(f"{task_metrics[task][setting][metric_name]:.5f}")
                else:
                    row.append("")
        
        if task == '__AVG__':
            table.add_row(*row, style="bold yellow")
        elif task in ['sustainable_living', 'pony', 'theoremqa_theorems']:
            table.add_row(*row)
            table.add_row(*(['-'] * len(row)))
        else:
            table.add_row(*row)

    console.rule(style="green")
    console.rule(f"[bold green]Retrieval Results", style="green")
    console.print(table)
    console.rule(style="green")