#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from functools import lru_cache
from pathlib import Path
import pandas as pd
import click

from utils import load_emb_distance

repo_dir = Path(__file__).resolve().parent.parent
model_dir = repo_dir/"BEAM_engine"/"models"

@click.command()
@click.option("--collector", "-c", type=str, default="wide", help="the name of RouteView collector that the route changes to evaluate are from")
@click.option("--year", "-y", type=int, required=True, help="the year of the route changes monitored, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month of the route changes monitored, e.g., 8")
@click.option("--beam-model", "-b", type=str, required=True, help="the trained BEAM model to use, e.g., 20240801.as-rel2.1000.10.128")
def evaluate_monthly_for(collector, year, month, beam_model):
    collector_result_dir = repo_dir/"routing_monitor"/"detection_result"/collector
    route_change_dir = collector_result_dir/"route_change"
    beam_metric_dir = collector_result_dir/"BEAM_metric"
    beam_metric_dir.mkdir(exist_ok=True, parents=True)

    emb_dir = model_dir/beam_model
    emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(emb_dir, return_emb=True)

    def dtw_d_only_exist(s, t):
        return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])

    for i in route_change_dir.glob(f"{year}{month:02d}*.csv"):
        beam_metric_file = beam_metric_dir/f"{i.stem}.bm.csv"
        if beam_metric_file.exists(): continue

        df = pd.read_csv(i)

        path1 = [s.split(" ") for s in df["path1"].values]
        path2 = [t.split(" ") for t in df["path2"].values]

        metrics = pd.DataFrame.from_dict({
            "diff": [dtw_d(s,t) for s,t in zip(path1, path2)], 
            "diff_only_exist": [dtw_d_only_exist(s,t) for s,t in zip(path1, path2)], 
            "path_d1": [path_d(i) for i in path1],
            "path_d2": [path_d(i) for i in path2],
            "path_l1": [len(i) for i in path1],
            "path_l2": [len(i) for i in path2],
            "head_tail_d1": [emb_d(i[0], i[-1]) for i in path1],
            "head_tail_d2": [emb_d(i[0], i[-1]) for i in path2],
        })
        
        metrics.to_csv(beam_metric_file, index=False)

if __name__ == "__main__":
    evaluate_monthly_for()
