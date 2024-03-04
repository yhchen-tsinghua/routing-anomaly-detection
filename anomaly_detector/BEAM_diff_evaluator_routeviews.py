#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from functools import lru_cache
from pathlib import Path
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed

from utils import load_emb_distance

repo_dir = Path(__file__).resolve().parent.parent
route_change_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"route_change"
beam_metric_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"BEAM_metric"
model_dir = repo_dir/"BEAM_engine"/"models"

beam_metric_dir.mkdir(exist_ok=True, parents=True)

train_dir = model_dir/"20230201.as-rel2.local.1000.10.128"
emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(train_dir, return_emb=True)
def dtw_d_only_exist(s, t):
    return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])

def evaluate_monthly_for(ym):
    for i in route_change_dir.glob(f"{ym}*.csv"):
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

Parallel(backend="multiprocessing", n_jobs=7, verbose=10)(
        delayed(evaluate_monthly_for)(f"2023{m:02}") for m in range(2,9))
