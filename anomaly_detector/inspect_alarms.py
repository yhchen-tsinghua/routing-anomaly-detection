#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path

metric = "diff_balance" # NOTE

reported_alarm_dir = Path(f"/opt/detection_result/reported_alarms/{metric}")

info = json.load(open(reported_alarm_dir/"info.json", "r"))

print(f"load from: {reported_alarm_dir}")

n_alarms = pd.Series([i["n_alarms"] for i in info])
print(f"#alarms in each one-hour window:")
print(n_alarms.describe())

for i,j in zip(*np.unique(n_alarms, return_counts=True)):
    print(f"#alarm={i}: {j}")

windows_top10 = [info[idx]["d1"] for idx in np.argsort(n_alarms)[::-1][:10]]
print(f"windows with most alarms: {windows_top10}")
print(f"total alarms: {np.sum(n_alarms)}")

info = {i["d1"]: i for i in info}

def inspect(ymdh):
    i = info[ymdh]
    print(json.dumps(i, indent=2))

    if i["save_path"] is None:
        print("no alarms reported in this window")
        return

    df = pd.read_csv(i["save_path"])

    for group_id, group in tuple(df.groupby("group_id")):
        print(f"alarm_id: {group_id}")
        for prefix_key, ev in tuple(group.groupby(["prefix1", "prefix2"])):
            print(f"* {' -> '.join(prefix_key)}")
            for _, row in ev.iterrows():
                print(f"  path1: {row['path1']}")
                print(f"  path2: {row['path2']}")
                print(f"  diff={row[metric]}")
                print(f"  culprit={row['culprit']}")
                print()
        input("..Enter to next")

while True:
    ymdh = input("Enter ymdh: ")
    try: inspect(ymdh)
    except KeyError as e:
        print(e)
        continue
