#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from utils import approx_knee_point, event_aggregate
import json
import pandas as pd
import numpy as np
import click

repo_dir = Path(__file__).resolve().parent.parent

def load_monthly_data(year, month, preprocessor=lambda df: df):
    route_change_files = sorted(route_change_dir.glob(f"{year}{month:02d}*.csv"))
    beam_metric_files = sorted(beam_metric_dir.glob(f"{year}{month:02d}*.bm.csv"))
    datetimes = [i.stem.replace(".","")[:-2] for i in route_change_files]

    bulk_datetimes, bulk_indices = np.unique(datetimes, return_index=True)
    bulk_ranges = zip(bulk_indices, bulk_indices[1:].tolist()+[len(datetimes)])

    def load_one_bulk(i,j):
        rc_df = pd.concat(list(map(pd.read_csv, route_change_files[i:j])))
        bm_df = pd.concat(list(map(pd.read_csv, beam_metric_files[i:j])))
        return pd.concat([rc_df, bm_df], axis=1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        bulks = list(executor.map(
                    lambda x: preprocessor(load_one_bulk(*x)), bulk_ranges))

    return bulk_datetimes, bulks

def metric_threshold(df, metric_col):
    values = df[metric_col]
    mu = np.mean(values)
    sigma = np.std(values)
    metric_th = mu+4*sigma

    print("reference metric: ")
    print(values.describe())
    print(f"metric threshold: {metric_th}")

    return metric_th

def forwarder_threshold(df, event_key):
    route_changes = tuple(df.groupby(event_key))
    forwarder_num = [len(j["forwarder"].unique()) for _, j in route_changes]
    forwarder_th, cdf = approx_knee_point(forwarder_num)

    print("reference forwarder: ")
    print(pd.Series(forwarder_num).describe())
    print(f"forwarder threshold: {forwarder_th}")

    return forwarder_th

def window(df0, df1, # df0 for reference, df1 for detection
        metric="diff", event_key=["prefix1", "prefix2"],
        dedup_index=["prefix1", "prefix2", "forwarder", "path1", "path2"]):

    if dedup_index is not None:
        df0 = df0.drop_duplicates(dedup_index, keep="first", inplace=False, ignore_index=True)

    with pd.option_context("mode.use_inf_as_na", True):
        df0 = df0.dropna(how="any")

    metric_th = metric_threshold(df0, metric)
    forwarder_th = forwarder_threshold(df0, event_key)

    events = {}
    for key,ev in tuple(df1.groupby(event_key)):
        if len(ev["forwarder"].unique()) <= forwarder_th: continue

        ev_sig = ev.sort_values(metric, ascending=False).drop_duplicates("forwarder")
        ev_anomaly = ev_sig.loc[ev_sig[metric]>metric_th]
        if ev_anomaly.shape[0] <= forwarder_th: continue

        events[key] = ev_anomaly

    if events:
        _, df = event_aggregate(events)
        n_alarms = len(df['group_id'].unique())
    else:
        df = None
        n_alarms = 0

    info = dict(
        metric=metric,
        event_key=event_key,
        metric_th=float(metric_th),
        forwarder_th=int(forwarder_th),
        n_raw_events=len(events),
        n_alarms=n_alarms,
    )

    return info, df

@click.command()
@click.option("--collector", "-c", type=str, default="wide", help="the name of RouteView collector to detect anomalies")
@click.option("--year", "-y", type=int, required=True, help="the year of the route changes monitored, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month of the route changes monitored, e.g., 8")
def report_alarm_monthly(collector, year, month):
    collector_result_dir = repo_dir/"routing_monitor"/"detection_result"/collector
    route_change_dir = collector_result_dir/"route_change"
    beam_metric_dir = collector_result_dir/"BEAM_metric"
    reported_alarm_dir = collector_result_dir/"reported_alarms"/f"{year}{month:02d}"
    reported_alarm_dir.mkdir(parents=True, exist_ok=True)

    def preprocessor(df):
        df["diff_balance"] = df["diff"]/(df["path_d1"]+df["path_d2"])
        return df

    datetimes, bulks = load_monthly_data(year, month, preprocessor)
    indices = np.arange(len(bulks))
    infos = []

    for i, j in list(zip(indices[:-1], indices[1:])):
        info = dict(d0=datetimes[i], d1=datetimes[j])
        _info, df = window(bulks[i], bulks[j], metric="diff_balance")
        info.update(**_info)

        if df is None:
            info.update(save_path=None)
        else: 
            save_path = reported_alarm_dir/f"{datetimes[i]}_{datetimes[j]}.alarms.csv"
            df.to_csv(save_path, index=False)
            info.update(save_path=str(save_path))

        infos.append(info)

    json.dump(infos, open(reported_alarm_dir/f"info_{year}{month:02d}.json", "w"), indent=2)

if __name__ == "__main__":
    report_alarm_monthly()
