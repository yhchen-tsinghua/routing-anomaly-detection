#!/usr/bin/env python
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import click

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
import sys; sys.path.append(str(root_dir))
from routing_monitor.monitor import Monitor
from data.routeviews.fetch_updates import get_all_collectors, get_archive_list, download_data, load_updates_to_df
from anomaly_detector.utils import load_emb_distance
from data.caida_as_org.fetch_data import get_most_recent as as_org_file
from data.caida_as_org.query import load as parse_as_org

event_info = pd.read_csv(script_dir/"event_information.csv", dtype=str)
event_info["ev_code"] = [re.search(r"\$\w+_\{([^}]*)\}\$", i).group(1).replace("\\", "") for i in event_info["name"]]
event_info.set_index("ev_code", inplace=True)

collectors2url = get_all_collectors()
result_dir = script_dir/"detection_result"
result_dir.mkdir(exist_ok=True, parents=True)

def detect_route_change(ev_code, ev_category, ev_time, collector="wide", half_span_hours=12, num_workers=24):
    print(f"Loading event data: {ev_code} ({ev_category})")
    print(f"Analyzed time span: {ev_time} ({chr(0x00B1)}{half_span_hours} hours)")
    print(f"Collector: {collector}")

    save_path = result_dir/f"{collector}.{ev_code}.csv"
    if save_path.exists():
        print(f"Load from {save_path}")
        return pd.read_csv(save_path)

    d0 = ev_time - timedelta(hours=half_span_hours)
    d1 = ev_time + timedelta(hours=half_span_hours)
    data = get_archive_list(collector, collectors2url, d0, d1)
    job = lambda url: download_data(url, collector)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        fpaths = executor.map(job, data)

    mon = Monitor()

    for fpath in fpaths:
        print(f"Detect route changes from {fpath}")
        df = load_updates_to_df(fpath)
        df = df.sort_values(by="timestamp")
        mon.consume(df, detect=True)

    route_change_df = pd.DataFrame.from_records(mon.route_changes)
    route_change_df.to_csv(save_path, index=False)
    print(f"{route_change_df.shape[0]} route changes detected")
    print(f"Saved to {save_path}")
    return route_change_df

def search_groundtruth(route_change_df, ev_code, ev_category, ev_time, hijk_as, collector="wide"):
    hijk_as = hijk_as.split(", ")

    def is_hijacked(aspath):
        for i in aspath:
            if i in hijk_as:
                return True
        return False

    def is_leaked(aspath):
        return len([i for i in hijk_as if i in aspath]) == len(hijk_as)

    get_same_org, get_asn_country = load_as_org(ev_time.strftime("%Y%m01"))
    is_same_org = lambda x,y: get_same_org(x,y) is not None

    is_positive = is_leaked if ev_category == "route_leak" else is_hijacked

    gt_a = []
    gt_b = []

    for fw, p1, p2 in tqdm(route_change_df[["forwarder", "path1", "path2"]].values, desc="Search GT"):
        p1 = p1.split(" ")
        p2 = p2.split(" ")

        pos1 = is_positive(p1)
        pos2 = is_positive(p2)
        if pos2 != pos1:
            if pos1: p1, p2 = p2, p1
            gt_a.append(dict(
                        forwarder=fw,
                        path1=" ".join(p1),
                        path2=" ".join(p2),
                        origin1=p1[-1],
                        origin2=p2[-1],
                        gt_type="anomalous"))
        elif not pos1 and not pos2:
            if p1[-1] != p2[-1] and p1[:-1] == p2[:-1] and is_same_org(p1[-1], p2[-1]):
                gt_b.append(dict(
                            forwarder=fw,
                            path1=" ".join(p1),
                            path2=" ".join(p2),
                            origin1=p1[-1],
                            origin2=p2[-1],
                            gt_type="legitimate"))

    def keep_multiple(df, th=5, cols=["forwarder", "origin1", "origin2"]):
        counts = df.groupby(cols).size()
        return df.merge(counts[counts > th].reset_index(),
                on=cols).drop_duplicates(subset=cols)

    df_a = keep_multiple(pd.DataFrame.from_records(gt_a))
    df_b = keep_multiple(pd.DataFrame.from_records(gt_b))

    print(f"Grount Truth: {df_a.shape[0]} anomalous, {df_b.shape[0]} legitimate")
    save_path = result_dir/f"{collector}.{ev_code}.gt.csv"
    df = pd.concat([df_a, df_b])
    df.to_csv(save_path, index=False)
    print(f"Save groundtruth to {save_path}")
    return df

def choose_as_rel(ev_time):
    date = int(ev_time.strftime("%Y%m01"))
    if date == 20150301: date = 20150201 # 20150301 is flawed
    return f"{date}.as-rel{'' if date < 20151201 else '2'}"

def load_as_org(time):
    time, fpath = as_org_file(time)
    as_info, org_info = parse_as_org(time)

    def get_org_id(asn):
        if asn not in as_info:
            return asn
        info = as_info[asn]
        return info["opaque_id"] if info["opaque_id"] != "" else info["org_id"]

    def get_same_org(asn1, asn2):
        if get_org_id(asn1) == get_org_id(asn2):
            return get_org_id(asn1)
        else:
            return None

    def get_asn_country(asn):
        org_id = get_org_id(asn)
        if org_id in org_info:
            return org_info[org_id]["country"]

    return get_same_org, get_asn_country

def evaluate_path_diff(df, ev_time, epoches=1000, Q=10, dimension=128):
    as_rel = choose_as_rel(ev_time)
    model_name = f"{as_rel}.{epoches}.{Q}.{dimension}"
    train_dir = root_dir/"BEAM_engine"/"models"/model_name
    assert train_dir.exists(), f"Train BEAM model {model_name} first."

    emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(train_dir, return_emb=True)

    def dtw_d_only_exist(s, t):
        return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])

    path1 = [s.split(" ") for s in df["path1"].values]
    path2 = [t.split(" ") for t in df["path2"].values]

    diff = np.array([dtw_d_only_exist(s,t) for s,t in zip(path1, path2)])
    path_d1 = np.array([path_d(i) for i in path1])
    df["diff"] = diff/path_d1

    print("Legitimate path difference:")
    print(df.loc[df["gt_type"] == "legitimate"]["diff"].describe())
    print("Anomalous path difference:")
    print(df.loc[df["gt_type"] == "anomalous"]["diff"].describe())
    return df

def plot_ecdf(data, ax, mk_kwargs={}, horiz=True, verti=True, line_kwargs={}):
    x, n = np.unique(data, return_counts=True)
    y = n.cumsum()/len(data)
    
    if mk_kwargs:
        ax.scatter(x, y, **mk_kwargs)

    if horiz and verti:
        _x = [j for i in zip(x,x) for j in i]
        _y = [0] + [j for i in zip(y,y) for j in i][:-1]
        ax.plot(_x, _y, **line_kwargs)
    else:
        if horiz:
            lines = [((x[i], y[i]), (x[i+1], y[i])) for i in range(len(x)-1)]
            ln_coll = LineCollection(lines, **line_kwargs)
            ax.add_collection(ln_coll)

        if verti:
            lines = [((x[i], (y[i-1]) if i>0 else 0), (x[i], y[i])) for i in range(len(x))]
            ln_coll = LineCollection(lines, **line_kwargs)
            ax.add_collection(ln_coll)

@click.command()
@click.option("--collector", "-c", type=str, default="wide", help="the name of RouteView collector to use")
@click.option("--ev-code", "-e", type=click.Choice(sorted(event_info.index.tolist())), default="pakistan", help="the code name of the event to analyze")
def main(ev_code, collector):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib as mpl

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ev_latex, ev_category, ev_time, hijk_as = event_info.loc[ev_code][["name", "category", "start_time", "hijack_as"]].values

    ev_time = datetime.strptime(ev_time, "%Y-%m-%dT%H:%M:%S")
    df = detect_route_change(ev_code, ev_category, ev_time, collector=collector)
    df = search_groundtruth(df, ev_code, ev_category, ev_time, hijk_as, collector=collector)
    df = evaluate_path_diff(df, ev_time)
    df_a = df.loc[df["gt_type"]=="anomalous"]
    df_b = df.loc[df["gt_type"]=="legitimate"]

    ax.plot([], [], marker="D", color="orange", label="anomalous")
    ax.plot([], [], marker="o", color="blue", label="legitimate")
    ax.legend(fontsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    mk_kwargs = dict(s=6, zorder=10)
    line_kwargs = dict(lw=1.2, zorder=5)

    plot_ecdf(df_a["diff"].values, ax, mk_kwargs=dict(color="orange", marker="D", **mk_kwargs), line_kwargs=dict(color="orange", **line_kwargs))
    plot_ecdf(df_b["diff"].values, ax, mk_kwargs=dict(color="dodgerblue", marker="o", **mk_kwargs), line_kwargs=dict(color="dodgerblue", **line_kwargs))
    ax.set_xlim((0, None))
    ax.set_ylim((0, 1))

    ax.axvspan(df_b["diff"].values.min(), df_b["diff"].values.max(), edgecolor=None, facecolor="dodgerblue", alpha=0.1, zorder=0)
    ax.axvspan(df_a["diff"].values.min(), df_a["diff"].values.max(), edgecolor=None, facecolor="orange", alpha=0.1, zorder=0)

    ax.set_xlabel("path difference score", fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True)
    ax.set_title(ev_latex, fontsize=16)

    fig.tight_layout()
    fig.savefig(f"{collector}.{ev_code}.ecdf.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
