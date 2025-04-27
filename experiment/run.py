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
event_dir = script_dir/"anomaly_gt"
import sys; sys.path.append(str(root_dir))
from routing_monitor.monitor import Monitor
from data.routeviews.fetch_updates import get_all_collectors, get_archive_list, download_data, load_updates_to_df
from anomaly_detector.utils import load_emb_distance
from anomaly_detector.report_anomaly_routeviews import window
from data.caida_as_org.fetch_data import get_most_recent as as_org_file
from data.caida_as_org.query import load as parse_as_org

event_info = pd.read_csv(event_dir/"event_information.csv", dtype=str)
event_info["ev_code"] = [re.search(r"\$\w+_\{([^}]*)\}\$", i).group(1).replace("\\", "") for i in event_info["name"]]
event_info.set_index("ev_code", inplace=True)

collectors2url = get_all_collectors()
result_dir = script_dir/"detection_result"
result_dir.mkdir(exist_ok=True, parents=True)
report_dir = script_dir/"report"
report_dir.mkdir(exist_ok=True, parents=True)

def detect_route_change(ev_code, ev_category, ev_time, collector="wide", half_span_hours=12, num_workers=1):
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

    if num_workers == 1:
        fpaths = list(map(job, data))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            fpaths = executor.map(job, data)

    mon = Monitor()

    for fpath in fpaths:
        print(f"Detect route changes from {fpath}")
        df = load_updates_to_df(fpath)
        df = df.sort_values(by="timestamp")
        mon.consume(df, detect=True)

    route_change_df = pd.DataFrame.from_records(mon.route_changes)

    # consolidating ground-truth anomalous updates
    print(f"Detect route changes with ground-truth updates")
    df = pd.read_csv(event_dir/f"{ev_code}.csv")
    df["peer-asn"] = [i.split(" ")[0] for i in df["as-path"].values]
    mon.route_changes = []
    mon.consume(df, detect=True)
    gt_df = pd.DataFrame.from_records(mon.route_changes)

    route_change_df = pd.concat([route_change_df, gt_df])
    route_change_df["timestamp"] = route_change_df["timestamp"].astype(str)
    route_change_df.sort_values("timestamp", inplace=True)
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
    tier1 = {"7018", "3320", "3257", "6830", "3356", "2914", "5511", "3491", "6453", "6762", "1299", "12956", "701", "6461"}
    is_both_tier1 = lambda x,y: x in tier1 and y in tier1 

    is_positive = is_leaked if ev_category == "route_leak" else is_hijacked

    gt_a = []
    gt_b = []

    for ts, pf1, pf2, fw, p1, p2 in tqdm(route_change_df[["timestamp", "prefix1", "prefix2", "forwarder", "path1", "path2"]].values, desc="Search GT"):
        p1 = p1.split(" ")
        p2 = p2.split(" ")

        pos1 = is_positive(p1)
        pos2 = is_positive(p2)
        if pos2 != pos1:
            if pos1: p1, p2 = p2, p1
            gt_a.append(dict(
                        timestamp=ts,
                        prefix1=pf1,
                        prefix2=pf2,
                        forwarder=fw,
                        path1=" ".join(p1),
                        path2=" ".join(p2),
                        origin1=p1[-1],
                        origin2=p2[-1],
                        gt_type="anomalous"))
        elif not pos1 and not pos2 and len(p1) == len(p2):
            is_benign = True
            for asn1, asn2 in zip(p1, p2):
                if not is_same_org(asn1, asn2) and not is_both_tier1(asn1, asn2):
                    is_benign = False
                    break
            if is_benign:
                gt_b.append(dict(
                            timestamp=ts,
                            prefix1=pf1,
                            prefix2=pf2,
                            forwarder=fw,
                            path1=" ".join(p1),
                            path2=" ".join(p2),
                            origin1=p1[-1],
                            origin2=p2[-1],
                            gt_type="legitimate"))

    df_a = pd.DataFrame.from_records(gt_a)
    df_b = pd.DataFrame.from_records(gt_b)

    if df_a.empty or df_b.empty:
        print(f"Groun truth inavailable in this collector")
        return None

    print(f"Ground Truth: {df_a.shape[0]} anomalous, {df_b.shape[0]} legitimate")
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

def gen_html(df, ev_code, collector):
    def group_html_checkout(group_id, group):
        timestamp = np.array([t.split(".")[0]
                for t in group["timestamp"].values.astype(str)]).astype(int)
        fmt = "%Y/%m/%d %H:%M:%S"
        start_time = datetime.fromtimestamp(timestamp.min()).strftime(fmt)
        end_time = datetime.fromtimestamp(timestamp.max()).strftime(fmt)

        events = []
        for prefix_key, ev in group.groupby(["prefix1", "prefix2"]):
            route_changes = []
            for _, row in ev.iterrows():
                timestamp = f"<p><b>timestamp:</b> {row['timestamp']}</p>"
                path1 = f"<p><b>path1:</b> {row['path1']}</p>"
                path2 = f"<p><b>path2:</b> {row['path2']}</p>"
                diff = f"<p><b>diff:</b> {row['diff']}</p>"
                culprit = f"<p><b>culprit:</b> {row['culprit']}</p>"

                rc_html = "    <li>\n"
                rc_html+= "    "+timestamp+"\n"
                rc_html+= "    "+path1+"\n"
                rc_html+= "    "+path2+"\n"
                rc_html+= "    "+diff+"\n"
                rc_html+= "    "+culprit+"\n"
                rc_html+= "    </li>"

                route_changes.append(rc_html)

            p0, p1 = prefix_key
            prefix_title = f'<p>{p0} -> {p1}</p>'
            route_change_part = "\n".join(route_changes)

            ev_html = "  <li>\n"
            ev_html+= "  "+prefix_title+"\n"
            ev_html+= "  <ul>\n"
            ev_html+= route_change_part+"\n"
            ev_html+= "  </ul>\n"
            ev_html+= "  </li>\n"

            events.append(ev_html)

        group_title = f"id: {group_id}, start: {start_time}, end: {end_time}, events: {len(events)}, route_changes: {group.shape[0]}"
        events_part = "".join(events)

        html = f'<button class="collapsible">{group_title}</button>\n'
        html+= '<div class="content">\n'
        html+= '<ul>\n'
        html+= events_part
        html+= '</ul>\n'
        html+= '</div>\n'

        return html

    template = open(script_dir/"report_template.html", "r").read()
    if df is None:
        html = template.replace("REPLACE_WITH_SECTIONS", "<p>No events reported.</p>")
        html = html.replace("REPLACE_WITH_TITLE", f"Report-{ev_code} (collector {collector})")

    else:
        sections = []
        for group_id, group in df.groupby("group_id"):
            html = group_html_checkout(group_id, group)
            sections.append(html)
        html = template.replace("REPLACE_WITH_SECTIONS", "\n".join(sections))
        html = html.replace("REPLACE_WITH_TITLE", f"Report-{ev_code} (collector {collector})")

    open(report_dir/f"report_{collector}_{ev_code}.html", "w").write(html)

def run(ev_code, collector):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib as mpl

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ev_latex, ev_category, ev_time, hijk_as = event_info.loc[ev_code][["name", "category", "start_time", "hijack_as"]].values

    ev_time = datetime.strptime(ev_time, "%Y-%m-%dT%H:%M:%S")
    df = detect_route_change(ev_code, ev_category, ev_time, collector=collector)

    df = search_groundtruth(df, ev_code, ev_category, ev_time, hijk_as, collector)
    if df is None: return
    df = evaluate_path_diff(df, ev_time)

    df_a = df.loc[df["gt_type"]=="anomalous"]
    df_b = df.loc[df["gt_type"]=="legitimate"]

    info, df_ev = window(df_b, df_a)
    gen_html(df_ev, ev_code, collector)

    ax.plot([], [], marker="D", color="orange", label="anomalous")
    ax.plot([], [], marker="o", color="blue", label="legitimate")
    ax.legend(fontsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    mk_kwargs = dict(s=6, zorder=10)
    line_kwargs = dict(lw=1.2, zorder=5)

    plot_ecdf(df_a["diff"].values, ax, mk_kwargs=dict(color="orange", marker="D", **mk_kwargs), line_kwargs=dict(color="orange", **line_kwargs))
    plot_ecdf(df_b["diff"].values, ax, mk_kwargs=dict(color="dodgerblue", marker="o", **mk_kwargs), line_kwargs=dict(color="dodgerblue", **line_kwargs))
    ax.set_ylim((0, 1))

    ax.axvspan(df_b["diff"].values.min(), df_b["diff"].values.max(), edgecolor=None, facecolor="dodgerblue", alpha=0.1, zorder=0)
    ax.axvspan(df_a["diff"].values.min(), df_a["diff"].values.max(), edgecolor=None, facecolor="orange", alpha=0.1, zorder=0)

    ax.set_xlabel("path difference score", fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True)
    ax.set_title(ev_latex, fontsize=16)

    fig.tight_layout()
    fig.savefig(report_dir/f"{collector}.{ev_code}.ecdf.pdf", bbox_inches="tight")

@click.command()
@click.option("--collector", "-c", type=str, default="wide", help="the name of RouteView collector to use")
@click.option("--ev-code", "-e", type=click.Choice(sorted(event_info.index.tolist()+["all"])), default="all", help="the code name of the event to analyze")
def main(ev_code, collector):
    if ev_code == "all":
        for ev_code in event_info.index.tolist():
            run(ev_code, collector)
    else:
        run(ev_code, collector)

if __name__ == "__main__":
    main()
