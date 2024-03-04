#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess 
import calendar

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from anomaly_detector.utils import event_aggregate

def main(month, metric):
    repo_dir = Path(__file__).resolve().parent.parent
    reported_alarm_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"reported_alarms"/metric/f"2023{month:02}"
    route_change_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"route_change"
    info = json.load(open(reported_alarm_dir/f"info_2023{month:02}.json", "r"))
    flags_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"reported_alarms"/metric/f"2023{month:02}.flags"

    summary_dir = Path(__file__).resolve().parent/"summary_output"
    summary_dir.mkdir(parents=True, exist_ok=True)

    html_dir = Path(__file__).resolve().parent/"html"

    def has_flag(v):
        return v != "-"
    v_flag = np.vectorize(has_flag)

    def invalid_asn(v):
        return v == "invalid_asn"
    v_invalid_asn = np.vectorize(invalid_asn)

    def invalid_len(v):
        return v == "invalid_length"
    v_invalid_len = np.vectorize(invalid_len)

    def valid(v):
        return v == "valid"
    v_valid = np.vectorize(valid)

    def summary():
        global_group_id = 0
        dfs = []
        for i in info:
            if i["save_path"] is None: continue
            df = pd.read_csv(i["save_path"])
            flags = pd.read_csv(flags_dir/f"{Path(i['save_path']).stem}.flags.csv")
            df = pd.concat([df, flags], axis=1)

            # highly possible origin hijack
            anomaly_t1 = df["origin_change"] \
                            & (~v_flag(df["origin_same_org"])) \
                            & (v_invalid_asn(df["origin_rpki_1"])
                                ^ v_invalid_asn(df["origin_rpki_2"]))

            # highly possible route leak
            anomaly_t2 = v_flag(df["non_valley_free_1"]) \
                            | v_flag(df["non_valley_free_2"])

            # highly possible path manipulation
            anomaly_t3 = v_flag(df["reserved_path_1"]) \
                            | v_flag(df["reserved_path_2"]) \
                            | v_flag(df["none_rel_1"]) \
                            | v_flag(df["none_rel_2"]) \
                            | np.isinf(df["diff"])
            exception_t3 = (~v_flag(df["reserved_path_1"])) \
                            & (~v_flag(df["reserved_path_2"])) \
                            & (v_flag(df["none_rel_1"]) \
                                | v_flag(df["none_rel_2"]))

            # highly possible ROA misconfiguration
            anomaly_t4 = v_flag(df["origin_same_org"]) \
                            & (v_invalid_len(df["origin_rpki_1"])
                                | v_invalid_len(df["origin_rpki_2"])
                                | (v_invalid_asn(df["origin_rpki_1"])
                                    ^ v_invalid_asn(df["origin_rpki_2"])))

            # highly possible benign MOAS
            benign_t1 = df["origin_change"] \
                            & (v_flag(df["origin_same_org"]) 
                                | v_flag(df["origin_connection"])
                                | (v_valid(df["origin_rpki_1"])
                                    & v_valid(df["origin_rpki_2"])))

            # highly possible AS prepending
            benign_t2 = (~df["origin_change"]) \
                            & (has_flag(df["as_prepend_1"])
                                ^ has_flag(df["as_prepend_2"]))

            # highly possible multi-homing
            benign_t3 = v_flag(df["origin_different_upstream"])

            # no any sign of anomaly
            benign_t4 = (~df["detour_country"]) \
                            & v_valid(df["origin_rpki_1"]) \
                            & v_valid(df["origin_rpki_2"])

            # possible false alarms due to the nature of diff computation
            benign_t5 = (df["path_l1"]+df["path_l2"])/2 <= 3

            # possible prefix transfer
            benign_t6 = df["path1_in_path2"]


            df["a1"] = anomaly_t1
            df["a2"] = anomaly_t2
            df["a3"] = anomaly_t3
            df["a4"] = anomaly_t4
            df["b1"] = benign_t1
            df["b2"] = benign_t2
            df["b3"] = benign_t3
            df["b4"] = benign_t4
            df["b5"] = benign_t5
            df["b6"] = benign_t6

            anomaly = anomaly_t1 | anomaly_t2 | anomaly_t3 | anomaly_t4
            benign = benign_t1 | benign_t2 | benign_t3 | benign_t4 | benign_t5 | benign_t6

            df["pattern"] = "unknown"
            df.loc[benign, ["pattern"]] = "benign"
            df.loc[anomaly, ["pattern"]] = "anomaly"

            df = df.loc[anomaly | (~benign)] # post-filtering
            df = df.loc[(~exception_t3)|anomaly_t1|anomaly_t2|anomaly_t4]

            event_key = i["event_key"]
            forwarder_th = i["forwarder_th"]

            events = {}
            for key,ev in df.groupby(event_key): # re-grouping and filtering
                if ev.shape[0] <= forwarder_th: continue
                events[key] = ev

            if events:
                _, df = event_aggregate(events)
                n_alarms = len(df["group_id"].unique())
                assert np.max(df["group_id"]) == n_alarms-1, f"{np.max(df['group_id'])}, {n_alarms-1}"
                df["group_id"] += global_group_id
                global_group_id += n_alarms
                dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(summary_dir/f"alarms_after_post_process_2023{month:02}.csv", index=False)
        return df

    df = summary()

    def reason(tag, row):
        if tag == "a1":
            fields = ["origin_rpki_1", "origin_rpki_2"]
        elif tag == "a2":
            fields = ["non_valley_free_1", "non_valley_free_2"]
        elif tag == "a3":
            fields = ["reserved_path_1", "reserved_path_2",
                        "none_rel_1", "none_rel_2", "unknown_asn_1", "unknown_asn_2"]
        elif tag == "a4":
            fields = ["origin_same_org", "origin_rpki_1", "origin_rpki_2"]
        elif tag == "b1":
            fields = ["origin_same_org", "origin_connection",
                        "origin_rpki_1", "origin_rpki_2"]
        elif tag == "b2":
            fields = ["as_prepend_1", "as_prepend_2"]
        elif tag == "b3":
            fields = ["origin_different_upstream"]
        elif tag == "b4":
            fields = ["origin_rpki_1", "origin_rpki_2"]
        elif tag == "b5":
            fields = []

        r = {i: str(row[i]) for i in fields if has_flag(row[i])}
        return r

    def terminal_checkout(group_id, group):
        tags = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "b5"]

        print(f"alarm_id: {group_id}")
        for prefix_key, ev in group.groupby(["prefix1", "prefix2"]):
            print(f"* {' -> '.join(prefix_key)}")
            for _, row in ev.iterrows():
                print(f"  path1: {row['path1']}")
                print(f"  path2: {row['path2']}")
                print(f"  diff={row[metric]}")
                print(f"  culprit={row['culprit']}")
                for k,v in zip(tags, row[tags]):
                    if v:
                        r = reason(k, row)
                        print(f"{k}: ", end="")
                        print(",".join([f"{x}={y}" for x,y in r.items()]))
                print()
        input("..Enter to next")

    def json_checkout(group_id, group):
        tags = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "b5"]

        timestamp = group["timestamp"].values
        fmt = "%a %d %b %Y, %I:%M%p"
        start_time = datetime.fromtimestamp(timestamp.min()).strftime(fmt)
        end_time = datetime.fromtimestamp(timestamp.max()).strftime(fmt)

        events = []
        for prefix_key, ev in group.groupby(["prefix1", "prefix2"]):
            route_changes = []
            for _, row in ev.iterrows():
                route_changes.append({
                    "timestamp": int(row["timestamp"]),
                    "path1": str(row["path1"]),
                    "path2": str(row["path2"]),
                    "diff": float(row[metric]),
                    "culprit": json.loads(str(row['culprit'])),
                    "patterns": {k: reason(k, row) for k,v in zip(tags, row[tags]) if v},
                })

            events.append({
                "prefix": prefix_key,
                "route_changes": route_changes
            })

        ret = {
            "group_id": group_id,
            "start_time": start_time,
            "end_time": end_time,
            "events": events,
        }
        # print(json.dumps(ret, indent=2))
        return ret

    def group_html_checkout(group_id, group):
        tags = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "b5"]

        timestamp = group["timestamp"].values
        fmt = "%Y/%m/%d %H:%M:%S"
        start_time = datetime.fromtimestamp(timestamp.min()).strftime(fmt)
        end_time = datetime.fromtimestamp(timestamp.max()).strftime(fmt)

        def text_color(s, color):
            return f'<span style="color:{color};">{s}</span>'

        events = []
        for prefix_key, ev in group.groupby(["prefix1", "prefix2"]):
            route_changes = []
            for _, row in ev.iterrows():
                timestamp = f"<p><b>timestamp:</b> {row['timestamp']}</p>"
                path1 = f"<p><b>path1:</b> {row['path1']}</p>"
                path2 = f"<p><b>path2:</b> {row['path2']}</p>"
                diff = f"<p><b>diff:</b> {row[metric]}</p>"
                culprit = f"<p><b>culprit:</b> {row['culprit']}</p>"

                patterns = []
                for k,v in zip(tags, row[tags]):
                    if v:
                        r = reason(k, row)
                        p = f"<p>{k}: "+",".join([f"{x}={y}" for x,y in r.items()])+"</p>"
                        patterns.append(p)
                pattern_part = "<p><b>patterns:</b> "
                if not patterns:
                    pattern_part += "none"
                pattern_part += "</p>"

                rc_html = "    <li>\n"
                rc_html+= "    "+timestamp+"\n"
                rc_html+= "    "+path1+"\n"
                rc_html+= "    "+path2+"\n"
                rc_html+= "    "+diff+"\n"
                rc_html+= "    "+culprit+"\n"
                rc_html+= "    "+pattern_part+"\n"
                if patterns: rc_html+= "    <ul>"+"".join(patterns)+"</ul>\n"
                rc_html+= "    </li>"

                route_changes.append(rc_html)


            c = "MediumSeaGreen" if (ev["pattern"] == "anomaly").any() else "Orange"
            p0, p1 = prefix_key
            prefix_title = f'<p>{text_color(p0,c)} -> {text_color(p1,c)}</p>'
            route_change_part = "\n".join(route_changes)

            ev_html = "  <li>\n"
            ev_html+= "  "+prefix_title+"\n"
            ev_html+= "  <ul>\n"
            ev_html+= route_change_part+"\n"
            ev_html+= "  </ul>\n"
            ev_html+= "  </li>\n"

            events.append(ev_html)

        mark = text_color("&#10004;", "MediumSeaGreen") \
                if (group["pattern"] == "anomaly").any() else \
                text_color('&#10007;', "Orange")
        group_title = f"{mark} id: {group_id}, start: {start_time}, end: {end_time}, events: {len(events)}, route_changes: {group.shape[0]}"
        events_part = "".join(events)

        html = f'<button class="collapsible">{group_title}</button>\n'
        html+= '<div class="content">\n'
        html+= '<ul>\n'
        html+= events_part
        html+= '</ul>\n'
        html+= '</div>\n'

        return html

    def gen_jsonl():
        anomaly_cnt = 0
        lines = []
        for group_id, group in df.groupby("group_id"):
            if (group["pattern"] == "anomaly").any():
                anomaly_cnt += 1
            jl = json_checkout(group_id, group)
            lines.append(json.dumps(jl))

        with open(summary_dir/f"alarms_2023{month:02}.jsonl", "w") as f:
            f.write("\n".join(lines)+"\n")

        print(f"total groups: {group_id+1}")
        print(f"anomaly: {anomaly_cnt}")

    def terminal_display():
        for group_id, group in df.groupby("group_id"):
            terminal_checkout(group_id, group)

    def stats_checkout(df):
        daily_cnts = np.zeros(calendar.monthrange(2023, int(month))[1], dtype=int)
        daily_cnts_a = daily_cnts.copy()

        days = [datetime.fromtimestamp(np.min(g["timestamp"])).day for _,g in df.groupby("group_id")]
        days_a = [datetime.fromtimestamp(np.min(g["timestamp"])).day for _,g in df.groupby("group_id")
                    if (g["pattern"] == "anomaly").any()]

        days, cnts = np.unique(days, return_counts=True)
        daily_cnts[days-1] = cnts

        days_a, cnts_a = np.unique(days_a, return_counts=True)
        daily_cnts_a[days_a-1] = cnts_a

        route_change_cnts = int(subprocess.run(f"wc -l {route_change_dir}/2023{month:02}*.csv", shell=True,
                                stdout=subprocess.PIPE, encoding='UTF-8').stdout.strip().split()[-2])

        return daily_cnts, daily_cnts_a, route_change_cnts

    def gen_html():
        sections = []
        for group_id, group in df.groupby("group_id"):
            html = group_html_checkout(group_id, group)
            sections.append(html)
        template = open(html_dir/"template_routeviews.html", "r").read()
        html = template.replace("REPLACE_WITH_SECTIONS", "\n".join(sections))
        html = html.replace("REPLACE_WITH_TITLE", f"Month-{month} Report（RouteViews）")
        

        daily_cnts, daily_cnts_a, route_change_cnts = stats_checkout(df)
        xvalues = "["+", ".join([f"{i+1:02}" for i in range(calendar.monthrange(2023, int(month))[1])])+"]"
        yvalues_a = "["+", ".join([f"{i+1:02}" for i in daily_cnts_a])+"]"
        yvalues_b = "["+", ".join([f"{i+1:02}" for i in daily_cnts-daily_cnts_a])+"]"

        html = html.replace("REPLACE_WITH_XVALUES", xvalues)
        html = html.replace("REPLACE_WITH_YVALUES_A", yvalues_a)
        html = html.replace("REPLACE_WITH_YVALUES_B", yvalues_b)

        exp = "<ul>\n"
        exp += f"  <li><p>Route change total：{route_change_cnts:,}，Alarm total：{daily_cnts.sum():,}（daily {daily_cnts.mean():.2f}）。Among them，{daily_cnts_a.sum():,} show known anomalous patterns, {daily_cnts.sum()-daily_cnts_a.sum():,} unknown anomalous patterns</p></li>\n"
        exp += "</ul>\n"

        html = html.replace("REPLACE_WITH_EXPLANATION", exp)

        open(html_dir/f"report_{month:02}.html", "w").write(html)

    gen_jsonl()
    # terminal_display()
    gen_html()

for m in range(2,9):
    main(m, "diff_balance")
