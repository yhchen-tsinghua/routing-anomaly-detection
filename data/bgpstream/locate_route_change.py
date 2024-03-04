#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import ipaddress
from joblib import Parallel, delayed

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.routeviews.fetch_updates import download_data, load_updates_to_df, get_all_collectors, get_archive_list as get_updates_list
from data.routeviews.fetch_rib import load_ribs_to_df, get_most_recent_rib
from routing_monitor.monitor import Monitor

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
EVENT_DIR = SCRIPT_DIR/"event"
EVENT_DIR.mkdir(parents=True, exist_ok=True)

collectors2url = get_all_collectors()

class RibMonitor(Monitor):
    def __init__(self, rib_df, checker):
       super().__init__()
       self.checker = checker
       self.consume(rib_df, detect=False)

    def update(self, timestamp, prefix_str, vantage_point, aspath_str, detect):
        prefix = ipaddress.ip_network(prefix_str)

        if prefix.version == 6:
            prefixlen = prefix.prefixlen
            prefix = int(prefix[0]) >> (128-prefixlen)
        else:
            prefixlen = prefix.prefixlen
            prefix = int(prefix[0]) >> (32-prefixlen)

        aspath = aspath_str.split(" ")
        forwarder = aspath[0] # forwarder could be vantage point or not

        n = self.root
        original_route = None
        for shift in range(prefixlen-1, -1, -1): # find the original route
            left = (prefix >> shift) & 1

            if left: n = n.get_left()
            else: n = n.get_right()
            
            if n.find_route(forwarder) is not None:
                original_route = [shift, n.find_route(forwarder)]

        if detect and original_route is not None:
            shift, original_path = original_route
            vict_prefix = ipaddress.ip_network(prefix_str) \
                            .supernet(new_prefix=prefixlen-shift)
            if aspath != original_path:
                route_change = {
                    "timestamp"    : timestamp,
                    "vantage_point": vantage_point,
                    "forwarder"    : forwarder,
                    "prefix1"      : str(vict_prefix),
                    "prefix2"      : prefix_str,
                    "path1"        : " ".join(original_path),
                    "path2"        : " ".join(aspath),
                }
                if self.checker(route_change):
                    self.route_changes.append(route_change)

        n.routes[forwarder] = aspath

def get_event_list():
    existing_ids = [int(i.stem) for i in EVENT_DIR.glob("*.csv")]
    last_id = max(existing_ids) if existing_ids else -1

    events = [json.loads(l.strip()) for jl in CACHE_DIR.glob("*.jsonl")
                                        for l in open(jl, "r").readlines()]
    events = [ev for ev in events if ev["event_type"] != "Outage"] # NOTE: ignore outage for now
    events = np.array(events)
    ids = np.array([int(i["event_id"]) for i in events])
    sort_idx = np.argsort(ids)
    events = events[sort_idx].tolist()
    ids = ids[sort_idx].tolist()

    start_idx = np.searchsorted(ids, last_id, "right")
    return events[start_idx:]

def process_event(collector, event):
    if event["event_type"] == "Outage" and len(event["asn"]) != 1:
        return # ignore the country-wide outage

    output_dir = EVENT_DIR/event["event_id"]
    if (output_dir/f"{collector}.csv").exists():
        return

    dtime1 = datetime.strptime(event["starttime"], "%Y-%m-%d %H:%M:%S")
    dtime2 = datetime.strptime(event["endtime"], "%Y-%m-%d %H:%M:%S") \
                        if event["endtime"] else dtime1+timedelta(hours=1)

    rib, _, stime = get_most_recent_rib(collector, collectors2url, dtime1)
    update_list = get_updates_list(collector, collectors2url, stime, dtime2)

    rib_fpath = download_data(rib, collector)
    update_fpaths = [download_data(url, collector) for url in update_list]

    def outage_process(event):
        target_asn, = event["asn"] # assert single asn here
        def checker(route_change):
            p1 = route_change["path1"].split(" ")
            p2 = route_change["path2"].split(" ")
            return ((target_asn in p1) and (target_asn not in p2)) \
                        or ((target_asn not in p1) and (target_asn in p2))
        def locator(df):
            return df
        return checker, locator

    def hijack_process(event):
        expected_asn = event["expected_asn"]
        detected_asn = event["detected_asn"]
        def checker(route_change):
            o1 = route_change["path1"].split(" ")[-1]
            o2 = route_change["path2"].split(" ")[-1]
            return (expected_asn == o1 and detected_asn == o2) \
                    or (detected_asn == o1 and expected_asn == o2)
        expected_prefix = event["expected_prefix"]
        detected_prefix = event["detected_prefix"]
        def locator(df):
            return df.loc[(df["prefix"] == expected_prefix)
                            | (df["prefix"] == detected_prefix)]
        return checker, locator

    def leak_process(event):
        origin = event["origin_asn"]
        leaker = event["leaker_asn"]
        leakedto = event["leaked_to"]
        def checker(route_change):
            p1 = route_change["path1"].split(" ")
            p2 = route_change["path2"].split(" ")
            return origin == p1[-1] and origin == p2[-1] \
                            and ((leaker in p1 and leakedto in p1)
                                    or (leaker in p2 and leakedto in p2))
        leaked_prefix = event["leaked_prefix"]
        def locator(df):
            return df.loc[df["prefix"] == leaked_prefix]
        return checker, locator

    if event["event_type"] == "Outage":
        checker, locator = outage_process(event)
    elif event["event_type"] == "Possible Hijack":
        checker, locator = hijack_process(event)
    elif event["event_type"] == "BGP Leak":
        checker, locator = leak_process(event)

    rib_df = locator(load_ribs_to_df(rib_fpath))

    mon = RibMonitor(rib_df, checker)
    for fp in update_fpaths:
        df = locator(load_updates_to_df(fp))
        mon.consume(df, detect=True)

    route_change_df = pd.DataFrame.from_records(mon.route_changes)

    output_dir.mkdir(exist_ok=True, parents=True)
    route_change_df.to_csv(output_dir/f"{collector}.csv", index=False)

events = get_event_list()
np.random.shuffle(events)

def process_event_safe(collector, event):
    try: process_event(collector, event)
    except Exception as e:
        print(f"{e} at {collector} {event}")

Parallel(n_jobs=12, backend="multiprocessing", verbose=10)(
        delayed(process_event_safe)("route-views4", ev) for ev in events)

# for ev in events:
#     process_event("route-views4", ev)
#     print("done")
#     input()
