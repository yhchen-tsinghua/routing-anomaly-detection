#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import json

from rpki_validation_request import rpki_valid

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.caida_as_org.fetch_data import get_most_recent as as_org_file
from data.caida_as_org.query import load as parse_as_org
from data.caida_as_rel.fetch_data import get as as_rel_file

def get_one_asn(asn):
    return asn.strip("{}").split(",")[0]

def load_as_org(time):
    time, fpath = as_org_file(time)
    as_info, org_info = parse_as_org(time)

    def get_org_id(asn):
        if asn not in as_info:
            return asn
        info = as_info[asn]
        return info["opaque_id"] if info["opaque_id"] != "" else info["org_id"]

    def from_same_org(asn1, asn2):
        if get_org_id(asn1) == get_org_id(asn2):
            return get_org_id(asn1)
        else:
            return "-"

    def get_asn_country(asn):
        org_id = get_org_id(asn)
        if org_id in org_info:
            return org_info[org_id]["country"]

    return as_info, org_info, from_same_org, get_asn_country

def load_as_rel(serial, time):
    target = as_rel_file(serial, time)
    as_rel_map = {}
    lines = open(target, "r").readlines()
    for l in lines:
        if l[0] == "#": continue
        as1, as2, rel = l.split("|")[:3]
        rel = int(rel)
        as_rel_map.setdefault(as1, {-1:set(), 0:set(), 1:set()})[+rel].add(as2)
        as_rel_map.setdefault(as2, {-1:set(), 0:set(), 1:set()})[-rel].add(as1)

    def get_as_rel(as1, as2):
        if as1 in as_rel_map:
            for rel, as_set in as_rel_map[as1].items():
                if as2 in as_set: return rel
        return None

    def get_all_ngbrs(asn):
        if asn not in as_rel_map: return None
        ret = set()
        for v in as_rel_map[asn].values():
            ret |= v
        return ret

    def have_connection(as1, as2):
        if as1 in as_rel_map and as2 in as_rel_map:
            for rel,ngbrs in as_rel_map[as1].items():
                if as2 in ngbrs:
                    return f"rel({rel})"
            ret = []
            for rel in [-1, 0, 1]:
                if as_rel_map[as1][rel]&as_rel_map[as2][rel]:
                    ret.append(f"{rel}")
            if ret:
                return f"ngbr({';'.join(ret)})"
        return "-"

    return as_rel_map, get_as_rel, have_connection

def different_origin_country(path1, path2, get_asn_country):
    cty1 = get_asn_country(path1[-1])
    cty2 = get_asn_country(path2[-1])
    if cty1 != cty2:
        return f"{cty1};{cty2}"
    else:
        return "-"

def have_origin_connection(path1, path2, have_connection):
    return have_connection(path1[-1], path2[-1])

def have_unknown_asn(path, as_rel_map):
    ret = []
    for i in path:
        if i not in as_rel_map:
            ret.append(str(i))
    if ret:
        return ";".join(set(ret))
    return "-"

def have_reserved_asn(path):
    ret = []
    for i in path:
        i = int(i)
        if i == 0 \
                or i == 112 \
                or i == 23456 \
                or (i >= 64496 and i <= 65534) \
                or i == 65535 \
                or (i >= 65536 and i <= 65551):
            ret.append(str(i))
    if ret:
        return ";".join(set(ret))
    return "-"

def non_valley_free_or_none_rel(path, get_as_rel):
    none_rel = []
    non_valley_free = False
    rel_seq = []
    state = 1
    for a,b in zip(path[:-1], path[1:]):
        if a == b:
            rel_seq.append("x")
            continue
        r = get_as_rel(a, b)
        if r is None:
            rel_seq.append("x")
            none_rel.append(f"({a} {b})")
            continue
        rel_seq.append(str(r))
        if state == 1:
            state = r
            continue
        if r != -1:
            non_valley_free = True

    if non_valley_free:
        non_valley_free = " ".join(rel_seq)
    else:
        non_valley_free = "-"

    if none_rel:
        none_rel = ";".join(set(none_rel))
    else:
        none_rel = "-"

    return non_valley_free, none_rel

def detour_country(path1, path2, get_asn_country):
    countries0 = set(filter(lambda x: x is not None,
                        map(get_asn_country, path1[:-1])))
    countries1 = set(filter(lambda x: x is not None,
                        map(get_asn_country, path2[:-1])))
    detour_new_country = len(countries1-countries0) > 0
    return detour_new_country

def as_prepend(path):
    ret = []
    for asn, cnt in zip(*np.unique(path, return_counts=True)):
        if cnt > 1: ret.append(asn)
    if ret:
        return ";".join(ret)
    return "-"

def origin_different_upstream(path1, path2, get_as_rel):
    path1 = [v for i,v in enumerate(path1) if i == 0 or v != path1[i-1]]
    path2 = [v for i,v in enumerate(path2) if i == 0 or v != path2[i-1]]
    if len(path1) >= 2 and len(path2) >= 2 \
            and path1[-1] == path2[-1] \
            and path1[-2] != path2[-2] \
            and get_as_rel(path1[-2], path1[-1]) is not None \
            and get_as_rel(path2[-2], path2[-1]) is not None:
        return f"{path1[-2]};{path2[-2]}"
    return "-"

def origin_rpki_valid(prefix, path):
    return rpki_valid(prefix, path[-1])

def path_superset(path1, path2):
    return ",".join(path1) in ",".join(path2)


def postprocess(metric, ym):
    as_info, org_info, from_same_org, get_asn_country = load_as_org("20230101")
    as_rel_map, get_as_rel, have_connection = load_as_rel("1", 20230101)

    repo_dir = Path(__file__).resolve().parent.parent
    reported_alarm_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"reported_alarms"/metric/ym
    info = json.load(open(reported_alarm_dir/f"info_{ym}.json", "r"))
    flags_dir = reported_alarm_dir.parent/f"{ym}.flags"
    flags_dir.mkdir(parents=True, exist_ok=True)
    
    for i in info:
        if i["save_path"] is None: continue
        df = pd.read_csv(i["save_path"])
        prefix1 = df["prefix1"].values
        prefix2 = df["prefix2"].values
        path1 = [list(map(get_one_asn, i.split(" "))) for i in df["path1"].values]
        path2 = [list(map(get_one_asn, i.split(" "))) for i in df["path2"].values]

        non_valley_free_1, none_rel_1 = np.array(list(map(lambda x: non_valley_free_or_none_rel(x, get_as_rel), path1))).T
        non_valley_free_2, none_rel_2 = np.array(list(map(lambda x: non_valley_free_or_none_rel(x, get_as_rel), path2))).T

        flags = pd.DataFrame.from_dict({
            "subprefix_change": [p1 != p2 for p1, p2 in zip(prefix1, prefix2)],
            "origin_change": [l1[-1] != l2[-1] for l1, l2 in zip(path1, path2)],
            "origin_same_org": [from_same_org(l1[-1], l2[-1]) for l1, l2 in zip(path1, path2)],
            "origin_country_change": [different_origin_country(l1, l2, get_asn_country) for l1, l2 in zip(path1, path2)],
            "origin_connection": [have_origin_connection(l1, l2, have_connection) for l1, l2 in zip(path1, path2)],
            "origin_different_upstream": [origin_different_upstream(l1, l2, get_as_rel) for l1, l2 in zip(path1, path2)],
            "origin_rpki_1": [origin_rpki_valid(p, l) for p, l in zip(prefix1, path1)],
            "origin_rpki_2": [origin_rpki_valid(p, l) for p, l in zip(prefix2, path2)],
            "unknown_asn_1": [have_unknown_asn(l, as_rel_map) for l in path1],
            "unknown_asn_2": [have_unknown_asn(l, as_rel_map) for l in path2],
            "reserved_path_1": [have_reserved_asn(l) for l in path1],
            "reserved_path_2": [have_reserved_asn(l) for l in path2],
            "non_valley_free_1": non_valley_free_1,
            "non_valley_free_2": non_valley_free_2,
            "none_rel_1": none_rel_1,
            "none_rel_2": none_rel_2,
            "as_prepend_1": [as_prepend(l) for l in path1],
            "as_prepend_2": [as_prepend(l) for l in path2],
            "detour_country": [detour_country(l1, l2, get_asn_country) for l1, l2 in zip(path1, path2)],
            "path1_in_path2": [path_superset(l1, l2) for l1, l2 in zip(path1, path2)],
            "path2_in_path1": [path_superset(l2, l1) for l1, l2 in zip(path1, path2)],
        })
        flags.to_csv(flags_dir/f"{Path(i['save_path']).stem}.flags.csv", index=False)

Parallel(backend="multiprocessing", n_jobs=7, verbose=10)(
        delayed(postprocess)("diff_balance", f"2023{m:02}") for m in range(2,9))
