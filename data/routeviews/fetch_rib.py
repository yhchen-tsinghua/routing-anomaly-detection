#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from io import StringIO
from urllib.parse import urljoin
import pandas as pd
import numpy as np
import subprocess
import re
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

current_ym = datetime.now().strftime("%Y.%m")
for cache_file in CACHE_DIR.glob(f"*{current_ym}*"): # remove incomplete cache files
    cache_file.unlink()

def get_all_collectors(url_index="http://routeviews.org/"):
    cache_path = CACHE_DIR/f"collectors2url.{url_index.replace('/', '+')}"
    if cache_path.exists():
        # print(f"load cache: {cache_path}")
        try: return json.load(open(cache_path, "r"))
        except: pass

    res = subprocess.check_output(["curl", "-s", url_index]).decode()
    res = re.sub(r"\s\s+", " ", res.replace("\n", " "))
    collectors2url = {}
    for a, b in re.findall(r'\<A HREF="(.+?)"\>.+?\([\w\s]+, from (.+?)\)', res):
        collector_name = b.split(".")[-3]
        if collector_name in collectors2url:
            idx = 2
            while f"{collector_name}{idx}" in collectors2url:
                idx += 1
            collector_name = f"{collector_name}{idx}"
        collectors2url[collector_name] = urljoin(url_index, a) + "/"

    # print(f"save cache: {cache_path}")
    json.dump(collectors2url, open(cache_path, "w"), indent=2)
    return collectors2url

def get_most_recent_rib(collector, collectors2url, dtime):
    if collector not in collectors2url: return []

    def pull_list():
        target_url = urljoin(collectors2url[collector], f"{ym}{subdir}") + "/"
        cache_path = CACHE_DIR/f"archive_list.{target_url.replace('/', '+')}"
        if cache_path.exists():
            # print(f"load cache: {cache_path}")
            try: return target_url, json.load(open(cache_path, "r"))
            except: pass
        res = subprocess.check_output(["curl", "-s", target_url]).decode()
        archive_list = re.findall(
            r'\<a href="(.+?(\d{4}).??(\d{2}).??(\d{2}).??(\d{4}).*?\.bz2)"\>', res)
        # print(f"save cache: {cache_path}")
        json.dump(archive_list, open(cache_path, "w"), indent=2)
        return target_url, archive_list

    ym = dtime.strftime("%Y.%m")
    subdir = "/RIBS"
    target_url, archive_list = pull_list()

    if not archive_list:
        subdir = ""
        target_url, archive_list = pull_list()
    if not archive_list: return []
    
    time_list = ["".join(i[1:]) for i in archive_list]
    t = dtime.strftime("%Y%m%d%H%M")
    idx = np.searchsorted(time_list, t)

    if idx == 0:
        data1 = urljoin(target_url, archive_list[0][0])
        dtime = dtime-relativedelta(months=1)
        ym = dtime.strftime("%Y.%m")
        target_url, archive_list = pull_list()
        if not archive_list: return []
        data0 = urljoin(target_url, archive_list[-1][0])
        stime = datetime.strptime("".join(archive_list[-1][1:]), "%Y%m%d%H%M")
        return data0, data1, stime

    if idx == len(time_list):
        data0 = urljoin(target_url, archive_list[-1][0])
        stime = datetime.strptime("".join(archive_list[-1][1:]), "%Y%m%d%H%M")
        dtime = dtime+relativedelta(months=1)
        ym = dtime.strftime("%Y.%m")
        target_url, archive_list = pull_list()
        if not archive_list: return []
        data1 = urljoin(target_url, archive_list[0][0])
        return data0, data1, stime

    data0 = urljoin(target_url, archive_list[idx-1][0])
    data1 = urljoin(target_url, archive_list[idx][0])
    stime = datetime.strptime("".join(archive_list[idx-1][1:]), "%Y%m%d%H%M")
    return data0, data1, stime

def download_data(url, collector):
    fname = url.split("/")[-1].strip()
    outpath = SCRIPT_DIR / "ribs" / collector / fname
    fpath = outpath.with_suffix("")
    if fpath.exists():
        # print(f"updates for {collector} {outpath.stem} already existed")
        return fpath
    outpath.parent.mkdir(exist_ok=True, parents=True)
    subprocess.run(["curl", "-s", url, "--output", str(outpath)], check=True)
    subprocess.run(["bzip2", "-d", str(outpath)], check=True)
    print(f"get ribs for {collector} {outpath.stem}")
    return fpath

def load_ribs_to_df(fpath):
    if fpath.suffix == ".dat":
        fd = open(fpath, "r")
        l = fd.readline()
        while l:
            if "Network" in l and "Path" in l:
                idx_network = l.find("Network")
                idx_path = l.find("Path")
                break
            l = fd.readline()

        data = []
        current_network = ""
        while l:
            if l[0] != "*": l = fd.readline(); continue
            if l[idx_network] != " ":
                current_network = l[idx_network:idx_network+l[idx_network:].find(" ")]
            if l[1] == ">":
                path = l[idx_path:-3]
                if "/" in current_network:
                    data.append(["0", current_network, path.split(" ")[0], path])
            l = fd.readline()
        df = pd.DataFrame(data, columns=["timestamp", "prefix", "peer-asn", "as-path"])
    else:
        bgpd = SCRIPT_DIR / 'bgpd'
        res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
        fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
        cols = fmt.split("|")
        df = pd.read_csv(StringIO(res), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)

    return df
