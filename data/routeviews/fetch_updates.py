#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from io import StringIO
from urllib.parse import urljoin
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import subprocess
import re
import json
import click

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

def get_archive_list(collector, collectors2url, dtime1, dtime2):
    if collector not in collectors2url: return []

    def pull_list(ym):
        target_url = urljoin(collectors2url[collector], f"{ym}/UPDATES") + "/"
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

    ym1 = dtime1.strftime("%Y.%m")
    ym2 = dtime2.strftime("%Y.%m")
    target_url1, archive_list1 = pull_list(ym1)
    target_url2, archive_list2 = pull_list(ym2)

    if not archive_list1 or not archive_list2:
        print(f"failed to get archive list: {dtime1} {dtime2}")
        exit(1)
    
    time_list1 = ["".join(i[1:]) for i in archive_list1]
    time_list2 = ["".join(i[1:]) for i in archive_list2]
    t1 = dtime1.strftime("%Y%m%d%H%M")
    t2 = dtime2.strftime("%Y%m%d%H%M")
    idx1 = np.searchsorted(time_list1, t1, side="left")
    idx2 = np.searchsorted(time_list2, t2, side="right")

    if time_list1 == time_list2:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:idx2]]
    else:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:]]

        current_month = datetime(dtime1.year, dtime1.month, 1)
        current_month += relativedelta(months=1)
        upper_bound = datetime(dtime2.year, dtime2.month, 1)
        while current_month < upper_bound:
            cur_ym = current_month.strftime("%Y.%m")
            cur_target_url, cur_archive_list = pull_list(cur_ym)
            data += [urljoin(cur_target_url, i[0]) for i in cur_archive_list]
            current_month += relativedelta(months=1)
        data += [urljoin(target_url2, i[0]) for i in archive_list2[:idx2]]

    return data

def download_data(url, collector):
    fname = url.split("/")[-1].strip()
    outpath = SCRIPT_DIR / "updates" / collector / fname
    fpath = outpath.with_suffix("")
    if fpath.exists():
        # print(f"updates for {collector} {outpath.stem} already existed")
        return fpath
    outpath.parent.mkdir(exist_ok=True, parents=True)
    subprocess.run(["curl", "-s", url, "--output", str(outpath)], check=True)
    subprocess.run(["bzip2", "-d", str(outpath)], check=True)
    print(f"get updates for {collector} {outpath.stem}")
    return fpath

def load_updates_to_df(fpath, bgpd=SCRIPT_DIR/"bgpd"):
    res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
    fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
    cols = fmt.split("|")
    df = pd.read_csv(StringIO(res), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)
    return df


@click.command()
@click.option("--collector", type=str, required=True, help="the collector name, e.g., route-views4")
@click.option("--dtime1", type=str, required=True, help="the starttime (included), e.g., 201812312330")
@click.option("--dtime2", type=str, required=True, help="the endtime (included), e.g., 201812312330")
@click.option("--download", type=bool, default=False, help="download the archives")
@click.option("--num-workers", type=int, default=1, help="number of workers")
def main(collector, dtime1, dtime2, download, num_workers):
    dtime1 = datetime.strptime(dtime1, "%Y%m%d%H%M")
    dtime2 = datetime.strptime(dtime2, "%Y%m%d%H%M")

    collectors2url = get_all_collectors()
    data = get_archive_list(collector, collectors2url, dtime1, dtime2)
    # print(data)

    if download:
        job = lambda url: download_data(url, collector)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(job, data)

        # for url in data:
            # fpath = download_data(url, collector)
            # print(fpath)
            # df = load_updates_to_df(fpath)
            # print(df)

if __name__ == "__main__":
    main()
