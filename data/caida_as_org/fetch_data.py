#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import json
import subprocess
import click
import re

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR/"fetched_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_archive_list(refresh=False):
    cache_path = CACHE_DIR/f"time2url"
    if cache_path.exists() and not refresh:
        try: return json.load(open(cache_path, "r"))
        except: pass

    url_index = "https://publicdata.caida.org/datasets/as-organizations/"
    res = subprocess.check_output(["curl", "-s", url_index]).decode()
    res = re.sub(r"\s\s+", " ", res.replace("\n", " "))
    time2url = {}
    for fname, time in re.findall(r'\<a href="((\d{8}).as-org2info.txt.gz)"\>', res):
        time2url[time] = urljoin(url_index, fname)

    json.dump(time2url, open(cache_path, "w"), indent=2)
    return time2url

def get_most_recent(time):
    time2url = get_archive_list()
    times = sorted(time2url.keys())
    idx = np.searchsorted(times, time, "right")

    target_time = times[idx-1]
    target_url = time2url[target_time]

    out = OUTPUT_DIR/target_url.split("/")[-1]
    if out.with_suffix("").exists():
        # print(f"as-organizations for {target_time} exists")
        return target_time, out.with_suffix("")

    subprocess.run(["curl", target_url, "--output", str(out)], check=True)
    subprocess.run(["gzip", "-d", str(out)], check=True)
    print(f"get as-organizations for {target_time}")
    return target_time, out.with_suffix("")

@click.command()
@click.option("--time", "-t", type=str, required=True, help="timestamp, e.g., 20200901")
def main(time):
    get_most_recent(time)

if __name__ == "__main__":
    main()
