#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json
import re

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

url_index="https://bgpstream.crosswork.cisco.com/"

def get_page(url):
    page = subprocess.check_output(["curl", "-s", url]).decode()
    return page

def item_parser(index_page):
    events = []
    for item_str in re.finditer(r'\<tr\>.+?\</tr\>', index_page, flags=re.DOTALL):
        try:
            item_str = item_str[0]
            item = dict()
            for k, v in re.findall(r'\<td class="(.+?)"\>(.+?)\</td\>',
                    item_str, flags=re.DOTALL):
                v = re.sub(r"\s\s+", " ", v.replace("\n", " ")).strip()

                if k == "asn":
                    asns = re.findall(r'\(AS (\d+?)\)', v, flags=re.DOTALL)

                    if item["event_type"] == "Outage":
                        item["asn"] = asns
                    elif item["event_type"] == "Possible Hijack":
                        expected, detected = asns
                        item["expected_asn"] = expected
                        item["detected_asn"] = detected
                    elif item["event_type"] == "BGP Leak":
                        origin, leaker = asns
                        item["origin_asn"] = origin
                        item["leaker_asn"] = leaker
                    else:
                        raise RuntimeError(
                                f"Uncovered event_type: {item['event_type']}")
                elif k == "country" and v:
                    item["country"] = v.split(" ")[0]
                elif k == "moredetail":
                    v = re.search(r'\<a href="(.+?(\d+))"\>', v)
                    if v is None:
                        item["moredetail"] = ""
                        item["event_id"] = ""
                    else:
                        item["moredetail"] = v[1]
                        item["event_id"] = v[2]
                else:
                    item[k] = v
            events.append(item)
        except Exception as e:
            print(e)
            continue
    ids = np.array([int(i["event_id"]) for i in events])
    sort_idx = np.argsort(ids)
    events = np.array(events)[sort_idx].tolist()
    ids = ids[sort_idx].tolist()
    return events, ids

def update_cache():
    index_page = get_page(url_index)
    events, ids = item_parser(index_page)

    current_id = [int(i.stem) for i in CACHE_DIR.glob("*.jsonl")]
    current_max_id = max(current_id) if current_id else -1
    start_idx = np.searchsorted(ids, current_max_id, "right")

    events = events[start_idx:]
    if not events:
        print("No need to update.")
        return

    def fetch_for_detail(ev):
        if ev["event_type"] == "Possible Hijack":
            detail_page = get_page(urljoin(url_index, ev["moredetail"]))
            pattern = r'Expected prefix: (.+?/\d{1,2})'
            expected = re.search(pattern, detail_page)
            if expected is not None:
                ev["expected_prefix"] = expected[1]
            else:
                print(f"unknown expected_prefix: {ev}")

            pattern = r'Detected advertisement: (.+?/\d{1,2})'
            detected = re.search(pattern, detail_page)
            if detected is not None:
                ev["detected_prefix"] = detected[1]
            else:
                print(f"unknown detected_prefix: {ev}")

        elif ev["event_type"] == "BGP Leak":
            detail_page = get_page(urljoin(url_index, ev["moredetail"]))
            pattern = r'Leaked prefix: (.+?/\d{1,2})'
            leaked = re.search(pattern, detail_page)
            if leaked is not None:
                ev["leaked_prefix"] = leaked[1]
            else:
                print(f"unknown leaked_prefix: {ev}")

            pattern = r'Leaked To:\<br\>\s+\<li\>(\d+)'
            leakedto = re.search(pattern, detail_page)
            if leakedto is not None:
                ev["leaked_to"] = leakedto[1]
            else:
                print(f"unknown leaked_to: {ev}")

    with ThreadPoolExecutor(max_workers=128) as executor:
        executor.map(fetch_for_detail, events)

    f = open(CACHE_DIR/f"{ids[-1]}.jsonl", "w")
    f.write("\n".join([json.dumps(ev) for ev in events])+"\n")
    f.close()

    print(f"Update {len(events)} items")
    print(f"Latest event_id: {ids[-1]}")

update_cache()
