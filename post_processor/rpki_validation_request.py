#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import requests
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"rpki_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def rpki_valid(prefix, asn):
    # valid, unknown, invalid_asn, invalid_length, query error
    cache_path = CACHE_DIR/f"{prefix}.{asn}".replace("/", "-")
    if cache_path.exists():
        try:
            r = json.load(open(cache_path, "r"))
        except json.decoder.JSONDecodeError as e:
            print(f"cache_path: {cache_path}")
            raise e
        return r["data"]["status"]

    payload = {"prefix": prefix, "resource": asn}
    url = "https://stat.ripe.net/data/rpki-validation/data.json"
    r = requests.get(url, params=payload)
    if r.status_code == 200:
        r = r.json()
        json.dump(r, open(cache_path, "w"))
        return r["data"]["status"]
    else:
        print(f"RPKI query error: {prefix}, {asn}")
        return "query error"
