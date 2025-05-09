#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import subprocess
from datetime import datetime
from pathlib import Path
import re

script_dir = Path(__file__).resolve().parent
cache_dir = script_dir/"whois_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

def whois_lookup(target, cache_date=datetime.now().strftime("%Y-%m-%d")):
    cache_file = cache_dir/f"{target.replace('/', '_')}.{cache_date}.txt"
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            content = f.read()
    else:
        try:
            result = subprocess.run(["whois", target],
                        text=True, capture_output=True, check=True)
            content = result.stdout
            with cache_file.open("w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to perform WHOIS lookup for {target}: {e}")
            content = ""
    return content

def whois_match(prefix_str, asn_str):
    whois_content = whois_lookup(prefix_str)
    for line in whois_content.split("\n"):
        if not line or line.startswith("%"): continue
        match = re.match(r"^(\S+):\s+(.*)$", line)
        if match:
            _, value = match.groups()
            for asn_value in re.findall(r"as\d+", value):
                if f"as{asn_str}" == asn_value:
                    return True
    return False
