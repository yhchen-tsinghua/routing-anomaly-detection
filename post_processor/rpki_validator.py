#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import requests
import ipaddress
import lzma
from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent
cache_dir = script_dir/"rpki_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

def fetch_and_uncompress_xz(url, output_path):
    if output_path.exists():
        return output_path
    try:
        temp_xz_file = output_path.with_suffix(output_path.suffix + ".xz")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with temp_xz_file.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with lzma.open(temp_xz_file, "rb") as xz_file:
            with output_path.open("wb") as out_file:
                out_file.write(xz_file.read())

        temp_xz_file.unlink()

        return output_path

    except requests.RequestException as e:
        print(f"Error fetching file from {url}: {e}")
    except lzma.LZMAError as e:
        print(f"Error decompressing the .xz file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# check this out if the current one is down: http://josephine.sobornost.net/
def sync_cache(year, month, day, source="https://ftp.ripe.net/rpki"):
    dfs = []
    for rir in ["apnic", "afrinic", "arin", "lacnic", "ripencc"]:
        url = f"{source}/{rir}.tal/{year}/{month:02d}/{day:02d}/roas.csv.xz"
        output_path = cache_dir/f"roas-{rir}-{year}{month:02d}{day:02d}.csv"
        df = pd.read_csv(fetch_and_uncompress_xz(url, output_path))
        df["TA"] = rir
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

class RPKI:
    class PrefixNode:
        def __init__(self):
            self.left       = None
            self.right      = None
            self.data       = []

        def get_left(self):
            if self.left is None:
                self.left = RPKI.PrefixNode()
            return self.left

        def get_right(self):
            if self.right is None:
                self.right = RPKI.PrefixNode()
            return self.right

        def update_data(self, **kwargs):
            self.data.append(kwargs)

    def __init__(self):
        self.root = RPKI.PrefixNode()

    def load_data(self, year, month, day):
        df = sync_cache(year, month, day)
        for _, row in df.iterrows():
            if row["IP Prefix"][-2:] == "/0": continue
            directions = self.prefix_to_dirs(row["IP Prefix"])
            if not directions: continue
            self.create_node(directions).update_data(**row.to_dict())
        return self

    @staticmethod
    def prefix_to_dirs(prefix_str):
        prefix = ipaddress.ip_network(prefix_str)
        if prefix.version == 6: return None
        prefixlen = prefix.prefixlen
        prefix = int(prefix[0]) >> (32-prefixlen)
        directions = [(prefix>>shift)&1
                        for shift in range(prefixlen-1, -1, -1)]
        return directions

    def create_node(self, directions):
        n = self.root
        for left in directions:
            if left: n = n.get_left()
            else: n = n.get_right()
        return n

    def match_node(self, directions):
        matched = []
        n = self.root
        for left in directions:
            if left: n = n.get_left()
            else: n = n.get_right()
            if n is None: break
            if n.data: matched += n.data
        return matched

    def validate(self, prefix_str, asn_str):
        directions = self.prefix_to_dirs(prefix_str)

        if not directions: return "Not Found"

        matched = self.match_node(directions)

        if not matched: return "Not Found"

        for roa in matched:
            if int(prefix_str.split("/")[-1]) <= int(roa["Max Length"]) \
                    and f"AS{asn_str}" == roa["ASN"]:
                return "Valid"

        return "Invalid"

    def all_matched(self, prefix_str):
        directions = self.prefix_to_dirs(prefix_str)

        if not directions: return []

        matched = self.match_node(directions)

        return matched
