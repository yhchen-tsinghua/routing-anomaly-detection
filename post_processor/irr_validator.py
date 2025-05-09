#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import ftplib
import shutil
from pathlib import Path
import gzip
import ipaddress
import re

script_dir = Path(__file__).resolve().parent
cache_dir = script_dir/"irr_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

def fetch_and_uncompress_gz(year, month, day, domain="ftp.radb.net"):
        output_path = cache_dir/f"radb-{year}{month:02d}{day:02d}.db"

        if output_path.exists():
            return output_path

        temp_gz_file = output_path.with_suffix(output_path.suffix + ".gz")

        ftp = ftplib.FTP(domain)
        ftp.login()

        remote_paths = [
            f"/radb/dbase/archive/{year}/radb.db.{year}{month:02d}{day:02d}.gz",
            f"/radb/dbase/archive/{year}/radb.db.{str(year)[-2:]}{month:02d}{day:02d}.gz"
        ]

        for remote in remote_paths:
            try:
                with temp_gz_file.open("wb") as file:
                    ftp.retrbinary(f"RETR {remote}", file.write)
                break
            except ftplib.error_perm as e:
                print(f"Remote file not found at {remote}. Trying the next path...")
            except Exception as e:
                print(f"Unexpected error while accessing {remote}: {e}")
        else:
            raise FileNotFoundError("Remote files unavailable.")

        with gzip.open(temp_gz_file, "rb") as f_in, output_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        temp_gz_file.unlink() 

        return output_path

def parse_route_blocks(file_path):
    with open(file_path, encoding='ISO-8859-1') as file:
        content = file.read()

    blocks = content.strip().split('\n\n')

    route_blocks = []
    for block in blocks:
        if block.startswith("route:"):
            route_dict = parse_route_block(block)
            assert route_dict
            route_dict["original_data"] = block
            route_blocks.append(route_dict)

    return route_blocks

def parse_route_block(block):
    block_dict = {}
    current_key = None

    for line in block.split('\n'):
        if not line: continue
        match = re.match(r'^(\S+):(.*)$', line)
        if match:
            current_key, value = match.groups()
        else:
            value = line # multi-line value
            assert current_key is not None
        block_dict.setdefault(current_key, []).append(value.strip())
    for k, v in block_dict.items():
        block_dict[k] = "\n".join(v)

    return block_dict

def sync_cache(year, month, day):
    route_objects = parse_route_blocks(fetch_and_uncompress_gz(year, month, day))
    return route_objects

class RADB:
    class PrefixNode:
        def __init__(self):
            self.left       = None
            self.right      = None
            self.data       = []

        def get_left(self):
            if self.left is None:
                self.left = RADB.PrefixNode()
            return self.left

        def get_right(self):
            if self.right is None:
                self.right = RADB.PrefixNode()
            return self.right

        def update_data(self, **kwargs):
            self.data.append(kwargs)

    def __init__(self):
        self.root = RADB.PrefixNode()

    def load_data(self, year, month, day):
        route_objects = sync_cache(year, month, day)
        for obj in route_objects:
            if obj["route"][-2:] == "/0": continue
            try:
                directions = self.prefix_to_dirs(obj["route"])
            except:
                print(obj)
                exit()
            if not directions: continue
            self.create_node(directions).update_data(**obj)
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
        matched = None
        n = self.root
        for left in directions:
            if left: n = n.get_left()
            else: n = n.get_right()
            if n is None: break
            if n.data: matched = n
        return matched

    def validate(self, prefix_str, asn_str):
        directions = self.prefix_to_dirs(prefix_str)

        if not directions: return "Not Found"

        matched = self.match_node(directions) # longest match

        if matched is None: return "Not Found"

        irrs = matched.data

        for irr in irrs:
            if f"AS{asn_str}" == irr["origin"]:
                return "Valid"

        return "Invalid"

    def all_matched(self, prefix_str):
        directions = self.prefix_to_dirs(prefix_str)

        if not directions: return []

        matched = self.match_node(directions) # longest match

        if matched is None: return []

        return matched.data
