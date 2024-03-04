#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import click

SCRIPT_DIR = Path(__file__).resolve().parent

def load(time):
    fname = f"{time}.as-org2info.txt"
    lines = open(SCRIPT_DIR/"fetched_data"/fname, "r").readlines()
    field1 = "aut|changed|aut_name|org_id|opaque_id|source".split("|")
    field2 = "org_id|changed|name|country|source".split("|")
    as_info = {}
    org_info = {}
    for l in lines:
        if l[0] == "#": continue
        values = l.strip().split("|")
        if len(values) == len(field1):
            if values[0] in as_info and values[1] < as_info[values[0]]["changed"]: continue
            as_info[values[0]] = dict(zip(field1[1:], values[1:]))
        if len(values) == len(field2):
            if values[0] in org_info and values[1] < org_info[values[0]]["changed"]: continue
            org_info[values[0]] = dict(zip(field2[1:], values[1:]))
    return as_info, org_info

@click.command()
@click.option("--time", "-t", type=int, required=True, help="timestamp, like 20200901")
def main(time):
    as_info, org_info = load(time)
    while True:
        inp = input("ASN or org_id: ")
        if inp in as_info:
            print(f"asn: {inp}, {as_info[inp]}")
        elif inp in org_info:
            print(f"org_id: {inp}, {org_info[inp]}")
        else:
            print("no result")

if __name__ == "__main__":
    main()
