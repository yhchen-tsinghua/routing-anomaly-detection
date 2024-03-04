#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import click

SCRIPT_DIR = Path(__file__).resolve().parent

def load(serial, time):
    f = SCRIPT_DIR/f"serial-{serial}"/f"{time}.as-rel{'' if serial == '1' else 2}.txt"

    ngbrs = dict()
    for line in open(f, "r"):
        if line[0] == "#": continue
        i, j, k = line.strip().split("|")[:3]
        ngbrs.setdefault(i, {-1: set(), 0: set(), 1: set()})[int(k)].add(j)
        ngbrs.setdefault(j, {-1: set(), 0: set(), 1: set()})[-int(k)].add(i)

    def query(i, j):
        if i not in ngbrs: print(f"Unknown AS: {i}"); return None
        if j not in ngbrs: print(f"Unknown AS: {j}"); return None
        for k,v in ngbrs[i].items():
            if j in v: return k
        return None

    return query


@click.command()
@click.option("--serial", "-s", type=click.Choice(["1", "2"]), default="1", help="serial 1 or 2")
@click.option("--time", "-t", type=int, required=True, help="timestamp, e.g., 20200901")
def main(serial, time):
    query = load(serial, time)

    while True:
        i = input("AS1: ")
        j = input("AS2: ")
        print(query(i, j))

if __name__ == "__main__":
    main()
