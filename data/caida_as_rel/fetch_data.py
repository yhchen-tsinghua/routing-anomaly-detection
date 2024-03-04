#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import click

SCRIPT_DIR = Path(__file__).resolve().parent

SERIAL_1_DIR = SCRIPT_DIR / "serial-1"
SERIAL_2_DIR = SCRIPT_DIR / "serial-2"

SERIAL_1_DIR.mkdir(exist_ok=True, parents=True)
SERIAL_2_DIR.mkdir(exist_ok=True, parents=True)

def get(serial: str, time: int):
    if serial == "1":
        fname = f"{time}.as-rel.txt.bz2"
        obj = f"https://publicdata.caida.org/datasets/as-relationships/serial-1/{fname}"
        out = SERIAL_1_DIR / fname
    elif serial == "2":
        fname = f"{time}.as-rel2.txt.bz2"
        obj = f"https://publicdata.caida.org/datasets/as-relationships/serial-2/{fname}"
        out = SERIAL_2_DIR / fname
    else:
        raise RuntimeError("bad argument")
    if out.with_suffix("").exists():
        # print(f"as-relationship for {serial} {time} already existed")
        return out.with_suffix("")
    subprocess.run(["curl", obj, "--output", str(out)], check=True)
    subprocess.run(["bzip2", "-d", str(out)], check=True)
    print(f"get as-relationship for {serial} {time}")
    return out.with_suffix("")

@click.command()
@click.option("--serial", "-s", type=click.Choice(["1", "2"]), default="1", help="serial 1 or 2")
@click.option("--time", "-t", type=int, required=True, help="timestamp, e.g., 20200901")
def main(serial, time):
    get(serial, time)

if __name__ == "__main__":
    main()
