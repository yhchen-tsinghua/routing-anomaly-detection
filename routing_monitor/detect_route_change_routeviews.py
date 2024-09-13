#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import pickle
import click

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.routeviews.fetch_updates import load_updates_to_df, get_all_collectors, get_archive_list, download_data
from monitor import Monitor

SCRIPT_DIR = Path(__file__).resolve().parent

def detect(data, route_change_dir, snapshot_dir):
    mon = Monitor()

    for fpath in data:
        _, date, time = fpath.name.split(".")

        df = load_updates_to_df(fpath)
        df = df.sort_values(by="timestamp")

        mon.consume(df, detect=True)

        route_change_df = pd.DataFrame.from_records(mon.route_changes)
        mon.route_changes = []

        route_change_df.to_csv(route_change_dir/f"{date}.{time}.csv", index=False)

        if time == "2345":
            pickle.dump(mon, open(snapshot_dir/f"{date}.end-of-the-day", "wb"))

@click.command()
@click.option("--collector", "-c", type=str, default="wide", help="the name of RouteView collector to use")
@click.option("--year", "-y", type=int, required=True, help="the year to monitor, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month to monitor, e.g., 8")
def detect_monthly_for(collector, year, month):
    result_dir = SCRIPT_DIR/"detection_result"/collector
    route_change_dir = result_dir/"route_change"
    snapshot_dir = result_dir/"snapshot"

    route_change_dir.mkdir(exist_ok=True, parents=True)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    collectors2url = get_all_collectors()

    d1 = datetime(year=year, month=month, day=1)
    d2 = (datetime(year=year, month=month, day=28) + timedelta(days=4)
            ).replace(day=1) - timedelta(minutes=15)

    monthly_data = list(map(lambda url: download_data(url, collector),
                    get_archive_list(collector, collectors2url, d1, d2)))

    detect(monthly_data, route_change_dir, snapshot_dir)

if __name__ == "__main__":
    detect_monthly_for()
