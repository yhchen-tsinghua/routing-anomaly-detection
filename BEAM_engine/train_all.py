#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from BEAM_model import BEAM
from datetime import datetime
import pandas as pd
import os 
import click

def job(date, method, retrain=False):
    serial = f"serial-{1 if date < 20151201 else 2}"
    as_rel = f"{date}.as-rel{'' if date < 20151201 else '2'}"
    epoches = 1000
    
    args = dict(Q=10, dimension=128, sample_method=method, p2p_link_error_only=False,
        edgeFile=f"/dataset/{serial}/{as_rel}.txt") # NOTE
    
    train_dir = Path(__file__).resolve().parent / f"bgp-{method}-{epoches}ep" # NOTE
    train_dir.mkdir(parents=True, exist_ok=True)
    args["train_dir"] = train_dir

    if not retrain and (train_dir/"link.emb").exists() \
        and (train_dir/"node.emb").exists() \
        and (train_dir/"rela.emb").exists():
        print(f"{as_rel} {method} existed")
        return

    model = BEAM(**args)
    model.train(epoches=epoches)
    model.saveEmbeddings(path=str(train_dir))


@click.command()
@click.option("--part", "-p", type=click.Choice(["0", "1", "2", "3"]), required=True, help="the part to run")
@click.option("--retrain", "-r", type=bool, default=False, help="retrain or not")
def main(part, retrain):
    part = int(part)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{9-part}"

    news = pd.read_csv("/dataset/event_info.csv") # TODO
    date = set([int(datetime.strptime(st, "%Y-%m-%dT%H:%M:%S").strftime("%Y%m01")) for st in news.start_time.values])
    # replace 20150301 with 20150201
    date.remove(20150301)
    date.add(20150201)
    date = sorted(list(date))[::-1]

    for i, d in enumerate(date):
        if i % 4 == part:
            print(f"job {i//4}...")
            job(d, "local", retrain)

if __name__ == "__main__":
    main()
