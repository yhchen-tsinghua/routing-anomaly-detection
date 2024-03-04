#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from BEAM_model import BEAM
from shutil import get_terminal_size
import click
import os

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.caida_as_rel.fetch_data import get as prepare_edge_file

@click.command()
@click.option("--serial", "-s", type=click.Choice(["1", "2"]), default="1", help="serial 1 or 2")
@click.option("--time", "-t", type=int, required=True, help="timestamp, e.g., 20200901")
@click.option("--Q", "Q", type=int, default=10, help="hyperparameter Q, e.g., 10")
@click.option("--dimension", type=int, default=128, help="hyperparameter dimension size, e.g., 128")
@click.option("--epoches", type=int, default=1000, help="epoches to train, e.g., 1000")
@click.option("--device", type=int, default=0, help="device to train on")
@click.option("--num-workers", type=int, default=1, help="number of workers")
def main(serial, time, device, **model_params):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

    edge_file = prepare_edge_file(serial, time)
    assert edge_file.exists(), f"fail to prepare {edge_file}"

    model_params["edge_file"] = edge_file

    for k, v in model_params.items():
        print(f"{k}: {v}")
    print("*"*get_terminal_size().columns)
    # input("Press Enter to start.")

    train_dir = Path(__file__).resolve().parent/"models"/ \
        f"{edge_file.stem}.{model_params['epoches']}.{model_params['Q']}.{model_params['dimension']}"
    train_dir.mkdir(parents=True, exist_ok=True)
    model_params["train_dir"] = train_dir
    epoches = model_params.pop("epoches")

    model = BEAM(**model_params)
    model.train(epoches=epoches)
    model.save_embeddings(path=str(train_dir))

if __name__ == "__main__":
    main()
