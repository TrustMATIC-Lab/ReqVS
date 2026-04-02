#!/usr/bin/env python3
import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_cfg_defaults
from dataloader import BenchmarktestDataset
from models import drugban  
from utils import graph_collate_func


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset CSV")
    p.add_argument("--weights", default=os.path.join(base_dir, "weights", "drugban_chembl36.pth"), help="Path to model weights .pth")
    p.add_argument("--config", default=os.path.join(base_dir, "configs", "drugban.yaml"), help="Config yaml path")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dataset_filename = os.path.basename(args.dataset)
    dataset_name = os.path.splitext(dataset_filename)[0]
    output_path = os.path.join(base_dir, "..","..", "test_results", dataset_name, f"{dataset_name}_drugban.csv")
    device = torch.device(args.device)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    model = drugban(**cfg).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    df = pd.read_csv(args.dataset)
    df.reset_index(drop=True, inplace=True)
    dataset = BenchmarktestDataset(df.index.values, df)
    loader = DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_collate_func,
    )

    all_preds = []
    with torch.no_grad():
        for v_d, v_p, _ in tqdm(loader):
            v_d, v_p = v_d.to(device), v_p.to(device)
            _, _, _, scores = model(v_d, v_p)
            probs = torch.sigmoid(scores).squeeze().cpu().numpy()
            all_preds.extend(probs)

    df["prediction_drugban"] = 0.0
    for pred, valid_id in zip(all_preds, dataset.valid_ids):
        if valid_id in df.index:
            df.loc[valid_id, "prediction_drugban"] = pred

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"drugban predictions of {args.dataset} saved to {output_path}")


if __name__ == "__main__":
    main()
