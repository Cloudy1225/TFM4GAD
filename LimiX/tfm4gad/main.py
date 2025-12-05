import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "29500"

# Local imports
from dataloader import load_data
from augment import augment_node_features
from misc import get_training_config, get_logger, set_seed, metrics

# Ensure project root is on path so inference.predictor can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference.predictor import LimiXPredictor


# ---------------- Helper ---------------- #
def run_single_test(predictor, X_train, y_train, X_test, y_test, batch_size, logger):
    """Run batched inference on test set."""
    n_test = X_test.shape[0]
    probs_all = np.zeros((n_test,), dtype=float)
    for i in tqdm(range(0, n_test, batch_size), desc="Batches", disable=True):
        X_batch = X_test[i:i + batch_size]
        pred = predictor.predict(X_train, y_train, X_batch, task_type="Classification")
        if pred.ndim == 2 and pred.shape[1] >= 2:
            probs = pred[:, 1]
        else:
            probs = pred.ravel()
        probs_all[i:i + batch_size] = probs
    score = metrics(y_test, probs_all)
    logger.info(f"AUROC={score['AUROC']:.2f}%, AUPRC={score['AUPRC']:.2f}%, RecK={score['RecK']:.2f}%")
    return score

# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Graph anomaly detection with LimiX classifier")
    parser.add_argument("--device", type=str, default="cuda:0",)
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices=['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
                                 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial'])
    parser.add_argument('--data_dir', type=str, default="../../datasets",
                        help="Path to dataset")
    parser.add_argument('--config', type=str, default='icl.conf.yaml',
                        help="Path to config YAML")
    parser.add_argument('--model_ckpt', type=str, default='../ckpts/LimiX-16M.ckpt',
                        choices=['../ckpts/LimiX-16M.ckpt', '../ckpts/LimiX-2M.ckpt'])
    parser.add_argument('--inference_config', type=str, default='../config/cls_default_noretrieval.json',)
    parser.add_argument('--master_port', type=str, default='29500',)
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    args = parser.parse_args()
    os.environ["MASTER_PORT"] = args.master_port

    # ---------------- Setup logging and seed ---------------- #
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join('log', f'{args.dataset}2.log')
    logger = get_logger(log_path, log_level=1, name="MainLogger", mode='a')
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # ---------------- Read config ---------------- #
    conf = get_training_config(args.dataset, config_path=args.config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    for key in ["pagerank", "laplacian_pe"]:
        if key in conf.get("augmentation", {}) and "cache_dir" in conf["augmentation"][key]:
            conf["augmentation"][key]["cache_dir"] = os.path.join(data_dir, conf["augmentation"][key]["cache_dir"])
    logger.info(f"Loaded config for dataset={args.dataset}: {conf}")

    # ---------------- Load graph ---------------- #
    g, x, y, train_masks, val_masks, test_masks = load_data(
        args.dataset,
        root=args.data_dir,
        n_anomalies=conf.get('n_anomalies', 20),
        feat_trans=conf.get('feat_trans', 'no')
    )
    n_splits = test_masks.shape[1]
    logger.info(f"Loaded graph with {g.num_nodes()} nodes, splits={n_splits}")

    # ---------------- Feature augmentation ---------------- #
    if conf.get('augmentation', {}).get('enable', False):
        logger.info("Applying configurable feature augmentation...")
        features = augment_node_features(g, x, conf['augmentation'])
        logger.info(f"Feature augmentation done, initial_dim={x.shape[1]}, final_dim={features.shape[1]}")
    else:
        logger.info(f"Using raw node features, final_dim={x.shape[1]}")
        features = x

    # Convert features/labels to numpy
    features_np = features.cpu().numpy()
    labels_np = y.cpu().numpy().ravel()

    # ---------------- Predictor init ---------------- #
    predictor = LimiXPredictor(
        device=args.device if torch.cuda.is_available() else 'cpu',
        model_path=args.model_ckpt,
        inference_config=args.inference_config
    )
    logger.info(f"Predictor initialized, model_path={args.model_ckpt}")

    # ---------------- Splits evaluation ---------------- #
    split_scores = []
    for split_id in range(n_splits):
        logger.info(f"\n=== Split {split_id+1}/{n_splits} ===")
        train_idx = np.where(train_masks[:, split_id].cpu().numpy())[0]
        test_idx = np.where(test_masks[:, split_id].cpu().numpy())[0]
        logger.info(f"Train size={len(train_idx)}, Test size={len(test_idx)}")

        X_train, y_train = features_np[train_idx], labels_np[train_idx].astype(int)
        X_test, y_test = features_np[test_idx], labels_np[test_idx].astype(int)

        score = run_single_test(
            predictor, X_train, y_train, X_test, y_test,
            batch_size=conf.get('batch_size', 2048),
            logger=logger,
        )
        split_scores.append(score)

    # ---------------- Aggregate metrics ---------------- #
    def agg(name):
        vals = np.array([s[name] for s in split_scores])
        return vals.mean(), vals.std()

    auroc_m, auroc_s = agg('AUROC')
    auprc_m, auprc_s = agg('AUPRC')
    reck_m, reck_s = agg('RecK')
    logger.info("\n=== Final Summary ===")
    logger.info(f"AUROC={auroc_m:.2f}±{auroc_s:.2f}, AUPRC={auprc_m:.2f}±{auprc_s:.2f}, RecK={reck_m:.2f}±{reck_s:.2f}\n")

if __name__ == "__main__":
    main()
