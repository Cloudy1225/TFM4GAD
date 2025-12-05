import logging
import os
import random
import shutil

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score


def metrics(labels, probs):
    """Compute AUROC, AUPRC, Recall@K (percentage)."""
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = float(roc_auc_score(labels, probs) * 100)
        score['AUPRC'] = float(average_precision_score(labels, probs) * 100)
        labels = np.array(labels)
        k = labels.sum()
    score['RecK'] = float(sum(labels[probs.argsort()[-int(k):]]) / sum(labels) * 100)
    return score

def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update nested dict (base ← overrides)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def get_training_config(dataset, config_path='train.conf.yaml'):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)

    default_config = full_config.get('default', {})
    dataset_config = full_config.get(dataset, {})

    # deep merge: dataset overrides default
    merged = deep_update(default_config.copy(), dataset_config)
    return merged


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):
        raise ValueError(f"No such file or directory! {path}")


def get_logger(filename, log_level=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[log_level])

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
