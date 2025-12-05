import numpy as np
import dgl, torch
from dgl.data.utils import load_graphs


def standard_scale(x, dim=0, eps=1e-6):
    m = x.mean(dim, keepdim=True)
    s = x.std(dim, unbiased=False, keepdim=True) + eps
    x -= m
    x /= s
    return x


def load_data(dataset, root, n_anomalies=20, feat_trans='no'):
    """load and preprocess dataset"""
    g = load_graphs(f'{root}/{dataset}')[0][0]
    x = g.ndata.pop('feature')
    y = g.ndata.pop('label')
    if n_anomalies == 20:
        train_masks = g.ndata.pop('train_masks')[:, 10:].bool()
        val_masks = g.ndata.pop('val_masks')[:, 10:].bool()
        test_masks = g.ndata.pop('test_masks')[:, 10:].bool()
    else:
        train_masks = g.ndata.pop('train_masks')[:, :10].bool()
        val_masks = g.ndata.pop('val_masks')[:, :10].bool()
        test_masks = g.ndata.pop('test_masks')[:, :10].bool()

    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    g.name = dataset

    if feat_trans == 'l1':
        x = torch.nn.functional.normalize(x, dim=1, p=1)
    elif feat_trans == 'l2':
        x = torch.nn.functional.normalize(x, dim=1, p=2)
    elif feat_trans == 'sc':
        x = standard_scale(x)
    elif feat_trans == 'row':
        x = x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.0))
    else:
        x = x

    return g, x, y, train_masks, val_masks, test_masks
