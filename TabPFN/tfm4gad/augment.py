import os
import time

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import scipy.special
import sympy
import torch
from scipy.sparse.linalg import lobpcg, LinearOperator
from sklearn.utils.extmath import randomized_svd


def standard_scale(x, dim=0, eps=1e-6):
    m = x.mean(dim, keepdim=True)
    s = x.std(dim, unbiased=False, keepdim=True) + eps
    x -= m
    x /= s
    return x


def sage_feature_augment(g: dgl.DGLGraph, feat: torch.Tensor,
                             num_layers: int=1, agg: str='mean', sc: bool=False) -> torch.Tensor:
    """
    Feature augmentation using non-parametric SAGE-style polynomial convolution.

    Parameters:
        g: DGLGraph
        feat: Tensor [num_nodes, feat_dim]
        agg: 'sum' / 'mean' / 'max'
        num_layers: number of aggregation hops
        sc: bool, apply standard scaler to aggregated features at each hop
    Returns:
        Tensor [num_nodes, feat_dim * (num_layers + 1)]
    """
    assert agg in ['sum', 'mean', 'max'], "agg must be one of 'sum', 'mean', 'max'"

    results = [feat if not sc else standard_scale(feat)]
    h = feat

    with g.local_scope():  # temporary node data
        for _ in range(num_layers):
            g.ndata['h'] = h
            if agg == 'sum':
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            elif agg == 'mean':
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            elif agg == 'max':
                g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))

            h = g.ndata['neigh']
            if sc:
                h = standard_scale(h)
            results.append(h)

    return torch.cat(results, dim=-1)


def calculate_theta(d):
    """
    Calculate fixed polynomial filter coefficients based on beta distribution.

    Args:
        d (int): maximum polynomial degree.

    Returns:
        list[list[float]]: shape (d+1, d+1), each sublist is a polynomial coefficient set.
    """
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) /
                       (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()  # length = d+1
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))  # reverse order
        thetas.append(inv_coeff)
    return thetas


def poly_conv(graph, feat, theta):
    """
    Perform graph polynomial convolution with fixed coefficients (no learnable parameters).

    Args:
        graph (DGLGraph): input graph.
        feat (Tensor): node feature matrix [num_nodes, feat_dim].
        theta (list[float]): polynomial filter coefficients.

    Returns:
        Tensor: updated node features after polynomial convolution.
    """

    def unn_laplacian(feat, D_invsqrt, graph):
        """
        Apply unnormalized Laplacian: feat - D^-1/2 * A * D^-1/2 * feat
        """
        graph.ndata['h'] = feat * D_invsqrt
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return feat - graph.ndata.pop('h') * D_invsqrt

    with graph.local_scope():
        # Calculate D^-1/2 for symmetric normalization
        D_invsqrt = torch.pow(
            graph.in_degrees().float().clamp(min=1), -0.5
        ).unsqueeze(-1).to(feat.device)

        h = theta[0] * feat
        for k in range(1, len(theta)):
            feat = unn_laplacian(feat, D_invsqrt, graph)
            h += theta[k] * feat
    return h


def bwgnn_feature_augment(graph: dgl.DGLGraph, feat: torch.Tensor, num_layers: int=2) -> torch.Tensor:
    """
    Feature augmentation using non-parametric BWGNN-style polynomial convolution.

    Args:
        graph (DGLGraph): input graph.
        feat (Tensor): node features [num_nodes, feat_dim].
        num_layers (int): maximum polynomial degree (d).

    Returns:
        Tensor [num_nodes, feat_dim * (d+1)]: concatenated convolution results.
    """
    # Generate fixed polynomial coefficients
    thetas = calculate_theta(d=num_layers)

    results = []
    # Apply convolution for each polynomial filter
    for theta in thetas:
        h_conv = poly_conv(graph, feat, theta)
        results.append(h_conv)

    # Concatenate all convolution results along feature dimension
    return torch.cat(results, dim=-1)


def compute_degree_encoding(
        graph: dgl.DGLGraph,
        log: bool = True,
) -> torch.Tensor:
    degrees = 0.5 * (graph.in_degrees() + graph.out_degrees())
    if log:
        degrees = torch.log(1 + degrees)
    return degrees.unsqueeze(-1)


def compute_pagerank(
        graph: dgl.DGLGraph,
        alpha: float = 0.85,
        max_iterations: int = 100,
        tol=1e-6,
        log: bool = True,
        cache_dir: str = "./pagerank_cache"
) -> torch.Tensor:
    """
    Compute PageRank scores with optional caching.
    Uses dgl.function for message passing.
    Caches only the raw (non-log) PageRank scores.

    Parameters:
        graph (dgl.DGLGraph): Input graph.
        alpha (float): Damping factor (> 0.5).
        max_iterations (int): Maximum iterations.
        tol (float): Convergence tolerance.
        log (bool): Apply log-transform to final scores before returning.
        cache_dir (str or None): Directory to save/load cached raw results.
    """
    assert alpha > 0.5, "Please make sure that you provide alpha, not 1-alpha"

    # Prepare cache path if cache_dir is given
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        g_name = getattr(graph, "name", "graph")
        cache_path = os.path.join(cache_dir, f"{g_name}_alpha{alpha}_pagerank.pt")
        if os.path.exists(cache_path):
            # Load raw PV from cache
            pv = torch.load(cache_path, weights_only=False)
            # Apply log if requested
            if log:
                pv = torch.log(tol + pv)
            return pv.unsqueeze(-1)

    n_nodes = graph.num_nodes()
    pv = torch.ones(n_nodes) / n_nodes
    degrees = graph.out_degrees().float().clamp(min=1)  # Prevent division by zero

    # Reset probability vector
    reset_prob = (1 - alpha) / n_nodes

    # -------- PageRank Iteration --------
    with graph.local_scope():
        for _ in range(max_iterations):
            prev_pv = pv.clone()

            graph.ndata['pv'] = pv / degrees
            graph.update_all(fn.copy_u('pv', 'm'), fn.sum('m', 'pv_new'))
            new_pv = graph.ndata.pop('pv_new')

            pv = alpha * new_pv + reset_prob

            # Convergence check
            if torch.abs(pv - prev_pv).sum() < tol:
                break

    # Save raw pv (non-log) to cache
    if cache_path is not None:
        torch.save(pv, cache_path)

    # Apply log transform if requested (for returning)
    if log:
        pv = torch.log(tol + pv)

    # Return column vector
    return pv.unsqueeze(-1)


def get_laplacian_pe(g, k, cache_dir="./lap_pe_cache",
                     approx_threshold=100_000, approx_method="randomized_svd") -> torch.Tensor:
    """
    Compute or load Laplacian Position Embedding (LapPE) for graph `g`.

    If a cache file exists and its dimensionality >= k:
        - Load from cache and return the first k dimensions.
    Else:
        - If num_nodes exceeds `approx_threshold`, use approximate spectral method.
        - Otherwise, compute LapPE using exact sparse eigendecomposition.

    Parameters
    ----------
    g : DGLGraph
        Input graph.
    k : int
        Number of Laplacian PE dimensions required.
    cache_dir : str
        Directory to store cache files.
    approx_threshold : int
        Number of nodes above which approximate method will be used.
    approx_method : str
        Approximation method to use: "randomized_svd" or "lobpcg".

    Returns
    -------
    pe : torch.Tensor
        Tensor of shape [num_nodes, k], Laplacian PE.
    """

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{g.name}_lap_pe.pt")

    # Try to load from cache
    cached_k = -1
    if os.path.exists(cache_path):
        start_time = time.time()
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        cached_k = cached.shape[1]
        if cached_k >= k:
            elapsed = time.time() - start_time
            print(f"[Cache] Loaded LapPE from '{cache_path}' (cached_k={cached_k}) in {elapsed:.2f} seconds")
            return cached[:, :k]
        else:
            print(f"[Cache] Found LapPE cache k={cached_k} but need k={k}, recomputing...")

    num_nodes = g.num_nodes()

    # Get adjacency matrix and normalized Laplacian in sparse format
    A = g.adj_external(scipy_fmt="csr")  # adjacency in CSR
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(g.in_degrees().numpy(), 1, None))
    D_half = sp.diags(deg_inv_sqrt, dtype=float)
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = sp.eye(num_nodes) - D_half @ A @ D_half

    # Choose computation method
    if num_nodes > approx_threshold:
        start_time = time.time()
        print(f"[Approx] Using {approx_method} for LapPE (n={num_nodes} > threshold={approx_threshold}, k={k})")

        if approx_method == "randomized_svd":
            # Randomized SVD for smallest eigenvectors of Laplacian
            U, Sigma, VT = randomized_svd(L, n_components=k+1, random_state=0)
            PE_vecs = U[:, 1:k+1]  # skip the trivial eigenvector
        elif approx_method == "lobpcg":
            # Define LinearOperator for Laplacian to avoid dense matrix
            def matvec(x):
                x = x.reshape(-1, 1)
                Ax = A @ (deg_inv_sqrt[:, None] * x)
                return (x - deg_inv_sqrt[:, None] * Ax).ravel()

            Lop = LinearOperator((num_nodes, num_nodes), matvec=matvec)
            X_init = np.random.rand(num_nodes, k+1)
            eigvals, eigvecs = lobpcg(Lop, X_init, largest=False, tol=1e-4, maxiter=200)
            PE_vecs = eigvecs[:, 1:k+1]
        else:
            raise ValueError(f"Unknown approx_method: {approx_method}")

        # Randomly flip signs for stability
        rand_sign = 2 * (np.random.rand(k) > 0.5) - 1.0
        PE = torch.tensor(PE_vecs * rand_sign, dtype=torch.float32)

        elapsed = time.time() - start_time
        print(f"[Approx] LapPE computed in {elapsed:.2f} seconds (method={approx_method})")

    else:
        start_time = time.time()
        print(f"[Exact] Calculating Laplacian PE (n={num_nodes}, k={k}) ...")

        if k + 1 < num_nodes - 1:
            # Use ARPACK for exact eigen-decomposition of sparse matrix
            EigVal, EigVec = sp.linalg.eigs(L, k=k+1, which="SR", ncv=4*k, tol=1e-2)
            topk_indices = EigVal.argsort()[1:]  # skip trivial eigenvalue
        else:
            # Fallback: dense eigendecomposition (only feasible for small graphs!)
            EigVal, EigVec = np.linalg.eig(L.toarray())
            max_freqs = min(num_nodes - 1, k)
            kpartition_indices = np.argpartition(EigVal, max_freqs)[:max_freqs+1]
            topk_eigvals = EigVal[kpartition_indices]
            topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]

        topk_EigVec = EigVec[:, topk_indices].real
        rand_sign = 2 * (np.random.rand(len(topk_indices)) > 0.5) - 1.0
        PE = torch.tensor(topk_EigVec * rand_sign, dtype=torch.float32)

        elapsed = time.time() - start_time
        print(f"[Exact] LapPE computed in {elapsed:.2f} seconds (exact method)")

    # Save to cache
    if k > cached_k:
        torch.save(PE, cache_path)
        print(f"[Cache] Saved LapPE to '{cache_path}' (k={k})")
    return PE


def augment_node_features(
    graph: dgl.DGLGraph,
    base_feat: torch.Tensor,
    conf: dict
) -> torch.Tensor:
    """
    Apply composable node feature augmentation on (graph, base_feat) according to config.

    Parameters
    ----------
    graph : dgl.DGLGraph
        Input graph (should be preprocessed: bidirectional, self-loops if needed).
    base_feat : torch.Tensor
        Initial node feature matrix, shape [num_nodes, feat_dim].
    conf : dict
        Augmentation configuration, for example:
        {
            "neighbor": {
                "enable": True,
                "type": "sage",          # 'sage' or 'bwgnn'
                "num_layers": 2,
                "agg": "mean",           # 'sum', 'mean', 'max' (only for type='sage')
                "sc": False
            },
            "degree": {
                "enable": True,
                "log": True
            },
            "pagerank": {
                "enable": True,
                "alpha": 0.85,
                "max_iterations": 100,
                "tol": 1e-6,
                "log": True,
                "cache_dir": "./pagerank_cache"
            },
            "laplacian_pe": {
                "enable": True,
                "k": 16,
                "cache_dir": "./lap_pe_cache",
                "approx_threshold": 100_000,
                "approx_method": "randomized_svd"
            }
        }

    Returns
    -------
    torch.Tensor
        Augmented feature tensor, shape [num_nodes, augmented_dim]
    """
    # --- store all feature blocks to concatenate later ---
    features = []

    # ---- 1. Neighbor-based augmentation ----
    if conf.get("neighbor", {}).get("enable", False):
        neigh_conf = conf["neighbor"]
        if neigh_conf["type"] == "sage":
            h_aug = sage_feature_augment(
                g=graph,
                feat=base_feat,
                num_layers=neigh_conf.get("num_layers", 1),
                agg=neigh_conf.get("agg", "mean"),
                sc=neigh_conf.get("sc", False)
            )
        elif neigh_conf["type"] == "bwgnn":
            h_aug = bwgnn_feature_augment(
                graph,
                feat=base_feat,
                num_layers=neigh_conf.get("num_layers", 2)
            )
        else:
            h_aug = base_feat
        features.append(h_aug)

    # ---- 2. Degree encoding ----
    if conf.get("degree", {}).get("enable", False):
        deg_conf = conf["degree"]
        deg_feat = compute_degree_encoding(graph, log=deg_conf.get("log", True))
        features.append(deg_feat)

    # ---- 3. PageRank ----
    if conf.get("pagerank", {}).get("enable", False):
        pr_conf = conf["pagerank"]
        pr_feat = compute_pagerank(
            graph,
            alpha=pr_conf.get("alpha", 0.85),
            max_iterations=pr_conf.get("max_iterations", 100),
            tol=pr_conf.get("tol", 1e-6),
            log=pr_conf.get("log", True),
            cache_dir=pr_conf.get("cache_dir", "./pagerank_cache")
        )
        features.append(pr_feat)

    # ---- 4. Laplacian PE ----
    if conf.get("laplacian_pe", {}).get("enable", False):
        lap_conf = conf["laplacian_pe"]
        if not hasattr(graph, "name"):
            graph.name = lap_conf.get("graph_name", "unnamed_graph")
        lap_feat = get_laplacian_pe(
            g=graph,
            k=lap_conf.get("k", 16),
            cache_dir=lap_conf.get("cache_dir", "./lap_pe_cache"),
            approx_threshold=lap_conf.get("approx_threshold", 100_000),
            approx_method=lap_conf.get("approx_method", "randomized_svd")
        )
        features.append(lap_feat)

    # ---- Concatenate all features along last dimension ----
    augmented_feat = torch.cat(
        [f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
         for f in features],
        dim=-1
    )

    return augmented_feat
