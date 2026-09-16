import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


GATORANCHOR_CONFIG = {
    "n_gnn": 3,
    "deep_split": 0.6,
    "lr_ft": 0.1,
    "k_fuse": 7,
    "drop_q": 0.8,
    "r_max": 0.5,
    "lambda_D": 0.1,
    "lambda_B": 0.5,
    "lambda_L": 1.0,
    "encoder_dropout": 0.15,
    "source_bias_observation": 1.5,
    "source_bias_reconstruction": 0.4,
    "observation_temperature": 0.2,
    "epsilon_cosine": 1e-8,
    "epsilon_precision": 1e-12,
    "epsilon_unreliability": 1e-4,
    "epsilon_source": 1e-12,
    "epsilon_attenuation": 1e-8,
    "epsilon_degree": 1e-8,
    "epsilon_curvature": 1e-6,
}


def _prep(A, n_hvg, is_adt):
    A = A.copy()
    A.X = A.X.astype("float32")
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    if (not is_adt) and A.n_vars > n_hvg:
        sc.pp.highly_variable_genes(A, n_top_genes=n_hvg, flavor="seurat")
        A = A[:, A.var["highly_variable"]].copy()
    sc.pp.scale(A)
    X = A.X
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype="float32")


def _pca(X, d):
    d = int(min(d, X.shape[1] - 1, X.shape[0] - 1))
    if d < 1:
        raise ValueError("PCA requires at least two cells and two retained features.")
    return PCA(n_components=d, random_state=0).fit_transform(X).astype("float32")


def _sym_norm(row, col, w, n, epsilon_degree=1e-8):
    row = np.asarray(row)
    col = np.asarray(col)
    w = np.asarray(w)
    nonself = row != col
    row, col, w = row[nonself], col[nonself], w[nonself]
    A = sp.coo_matrix((w, (row, col)), shape=(n, n))
    A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel()
    if epsilon_degree is None:
        if np.any(d <= 0) or not np.isfinite(d).all():
            raise ValueError("The initial graph must have finite, positive weighted degrees.")
        degree = d
    else:
        degree = np.maximum(d, epsilon_degree)
    dinv = sp.diags(1.0 / np.sqrt(degree))
    return (dinv @ A @ dinv).tocsr()


def _multihop(H, An, hops):
    feats = [H]
    cur = H
    for _ in range(hops):
        cur = An @ cur
        feats.append(cur)
    return np.concatenate(feats, 1).astype("float32")


def _sp_to_torch(A, dev):

    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=dev)
    val = torch.tensor(A.data, dtype=torch.float32, device=dev)
    return torch.sparse_coo_tensor(idx, val, A.shape, device=dev).coalesce()


def _graph_mm(A, H):

    if not torch.are_deterministic_algorithms_enabled():
        return torch.sparse.mm(A, H)
    A = A.coalesce()
    row, col = A.indices()
    contribution = H.index_select(0, col) * A.values().unsqueeze(1)
    lengths = torch.bincount(row, minlength=A.shape[0])
    return torch.segment_reduce(contribution, "sum", lengths=lengths)


class DeepEnc(nn.Module):

    def __init__(self, din_mh, din_pca, dz, n_gnn=2, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.mu_s = nn.Linear(din_mh, dz)
        self.lv_s = nn.Linear(din_mh, dz)
        dims = [din_pca] + [dz] * n_gnn
        self.gcn = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_gnn)]
        )
        self.bn = nn.ModuleList([nn.BatchNorm1d(dz) for _ in range(n_gnn)])
        self.mu_d = nn.Linear(dz, dz)
        self.lv_d = nn.Linear(dz, dz)
        self.gate = nn.Parameter(torch.zeros(1))
        self.use_deep = False

    def freeze_shallow(self, flag):
        for p in list(self.mu_s.parameters()) + list(self.lv_s.parameters()):
            p.requires_grad = not flag

    def forward(self, x_mh, x_pca, A):
        xs = self.drop(x_mh)
        mu_s, lv_s = self.mu_s(xs), self.lv_s(xs)
        if not self.use_deep:
            return mu_s, lv_s
        h = x_pca
        for lin, bn in zip(self.gcn, self.bn):
            h2 = F.relu(bn(_graph_mm(A, lin(h))))
            h = h2 + h if h2.shape == h.shape else h2
        return mu_s + self.gate * self.mu_d(h), lv_s + self.gate * self.lv_d(h)


class Dec(nn.Module):
    def __init__(self, dz, dout):
        super().__init__()
        self.net = nn.Linear(dz, dout)

    def forward(self, z):
        return self.net(z)


class EdgeDec(nn.Module):

    def __init__(self, dz):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dz, dz), nn.ReLU(), nn.Linear(dz, 1))

    def pair(self, zi, zj):
        return torch.cat([zi, zj, (zi - zj).abs(), zi * zj], -1)

    def feat(self, zi, zj):
        return self.net[1](self.net[0](self.pair(zi, zj)))

    def head(self, phi):
        return self.net[2](phi).squeeze(-1)

    def forward(self, zi, zj):
        return self.net(self.pair(zi, zj)).squeeze(-1)


class GatorAnchor(nn.Module):
    def __init__(self, in_dims, rec_dims, dz=128, lamD=0.1, p_drop=0.15, n_gnn=3):
        super().__init__()
        self.lamD = lamD
        self.enc = nn.ModuleList(
            [
                DeepEnc(in_dims[i], rec_dims[i], dz, n_gnn, p_drop)
                for i in range(len(in_dims))
            ]
        )
        self.dec = nn.ModuleList([Dec(dz, d) for d in rec_dims])
        self.edge = EdgeDec(dz)
        self._pcas = None
        self._A = None

    def set_graph(self, pcas, A):
        self._pcas = pcas
        self._A = A

    def set_deep(self, on):

        for e in self.enc:
            if isinstance(e, DeepEnc):
                e.use_deep = bool(on)

    def freeze_shallow(self, flag):
        for e in self.enc:
            if isinstance(e, DeepEnc):
                e.freeze_shallow(flag)

    def gates(self):
        return [
            float(e.gate.detach().abs().mean())
            for e in self.enc
            if isinstance(e, DeepEnc)
        ]

    def poe(self, mus, lvs):
        P = torch.ones_like(mus[0])
        Pmu = torch.zeros_like(mus[0])
        for mu, lv in zip(mus, lvs):
            p = torch.exp(-lv)
            P = P + p
            Pmu = Pmu + p * mu
        var = 1.0 / P
        mu = var * Pmu
        if len(mus) > 1:
            var = var + self.lamD * torch.stack(mus, 0).var(0)
        return mu, var

    def encode(self, feats):
        mus, lvs = [], []
        for i, (e, x) in enumerate(zip(self.enc, feats)):
            m, l = e(x, self._pcas[i], self._A)
            mus.append(m)
            lvs.append(l)
        return self.poe(mus, lvs) + (mus, lvs)


def _knn_edges(F, k, n):
    if int(k) != k:
        raise ValueError("The neighbour count must be an integer.")
    k = int(k)
    if not 1 <= k < n:
        raise ValueError("The neighbour count must satisfy 1 <= k < n.")
    nn_ = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(F)
    d, idx = nn_.kneighbors(F)
    selected_index = np.empty((n, k), dtype=np.int64)
    selected_distance = np.empty((n, k), dtype=d.dtype)
    for cell in range(n):
        nonself = idx[cell] != cell
        selected_index[cell] = idx[cell][nonself][:k]
        selected_distance[cell] = d[cell][nonself][:k]
    return (
        np.repeat(np.arange(n), k),
        selected_index.reshape(-1),
        (1.0 - selected_distance.reshape(-1)).astype("float32"),
    )


def _candidate_knn_edges(F, k, n):
    if int(k) != k or not 1 <= k < n:
        raise ValueError("The candidate neighbour count must be an integer in [1, n-1].")
    k = int(k)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(F)
    distances, indices = neighbors.kneighbors(F)
    return (
        np.repeat(np.arange(n), k),
        indices[:, 1:].reshape(-1),
        (1.0 - distances[:, 1:].reshape(-1)).astype("float32"),
    )


def _laplace_hinv(net, mu, ii, jj, dev, damping=1.0, epsilon_curvature=1e-6):

    with torch.no_grad():
        Phi = torch.cat(
            [net.edge.feat(mu[ii[0]], mu[ii[1]]), net.edge.feat(mu[ii[0]], mu[jj])], 0
        )
        p = torch.sigmoid(net.edge.head(Phi))
        w = (p * (1 - p)).clamp_min(epsilon_curvature)
        H = (Phi * w[:, None]).T @ Phi
        H = H + damping * torch.eye(H.shape[0], device=dev, dtype=H.dtype)
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except Exception:
            Hinv = torch.linalg.pinv(H)
    return Hinv


def _laplace_var(net, Hinv, zi, zj):

    with torch.no_grad():
        phi = net.edge.feat(zi, zj)
        v_logit = ((phi @ Hinv) * phi).sum(-1).clamp_min(0.0)
        p = torch.sigmoid(net.edge.head(phi))
        return (p * (1 - p)) ** 2 * v_logit


def _fusion_weights(penalties, biases, epsilon_source=1e-12):
    normalized = penalties / (penalties.mean(1, keepdims=True) + float(epsilon_source))
    precision = np.asarray(biases, dtype="float64")[:, None] / normalized
    weights = precision / precision.sum(0, keepdims=True)
    return normalized, weights


def _evidence_fusion(
    net,
    feats,
    mu0,
    lvs,
    Pr,
    Pm,
    is_atac,
    row,
    col,
    n,
    k_graph,
    T,
    dev,
    drop_q=0.9,
    k_fuse=None,
    b_obs=1.5,
    b_rec=0.4,
    lambda_L=1.0,
    epsilon_cosine=1e-8,
    epsilon_precision=1e-12,
    epsilon_unreliability=1e-4,
    epsilon_source=1e-12,
    epsilon_attenuation=1e-8,
    epsilon_degree=1e-8,
    epsilon_curvature=1e-6,
):

    z = mu0.detach().cpu().numpy()
    mods = [Pr, Pm]
    mod_names = ["obs:RNA", "obs:ATAC" if is_atac else "obs:ADT"]

    kf = int(k_fuse if k_fuse is not None else max(1, k_graph // 2))
    if not 1 <= kf < n:
        raise ValueError("The fusion neighbour count must satisfy 1 <= k_fuse < n.")
    rr, rc, _ = _candidate_knn_edges(z, kf, n)
    ci = [row, rr]
    cj = [col, rc]
    for Pmod in mods:
        a, b, _ = _candidate_knn_edges(Pmod, kf, n)
        ci.append(a)
        cj.append(b)
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)
    key = np.minimum(ci, cj).astype(np.int64) * n + np.maximum(ci, cj)
    _, u = np.unique(key, return_index=True)
    ci, cj = ci[u], cj[u]
    cei = torch.tensor(np.stack([ci, cj]), device=dev, dtype=torch.long)

    def cos(F):
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + epsilon_cosine)
        return np.clip((Fn[ci] * Fn[cj]).sum(1), 0.0, 1.0).astype("float32")

    with torch.no_grad():
        prec = np.array([float(torch.exp(-lv).mean()) for lv in lvs], dtype="float64")
    om = prec / (prec.mean() + epsilon_precision)
    MU, V, names, bias = [], [], [], []
    for Pmod, w_m, nm in zip(mods, om, mod_names):
        s = cos(Pmod)
        MU.append(s)
        V.append(
            ((1.0 - s) + epsilon_unreliability) / max(float(w_m), 1e-3)
        )
        names.append(nm)
        bias.append(b_obs)

    net.eval()
    with torch.no_grad():
        ms = net.encode(feats)[0]
        nj = torch.randint(0, n, (cei.shape[1],), device=dev)
        Hinv = _laplace_hinv(
            net,
            ms,
            cei,
            nj,
            dev,
            damping=lambda_L,
            epsilon_curvature=epsilon_curvature,
        )
        mu_rec = torch.sigmoid(net.edge(ms[cei[0]], ms[cei[1]])).cpu().numpy()
        v_rec = (
            _laplace_var(net, Hinv, ms[cei[0]], ms[cei[1]]).cpu().numpy()
            + epsilon_unreliability
        )
    MU.append(mu_rec)
    V.append(v_rec)
    names.append("rec")
    bias.append(b_rec)

    MU = np.stack(MU)
    V = np.stack(V)
    Vn, W = _fusion_weights(V, bias, epsilon_source)
    mu_f = (W * MU).sum(0)
    conflict = (W * (MU - mu_f) ** 2).sum(0)
    v_f = (W * Vn).sum(0) + conflict

    nm_ = len(mods)
    Wm = W[:nm_] / (W[:nm_].sum(0, keepdims=True) + epsilon_source)
    mu_mod = (Wm * MU[:nm_]).sum(0)
    conf_mod = (Wm * (MU[:nm_] - mu_mod) ** 2).sum(0)
    conf_rec = (mu_rec - mu_mod) ** 2
    c = np.exp(-v_f / (T * (v_f.mean() + epsilon_attenuation)))
    w = (mu_f * c).astype("float32")
    keep = v_f < np.quantile(v_f, drop_q)
    An2 = _sym_norm(ci[keep], cj[keep], w[keep], n, epsilon_degree=epsilon_degree)
    return An2, {
        "mean_cross_source_conflict": float(conflict.mean()),
        "mean_candidate_unreliability": float(v_f.mean()),
        "mean_modality_conflict": float(conf_mod.mean()),
        "mean_decoder_observation_disagreement": float(conf_rec.mean()),
        "n_refined_edges": int(keep.sum()),
        "n_candidate_edges": int(len(ci)),
        "score_variance_approximation": "last-layer",
        "normalized_modality_precision": [float(x) for x in om],
        "source_names": names,
        "mean_source_weight": {nm: float(x) for nm, x in zip(names, W.mean(1))},
    }


def precompute(
    ds,
    cache,
    k_graph=30,
    pca_dim=30,
    n_hvg_rna=4000,
    n_hvg_atac=5000,
    observation_temperature=0.2,
):

    import os

    is_atac = str(ds.mod2_type).upper() == "ATAC"
    Xr = _prep(ds.rna, n_hvg_rna, is_adt=False)
    Xm = _prep(ds.mod2, n_hvg_atac if is_atac else 0, is_adt=(not is_atac))
    Pr = _pca(Xr, pca_dim)
    Pm = _pca(Xm, pca_dim)
    n = Xr.shape[0]
    row, col, similarity = _knn_edges(np.concatenate([Pr, Pm], 1), k_graph, n)
    w_obs = (1.0 / (1.0 + np.exp(-similarity / observation_temperature))).astype(
        "float32"
    )
    cache_parent = os.path.dirname(cache)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)
    np.savez_compressed(
        cache,
        Pr=Pr,
        Pm=Pm,
        row=row.astype("int64"),
        col=col.astype("int64"),
        w_obs=w_obs,
    )
    return cache


def run_gatoranchor(
    ds,
    cfg=None,
    seed=0,
    device="cuda:0",
    cache=None,
    dz=128,
    k_graph=30,
    hops=1,
    pca_dim=30,
    n_hvg_rna=4000,
    n_hvg_atac=5000,
    epochs_pre=500,
    epochs_ref=250,
    lr=1e-3,
    beta=1e-5,
    T=1.0,
    return_net=False,
):

    import os

    options = dict(GATORANCHOR_CONFIG)
    if cfg:
        options.update(cfg)
    n_gnn = int(options["n_gnn"])
    deep_split = float(options["deep_split"])
    lr_ft = float(options["lr_ft"])
    r_max = float(options["r_max"])
    lambda_b = float(options["lambda_B"])
    is_atac = str(ds.mod2_type).upper() == "ATAC"

    if int(options["n_gnn"]) < 1:
        raise ValueError("The residual GCN must contain at least one layer.")
    if not 0 <= float(options["deep_split"]) <= 1:
        raise ValueError("The Stage-B allocation must lie in [0, 1].")
    if not 0 < float(options["drop_q"]) <= 1:
        raise ValueError("The pruning quantile must lie in (0, 1].")
    if float(options["observation_temperature"]) <= 0:
        raise ValueError("The observation-graph temperature must be positive.")
    if float(options["lambda_D"]) < 0 or lambda_b < 0:
        raise ValueError("Variance and Brier coefficients must be nonnegative.")
    if float(options["lambda_L"]) <= 0:
        raise ValueError("The last-layer curvature damping must be positive.")
    if float(options["r_max"]) < 0:
        raise ValueError(
            "The terminal modality-specific edge weight must be nonnegative."
        )
    if (
        float(options["source_bias_observation"]) <= 0
        or float(options["source_bias_reconstruction"]) <= 0
    ):
        raise ValueError("Source-bias factors must be positive.")
    if int(hops) < 0:
        raise ValueError("The propagation order must be nonnegative.")
    if int(epochs_pre) < 1:
        raise ValueError("Stage A requires at least one warm-up epoch.")
    if int(epochs_ref) < 0:
        raise ValueError("The post-refinement epoch budget must be nonnegative.")
    if float(T) <= 0:
        raise ValueError("The uncertainty temperature must be positive.")
    if float(beta) < 0:
        raise ValueError("The modality-specific KL coefficient must be nonnegative.")
    for name, value in options.items():
        if name.startswith("epsilon_") and (not np.isfinite(value) or float(value) <= 0):
            raise ValueError(f"The numerical stabilizer {name} must be finite and positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if cache and os.path.exists(cache):
        raise FileExistsError(
            "Existing preprocessing caches have no verified input provenance. "
            "Use a new cache path or cache=None; unverified caches are never reused."
        )
    else:
        if cache:
            precompute(
                ds,
                cache,
                k_graph,
                pca_dim,
                n_hvg_rna,
                n_hvg_atac,
                observation_temperature=float(options["observation_temperature"]),
            )
            cached = np.load(cache)
            Pr, Pm = cached["Pr"], cached["Pm"]
            row, col, w_obs = cached["row"], cached["col"], cached["w_obs"]
        else:
            Xr = _prep(ds.rna, n_hvg_rna, is_adt=False)
            Xm = _prep(ds.mod2, n_hvg_atac if is_atac else 0, is_adt=not is_atac)
            Pr, Pm = _pca(Xr, pca_dim), _pca(Xm, pca_dim)
            Pc = np.concatenate([Pr, Pm], axis=1)
            row, col, similarity = _knn_edges(Pc, k_graph, Pr.shape[0])
            w_obs = (
                1.0
                / (
                    1.0
                    + np.exp(-similarity / float(options["observation_temperature"]))
                )
            ).astype("float32")

    n = Pr.shape[0]
    initial_graph = _sym_norm(
        row,
        col,
        w_obs,
        n,
        epsilon_degree=None,
    )
    Fr = _multihop(Pr, initial_graph, hops)
    Fm = _multihop(Pm, initial_graph, hops)

    def to_features(rna_features, modality2_features):
        return [
            torch.tensor(rna_features, device=dev),
            torch.tensor(modality2_features, device=dev),
        ]

    features = to_features(Fr, Fm)
    reconstruction_targets = [
        torch.tensor(Pr, device=dev),
        torch.tensor(Pm, device=dev),
    ]
    positive_pairs = torch.tensor(np.stack([row, col]), device=dev, dtype=torch.long)
    n_positive = positive_pairs.shape[1]

    model = GatorAnchor(
        [Fr.shape[1], Fm.shape[1]],
        [Pr.shape[1], Pm.shape[1]],
        dz=dz,
        lamD=float(options["lambda_D"]),
        p_drop=float(options["encoder_dropout"]),
        n_gnn=n_gnn,
    ).to(dev)
    model.set_graph(reconstruction_targets, _sp_to_torch(initial_graph, dev))
    model.set_deep(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    def edge_score_loss(representation, negative_destination, labels):
        logits = torch.cat(
            [
                model.edge(
                    representation[positive_pairs[0]], representation[positive_pairs[1]]
                ),
                model.edge(
                    representation[positive_pairs[0]],
                    representation[negative_destination],
                ),
            ]
        )
        probabilities = torch.sigmoid(logits)
        return F.binary_cross_entropy_with_logits(
            logits, labels
        ) + lambda_b * F.mse_loss(probabilities, labels)

    def edge_loss(fused_mean, modality_means, r_t):

        negative_destination = torch.randint(0, n, (n_positive,), device=dev)
        labels = torch.cat(
            [torch.ones(n_positive, device=dev), torch.zeros(n_positive, device=dev)]
        )
        return edge_score_loss(fused_mean, negative_destination, labels) + r_t * sum(
            edge_score_loss(view, negative_destination, labels)
            for view in modality_means
        ) / len(modality_means)

    def reconstruction_loss(latent_sample):
        return sum(
            F.mse_loss(model.dec[m](latent_sample), reconstruction_targets[m])
            for m in range(len(reconstruction_targets))
        )

    def optimization_step(active_optimizer, r_t):
        active_optimizer.zero_grad()
        fused_mean, fused_variance, modality_means, modality_logvars = model.encode(
            features
        )
        latent_sample = (
            fused_mean + torch.randn_like(fused_mean) * fused_variance.sqrt()
        )
        kl_loss = sum(
            -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            for mean, logvar in zip(modality_means, modality_logvars)
        )
        total = (
            reconstruction_loss(latent_sample)
            + edge_loss(fused_mean, modality_means, r_t)
            + beta * kl_loss
        )
        total.backward()
        active_optimizer.step()

    model.train()
    for epoch in range(epochs_pre):
        r_t = r_max * epoch / max(1, epochs_pre - 1)
        optimization_step(optimizer, r_t)

    model.eval()
    with torch.no_grad():
        warmup_mean, _, _, warmup_logvars = model.encode(features)
    refined_graph, diagnostics = _evidence_fusion(
        model,
        features,
        warmup_mean,
        warmup_logvars,
        Pr,
        Pm,
        is_atac,
        row,
        col,
        n,
        k_graph,
        T,
        dev,
        drop_q=float(options["drop_q"]),
        k_fuse=(None if options.get("k_fuse") is None else int(options["k_fuse"])),
        b_obs=float(options["source_bias_observation"]),
        b_rec=float(options["source_bias_reconstruction"]),
        lambda_L=float(options["lambda_L"]),
        epsilon_cosine=float(options["epsilon_cosine"]),
        epsilon_precision=float(options["epsilon_precision"]),
        epsilon_unreliability=float(options["epsilon_unreliability"]),
        epsilon_source=float(options["epsilon_source"]),
        epsilon_attenuation=float(options["epsilon_attenuation"]),
        epsilon_degree=float(options["epsilon_degree"]),
        epsilon_curvature=float(options["epsilon_curvature"]),
    )
    Fr = _multihop(Pr, refined_graph, hops)
    Fm = _multihop(Pm, refined_graph, hops)
    features = to_features(Fr, Fm)
    model.set_graph(reconstruction_targets, _sp_to_torch(refined_graph, dev))

    epochs_b = int(round(epochs_ref * deep_split))
    epochs_c = epochs_ref - epochs_b
    model.set_deep(True)
    model.freeze_shallow(True)
    optimizer_b = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    model.train()
    for _ in range(epochs_b):
        optimization_step(optimizer_b, r_max)
    diagnostics["gate_after_stage_b"] = model.gates()

    model.freeze_shallow(False)
    optimizer_c = torch.optim.Adam(model.parameters(), lr=lr * lr_ft, weight_decay=0.0)
    for _ in range(epochs_c):
        optimization_step(optimizer_c, r_max)
    diagnostics["gate_final"] = model.gates()
    diagnostics["post_refinement_epochs"] = [epochs_b, epochs_c]

    model.eval()
    with torch.no_grad():
        z_out = model.encode(features)[0].cpu().numpy()
    if return_net:
        return z_out, diagnostics, model
    return z_out, diagnostics


__all__ = ["GatorAnchor", "GATORANCHOR_CONFIG", "precompute", "run_gatoranchor"]
