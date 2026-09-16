import csv
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import GatorAnchor as source


GatorAnchor = source.GatorAnchor
GATORANCHOR_CONFIG = dict(source.GATORANCHOR_CONFIG)
GATORANCHOR_CONFIG.update(
    ablation="full", loss_scaling="two_over_m", edge_chunk_size=2048,
    source_bias_mode="fixed_observation_total",
    logvar_initialization="default",
)
TRAINING_DEFAULTS = dict(
    dz=128, k_graph=30, hops=1, epochs_pre=500, epochs_ref=250,
    lr=1e-3, beta=1e-5, T=1.0,
)
MODALITY_ORDER = ("RNA", "ATAC", "ADT")


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _edge_loss_chunk(model, representation, src, positive_dst, negative_dst, lambda_b):
    logits = torch.cat([
        model.edge(representation[src], representation[positive_dst]),
        model.edge(representation[src], representation[negative_dst]),
    ])
    count = len(src)
    labels = torch.cat([torch.ones(count, device=logits.device), torch.zeros(count, device=logits.device)])
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            + lambda_b * F.mse_loss(torch.sigmoid(logits), labels, reduction="sum"))


def edge_score_loss(model, representation, positive_pairs, negative_destination,
                    lambda_b=0.5, chunk_size=0):
    n_pairs = positive_pairs.shape[1]
    if chunk_size <= 0:
        logits = torch.cat([
            model.edge(representation[positive_pairs[0]], representation[positive_pairs[1]]),
            model.edge(representation[positive_pairs[0]], representation[negative_destination]),
        ])
        labels = torch.cat([torch.ones(n_pairs, device=logits.device), torch.zeros(n_pairs, device=logits.device)])
        return (F.binary_cross_entropy_with_logits(logits, labels)
                + lambda_b * F.mse_loss(torch.sigmoid(logits), labels))
    result = representation.new_zeros(())
    for start in range(0, n_pairs, chunk_size):
        src, dst = positive_pairs[:, start:start + chunk_size]
        neg = negative_destination[start:start + chunk_size]
        def evaluate(rep, left, right, random_right):
            return _edge_loss_chunk(model, rep, left, right, random_right, lambda_b)
        if torch.is_grad_enabled():
            value = activation_checkpoint(evaluate, representation, src, dst, neg, use_reentrant=False)
        else:
            value = evaluate(representation, src, dst, neg)
        result = result + value / (2 * n_pairs)
    return result


def _laplace_hinv(model, mu, pairs, negative_destination, damping, epsilon_curvature, chunk_size):
    if chunk_size <= 0:
        return source._laplace_hinv(model, mu, pairs, negative_destination, mu.device,
                                    damping=damping, epsilon_curvature=epsilon_curvature)
    h = torch.zeros((mu.shape[1], mu.shape[1]), dtype=mu.dtype, device=mu.device)
    with torch.no_grad():
        for destinations in (pairs[1], negative_destination):
            for start in range(0, pairs.shape[1], chunk_size):
                left = pairs[0, start:start + chunk_size]
                right = destinations[start:start + chunk_size]
                phi = model.edge.feat(mu[left], mu[right])
                p = torch.sigmoid(model.edge.head(phi))
                weight = (p * (1 - p)).clamp_min(epsilon_curvature)
                h += (phi * weight[:, None]).T @ phi
        h += damping * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
        try:
            return torch.cholesky_inverse(torch.linalg.cholesky(h))
        except RuntimeError:
            return torch.linalg.pinv(h)


def evidence_fusion(model, features, warmup_mean, logvars, pcas, row, col,
                    initial_weights, n, k_graph, temperature, options):
    names = list(pcas)
    matrices = list(pcas.values())
    k_fuse = options["k_fuse"]
    k_fuse = max(1, k_graph // 2) if k_fuse is None else int(k_fuse)
    rr, rc, _ = source._candidate_knn_edges(warmup_mean.detach().cpu().numpy(), k_fuse, n)
    ci, cj = [row, rr], [col, rc]
    for matrix in matrices:
        rr, rc, _ = source._candidate_knn_edges(matrix, k_fuse, n)
        ci.append(rr)
        cj.append(rc)
    ci, cj = np.concatenate(ci), np.concatenate(cj)
    keys = np.minimum(ci, cj).astype(np.int64) * n + np.maximum(ci, cj)
    unique_keys, first = np.unique(keys, return_index=True)
    ci, cj = ci[first], cj[first]
    pairs = torch.tensor(np.stack([ci, cj]), device=warmup_mean.device, dtype=torch.long)
    precision = np.array([float(torch.exp(-lv).mean()) for lv in logvars], dtype="float64")
    omega = precision / (precision.mean() + options["epsilon_precision"])
    supports, penalties = [], []
    for matrix, modality_precision in zip(matrices, omega):
        normalized = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + options["epsilon_cosine"])
        support = np.clip((normalized[ci] * normalized[cj]).sum(1), 0, 1).astype("float32")
        supports.append(support)
        penalties.append(((1 - support) + options["epsilon_unreliability"]) / max(float(modality_precision), 1e-3))
    model.eval()
    with torch.no_grad():
        mu = model.encode(features)[0]
        negative = torch.randint(0, n, (len(ci),), device=mu.device)
        hinv = _laplace_hinv(model, mu, pairs, negative, options["lambda_L"],
                            options["epsilon_curvature"], options["edge_chunk_size"])
        chunk = options["edge_chunk_size"] or len(ci)
        rec_support, rec_variance = [], []
        for start in range(0, len(ci), chunk):
            left, right = pairs[:, start:start + chunk]
            rec_support.append(torch.sigmoid(model.edge(mu[left], mu[right])).cpu().numpy())
            rec_variance.append(source._laplace_var(model, hinv, mu[left], mu[right]).cpu().numpy())
        rec_support = np.concatenate(rec_support)
        rec_variance = np.concatenate(rec_variance)
    supports.append(rec_support)
    penalties.append(rec_variance + options["epsilon_unreliability"])
    supports, penalties = np.stack(supports), np.stack(penalties)
    observation_bias = options["source_bias_observation"]
    if options["source_bias_mode"] == "fixed_observation_total":
        observation_bias *= 2 / len(names)
    biases = [observation_bias] * len(names) + [options["source_bias_reconstruction"]]
    normalized_penalties, weights = source._fusion_weights(penalties, biases, options["epsilon_source"])
    fused_support = (weights * supports).sum(0)
    conflict = (weights * (supports - fused_support) ** 2).sum(0)
    applied_conflict = np.zeros_like(conflict) if options["ablation"] == "no_conflict" else conflict
    unreliability = (weights * normalized_penalties).sum(0) + applied_conflict
    observational_weights = weights[:-1] / (weights[:-1].sum(0, keepdims=True) + options["epsilon_source"])
    observational_support = (observational_weights * supports[:-1]).sum(0)
    observational_conflict = (observational_weights * (supports[:-1] - observational_support) ** 2).sum(0)
    decoder_conflict = (rec_support - observational_support) ** 2
    attenuation = np.exp(-unreliability / (temperature * (unreliability.mean() + options["epsilon_attenuation"])))
    proposed_weights = (fused_support * attenuation).astype("float32")
    threshold = float(np.quantile(unreliability, options["drop_q"]))
    proposed_keep = unreliability < threshold
    initial_keys = np.minimum(row, col).astype(np.int64) * n + np.maximum(row, col)
    initial_unique, inverse = np.unique(initial_keys, return_inverse=True)
    initial_max = np.zeros(len(initial_unique), dtype=initial_weights.dtype)
    np.maximum.at(initial_max, inverse, initial_weights)
    initial_mask = np.isin(unique_keys, initial_unique)
    applied_weights = proposed_weights.copy()
    applied_keep = proposed_keep.copy()
    if options["ablation"] == "no_refinement":
        applied_keep = initial_mask
        applied_weights = np.zeros_like(proposed_weights)
        applied_weights[initial_mask] = initial_max[np.searchsorted(initial_unique, unique_keys[initial_mask])]
        refined = source._sym_norm(row, col, initial_weights, n, epsilon_degree=None)
    else:
        refined = source._sym_norm(ci[applied_keep], cj[applied_keep], applied_weights[applied_keep], n,
                                  epsilon_degree=options["epsilon_degree"])
    source_names = ["obs:" + name for name in names] + ["rec"]
    diagnostics = {
        "mean_cross_source_conflict": float(conflict.mean()),
        "mean_candidate_unreliability": float(unreliability.mean()),
        "mean_modality_conflict": float(observational_conflict.mean()),
        "mean_decoder_observation_disagreement": float(decoder_conflict.mean()),
        "n_refined_edges": int(applied_keep.sum()), "n_candidate_edges": len(ci),
        "score_variance_approximation": "last-layer",
        "normalized_modality_precision": omega.tolist(), "source_names": source_names,
        "mean_source_weight": dict(zip(source_names, weights.mean(1).tolist())),
        "source_biases": dict(zip(source_names, biases)), "strict_quantile_threshold": threshold,
        "ablation": options["ablation"], "candidate_self_pairs": int((ci == cj).sum()),
        "n_proposed_retained_edges": int(proposed_keep.sum()),
    }
    per_edge = dict(
        cell_i=ci, cell_j=cj, unordered_key=unique_keys, source_names=np.array(source_names),
        source_support=supports, source_penalty=penalties,
        source_normalized_penalty=normalized_penalties, source_weight=weights,
        fused_support=fused_support, cross_source_conflict=conflict,
        applied_cross_source_conflict=applied_conflict, unreliability=unreliability,
        observation_conflict=observational_conflict, decoder_observation_disagreement=decoder_conflict,
        decoder_score_variance=rec_variance, attenuation=attenuation,
        proposed_edge_weight=proposed_weights, proposed_keep=proposed_keep,
        is_initial_edge=initial_mask, keep=applied_keep, edge_weight=applied_weights,
        used_in_propagation=applied_keep & (ci != cj) & (applied_weights > 0),
        curvature_negative_destination=negative.cpu().numpy(), curvature_inverse=hinv.cpu().numpy(),
    )
    for name, value in per_edge.items():
        if np.asarray(value).dtype.kind == "f" and not np.isfinite(value).all():
            raise FloatingPointError("Non-finite refinement quantity: " + name)
    return refined, diagnostics, per_edge


def _validate(pcas, cell_ids, cfg, training):
    if not isinstance(pcas, dict) or set(pcas) != set(MODALITY_ORDER):
        raise ValueError("GatorAnchor-Tri requires exactly RNA, ATAC and ADT PCA matrices.")
    if any(name not in {"RNA", "ATAC", "ADT"} for name in pcas):
        raise ValueError("Modality names must be RNA, ATAC or ADT.")
    if len(cell_ids) < 3 or len(set(cell_ids)) != len(cell_ids) or any(not x.strip() for x in cell_ids):
        raise ValueError("At least three unique cell IDs are required.")
    for name, matrix in pcas.items():
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional float32 NumPy array.")
        if matrix.shape[0] != len(cell_ids) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
            raise ValueError(f"{name} has mismatched cells, empty features or non-finite values.")
    if cfg["ablation"] != "full":
        raise ValueError("The public Tri workflow implements the full manuscript model.")
    if cfg["loss_scaling"] != "two_over_m":
        raise ValueError("The Tri manuscript requires reconstruction and KL scaling by 2/3.")
    if cfg["source_bias_mode"] != "fixed_observation_total":
        raise ValueError("The Tri manuscript requires observation-source biases of 2*b_obs/3.")
    if cfg["logvar_initialization"] not in {"default", "zero"}:
        raise ValueError("logvar_initialization must be default or zero.")
    for name in ("n_gnn", "edge_chunk_size"):
        if int(cfg[name]) != cfg[name] or cfg[name] < (1 if name == "n_gnn" else 0):
            raise ValueError(f"Invalid integer configuration {name}.")
    if not 0 <= cfg["deep_split"] <= 1 or not 0 < cfg["drop_q"] <= 1:
        raise ValueError("Invalid split or drop quantile.")
    if not 0 <= cfg["encoder_dropout"] < 1:
        raise ValueError("encoder_dropout must be in [0, 1).")
    for name in ("source_bias_observation", "source_bias_reconstruction", "observation_temperature", "lambda_L", "lr_ft"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("lambda_D", "lambda_B", "r_max"):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name, value in cfg.items():
        if name.startswith("epsilon_") and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"Invalid stabilizer {name}.")
    for name in ("dz", "k_graph", "hops", "epochs_pre", "epochs_ref"):
        minimum = 0 if name in {"hops", "epochs_ref"} else 1
        if int(training[name]) != training[name] or training[name] < minimum:
            raise ValueError(f"Invalid training integer {name}.")
    if training["k_graph"] >= len(cell_ids):
        raise ValueError("k_graph must be smaller than the number of cells.")
    if cfg["k_fuse"] is not None and (int(cfg["k_fuse"]) != cfg["k_fuse"] or not 1 <= cfg["k_fuse"] < len(cell_ids)):
        raise ValueError("k_fuse must be an integer in [1, n-1].")
    for name in ("lr", "T"):
        if not np.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(training["beta"]) or training["beta"] < 0:
        raise ValueError("beta must be finite and nonnegative.")


def run_gatoranchor_tri(pcas, output_dir, cell_ids, seed=0, device="cpu", cfg=None, training=None):
    cfg = {} if cfg is None else cfg
    training = {} if training is None else training
    unknown = set(cfg) - set(GATORANCHOR_CONFIG)
    unknown_training = set(training) - set(TRAINING_DEFAULTS)
    if unknown or unknown_training:
        raise ValueError(f"Unknown config/training keys: {unknown}, {unknown_training}")
    options, train = dict(GATORANCHOR_CONFIG, **cfg), dict(TRAINING_DEFAULTS, **training)
    cell_ids = [str(value) for value in cell_ids]
    _validate(pcas, cell_ids, options, train)
    pcas = {name: pcas[name] for name in MODALITY_ORDER}
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty run directory: " + str(destination))
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no silent CPU fallback.")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    names, matrices, n = list(pcas), list(pcas.values()), len(cell_ids)
    manifest = dict(
        model_name="GatorAnchor-Tri", modality_order=names,
        pca_shapes={name: list(pcas[name].shape) for name in names},
        seed=seed, device=str(dev), cfg=options, training=train,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        labels_used_for_training=False, status="training",
    )
    _write_json(destination / "config.json", manifest)
    with (destination / "cell_ids.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["cell_index", "cell_id"])
        writer.writerows(enumerate(cell_ids))
    row, col, similarity = source._knn_edges(np.concatenate(matrices, 1), train["k_graph"], n)
    observation_weight = (1 / (1 + np.exp(-similarity / options["observation_temperature"]))).astype("float32")
    initial_graph = source._sym_norm(row, col, observation_weight, n, epsilon_degree=None)
    sp.save_npz(destination / "initial_graph.npz", initial_graph)
    targets = [torch.tensor(matrix, device=dev) for matrix in matrices]

    def features_for(graph):
        return [torch.tensor(source._multihop(matrix, graph, train["hops"]), device=dev) for matrix in matrices]

    features = features_for(initial_graph)
    pairs = torch.tensor(np.stack([row, col]), device=dev, dtype=torch.long)
    model = GatorAnchor([feature.shape[1] for feature in features], [matrix.shape[1] for matrix in matrices],
                        dz=train["dz"], lamD=options["lambda_D"], p_drop=options["encoder_dropout"],
                        n_gnn=options["n_gnn"]).to(dev)
    if options["logvar_initialization"] == "zero":
        for encoder in model.enc:
            for head in (encoder.lv_s, encoder.lv_d):
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)
    model.set_graph(targets, source._sp_to_torch(initial_graph, dev))
    history = []
    scale = 2 / len(matrices)

    def write_history():
        if history:
            with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)

    def step(optimizer, r_t, stage, epoch):
        tick = time.monotonic()
        optimizer.zero_grad()
        fused, variance, mus, logvars = model.encode(features)
        sample = fused + torch.randn_like(fused) * variance.sqrt()
        kl = sum(-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()) for mu, lv in zip(mus, logvars))
        reconstruction = sum(F.mse_loss(model.dec[index](sample), target) for index, target in enumerate(targets))
        negative = torch.randint(0, n, (pairs.shape[1],), device=dev)
        joint = edge_score_loss(model, fused, pairs, negative, options["lambda_B"], options["edge_chunk_size"])
        branch = sum(edge_score_loss(model, mu, pairs, negative, options["lambda_B"], options["edge_chunk_size"]) for mu in mus) / len(mus)
        total = scale * reconstruction + (joint + r_t * branch) + train["beta"] * scale * kl
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite objective in Stage {stage}, epoch {epoch}.")
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Non-finite gradient in Stage {stage}, epoch {epoch}.")
        optimizer.step()
        history.append(dict(stage=stage, epoch=epoch, r_t=r_t, total=float(total.detach()),
                            reconstruction=float(reconstruction.detach()), edge_joint=float(joint.detach()),
                            edge_branch_mean=float(branch.detach()), kl_sum=float(kl.detach()),
                            loss_scale=scale, seconds=time.monotonic() - tick))

    def stage_optimizer(stage):
        model.set_deep(stage != "A")
        model.freeze_shallow(stage == "B")
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                lr=train["lr"] * (options["lr_ft"] if stage == "C" else 1.0), weight_decay=0.0)

    try:
        optimizer = stage_optimizer("A")
        model.train()
        for epoch in range(train["epochs_pre"]):
            step(optimizer, options["r_max"] * epoch / max(1, train["epochs_pre"] - 1), "A", epoch)
        model.eval()
        with torch.no_grad():
            warmup_mean, _, _, warmup_logvars = model.encode(features)
        refined_graph, diagnostics, edges = evidence_fusion(
            model, features, warmup_mean, warmup_logvars, pcas, row, col, observation_weight,
            n, train["k_graph"], train["T"], options,
        )
        np.savez_compressed(destination / "candidate_edges.npz", **edges)
        sp.save_npz(destination / "refined_graph.npz", refined_graph)
        features = features_for(refined_graph)
        model.set_graph(targets, source._sp_to_torch(refined_graph, dev))
        epochs_b = int(round(train["epochs_ref"] * options["deep_split"]))
        epochs_c = train["epochs_ref"] - epochs_b
        optimizer = stage_optimizer("B")
        model.train()
        for epoch in range(epochs_b):
            step(optimizer, options["r_max"], "B", epoch)
        diagnostics["gate_after_stage_b"] = model.gates()
        optimizer = stage_optimizer("C")
        for epoch in range(epochs_c):
            step(optimizer, options["r_max"], "C", epoch)
        diagnostics["gate_final"] = model.gates()
        diagnostics["post_refinement_epochs"] = [epochs_b, epochs_c]
        diagnostics["n_parameters"] = sum(p.numel() for p in model.parameters())
        diagnostics["training_seconds"] = time.monotonic() - started
        model.eval()
        with torch.no_grad():
            embedding = model.encode(features)[0].cpu().numpy()
        if not np.isfinite(embedding).all():
            raise FloatingPointError("Non-finite final embedding.")
        np.save(destination / "embedding.npy", embedding)
        _write_json(destination / "diagnostics.json", diagnostics)
        manifest["status"] = "complete_training_no_clustering"
        _write_json(destination / "config.json", manifest)
        torch.save(dict(
            state_dict=model.state_dict(), modality_order=names,
            in_dims=[feature.shape[1] for feature in features],
            rec_dims=[matrix.shape[1] for matrix in matrices],
            dz=train["dz"], n_gnn=options["n_gnn"], p_drop=options["encoder_dropout"],
            lamD=options["lambda_D"], hops=train["hops"], config=manifest,
        ), destination / "model.pt")
        write_history()
        return embedding, diagnostics, model
    except BaseException as error:
        write_history()
        manifest.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        _write_json(destination / "config.json", manifest)
        raise


__all__ = ["run_gatoranchor_tri", "GATORANCHOR_CONFIG", "TRAINING_DEFAULTS"]
