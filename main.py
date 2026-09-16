from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from GatorAnchor import GATORANCHOR_CONFIG, run_gatoranchor
from clustering import check_clustering_dependencies, cluster_mclust
from utils import (
    clustering_metrics,
    json_ready,
    load_paired_h5ad,
    load_reference_labels,
    resolve_reference_cluster_count,
    save_embedding_h5ad,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learn a GatorAnchor representation from paired RNA+ADT or RNA+ATAC data."
    )
    parser.add_argument("--rna", required=True, help="RNA AnnData file (.h5ad).")
    parser.add_argument(
        "--modality2", required=True, help="Paired ADT or ATAC AnnData file (.h5ad)."
    )
    parser.add_argument(
        "--modality2-type", required=True, choices=("ADT", "ATAC", "adt", "atac")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default=None, help="Optional JSON configuration file."
    )
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Known number of reference classes; inferred when --label-key is provided.",
    )
    parser.add_argument(
        "--cluster-method",
        default="mclust",
        choices=("mclust",),
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="RNA obs reference labels for the downstream class count and optional metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--graph-neighbors", type=int, default=None)
    parser.add_argument("--propagation-order", type=int, default=None)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--rna-features", type=int, default=None)
    parser.add_argument("--modality2-features", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--refinement-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--uncertainty-temperature", type=float, default=None)
    return parser


def _load_configuration(path: str | None) -> dict:
    base = {
        "model": dict(GATORANCHOR_CONFIG),
        "training": {
            "latent_dim": 128,
            "graph_neighbors": 30,
            "propagation_order": 1,
            "pca_dim": 30,
            "rna_features": 4000,
            "modality2_features": 5000,
            "warmup_epochs": 500,
            "refinement_epochs": 250,
            "learning_rate": 1e-3,
            "beta": 1e-5,
            "uncertainty_temperature": 1.0,
        },
    }
    if path is None:
        return base
    with open(path, "r", encoding="utf-8") as handle:
        supplied = json.load(handle)
    if not isinstance(supplied, dict) or set(supplied) - set(base):
        raise ValueError("Configuration allows only model and training sections.")
    for section, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(base[section]):
            raise ValueError(f"Unknown or malformed configuration in {section}.")
    base["model"].update(supplied.get("model", {}))
    base["training"].update(supplied.get("training", {}))
    return base


def _override(config: dict, args: argparse.Namespace) -> None:
    mapping = {
        "latent_dim": args.latent_dim,
        "graph_neighbors": args.graph_neighbors,
        "propagation_order": args.propagation_order,
        "pca_dim": args.pca_dim,
        "rna_features": args.rna_features,
        "modality2_features": args.modality2_features,
        "warmup_epochs": args.warmup_epochs,
        "refinement_epochs": args.refinement_epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "uncertainty_temperature": args.uncertainty_temperature,
    }
    for key, value in mapping.items():
        if value is not None:
            config["training"][key] = value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_clusters is None and args.label_key is None:
        parser.error("Provide --n-clusters or --label-key to specify the reference class count.")
    check_clustering_dependencies(args.cluster_method)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty run directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)

    config = _load_configuration(args.config)
    _override(config, args)
    dataset = load_paired_h5ad(args.rna, args.modality2, args.modality2_type)
    reference = (
        load_reference_labels(args.rna, args.label_key, dataset.rna.obs_names)
        if args.label_key is not None else None
    )
    n_clusters = resolve_reference_cluster_count(args.n_clusters, reference)
    if n_clusters > dataset.rna.n_obs:
        raise ValueError("The reference class count cannot exceed the number of cells.")
    train = config["training"]
    cache = output / "preprocessing_cache.npz"

    embedding, diagnostics, model = run_gatoranchor(
        dataset,
        cfg=config["model"],
        seed=args.seed,
        device=args.device,
        cache=str(cache),
        dz=int(train["latent_dim"]),
        k_graph=int(train["graph_neighbors"]),
        hops=int(train["propagation_order"]),
        pca_dim=int(train["pca_dim"]),
        n_hvg_rna=int(train["rna_features"]),
        n_hvg_atac=int(train["modality2_features"]),
        epochs_pre=int(train["warmup_epochs"]),
        epochs_ref=int(train["refinement_epochs"]),
        lr=float(train["learning_rate"]),
        beta=float(train["beta"]),
        T=float(train["uncertainty_temperature"]),
        return_net=True,
    )

    clustered, clustering_details = cluster_mclust(
        embedding,
        n_clusters,
        seed=args.seed,
        return_details=True,
    )
    predicted = np.asarray(clustered, dtype=int) + 1

    np.save(output / "GatorAnchor_embedding.npy", embedding)
    table = pd.DataFrame(
        {"cell_id": dataset.rna.obs_names.astype(str), "predicted_cluster": predicted}
    )
    table.to_csv(output / "GatorAnchor_clusters.csv", index=False)
    save_embedding_h5ad(
        embedding,
        dataset.rna.obs_names,
        str(output / "GatorAnchor_embedding.h5ad"),
        predicted,
    )

    run_record = {
        "model_name": "GatorAnchor",
        "modality2_type": dataset.mod2_type,
        "n_cells": int(dataset.rna.n_obs),
        "seed": args.seed,
        "device_requested": args.device,
        "cluster_method": args.cluster_method,
        "mclust_model": clustering_details["covariance_model"],
        "clustering": clustering_details,
        "n_clusters": n_clusters,
        "cluster_count_source": (
            "reference_labels" if reference is not None else "explicit_reference_class_count"
        ),
        "configuration": config,
    }
    with open(output / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(run_record), handle, indent=2)
    with open(output / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(diagnostics), handle, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "model_name": "GatorAnchor",
                "in_dims": [encoder.mu_s.in_features for encoder in model.enc],
                "rec_dims": [decoder.net.out_features for decoder in model.dec],
                "dz": int(train["latent_dim"]),
                "n_gnn": int(config["model"]["n_gnn"]),
                "run": json_ready(run_record),
            },
        },
        output / "GatorAnchor_model.pt",
    )

    if reference is not None:
        metrics = clustering_metrics(reference, predicted)
        with open(output / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2))

    print(f"Saved GatorAnchor outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
