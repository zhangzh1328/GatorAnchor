import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from GatorAnchorTri import GATORANCHOR_CONFIG, TRAINING_DEFAULTS, run_gatoranchor_tri
from clustering import check_clustering_dependencies, cluster_mclust
from trimodal_data import load_label_table, load_pca_npz, load_trimodal_h5ad, prepare_trimodal_pcas
from utils import clustering_metrics, file_sha256, json_ready, load_reference_labels, resolve_reference_cluster_count, save_embedding_h5ad

PREPROCESSING_DEFAULTS = dict(pca_dim=30, rna_features=4000, atac_features=5000)


def build_parser():
    p = argparse.ArgumentParser(description='Train GatorAnchor-Tri and cluster the deterministic embedding with mclust.')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--rna", help="RNA .h5ad, count-scale X; also supply --atac and --adt.")
    inputs.add_argument("--pcas", help="Aligned NPZ: RNA, ATAC, ADT, cell_ids; no preprocessing is applied.")
    p.add_argument("--atac", help="Paired ATAC .h5ad, count-scale X.")
    p.add_argument("--adt", help="Paired ADT .h5ad, count-scale X (not DSB/CLR values).")
    p.add_argument("--labels", help="Optional CSV/TSV with cell_id and --label-key, used only downstream.")
    p.add_argument("--label-key", help="Reference column in --labels, or in RNA.obs for raw input.")
    p.add_argument("--n-clusters", type=int, help="Known reference-class count for fixed-G mclust.")
    p.add_argument("--cluster-method", choices=["mclust"], default="mclust")
    p.add_argument("--config", help="JSON sections: model, training, preprocessing.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clustering-seed", type=int, help="R/mclust seed; defaults to --seed.")
    p.add_argument("--threads", type=int, default=1)
    for flag, kind in [("latent-dim", int), ("graph-neighbors", int), ("propagation-order", int),
                       ("warmup-epochs", int), ("refinement-epochs", int), ("learning-rate", float),
                       ("beta", float), ("uncertainty-temperature", float), ("pca-dim", int),
                       ("rna-features", int), ("atac-features", int)]:
        p.add_argument("--" + flag, type=kind)
    return p


def load_configuration(path=None):
    config = dict(model=dict(GATORANCHOR_CONFIG), training=dict(TRAINING_DEFAULTS),
                  preprocessing=dict(PREPROCESSING_DEFAULTS))
    if path:
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("Configuration allows only model, training and preprocessing sections.")
        for section, values in supplied.items():
            if not isinstance(values, dict) or set(values) - set(config[section]):
                raise ValueError(f"Unknown or malformed configuration in {section}.")
            config[section].update(values)
    return config


def write_record(path, value):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(json_ready(value), f, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rna and not (args.atac and args.adt):
        parser.error("--rna requires both --atac and --adt.")
    if args.pcas and (args.atac or args.adt):
        parser.error("--pcas cannot be mixed with raw modality files.")
    if args.labels and not args.label_key:
        parser.error("--labels requires --label-key.")
    if args.pcas and args.label_key and not args.labels:
        parser.error("With --pcas, --label-key requires an aligned --labels table.")
    if args.n_clusters is None and args.label_key is None:
        parser.error("Provide --n-clusters or --label-key for fixed-G mclust.")
    if args.threads < 1 or args.seed < 0 or (args.clustering_seed is not None and args.clustering_seed < 0):
        parser.error("Threads must be positive and seeds nonnegative.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty run directory: " + str(output))
    config = load_configuration(args.config)
    mapping = dict(latent_dim="dz", graph_neighbors="k_graph", propagation_order="hops",
                   warmup_epochs="epochs_pre", refinement_epochs="epochs_ref", learning_rate="lr",
                   beta="beta", uncertainty_temperature="T")
    for argument, key in mapping.items():
        if getattr(args, argument) is not None:
            config["training"][key] = getattr(args, argument)
    for key in PREPROCESSING_DEFAULTS:
        if getattr(args, key) is not None:
            config["preprocessing"][key] = getattr(args, key)
        value = config["preprocessing"][key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    check_clustering_dependencies("mclust")
    torch.set_num_threads(args.threads)
    paths = [args.pcas] if args.pcas else [args.rna, args.atac, args.adt]
    inputs = [{"name": Path(p).name, "sha256": file_sha256(p)} for p in paths]
    if args.pcas:
        pcas, cells = load_pca_npz(args.pcas)
        preprocessing = "Caller-prepared aligned PCA coordinates; no transform applied"
    else:
        modalities = load_trimodal_h5ad(args.rna, args.atac, args.adt)
        pcas, cells = prepare_trimodal_pcas(modalities, **config["preprocessing"])
        preprocessing = "Count X; normalize_total(1e4), log1p, RNA/ATAC HVG, scale, independent PCA"
    reference = None
    if args.labels:
        reference = load_label_table(args.labels, args.label_key, cells)
    elif args.label_key:
        reference = load_reference_labels(args.rna, args.label_key, cells)
    count = resolve_reference_cluster_count(args.n_clusters, reference)
    if count > len(cells):
        raise ValueError("The reference class count cannot exceed the number of cells.")
    clustering_seed = args.seed if args.clustering_seed is None else args.clustering_seed
    record = dict(model_name="GatorAnchor-Tri", status="training", inputs=inputs,
                  modality_order=["RNA", "ATAC", "ADT"], configuration=config,
                  preprocessing=preprocessing, seed=args.seed, clustering_seed=clustering_seed,
                  device_requested=args.device, n_cells=len(cells), n_clusters=count,
                  cluster_method="mclust", labels_used_for_training=False,
                  cluster_count_source="reference_labels" if reference is not None else "explicit_reference_class_count",
                  reference_file_sha256=file_sha256(args.labels) if args.labels else None,
                  label_key=args.label_key, figure_9_reproduction_claim=False)
    record_path = output / "run_config.json"
    output.mkdir(parents=True, exist_ok=True)
    write_record(record_path, record)
    np.savez_compressed(output / "preprocessing_cache.npz", **pcas, cell_ids=np.asarray(cells, dtype=str))
    try:
        embedding, diagnostics, model = run_gatoranchor_tri(
            pcas, output / "artifacts", cells, seed=args.seed, device=args.device,
            cfg=config["model"], training=config["training"],
        )
        record["status"] = "training_complete_clustering_pending"
        write_record(record_path, record)
        labels, details = cluster_mclust(embedding, count, seed=clustering_seed, return_details=True)
        predicted = labels + 1
        np.save(output / "GatorAnchorTri_embedding.npy", embedding)
        pd.DataFrame({"cell_id": cells, "predicted_cluster": predicted}).to_csv(output / "GatorAnchorTri_clusters.csv", index=False)
        save_embedding_h5ad(embedding, cells, str(output / "GatorAnchorTri_embedding.h5ad"), predicted)
        if reference is not None:
            write_record(output / "metrics.json", clustering_metrics(reference, predicted))
        record.update(status="complete", clustering=details,
                      embedding_sha256=file_sha256(output / "GatorAnchorTri_embedding.npy"))
        write_record(record_path, record)
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        write_record(record_path, record)
        raise
    print(f"Saved GatorAnchor-Tri outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
