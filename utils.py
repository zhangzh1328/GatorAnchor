from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PairedDataset:

    rna: ad.AnnData
    mod2: ad.AnnData
    mod2_type: str


def validate_count_matrix(matrix, name="Modality"):
    if matrix.n_obs < 3 or matrix.n_vars < 2:
        raise ValueError(f"{name} requires at least three cells and two features.")
    values = matrix.X.data if sp.issparse(matrix.X) else np.asarray(matrix.X)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name}.X must contain finite non-negative counts; do not use DSB/CLR/scaled input.")
    if np.any(np.asarray(matrix.X.sum(axis=1)).ravel() <= 0):
        raise ValueError(f"{name} contains cells with zero total counts; filter paired cells before training.")


def load_paired_h5ad(
    rna_path: str,
    modality2_path: str,
    modality2_type: str,
) -> PairedDataset:

    modality2_type = modality2_type.upper()
    if modality2_type not in {"ADT", "ATAC"}:
        raise ValueError("modality2_type must be either 'ADT' or 'ATAC'.")

    rna = ad.read_h5ad(rna_path)
    mod2 = ad.read_h5ad(modality2_path)
    if not rna.obs_names.is_unique or not mod2.obs_names.is_unique:
        raise ValueError("Cell identifiers in both AnnData objects must be unique.")

    missing = rna.obs_names.difference(mod2.obs_names)
    extra = mod2.obs_names.difference(rna.obs_names)
    if len(missing) or len(extra):
        raise ValueError(
            "The two modalities must contain the same paired cells. "
            f"Missing from modality 2: {len(missing)}; extra in modality 2: {len(extra)}."
        )

    mod2 = mod2[rna.obs_names].copy()
    validate_count_matrix(rna, "RNA")
    validate_count_matrix(mod2, modality2_type)
    rna.obs = pd.DataFrame(index=rna.obs_names.copy())
    mod2.obs = pd.DataFrame(index=mod2.obs_names.copy())
    return PairedDataset(rna=rna, mod2=mod2, mod2_type=modality2_type)


def load_reference_labels(
    rna_path: str,
    label_key: str,
    cell_ids: pd.Index,
) -> np.ndarray:
    rna = ad.read_h5ad(rna_path, backed="r")
    try:
        if label_key not in rna.obs:
            raise KeyError(f"RNA obs does not contain label key {label_key!r}.")
        missing = cell_ids.difference(rna.obs_names)
        if len(missing):
            raise ValueError(
                f"RNA label table is missing {len(missing)} expected cell identifiers."
            )
        return rna.obs.loc[cell_ids, label_key].to_numpy(copy=True)
    finally:
        rna.file.close()


def resolve_reference_cluster_count(
    n_clusters: Optional[int] = None,
    reference: Optional[np.ndarray] = None,
) -> int:
    if n_clusters is not None and (int(n_clusters) != n_clusters or int(n_clusters) < 1):
        raise ValueError("The reference class count must be a positive integer.")
    if reference is None:
        if n_clusters is None:
            raise ValueError("Supply reference labels or their known number of classes.")
        return int(n_clusters)
    labels = np.asarray(reference)
    if labels.ndim != 1 or labels.size == 0 or pd.isna(labels).any():
        raise ValueError("Reference labels must be a nonempty vector without missing values.")
    observed_count = len(pd.unique(labels))
    if n_clusters is not None and int(n_clusters) != observed_count:
        raise ValueError(
            f"The manuscript requires {observed_count} mixture components for the "
            f"{observed_count} reference classes, but --n-clusters={n_clusters} was supplied."
        )
    return observed_count


def clustering_metrics(
    reference: np.ndarray, predicted: np.ndarray
) -> Dict[str, float]:

    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted must have the same length")
    if pd.isna(reference).any() or pd.isna(predicted).any():
        raise ValueError(
            "reference and predicted labels must not contain missing values"
        )

    reference_codes, reference_levels = pd.factorize(reference, sort=True)
    predicted_codes, predicted_levels = pd.factorize(predicted, sort=True)
    table = np.zeros((len(reference_levels), len(predicted_levels)), dtype=np.int64)
    np.add.at(table, (reference_codes, predicted_codes), 1)
    row, col = linear_sum_assignment(table.max() - table)
    acc = float(table[row, col].sum() / table.sum())
    return {
        "ARI": float(adjusted_rand_score(reference, predicted)),
        "NMI": float(
            normalized_mutual_info_score(
                reference, predicted, average_method="arithmetic"
            )
        ),
        "AMI": float(
            adjusted_mutual_info_score(
                reference, predicted, average_method="arithmetic"
            )
        ),
        "ACC": acc,
    }


def save_embedding_h5ad(
    embedding: np.ndarray,
    cell_ids: pd.Index,
    output_path: str,
    predicted: Optional[np.ndarray] = None,
) -> None:

    out = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    out.obs_names = cell_ids.astype(str)
    out.var_names = [f"GatorAnchor_{i + 1}" for i in range(out.n_vars)]
    if predicted is not None:
        out.obs["GatorAnchor_cluster"] = pd.Categorical(
            np.asarray(predicted).astype(str)
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output_path)


def json_ready(value: Any) -> Any:

    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
