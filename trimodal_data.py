from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from GatorAnchor import _pca, _prep
from utils import validate_count_matrix

MODALITIES = ("RNA", "ATAC", "ADT")


def load_trimodal_h5ad(rna, atac, adt):
    modalities = {name: ad.read_h5ad(path) for name, path in zip(MODALITIES, (rna, atac, adt))}
    cells = modalities["RNA"].obs_names
    if len(cells) < 3:
        raise ValueError("At least three paired cells are required.")
    for name, matrix in modalities.items():
        if not matrix.obs_names.is_unique or any(not str(x).strip() for x in matrix.obs_names):
            raise ValueError(f"{name} cell identifiers must be unique and nonempty.")
        if len(cells.difference(matrix.obs_names)) or len(matrix.obs_names.difference(cells)):
            raise ValueError(f"{name} must contain exactly the same cells as RNA.")
        matrix = matrix[cells].copy()
        validate_count_matrix(matrix, name)
        modalities[name] = ad.AnnData(
            X=matrix.X.copy(), obs=pd.DataFrame(index=cells.copy()),
            var=pd.DataFrame(index=matrix.var_names.copy()),
        )
    return modalities


def prepare_trimodal_pcas(modalities, pca_dim=30, rna_features=4000, atac_features=5000):
    pcas = {}
    for name in MODALITIES:
        n_features = rna_features if name == "RNA" else atac_features
        scaled = _prep(modalities[name], n_features, is_adt=name == "ADT")
        pcas[name] = _pca(scaled, pca_dim)
        if not np.isfinite(pcas[name]).all():
            raise ValueError(f"{name} preprocessing produced non-finite PCA coordinates.")
    return pcas, modalities["RNA"].obs_names.copy()


def load_pca_npz(path):
    with np.load(path, allow_pickle=False) as data:
        required = {*MODALITIES, "cell_ids"}
        if not required.issubset(data.files):
            raise ValueError(f"PCA NPZ is missing keys: {sorted(required.difference(data.files))}")
        ids = data["cell_ids"]
        if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"}:
            raise ValueError("cell_ids must be a one-dimensional string array, not pickled objects.")
        if ids.dtype.kind == "S":
            ids = np.char.decode(ids, "utf-8")
        cells = pd.Index(ids.astype(str))
        if len(cells) < 3 or not cells.is_unique or any(not x.strip() for x in cells):
            raise ValueError("PCA cell identifiers must be unique and nonempty (at least three).")
        pcas = {}
        for name in MODALITIES:
            value = data[name]
            if value.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"{name} PCA coordinates must be real numeric values.")
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 2 or value.shape[0] != len(cells) or value.shape[1] < 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} PCA coordinates have invalid dimensions or non-finite values.")
            pcas[name] = value
    return pcas, cells


def load_label_table(path, key, cells):
    sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if "cell_id" not in table or key not in table:
        raise ValueError(f"Label table requires cell_id and {key!r} columns.")
    if table.cell_id.duplicated().any() or table.cell_id.eq("").any():
        raise ValueError("Label-table cell identifiers must be unique and nonempty.")
    table = table.set_index("cell_id")
    if set(table.index) != set(cells):
        raise ValueError("Label table must contain exactly the same cells as the model input.")
    labels = table.loc[cells, key].to_numpy()
    if any(not str(label).strip() for label in labels):
        raise ValueError("Reference labels must not be empty.")
    return labels
