# GatorAnchor
![model](https://github.com/zhangzh1328/GatorAnchor/blob/main/GatorAnchor.png)

## Project Structure

```text
.
├── main.py                 # Dual-modal entry point
├── main_tri.py             # Tri-modal entry point
├── GatorAnchor.py          # Shared model and dual-modal training
├── GatorAnchorTri.py       # Tri-modal training
├── trimodal_data.py        # Tri-modal loading and preprocessing
├── utils.py                # Data, output and evaluation utilities
├── clustering.py           # R/mclust clustering
└── README.md
```

Keep the seven Python files together. Prepare input data locally; output
directories are created automatically. Built-in defaults require no JSON file.

## Usage

Install the dependency versions used for local verification:

```bash
conda create -n gatoranchor -c conda-forge python=3.9 r-base=4.3 r-mclust=6.1.1 pip
conda activate gatoranchor
python -m pip install \
  anndata==0.10.8 h5py==3.14.0 numpy==1.26.4 pandas==2.3.3 \
  scanpy==1.10.2 scikit-learn==1.5.1 scipy==1.13.1 \
  rpy2==3.5.12 torch==2.4.1
```

Run commands from the source directory. GPU execution uses `--device cuda:0`
and requires a compatible PyTorch build and driver. R/mclust and rpy2 are required.

### **1. Prepare your input data**

Provide one `.h5ad` file per modality:

- All files must contain the same cells with unique `adata.obs_names`; ATAC/ADT
  are automatically reordered to match RNA.
- `adata.X` must contain finite, non-negative counts, with a positive total per
  cell and at least two features per modality. Do not supply DSB, CLR or scaled values.
- Input requires at least three cells; default neighbourhood settings require
  more than 30 cells.

Count inputs undergo normalization to 10,000 per cell, `log1p`, feature selection
(RNA: 4,000; ATAC: 5,000; all ADTs), scaling and separate PCA (up to 30 dimensions).
Feature selection applies when the input exceeds the specified feature count.

For tri-modal PCA input, supply an NPZ containing finite `RNA`, `ATAC`, `ADT`
matrices and unique string `cell_ids`, all in the same cell order. No further
preprocessing is applied in this mode.

---

### **2. Run training and evaluation**

RNA–ATAC:

```bash
python main.py \
  --rna data/pbmc_unsorted_3k_RNA.h5ad \
  --modality2 data/pbmc_unsorted_3k_ATAC.h5ad --modality2-type ATAC \
  --n-clusters 12 --device cpu --seed 4 \
  --output-dir saved_results/pbmc_unsorted_3k
```

RNA–ADT:

```bash
python main.py \
  --rna data/GSE128639_RNA.h5ad \
  --modality2 data/GSE128639_ADT.h5ad --modality2-type ADT \
  --n-clusters 27 --device cpu --seed 0 \
  --output-dir saved_results/GSE128639
```

RNA–ATAC–ADT:

```bash
python main_tri.py \
  --rna data/trimodal_RNA.h5ad --atac data/trimodal_ATAC.h5ad \
  --adt data/trimodal_ADT.h5ad --n-clusters 4 --device cpu --seed 0 \
  --output-dir saved_results/trimodal
```

Alternatively, use precomputed tri-modal PCA coordinates:

```bash
python main_tri.py --pcas data/trimodal_pcas.npz \
  --n-clusters 4 --device cpu --seed 0 \
  --output-dir saved_results/trimodal_pca
```

Set `--n-clusters` to the reference-class count for your data; example counts
are not automatically inferred biological results. Only mclust is supported,
with a fixed component count and default BIC-based covariance-model selection.

To infer the count and report ARI/NMI/AMI/ACC, use `--label-key cell_type`
with an RNA `obs['cell_type']` column. For tri-modal PCA input, also supply
`--labels data/reference.csv`, containing `cell_id` and `cell_type` columns
for exactly the same cells. If `--n-clusters` is also supplied, the counts must
agree. Reference labels affect the downstream count and evaluation, but never
model training or graph refinement.

Use a new or empty output directory for every run. Training starts from
initialization; interrupted runs cannot be resumed.

#### Optional configuration

Common options include `--latent-dim`, `--graph-neighbors`, `--pca-dim`,
`--warmup-epochs` and `--refinement-epochs`. `--config` accepts a user-prepared
JSON file; command-line values override its settings. See all options with:

```bash
python main.py --help
python main_tri.py --help
```

---

### **3. Output files**

| Output | Contents |
|---|---|
| `GatorAnchor*_embedding.npy` / `.h5ad` | Integrated representation; h5ad also contains cell IDs and predicted clusters |
| `GatorAnchor*_clusters.csv` | Cell IDs and one-based cluster assignments |
| `GatorAnchor_model.pt` / `artifacts/model.pt` | Final dual-/tri-modal model parameters and metadata |
| `run_config.json` | Resolved settings, seeds and mclust configuration |
| `preprocessing_cache.npz` | PCA inputs; paired runs also include initial edges and weights |
| `diagnostics.json` / `artifacts/diagnostics.json` | Graph-refinement summaries |
| `metrics.json` | ARI, NMI, AMI and ACC when reference labels are supplied |

Embedding/cluster filenames start with `GatorAnchor` for paired runs and
`GatorAnchorTri` for tri-modal runs. Tri-modal `artifacts/` additionally stores
cell IDs, graphs, per-edge evidence, training history and the final embedding.

The historical ALTRA Figure 9 uses a selected weighted Leiden partition and
specific preprocessing. These mclust workflows do not reproduce its partition
or reported metrics.

## Python API

Dual-modal integration:

```python
from GatorAnchor import run_gatoranchor
from clustering import cluster_mclust
from utils import load_paired_h5ad

dataset = load_paired_h5ad("data/RNA.h5ad", "data/ATAC.h5ad", "ATAC")
z, diagnostics = run_gatoranchor(dataset, seed=0, device="cpu")
labels = cluster_mclust(z, k=12, seed=0)
```

Tri-modal integration from aligned PCA inputs:

```python
from GatorAnchorTri import run_gatoranchor_tri
from clustering import cluster_mclust
from trimodal_data import load_pca_npz

pcas, cells = load_pca_npz("data/trimodal_pcas.npz")
z, diagnostics, model = run_gatoranchor_tri(
    pcas, "saved_results/tri_api", cells, seed=0, device="cpu"
)
labels = cluster_mclust(z, k=4, seed=0)
```

API clustering returns zero-based labels; CLI outputs use one-based labels.
The paired API returns results in memory; the tri-modal API also saves training
artifacts. Use the CLI to save the complete result set, including cluster tables.

