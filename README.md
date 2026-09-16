# GatorAnchor
![model](https://github.com/zhangzh1328/GatorAnchor/blob/main/GatorAnchor.png)

## Project Structure

```text
.
├── main.py                 # Dual-modal command-line entry point
├── main_tri.py             # Tri-modal command-line entry point
├── GatorAnchor.py          # Shared model, graph refinement and dual-modal training
├── GatorAnchorTri.py       # Tri-modal training and final result export
├── trimodal_data.py        # Tri-modal input alignment and preprocessing
├── utils.py                # Data loading, output and evaluation utilities
├── clustering.py           # R/mclust clustering
└── README.md               # Installation and usage
```

Keep the seven Python files in the same directory. The `data/` directory is
prepared by the user. The output directory specified by `--output-dir`, including
any required parent directories, is created automatically. Use a new or empty
output directory for each run; existing nonempty directories are rejected.

Default settings are defined in the Python files. The examples below do not
require a separate configuration file.

## Usage

The following commands specify the dependency versions used for local
verification, without requiring an additional environment or requirements file:

```bash
conda create -n gatoranchor -c conda-forge python=3.9 r-base=4.3 r-mclust=6.1.1 pip
conda activate gatoranchor
python -m pip install \
  anndata==0.10.8 h5py==3.14.0 numpy==1.26.4 pandas==2.3.3 \
  scanpy==1.10.2 scikit-learn==1.5.1 scipy==1.13.1 \
  rpy2==3.5.12 torch==2.4.1
```

Run the commands below from the directory containing the seven Python files.
Use `--device cpu` for CPU execution. CUDA execution requires a compatible
PyTorch build and GPU driver. Both clustering workflows require R, the R package
mclust and rpy2; no alternative clustering algorithm is used if mclust fails.

### **1. Prepare your input data**

Place the paired omics `.h5ad` files in the `./data/` directory.

Example dataset folder:

```text
data/
 ├── pbmc_unsorted_3k_RNA.h5ad
 ├── pbmc_unsorted_3k_ATAC.h5ad
 ├── GSE128639_RNA.h5ad
 ├── GSE128639_ADT.h5ad
 ├── trimodal_RNA.h5ad
 ├── trimodal_ATAC.h5ad
 └── trimodal_ADT.h5ad
```

Each set of `.h5ad` files must satisfy the following requirements:

- dual-modal input contains RNA and either ADT or ATAC; tri-modal input contains
  RNA, ATAC and ADT;
- all modalities contain exactly the same cells, identified by unique cell names
  in `adata.obs_names`;
- `adata.X` contains finite, non-negative count-scale measurements;
- each modality contains at least three cells and two features, and every cell
  has a positive total count in every modality;
- different modalities may contain different numbers of features;
- cell-type annotations and spatial coordinates are not required for model
  training or graph refinement.

If the same cells are stored in different orders, the loaders reorder ATAC/ADT
to match RNA. They reject duplicated, missing or unmatched cell identifiers.
Filter cells consistently across modalities before training. The default initial
graph uses 30 neighbours, so its default settings require more than 30 cells;
neighbourhood sizes must be smaller than the number of input cells.

For count input, measurements are normalized to a total of 10,000 per cell and
transformed using `log1p`. Variable features are selected for RNA and ATAC when
needed, whereas all supplied ADT features are retained. Each modality is then
scaled and projected separately to PCA. Defaults retain up to 4,000 RNA features,
5,000 ATAC features and 30 PCA dimensions per modality, subject to input size.
Do not supply DSB, CLR, scaled values or PCA coordinates as count-scale `X`.

GatorAnchor-Tri also accepts precomputed PCA coordinates through `--pcas`.
The NPZ file must contain `RNA`, `ATAC` and `ADT` matrices, with identical cell
order, plus a one-dimensional string array named `cell_ids`. The matrices must
contain finite real values and have one row per cell. Cell identifiers must be
unique and nonempty. This mode applies no further normalization, feature
selection, scaling or PCA; the caller is responsible for alignment and
preprocessing provenance.

---

### **2. Run training and evaluation**

Run GatorAnchor on an RNA–ATAC dataset:

```bash
python main.py \
  --rna data/pbmc_unsorted_3k_RNA.h5ad \
  --modality2 data/pbmc_unsorted_3k_ATAC.h5ad \
  --modality2-type ATAC \
  --n-clusters 12 \
  --cluster-method mclust \
  --device cpu \
  --seed 4 \
  --output-dir saved_results/pbmc_unsorted_3k
```

Run GatorAnchor on an RNA–ADT dataset:

```bash
python main.py \
  --rna data/GSE128639_RNA.h5ad \
  --modality2 data/GSE128639_ADT.h5ad \
  --modality2-type ADT \
  --n-clusters 27 \
  --cluster-method mclust \
  --device cpu \
  --seed 0 \
  --output-dir saved_results/GSE128639
```

Run GatorAnchor-Tri on RNA–ATAC–ADT data:

```bash
python main_tri.py \
  --rna data/trimodal_RNA.h5ad \
  --atac data/trimodal_ATAC.h5ad \
  --adt data/trimodal_ADT.h5ad \
  --n-clusters 4 \
  --cluster-method mclust \
  --device cpu \
  --seed 0 \
  --threads 1 \
  --output-dir saved_results/trimodal
```

Alternatively, use aligned PCA inputs for GatorAnchor-Tri:

```bash
python main_tri.py \
  --pcas data/trimodal_pcas.npz \
  --n-clusters 4 \
  --device cpu \
  --seed 0 \
  --output-dir saved_results/trimodal_pca
```

The example component counts must match the reference-class count for the actual
input data. The paired examples use the manuscript's 12-class PBMC unsorted 3k
and 27-class GSE128639 settings. The tri-modal example assumes four classes;
four is a supplied count, not a result automatically discovered by the model.
Filtering cells or changing the annotation level may require a different count.

The required arguments are:

- for `main.py`: `--rna`, `--modality2`, `--modality2-type` and `--output-dir`;
- for `main_tri.py`: either all three of `--rna`, `--atac` and `--adt`, or
  `--pcas`, together with `--output-dir`;
- for either entry point: supply `--n-clusters`, or reference labels through
  `--label-key` so the component count can be inferred. PCA input additionally
  requires `--labels` when using reference labels.

Both entry points support **mclust only**, which is also the default. The mixture
component count is fixed to the supplied or inferred reference-class count;
mclust selects the covariance model using its default BIC procedure. If both
`--n-clusters` and reference labels are supplied, their class counts must agree.

#### Optional configuration

- `--config`: path to a user-prepared JSON configuration; omit it to use the
  defaults defined in the Python files;
- `--cluster-method`: `mclust`, the only supported value;
- `--label-key`: reference-label column used to determine or check the component
  count and calculate post-hoc ARI, NMI, AMI and ACC;
- `--latent-dim`: dimensionality of the fused representation (default: 128);
- `--graph-neighbors`: initial graph neighbour count (default: 30);
- `--propagation-order`: multi-hop propagation order (default: 1);
- `--pca-dim`: PCA dimension per modality for count input (default: 30);
- `--rna-features`: variable RNA feature count (default: 4,000);
- `--modality2-features`: ATAC feature count in `main.py` (default: 5,000);
  this does not reduce ADT features;
- `--atac-features`: ATAC feature count in `main_tri.py` (default: 5,000);
- `--warmup-epochs`: Stage-A epoch count (default: 500);
- `--refinement-epochs`: combined Stage-B/C epoch count (default: 250, split
  into 150 and 100 with the default model configuration);
- `--learning-rate`: initial Adam learning rate (default: 0.001);
- `--beta`: modality-specific KL coefficient (default: 0.00001);
- `--uncertainty-temperature`: graph-weight attenuation temperature (default: 1.0);
- `--seed`: model and default clustering random seed (default: 0);
- `--device`: `cpu` or a CUDA device such as `cuda:0`. The default is `cuda:0`
  for `main.py` and `cpu` for `main_tri.py`;
- `--clustering-seed`: separate R/mclust seed for `main_tri.py`; defaults to `--seed`;
- `--threads`: PyTorch CPU thread count for `main_tri.py` (default: 1);
- `--labels`: CSV/TSV reference table for `main_tri.py`, used with `--label-key`.

For custom JSON files, `main.py` accepts `model` and `training` sections;
`main_tri.py` accepts `model`, `training` and `preprocessing`. The section defaults
and accepted keys are defined in `main._load_configuration()` and
`main_tri.load_configuration()`, respectively. Their training key names differ.
Command-line values override the corresponding configuration entries. Preprocessing
options do not transform inputs supplied through `--pcas`.

Display all available parameters with:

```bash
python main.py --help
python main_tri.py --help
```

Reference annotations remain optional. For a paired dataset with an RNA
`obs['cell_type']` column:

```bash
python main.py \
  --rna data/pbmc_unsorted_3k_RNA.h5ad \
  --modality2 data/pbmc_unsorted_3k_ATAC.h5ad \
  --modality2-type ATAC \
  --label-key cell_type \
  --device cpu \
  --seed 4 \
  --output-dir saved_results/pbmc_unsorted_3k_evaluated
```

For tri-modal count input, add `--label-key cell_type` to read the reference
column from the RNA file. For PCA input, provide an aligned label table:

```bash
python main_tri.py \
  --pcas data/trimodal_pcas.npz \
  --labels data/reference.csv \
  --label-key cell_type \
  --device cpu \
  --seed 0 \
  --output-dir saved_results/trimodal_pca_evaluated
```

The reference table must contain a unique `cell_id` column and the requested
label column, with exactly one row for every input cell and no empty labels.
Reference labels never enter model training or graph refinement. They determine
or verify the downstream component count and support evaluation after clustering;
they therefore affect the clustering count when it is inferred from annotations.

Each CLI run trains from initialization and saves final model parameters and
analysis results. Interrupted training must be restarted in a new directory;
there is no resume option.

---

### **3. Output files**

After successful training and clustering, the output directory contains:

```text
saved_results/
 ├── pbmc_unsorted_3k/
 │   ├── GatorAnchor_model.pt
 │   ├── run_config.json
 │   ├── preprocessing_cache.npz
 │   ├── GatorAnchor_embedding.npy
 │   ├── GatorAnchor_embedding.h5ad
 │   ├── GatorAnchor_clusters.csv
 │   ├── diagnostics.json
 │   └── metrics.json (only with reference labels)
 └── trimodal/
     ├── run_config.json
     ├── preprocessing_cache.npz
     ├── GatorAnchorTri_embedding.npy
     ├── GatorAnchorTri_embedding.h5ad
     ├── GatorAnchorTri_clusters.csv
     ├── metrics.json (only with reference labels)
     └── artifacts/
         ├── model.pt
         ├── config.json
         ├── cell_ids.tsv
         ├── initial_graph.npz
         ├── refined_graph.npz
         ├── candidate_edges.npz
         ├── history.csv
         ├── diagnostics.json
         └── embedding.npy
```

The shared output types contain:

- model `.pt` files: final trained parameters, architecture information and run
  metadata; these are not resumable optimizer checkpoints;
- `run_config.json`: resolved configuration, component count, seed, requested
  device and mclust settings, including the selected covariance model. Tri-modal
  runs also record input hashes, preparation mode, clustering seed and run status;
- `preprocessing_cache.npz`: paired runs save `Pr`, `Pm`, initial directed
  neighbour pairs and observation weights; tri-modal runs save aligned `RNA`,
  `ATAC`, `ADT` PCA matrices and string `cell_ids`;
- embedding `.npy` files: deterministic fused latent means (`Z_out`), used for
  clustering and downstream analysis;
- embedding `.h5ad` files: the same representation with aligned cell identifiers
  and predictions in `obs['GatorAnchor_cluster']`;
- cluster `.csv` files: `cell_id` and one-based `predicted_cluster` assignments;
- `diagnostics.json`: candidate/refined-edge counts, modality precisions, source
  weights and graph-refinement uncertainty summaries;
- `metrics.json`: ARI, NMI, AMI and ACC, generated only when reference labels are
  supplied. ACC uses optimal label matching; metrics do not alter the partition.

The additional tri-modal artifacts contain the aligned cell order, resolved
training configuration, initial and refined propagation graphs, per-edge fusion
evidence, training-loss history and final embedding. The artifact embedding is
also retained if training succeeds but the subsequent mclust step fails.

Cluster numbers are arbitrary component identifiers. They do not automatically
carry cell-type names or the historical ALTRA P1–P4 biological interpretation.
The manuscript's historical ALTRA Figure 9 uses a selected weighted Leiden
partition and specific preprocessing settings. The generic mclust workflows
documented here do not reproduce that partition or its reported metrics.

## Python API

GatorAnchor can also be run directly from Python:

```python
from GatorAnchor import run_gatoranchor
from clustering import cluster_mclust
from utils import load_paired_h5ad

dataset = load_paired_h5ad(
    "data/pbmc_unsorted_3k_RNA.h5ad",
    "data/pbmc_unsorted_3k_ATAC.h5ad",
    "ATAC",
)

z_out, diagnostics = run_gatoranchor(
    dataset,
    seed=4,
    device="cpu",
    dz=128,
    k_graph=30,
    hops=1,
    pca_dim=30,
    n_hvg_rna=4000,
    n_hvg_atac=5000,
    epochs_pre=500,
    epochs_ref=250,
    beta=1e-5,
)

predicted_clusters = cluster_mclust(z_out, k=12, seed=4)
```

For GatorAnchor-Tri with aligned PCA inputs:

```python
from GatorAnchorTri import run_gatoranchor_tri
from clustering import cluster_mclust
from trimodal_data import load_pca_npz

pcas, cell_ids = load_pca_npz("data/trimodal_pcas.npz")

z_out, diagnostics, model = run_gatoranchor_tri(
    pcas,
    output_dir="saved_results/trimodal_api",
    cell_ids=cell_ids,
    seed=0,
    device="cpu",
)

predicted_clusters = cluster_mclust(z_out, k=4, seed=0)
```

The paired API returns the embedding and diagnostics; `return_net=True` also
returns the trained model. It does not create the CLI result files automatically.
The tri-modal API saves training artifacts to its output directory; the separate
clustering call returns labels without writing CLI cluster tables or metrics.
Use `main.py` or `main_tri.py` to produce the complete output sets shown above.

`z_out` is the final deterministic representation. During model fitting,
stochastic latent samples are used for reconstruction, whereas deterministic
fused means are used for edge discrimination and graph refinement. The direct
clustering functions return zero-based labels; the command-line workflows save
one-based cluster identifiers.
