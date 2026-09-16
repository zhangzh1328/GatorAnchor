# GatorAnchor


## Project Structure

```bash
.
├── main.py                 # Command-line entry point
├── GatorAnchor.py          # Core model, graph refinement and three-stage training
├── utils.py                # Paired-data loading, output and evaluation utilities
├── clustering.py           # Post-hoc clustering methods
├── requirements.txt        # Python dependencies
├── configs/default.json    # Full-model starting configuration
├── data/                   # Input-data specification
└── saved_results/          # Model checkpoints and inference outputs
```

The `data/` directory is prepared by the user. The output directory specified by `--output-dir`, including any required parent directories, is created automatically when training starts.


## Usage

### **1. Prepare your input data**

Place the two paired omics `.h5ad` files in the `./data/` directory.

Example dataset folder:

```text
data/
 ├── pbmc_unsorted_3k_RNA.h5ad
 ├── pbmc_unsorted_3k_ATAC.h5ad
 ├── GSE128639_RNA.h5ad
 ├── GSE128639_ADT.h5ad
 ...
```

Each pair of `.h5ad` files must satisfy the following requirements:

- one file contains RNA and the other contains paired ADT or ATAC measurements;
- both modalities contain the same uniquely named cells in `adata.obs_names`;
- `adata.X` contains the count-scale, non-negative feature matrix for that modality;
- the two modalities may contain different numbers of features;
- cell-type annotations are not required and are not used during representation
  learning, graph construction or graph refinement;
- spatial coordinates are not required because GatorAnchor is designed for paired
  single-cell multi-omics integration.

If the same cells are stored in different orders, GatorAnchor automatically
reorders the second modality to match the RNA file using `obs_names`. It raises an
error when cell identifiers are duplicated, missing or unmatched.

GatorAnchor performs modality-specific preprocessing internally. Measurements are
normalized to a total of 10,000 per cell and transformed using `log1p`. Variable
features are selected for RNA and ATAC when necessary, whereas all ADT features are
retained. Each modality is then standardized and projected separately using PCA.
Do not supply already standardized or PCA-transformed matrices unless intentionally
reproducing a separately documented preprocessing workflow. 

---

### **2. Run training and evaluation**

Run GatorAnchor on an RNA–ATAC dataset:

```bash
python main.py \
  --rna data/pbmc_unsorted_3k_RNA.h5ad \
  --modality2 data/pbmc_unsorted_3k_ATAC.h5ad \
  --modality2-type ATAC \
  --n-clusters 12 \
  --cluster-method louvain \
  --config configs/default.json \
  --device cuda:0 \
  --seed 4 \
  --output-dir saved_results/pbmc_unsorted_3k
```

Run GatorAnchor on an RNA–ADT dataset:

```bash
python main.py \
  --rna data/GSE128639_RNA.h5ad \
  --modality2 data/GSE128639_ADT.h5ad \
  --modality2-type ADT \
  --n-clusters 19 \
  --cluster-method leiden \
  --output-dir saved_results/GSE128639
```

The required arguments are:

- `--rna`: path to the RNA `.h5ad` file;
- `--modality2`: path to the paired ADT or ATAC `.h5ad` file;
- `--modality2-type`: second-modality type, either `ADT` or `ATAC`;
- `--n-clusters`: target number of clusters for post-hoc clustering;
- `--output-dir`: directory used to save the model and inference results.

#### Optional configuration

- `--config`: JSON configuration file, such as `configs/default.json`;
- `--cluster-method`: post-hoc clustering method (`kmeans`, `mclust`, `leiden`
  or `louvain`); the default is `kmeans`;
- `--label-key`: optional RNA `obs` column used only for post-hoc ARI, NMI, AMI
  and ACC evaluation;
- `--latent-dim`: dimensionality of the fused GatorAnchor representation;
- `--graph-neighbors`: number of nearest neighbours in the initial observation
  graph;
- `--propagation-order`: maximum graph-propagation order used to construct the
  multi-hop encoder input;
- `--pca-dim`: maximum number of retained PCA dimensions per modality;
- `--rna-features`: number of variable RNA features retained when feature
  selection is required;
- `--modality2-features`: number of variable ATAC features retained; this option
  does not reduce ADT features;
- `--warmup-epochs`: number of Stage-A observation-graph warm-up epochs;
- `--refinement-epochs`: total number of post-refinement epochs divided between
  Stages B and C;
- `--learning-rate`: Adam learning rate;
- `--beta`: coefficient of the modality-specific KL regularization term;
- `--seed`: random seed;
- `--device`: `cpu` or a CUDA device such as `cuda:0`.

Values supplied on the command line override the corresponding entries in the
JSON configuration. Display all available parameters with:

```bash
python main.py --help
```

Reference annotations remain optional. For example:

```bash
python main.py \
  --rna data/pbmc_unsorted_3k_RNA.h5ad \
  --modality2 data/pbmc_unsorted_3k_ATAC.h5ad \
  --modality2-type ATAC \
  --n-clusters 12 \
  --label-key cell_type \
  --output-dir saved_results/pbmc_unsorted_3k
```

Supplying `--label-key` only adds post-hoc evaluation after the representation and
cluster assignments have been generated. The reference labels are not stored in
the model input object and cannot affect model training or graph refinement.

---

### **3. Output files**

After training completes, the selected output directory contains the trained model,
the integrated representation, post-hoc cluster assignments and graph-refinement
diagnostics:

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
 │   └── metrics.json (optional)
 └── GSE128639/
     ├── GatorAnchor_model.pt
     ├── run_config.json
     ├── preprocessing_cache.npz
     ├── GatorAnchor_embedding.npy
     ├── GatorAnchor_embedding.h5ad
     ├── GatorAnchor_clusters.csv
     └── diagnostics.json
```

The files contain:

- `GatorAnchor_model.pt`: trained model parameters, architecture information and
  run metadata;
- `run_config.json`: resolved input-independent hyperparameters, modality type,
  random seed, clustering settings and requested device;
- `preprocessing_cache.npz`: modality-specific PCA coordinates, initial directed
  neighbour pairs and observation-edge weights;
- `GatorAnchor_embedding.npy`: deterministic fused latent means
  (`Z_out` in the manuscript) used for downstream clustering and analysis;
- `GatorAnchor_embedding.h5ad`: the same integrated representation with aligned
  cell identifiers and predicted clusters;
- `GatorAnchor_clusters.csv`: cell identifiers and one-based post-hoc cluster
  assignments;
- `diagnostics.json`: candidate-edge counts, refined-edge counts, modality
  precisions, source weights and graph-refinement uncertainty diagnostics;
- `metrics.json`: optional ARI, NMI, AMI and ACC values, generated only when
  `--label-key` is supplied.

## Python API

GatorAnchor can also be run directly from Python:

```python
from GatorAnchor import run_gatoranchor
from clustering import cluster_all
from utils import load_paired_h5ad

dataset = load_paired_h5ad(
    "data/pbmc_unsorted_3k_RNA.h5ad",
    "data/pbmc_unsorted_3k_ATAC.h5ad",
    "ATAC",
)

z_out, diagnostics = run_gatoranchor(
    dataset,
    seed=4,
    device="cuda:0",
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

predicted_clusters = cluster_all(
    z_out,
    k=12,
    seed=4,
    methods=("louvain",),
)["louvain"]
```

`z_out` is the final deterministic GatorAnchor representation used for clustering
and downstream analyses. During model fitting, stochastic latent samples are used
for reconstruction, whereas deterministic fused means are used for edge
discrimination and graph refinement. The direct clustering functions return
zero-based labels; the command-line workflow saves one-based cluster identifiers.
