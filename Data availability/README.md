## Datasets

The public data sources for the ten benchmark datasets in Table 1 and the ALTRA TEA-seq dataset used for the three-modality analysis in Figure 9 are listed below.

| Dataset | Modalities | Public source |
| --- | --- | --- |
| 10Xmalt | RNA + ADT | [10x Genomics: MALT tumor](https://www.10xgenomics.com/datasets/10-k-cells-from-a-malt-tumor-gene-expression-and-cell-surface-protein-3-standard-3-0-0) |
| GSE128639 | RNA + ADT | [GEO: GSE128639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639) |
| GSE194122 s3d6 | RNA + ADT | [GEO: GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) |
| 10X 1k PBMC | RNA + ADT | [10x Genomics: 1k PBMCs](https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-gene-expression-and-cell-surface-protein-3-standard-3-0-0) |
| GSE158013 | RNA + ADT | [GEO: GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) |
| GSE194122 s4d8 | RNA + ATAC | [GEO: GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) |
| GSE201402 | RNA + ATAC | [GEO: GSE201402](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201402) |
| PBMC unsorted 3k | RNA + ATAC | [10x Genomics: PBMCs, no cell sorting, 3k](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-3-k-1-standard-2-0-0) |
| GSE156478 Stim | RNA + ATAC | [GEO: GSE156478 (SuperSeries)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478); [GSE166188 (DOGMA-seq SubSeries)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166188) |
| Brain SNARE | RNA + ATAC | [GEO: GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074) |
| ALTRA TEA-seq | RNA + ATAC + ADT | [Allen Institute HISE](https://apps.allenimmunology.org/aifi/insights/ra-progression/downloads/tea-seq/) |

For GSE194122, the `s3d6` RNA+ADT subset corresponds to the CITE-seq processed data (`GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz`), whereas the `s4d8` RNA+ATAC subset corresponds to the Multiome processed data (`GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz`). Both files are listed in the linked GEO series record.
