# Supplementary figure notebooks

Seven editable notebooks cover Supplementary Figures S1, S2, S3, S4 and S7.
S5 contains implementation parameters without a figure; S6 is excluded from this
release. PARE is named Lasso-ARE / LassoARE in historical experimental files.

## Figure index

| Panels | Contents | Notebook |
| --- | --- | --- |
| S1a-f | Prior selections across six datasets | [figS1_prior_selections.ipynb](figS1_prior_selections.ipynb) |
| S2a-d | HBCA stripped-nuclei selection and method UMAPs | [figS2a_d_hbca_umap.ipynb](figS2a_d_hbca_umap.ipynb) |
| S2e-h | Clustering, marker statistics, donor comparisons and pathways | [figS2e_h_hbca_statistics.ipynb](figS2e_h_hbca_statistics.ipynb) |
| S3a-b | Batch effects across input representations | [figS3a_b_batch_umap.ipynb](figS3a_b_batch_umap.ipynb) |
| S3c-d | Runtime comparison and scaling | [figS3c_d_runtime.ipynb](figS3c_d_runtime.ipynb) |
| S4 | Robustness to erroneous priors | [figS4_prior_robustness.ipynb](figS4_prior_robustness.ipynb) |
| S7 | Six TLS markers in the Visium HD ROI | [figS7_tls_marker_violin.ipynb](figS7_tls_marker_violin.ipynb) |

## Running

Use an existing Scanpy environment or install the dependencies:

```bash
python -m pip install -r figures_res/supplementary/requirements.txt
```

Open a notebook in Jupyter or VS Code, select that environment and execute cells
in order. All plotting runs on CPU without model training. S1 sections run
independently after the setup/function cells.

The first code cell prints the resolved notebook folder, data root, each H5AD
input with an OK/MISSING status, and the output folder. With the default settings,
no environment variables or working-directory changes are needed for this layout:

```text
analysis_workspace/
├── pub_epithelial_pca500.h5ad
├── NKTCL-NK-10.22.h5ad
├── 10x_tmp/WT.h5ad
├── ... other experimental data directories ...
└── git_version/Lasso-ARE/figures_res/supplementary/
```

- The notebook folder is found from the kernel working directory or the VS Code
  notebook location. Supported working directories include the notebook folder,
  `figures_res`, the repository root, `git_version`, and the enclosing analysis
  workspace. If Jupyter starts elsewhere, set `SUPP_DIR` in the first code cell
  to the absolute supplementary folder, or set `LASSOARE_SUPPLEMENTARY_DIR`.
- H5AD discovery checks the notebook folder and its ancestors against the exact
  relative paths in `DATA_FILES`. It uses the nearest root containing all required
  files. If none is complete, it uses the nearest root containing any required
  file, allowing available S1 panels to run. It does not combine data from multiple
  roots or search for files by basename.
- To choose another data location, set `DATA_ROOT = Path('/path/to/data/root')`
  in the first code cell. This takes priority over `LASSOARE_DATA_ROOT`, which
  takes priority over automatic discovery. Relative root overrides are resolved
  against `SUPP_DIR`, not the kernel working directory. Invalid explicit roots
  raise an error. Individual `DATA_FILES` entries can also be absolute paths.
  After changing any path, rerun from the first code cell.
- `inputs/` and `output/figS*/` always belong to the resolved supplementary folder,
  independently of where the H5AD files are stored. Shared helpers are loaded
  directly from that folder, avoiding imports from another repository copy.

Outputs are PDF, 300 dpi PNG and selected CSV files. Distributed notebooks have
empty execution outputs. PDF scatter points are rasterized; text is editable.

**S2e-h, S3c-d and S4 require this folder alone:** their result tables are included
under `inputs/` (approximately 5 MB). Plotting functions are visible inside the
notebooks; [supplementary_utils.py](supplementary_utils.py) supplies shared I/O,
selection, embedding plotting and saving helpers.

## H5AD inputs

Large experimental H5AD files are not bundled. Default H5AD paths are relative to DATA_ROOT; they are editable in DATA_FILES.
Saved coordinates are used without recomputing neighbors or embeddings.

| Notebook | Input files |
| --- | --- |
| S1 | `pub_epithelial_pca500.h5ad`; `datasets_260407/adata_fib.h5ad`; `10x_tmp/WT.h5ad`; `datasets_260407/spatial_adata/merged_visium_spatial.h5ad`; `merfish/merfish_downsampled.h5ad`; `atac_true/atac_processed.h5ad` |
| S2a–d | `pub_epithelial_pca500.h5ad`; `new_evaluations/new_scRNA/data/reclustered_adata_test_42_rsc_releiden_downstream.h5ad`, `reclustered_adata_test_woLasso_42_rsc_releiden_downstream.h5ad`, `scNET_breast_downstream.h5ad`, `scvi_NK_releiden_downstream.h5ad`; `other_approaches/scFoundation/do_foundation_breast/scFoundation_breast.h5ad` |
| S3a–b | `NKTCL-NK-10.22.h5ad`; `time_batch/output_batch/reclustered_adata_test_no_harmony_no_pca.h5ad`, `reclustered_adata_test_no_harmony_pca.h5ad`, `reclustered_adata_test_harmony_pca.h5ad` |
| S7 | `10x_tmp/WT.h5ad` |

Preserve the original cell order and preprocessing. S1 selections are zero-based
row positions, not barcodes; do not apply them after filtering or reordering.
S2 aligns methods by barcode. Missing files, coordinates, genes or invalid
positions raise explicit errors. Most panels read metadata only. S7 streams six
genes in blocks; S1d loads the full matrix for score_genes and needs several GB
of memory. Source H5AD files are read only.

## Provenance and reproduction details

[inputs/manifest.json](inputs/manifest.json) records source locations,
transformations and SHA-256 checksums. Historical source paths refer to the
authors' analysis workspace, not necessarily this GitHub checkout. The notebooks
identify their original plotting scripts. Timing labels are translated to English;
numeric values are unchanged. Both original and distributed checksums are retained
for translated timing tables.

- S1b: the text describes an MMP1/MMP3 intersection, but the historical code saved
  a union (`|`). The saved experimental selection is retained.
- S1d: MMP3 colors use score_genes and selection uses the first line of top_cells.txt.
  Spatial/MERFISH plots use saved coordinates without tissue background images.
- S2: numbering follows the formal caption on page 6; some body-text references
  differ. Set INCLUDE_WOLASSO=True to include that configuration in S2e. Clustering
  CIs use seeded seaborn bootstrap. Marker bars are pooled-cell Welch t statistics;
  stars use saved donor-level one-sided Wilcoxon tests and within-population BH.
- S3c: colored intervals are mean +/- SD; black lines are 95% CIs, reversing the
  caption's box description. The source convention is retained and explained in
  the legend. S3d omits a CI for n=1.
- S4: duplicate experiment keys retain the last record, matching the source.
  Shading uses 1.96 * SD / sqrt(n). Each metric validates 21 error rates and five
  repeats per method/error rate.
- S7: the source applies log1p only when X.max()>100, without normalize_total.
  The default preserves this rule; explicit counts/log1p modes are also available.

Panels are regenerated without reproducing the PDF's final page layout, arrows
or font placement. Share this entire folder with both Python helpers and inputs. Git ignores
outputs, caches and notebook checkpoints. See [VALIDATION.md](VALIDATION.md).
