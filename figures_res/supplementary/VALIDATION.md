# Validation

Checks use saved experimental inputs without synthetic results or model training.

- All seven notebooks pass nbformat validation and code-cell syntax compilation.
  Distributed notebooks have empty outputs and null execution counts.
- All code cells execute in order in a fresh namespace per notebook, with
  Matplotlib Agg. This executes code directly rather than via Jupyter/nbclient.
- Execution produces 40 PDF/PNG pairs. H5ADs are read only; S1 scoring modifies
  an in-memory object. All seven notebooks execute with the default path settings
  and LASSOARE_DATA_ROOT / LASSOARE_SUPPLEMENTARY_DIR unset. The external analysis
  workspace is discovered automatically from the nested GitHub checkout.
- All seven setup cells pass from five working directories (35 combinations):
  the notebook folder, figures_res, repository root, git_version, and the enclosing
  analysis workspace. Shared helpers and bundled inputs resolve to the GitHub
  copy even when the workspace contains the original supplementary notebooks.
- S2e-h, S3c-d and S4 also execute from a standalone copy of this folder under
  an unrelated temporary directory, with no H5AD files or data-root settings.
- Additional path checks cover explicit overrides, relative data roots, absolute
  per-file paths, incomplete datasets, invalid directories, VS Code notebook
  locations, and an already-imported helper from a different checkout.
- The path fix preserves plotting and statistics code, apart from input-path
  references and loading the existing S2 save helper directly from this folder.
- English timing labels preserve numerical fields and original-source checksums.

| Check | Result |
| --- | --- |
| HBCA LHS3 selection | 11,373 / 130,171 |
| HBCA stripped nuclei | 5,748 / 130,171; saved list matches level2 exactly |
| scRNA DFU selection | 3,447 / 60,727 |
| Visium HD selection | 9,249 / 38,993 |
| Spatial DFU selection | 2,523 / 25,913 |
| MERFISH selection | 788 / 40,886 |
| ATAC selection | 3,088 / 259,011 |
| S3d runtimes | 172.430145 s (n=10), 918.449614 s (n=10), 6153.948169 s (n=1) |
| S4 duplicate records | 2,198 removed, retaining the last occurrence |
| S4 groups | 4 metrics x 2 methods x 21 error rates; 5 repeats each; 4 manual baselines |
| S7 ROI | 5,659 spots; log1p applied; all six genes present |

DFU source observations have duplicate names, producing an AnnData warning.
S1b uses saved row positions; S2 validates barcode uniqueness before alignment.

Environment: Python 3.10, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3,
Matplotlib 3.10.8, seaborn 0.13.2, AnnData 0.11.4, Scanpy 1.11.5, h5py 3.16.0.
H5AD data are not bundled. Validation covers saved data and coordinates, not an
end-to-end rerun from sequencing reads. Other versions may affect rendering,
bootstrap draws and score_genes. S6 is excluded.
