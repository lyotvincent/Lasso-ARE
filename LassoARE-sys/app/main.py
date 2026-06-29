from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.execution import resolve_execution_policy
from app.samples import SampleError, SampleManager
from app.runtime import RuntimeSettings, module_command


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
RUNTIME_SETTINGS = RuntimeSettings.from_env(project_root=ROOT_DIR)
TMP_DIR = RUNTIME_SETTINGS.data_dir
DATASET_ROOT = TMP_DIR / "datasets"
JOB_ROOT = TMP_DIR / "jobs"
UPLOAD_DIR = TMP_DIR / "uploads"

for directory in (TMP_DIR, DATASET_ROOT, JOB_ROOT, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PREFERRED_COLOR_COLUMNS = ["leiden", "level2", "level1"]
PREFERRED_UMAP_KEYS = ["X_umap", "umap", "UMAP", "X_UMAP"]
sample_manager = SampleManager(RUNTIME_SETTINGS.sample_dir)

ANALYSIS_TYPES = {
    "lasso_view",
    "downsample",
    "lassoare",
    "lassoare_x",
    "reconstruct_embedding",
}
RECONSTRUCTION_TYPES = {"lassoare", "lassoare_x", "reconstruct_embedding"}


class PlotRequest(BaseModel):
    embedding_key: str | None = None
    color_by: str | None = None


class MarkerPlotRequest(BaseModel):
    obs_col: str
    method: Literal["t-test", "t-test_overestim_var", "wilcoxon"] = "t-test"
    top_n: int = Field(default=5, ge=1, le=20)
    use_raw: bool | None = None


class GeneExpressionPlotRequest(BaseModel):
    gene: str
    embedding_key: str | None = None
    use_raw: bool | None = None


class AnalysisJobRequest(BaseModel):
    analysis_type: Literal["lasso_view", "downsample", "lassoare", "lassoare_x", "reconstruct_embedding"]
    lassoare_mode: Literal["generate", "reconstruct_embedding"] = "generate"
    selected_ids: list[int] = Field(default_factory=list)
    selected_groups: list[list[int]] = Field(default_factory=list)
    embedding_key: str | None = None
    color_by: str | None = None
    obs_col: str | None = None
    cluster_key: str | None = None
    sample_rate: float = 0.1
    uniform_rate: float = 0.5
    leiden_resolution: float = 1.0
    n_neighbors: int = 100
    do_correct: bool = True
    n_clusters: int | None = None
    enc_layers: list[int] | None = None
    disc_layers: list[int] | None = None
    enc_pretrain_epoch: int = 20
    disc_pretrain_epoch: int = 20
    gan_epoch: int = 20
    batch_size: int = 256
    z_dim: int = 32
    is_pca: bool = True
    pca_dimension: int = 500
    lambda_attention: float = 0.1
    lambda_ref: float = 0.3


class RecoverSelectionRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)


@dataclass
class DatasetRecord:
    dataset_id: str
    dataset_name: str
    source_path: str
    parent_dataset_id: str | None = None
    analysis_type: str | None = None
    is_derived: bool = False
    created_at: str = field(default_factory=lambda: _iso_now())


@dataclass
class AnalysisJob:
    job_id: str
    dataset_id: str
    analysis_type: str
    status: str
    message: str
    progress: float
    job_dir: str
    created_at: str = field(default_factory=lambda: _iso_now())
    updated_at: str = field(default_factory=lambda: _iso_now())
    result_dataset_id: str | None = None
    interactive_kind: str | None = None
    error: str | None = None
    result_info: dict[str, Any] | None = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: dict[str, ad.AnnData] = {}
        self._meta: dict[str, DatasetRecord] = {}
        self._lock = threading.Lock()

    def add(
        self,
        dataset_name: str,
        adata: ad.AnnData,
        *,
        parent_dataset_id: str | None = None,
        analysis_type: str | None = None,
        is_derived: bool = False,
    ) -> str:
        dataset_id = uuid.uuid4().hex
        dataset_dir = DATASET_ROOT / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_path = dataset_dir / "source.h5ad"
        adata.write_h5ad(source_path)
        record = DatasetRecord(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_path=str(source_path),
            parent_dataset_id=parent_dataset_id,
            analysis_type=analysis_type,
            is_derived=is_derived,
        )
        with self._lock:
            self._datasets[dataset_id] = adata
            self._meta[dataset_id] = record
        return dataset_id

    def add_from_path(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        parent_dataset_id: str | None = None,
        analysis_type: str | None = None,
        is_derived: bool = False,
    ) -> str:
        adata = ad.read_h5ad(source_path)
        return self.add(
            dataset_name,
            adata,
            parent_dataset_id=parent_dataset_id,
            analysis_type=analysis_type,
            is_derived=is_derived,
        )

    def get(self, dataset_id: str) -> ad.AnnData:
        with self._lock:
            adata = self._datasets.get(dataset_id)
        if adata is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return adata

    def get_meta(self, dataset_id: str) -> DatasetRecord:
        with self._lock:
            record = self._meta.get(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return record

    def get_name(self, dataset_id: str) -> str:
        return self.get_meta(dataset_id).dataset_name

    def get_source_path(self, dataset_id: str) -> Path:
        return Path(self.get_meta(dataset_id).source_path)

    def persist(self, dataset_id: str) -> None:
        source_path = self.get_source_path(dataset_id)
        adata = self.get(dataset_id)
        adata.write_h5ad(source_path)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()

    def create(self, job: AnalysisJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> AnalysisJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    def update(self, job_id: str, **updates: Any) -> AnalysisJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _iso_now()
            return job

    def snapshot(self, job_id: str) -> dict[str, Any]:
        return asdict(self.get(job_id))


store = DatasetStore()
job_store = JobStore()
analysis_job_lock = threading.Lock()
runtime_state_lock = threading.Lock()
runtime_degraded_reason: str | None = None


def _set_runtime_degraded(reason: str | None) -> None:
    global runtime_degraded_reason
    with runtime_state_lock:
        runtime_degraded_reason = reason


def _runtime_degradation() -> str | None:
    with runtime_state_lock:
        return runtime_degraded_reason


app = FastAPI(title="Bioinformatics Visualizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets", check_dir=False), name="assets")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_series(values: pd.Series) -> list[Any]:
    clean = values.astype("object").where(~values.isna(), None).tolist()
    normalized: list[Any] = []
    for value in clean:
        if isinstance(value, (np.integer, np.floating)):
            normalized.append(value.item())
        else:
            normalized.append(value)
    return normalized


def _available_embeddings(adata: ad.AnnData) -> list[str]:
    embeddings: list[str] = []
    for key, value in adata.obsm.items():
        if value is not None and len(value.shape) == 2 and value.shape[1] >= 2:
            embeddings.append(key)
    return embeddings


def _default_embedding_key(adata: ad.AnnData) -> str | None:
    available = _available_embeddings(adata)
    lowered = {key.lower(): key for key in available}

    for candidate in PREFERRED_UMAP_KEYS:
        if candidate in available:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for key in available:
        if "umap" in key.lower():
            return key
    return available[0] if available else None


def _default_color_column(adata: ad.AnnData) -> str | None:
    obs_columns = list(adata.obs.columns)
    for column in PREFERRED_COLOR_COLUMNS:
        if column in obs_columns:
            return column
    return obs_columns[0] if obs_columns else None


def _resolve_embedding_key(adata: ad.AnnData, requested: str | None) -> str | None:
    if requested:
        if requested not in _available_embeddings(adata):
            raise HTTPException(status_code=400, detail=f"Embedding '{requested}' is not available.")
        return requested
    return _default_embedding_key(adata)


def _resolve_color_column(adata: ad.AnnData, requested: str | None) -> str | None:
    if requested:
        if requested not in adata.obs.columns:
            raise HTTPException(status_code=400, detail=f"Column '{requested}' is not present in adata.obs.")
        return requested
    return _default_color_column(adata)


def _dataset_summary(dataset_id: str, adata: ad.AnnData) -> dict[str, Any]:
    default_embedding = _default_embedding_key(adata)
    color_column = _default_color_column(adata)
    embeddings = _available_embeddings(adata)
    meta = store.get_meta(dataset_id)
    return {
        "dataset_id": dataset_id,
        "dataset_name": meta.dataset_name,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "available_embeddings": embeddings,
        "default_embedding": default_embedding,
        "default_color_by": color_column,
        "needs_umap_choice": default_embedding is None,
        "has_other_embeddings": len(embeddings) > 0,
        "is_derived": meta.is_derived,
        "analysis_type": meta.analysis_type,
        "parent_dataset_id": meta.parent_dataset_id,
    }


def _build_plot_payload(
    adata: ad.AnnData,
    embedding_key: str | None = None,
    color_by: str | None = None,
) -> dict[str, Any]:
    available_embeddings = _available_embeddings(adata)
    if not available_embeddings:
        raise HTTPException(
            status_code=400,
            detail="No two-dimensional embedding is available. Please compute UMAP first.",
        )

    resolved_embedding = embedding_key or _default_embedding_key(adata) or available_embeddings[0]
    if resolved_embedding not in available_embeddings:
        raise HTTPException(status_code=400, detail=f"Embedding '{resolved_embedding}' is not available.")

    coords = np.asarray(adata.obsm[resolved_embedding])[:, :2]
    color_column = color_by or _default_color_column(adata)
    color_values: list[Any] | None = None
    if color_column:
        if color_column not in adata.obs.columns:
            raise HTTPException(status_code=400, detail=f"Column '{color_column}' is not present in adata.obs.")
        color_values = _normalize_series(adata.obs[color_column])

    return {
        "embedding_key": resolved_embedding,
        "color_by": color_column,
        "x_label": f"{resolved_embedding}_1",
        "y_label": f"{resolved_embedding}_2",
        "points": {
            "x": coords[:, 0].astype(float).tolist(),
            "y": coords[:, 1].astype(float).tolist(),
            "ids": list(range(int(adata.n_obs))),
            "color_values": color_values,
        },
    }



def _matrix_to_vector(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).reshape(-1)


def _series_group_labels(values: pd.Series) -> pd.Series:
    return values.astype("object").where(~values.isna(), "Unassigned").astype(str)


def _rank_group_value(container: Any, group: str, index: int) -> Any:
    if container is None:
        return None
    try:
        return container[group][index]
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _build_marker_plot_payload(adata: ad.AnnData, request: MarkerPlotRequest) -> dict[str, Any]:
    obs_col = request.obs_col
    if obs_col not in adata.obs.columns:
        raise HTTPException(status_code=400, detail=f"Column '{obs_col}' is not present in adata.obs.")
    if adata.n_vars < 1:
        raise HTTPException(status_code=400, detail="This dataset does not contain genes to rank.")

    valid_mask = ~adata.obs[obs_col].isna()
    if int(valid_mask.sum()) < 2:
        raise HTTPException(status_code=400, detail="At least two cells with non-empty group labels are required.")

    analysis_adata = adata[valid_mask].copy()
    group_labels = _series_group_labels(analysis_adata.obs[obs_col])
    group_counts = group_labels.value_counts(sort=False)
    groups = [str(group) for group in group_counts.index.tolist()]
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="Choose an obs column with at least two groups.")
    if len(groups) > 80:
        raise HTTPException(status_code=400, detail="Choose an obs column with 80 or fewer groups for marker plotting.")

    analysis_adata.obs[obs_col] = pd.Categorical(group_labels, categories=groups, ordered=True)
    use_raw = bool(analysis_adata.raw is not None) if request.use_raw is None else bool(request.use_raw and analysis_adata.raw is not None)
    n_genes = min(int(request.top_n), int(analysis_adata.n_vars))

    try:
        sc.tl.rank_genes_groups(
            analysis_adata,
            groupby=obs_col,
            method=request.method,
            n_genes=n_genes,
            use_raw=use_raw,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to rank marker genes: {exc}") from exc

    ranked = analysis_adata.uns.get("rank_genes_groups", {})
    names = ranked.get("names")
    if names is None or not getattr(names, "dtype", None) or not names.dtype.names:
        raise HTTPException(status_code=400, detail="Scanpy did not return marker genes for this column.")

    ranked_groups = [str(group) for group in names.dtype.names]
    scores = ranked.get("scores")
    pvals = ranked.get("pvals")
    pvals_adj = ranked.get("pvals_adj")
    logfoldchanges = ranked.get("logfoldchanges")

    marker_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    markers_by_group: list[dict[str, Any]] = []
    marker_genes: list[str] = []
    seen_genes: set[str] = set()

    for group in ranked_groups:
        group_markers: list[dict[str, Any]] = []
        for index in range(n_genes):
            gene = str(names[group][index])
            if not gene:
                continue
            marker = {
                "gene": gene,
                "rank": index + 1,
                "score": _float_or_none(_rank_group_value(scores, group, index)),
                "pval": _float_or_none(_rank_group_value(pvals, group, index)),
                "pval_adj": _float_or_none(_rank_group_value(pvals_adj, group, index)),
                "logfoldchange": _float_or_none(_rank_group_value(logfoldchanges, group, index)),
            }
            group_markers.append(marker)
            marker_lookup[(group, gene)] = marker
            if gene not in seen_genes:
                seen_genes.add(gene)
                marker_genes.append(gene)
        markers_by_group.append({"group": group, "markers": group_markers})

    if not marker_genes:
        raise HTTPException(status_code=400, detail="No marker genes were found for this column.")

    try:
        if use_raw and analysis_adata.raw is not None:
            expression = analysis_adata.raw[:, marker_genes].X
        else:
            expression = analysis_adata[:, marker_genes].X
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to collect marker expression values: {exc}") from exc

    group_labels = _series_group_labels(analysis_adata.obs[obs_col])
    points: list[dict[str, Any]] = []
    max_mean_expression = 0.0
    for group in ranked_groups:
        mask = group_labels.to_numpy() == group
        if not bool(mask.any()):
            continue
        group_expression = expression[mask, :]
        means = _matrix_to_vector(group_expression.mean(axis=0))
        fractions = _matrix_to_vector((group_expression > 0).mean(axis=0))
        for gene_index, gene in enumerate(marker_genes):
            marker = marker_lookup.get((group, gene))
            if marker is None:
                continue
            mean_expression = _float_or_none(means[gene_index]) or 0.0
            fraction = _float_or_none(fractions[gene_index]) or 0.0
            max_mean_expression = max(max_mean_expression, mean_expression)
            points.append({
                "group": group,
                "gene": gene,
                "mean_expression": mean_expression,
                "fraction": fraction,
                "is_marker": True,
                "rank": marker.get("rank"),
                "score": marker.get("score"),
                "pval_adj": marker.get("pval_adj"),
                "logfoldchange": marker.get("logfoldchange"),
            })

    return {
        "obs_col": obs_col,
        "method": request.method,
        "top_n": n_genes,
        "use_raw": use_raw,
        "groups": ranked_groups,
        "genes": marker_genes,
        "markers_by_group": markers_by_group,
        "points": points,
        "max_mean_expression": max_mean_expression,
        "group_counts": {str(group): int(count) for group, count in group_counts.items() if str(group) in ranked_groups},
    }



def _gene_candidates(adata: ad.AnnData, use_raw: bool) -> pd.Index:
    if use_raw and adata.raw is not None:
        return adata.raw.var_names
    return adata.var_names


def _resolve_gene_name(adata: ad.AnnData, requested_gene: str, use_raw: bool) -> str:
    gene = requested_gene.strip()
    if not gene:
        raise HTTPException(status_code=400, detail="Please enter a gene name.")

    candidates = _gene_candidates(adata, use_raw)
    if gene in candidates:
        return gene

    lowered = {str(candidate).lower(): str(candidate) for candidate in candidates}
    match = lowered.get(gene.lower())
    if match:
        return match

    raise HTTPException(status_code=404, detail=f"Gene '{gene}' was not found in this dataset.")


def _expression_vector_for_gene(adata: ad.AnnData, gene: str, use_raw: bool) -> np.ndarray:
    if use_raw and adata.raw is not None:
        values = adata.raw[:, [gene]].X
    else:
        values = adata[:, [gene]].X
    vector = _matrix_to_vector(values).astype(float)
    vector[~np.isfinite(vector)] = 0.0
    return vector


def _build_gene_expression_plot_payload(adata: ad.AnnData, request: GeneExpressionPlotRequest) -> dict[str, Any]:
    available_embeddings = _available_embeddings(adata)
    if not available_embeddings:
        raise HTTPException(
            status_code=400,
            detail="No two-dimensional embedding is available. Please compute UMAP first.",
        )

    resolved_embedding = request.embedding_key or _default_embedding_key(adata) or available_embeddings[0]
    if resolved_embedding not in available_embeddings:
        raise HTTPException(status_code=400, detail=f"Embedding '{resolved_embedding}' is not available.")

    use_raw = bool(adata.raw is not None) if request.use_raw is None else bool(request.use_raw and adata.raw is not None)
    gene = _resolve_gene_name(adata, request.gene, use_raw)
    expression = _expression_vector_for_gene(adata, gene, use_raw)
    coords = np.asarray(adata.obsm[resolved_embedding])[:, :2]

    return {
        "embedding_key": resolved_embedding,
        "color_by": gene,
        "color_mode": "continuous",
        "color_label": f"{gene} expression",
        "expression_gene": gene,
        "expression_min": float(np.min(expression)) if expression.size else 0.0,
        "expression_max": float(np.max(expression)) if expression.size else 0.0,
        "use_raw": use_raw,
        "x_label": f"{resolved_embedding}_1",
        "y_label": f"{resolved_embedding}_2",
        "points": {
            "x": coords[:, 0].astype(float).tolist(),
            "y": coords[:, 1].astype(float).tolist(),
            "ids": list(range(int(adata.n_obs))),
            "color_values": None,
            "expression_values": expression.astype(float).tolist(),
        },
    }


def _compute_umap(adata: ad.AnnData) -> None:
    if "X_umap" in adata.obsm and adata.obsm["X_umap"].shape[1] >= 2:
        return

    if adata.n_obs < 3:
        raise HTTPException(status_code=400, detail="At least 3 cells are required to compute UMAP.")

    use_rep = None
    if "X_pca" not in adata.obsm:
        n_comps = min(50, int(adata.n_vars), max(2, int(adata.n_obs) - 1))
        if n_comps >= 2:
            sc.pp.pca(adata, n_comps=n_comps)
            use_rep = "X_pca"
    else:
        use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)


def _job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def _run_python_module(
    *,
    python_executable: Path,
    module_name: str,
    input_h5ad: Path,
    spec_path: Path,
    output_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    command = module_command(
        python_executable,
        module_name,
        [
            "--input-h5ad",
            input_h5ad,
            "--spec",
            spec_path,
            "--output-dir",
            output_dir,
        ],
    )
    with stdout_path.open("a") as stdout_file, stderr_path.open("a") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{module_name} failed with exit code {completed.returncode}.")


def _refresh_job_progress(job_id: str) -> None:
    job = job_store.get(job_id)
    if job.status not in {"preprocessing", "running", "postprocessing"}:
        return

    stdout_path = Path(job.job_dir) / "stdout.log"
    if not stdout_path.exists():
        return

    content = stdout_path.read_text()
    if not content.strip():
        return

    spec_path = Path(job.job_dir) / "spec.json"
    spec = _read_json(spec_path) if spec_path.exists() else {}
    is_reconstruct = job.analysis_type in RECONSTRUCTION_TYPES and spec.get("lassoare_mode") == "reconstruct_embedding"

    if job.status == "postprocessing":
        backend = spec.get("execution_backend")
        message = (
            "Running Scanpy fallback postprocessing."
            if backend == "scanpy-fallback"
            else "Running RAPIDS postprocessing."
        )
        job_store.update(job_id, message=message, progress=0.95)
        return

    patterns: list[tuple[str, str, float, float]] = [
        (r"Ref encoder pretrain \[(\d+)/(\d+)\]", "Pretraining Reference Encoder", 0.05, 0.25),
        (r"Pretrain epoch \[(\d+)/(\d+)\]", "Pretraining Encoder/Decoder", 0.3 if is_reconstruct else 0.1, 0.3),
        (r"Pretrain Disc Epoch (\d+)/(\d+)", "Pretraining Discriminator", 0.6 if is_reconstruct else 0.4, 0.1),
        (r"Disc pretrain \[(\d+)/(\d+)\]", "Pretraining Discriminator", 0.6 if is_reconstruct else 0.4, 0.1),
        (r"Epoch \[(\d+)/(\d+)\]", "Training Adversarial Model", 0.7 if is_reconstruct else 0.5, 0.25),
        (r"--- Epoch (\d+)/(\d+) ---", "Training Adversarial Model", 0.7 if is_reconstruct else 0.5, 0.25),
    ]

    latest_update: tuple[int, str, float] | None = None
    for pattern, label, offset, span in patterns:
        matches = list(re.finditer(pattern, content))
        if not matches:
            continue
        match = matches[-1]
        current = int(match.group(1))
        total = max(int(match.group(2)), 1)
        progress = min(offset + span * (current / total), 0.98)
        candidate = (match.start(), f"{label} ({current}/{total})", progress)
        if latest_update is None or candidate[0] > latest_update[0]:
            latest_update = candidate

    stage_markers: list[tuple[str, str, float]] = [
        ("Running RAPIDS PCA preprocessing on expression matrix", "Running RAPIDS PCA", 0.08),
        ("Running RAPIDS PCA preprocessing on raw counts", "Running RAPIDS Raw PCA", 0.12),
        ("Running RAPIDS PCA preprocessing on embedding", "Running RAPIDS PCA", 0.1),
        ("Writing RAPIDS PCA preprocessed h5ad...", "Writing RAPIDS PCA h5ad", 0.18),
        ("Inferring LassoARE cluster count...", "Inferring Clusters", 0.26),
        ("Computing neighbor graph for cluster inference...", "Inferring Clusters: Neighbors", 0.27),
        ("Running Leiden for cluster inference...", "Inferring Clusters: Leiden", 0.28),
        ("Running PCA on input matrix", "Running PCA", 0.3),
        ("Running PCA on embedding", "Running PCA", 0.3),
        ("Running PCA on raw counts", "Running Raw PCA", 0.32),
        ("Preparing LassoARE matrices...", "Preparing Matrices", 0.34),
        ("Preparing expression reference matrices...", "Preparing Reference Matrices", 0.34),
        ("Initializing LassoARE model...", "Initializing Model", 0.36),
        ("Initializing reference-guided LassoARE model...", "Initializing Model", 0.36),
        ("Pretraining reference encoder on expression data...", "Pretraining Reference Encoder", 0.08),
        ("Pre-computing reference latent representations z_ref", "Preparing Reference Latents", 0.28),
        ("Pretraining main autoencoder (embedding → expression)...", "Pretraining Encoder/Decoder", 0.35),
        ("Pretraining autoencoder...", "Pretraining Encoder/Decoder", 0.12),
        ("Encoding data for KMeans initialization...", "Preparing KMeans Input", 0.5 if is_reconstruct else 0.36),
        ("Running KMeans initialization", "Running KMeans Initialization", 0.52 if is_reconstruct else 0.38),
        ("Pretraining discriminators...", "Pretraining Discriminator", 0.62 if is_reconstruct else 0.42),
        ("Training adversarial model with expression reference guidance...", "Training Adversarial Model", 0.72),
        ("Collecting final LassoARE latent representation...", "Collecting Final Latents", 0.94 if is_reconstruct else 0.76),
        ("Storing LassoARE outputs in AnnData...", "Storing LassoARE Outputs", 0.95 if is_reconstruct else 0.78),
        ("Writing LassoARE intermediate h5ad...", "Writing Intermediate h5ad", 0.98),
    ]
    for marker, label, progress in stage_markers:
        position = content.rfind(marker)
        if position == -1:
            continue
        candidate = (position, label, progress)
        if latest_update is None or candidate[0] > latest_update[0]:
            latest_update = candidate

    if latest_update is not None:
        _, message, progress = latest_update
        job_store.update(job_id, message=message, progress=progress)
        return


def _job_snapshot(job_id: str) -> dict[str, Any]:
    _refresh_job_progress(job_id)
    snapshot = job_store.snapshot(job_id)
    result_dataset_id = snapshot.get("result_dataset_id")
    result_info = snapshot.get("result_info") or {}
    if result_dataset_id:
        adata = store.get(result_dataset_id)
        preferred_embedding = result_info.get("preferred_embedding")
        preferred_color = result_info.get("preferred_color_by")
        snapshot["result_summary"] = _dataset_summary(result_dataset_id, adata)
        snapshot["result_plot"] = _build_plot_payload(adata, preferred_embedding, preferred_color)
        snapshot["interactive_kind"] = snapshot.get("interactive_kind") or result_info.get("interactive_kind")
    return snapshot


def _complete_job(job_id: str, result_info: dict[str, Any]) -> None:
    result_path = Path(result_info["result_h5ad"])
    source_name = store.get_name(job_store.get(job_id).dataset_id)
    derived_name = result_info.get("dataset_name") or f"{source_name} ({job_store.get(job_id).analysis_type})"
    result_dataset_id = store.add_from_path(
        derived_name,
        result_path,
        parent_dataset_id=job_store.get(job_id).dataset_id,
        analysis_type=job_store.get(job_id).analysis_type,
        is_derived=True,
    )
    job_store.update(
        job_id,
        status="completed",
        message="Analysis completed.",
        progress=1.0,
        result_dataset_id=result_dataset_id,
        interactive_kind=result_info.get("interactive_kind"),
        result_info=result_info,
        error=None,
    )


def _fail_job(job_id: str, message: str) -> None:
    job_store.update(job_id, status="failed", message=message, error=message)


def _run_analysis_job(job_id: str) -> None:
    job = job_store.get(job_id)
    spec_path = Path(job.job_dir) / "spec.json"
    stdout_path = Path(job.job_dir) / "stdout.log"
    stderr_path = Path(job.job_dir) / "stderr.log"
    input_h5ad = store.get_source_path(job.dataset_id)
    is_reconstruction = job.analysis_type in RECONSTRUCTION_TYPES
    spec = _read_json(spec_path) if spec_path.exists() else {}
    policy = resolve_execution_policy(
        profile=RUNTIME_SETTINGS.profile,
        is_reconstruction=is_reconstruction,
        is_pca=bool(spec.get("is_pca", True)),
    )
    spec["execution_backend"] = policy.postprocess_backend
    _write_json(spec_path, spec)

    try:
        if is_reconstruction:
            job_store.update(
                job_id,
                status="queued",
                message=f"Waiting for {policy.slot_label} slot.",
                progress=0.05,
            )

        with analysis_job_lock if is_reconstruction else nullcontext():
            if RUNTIME_SETTINGS.profile == "cuda" and is_reconstruction:
                _set_runtime_degraded(None)
            if policy.use_rsc_pca:
                rsc_python = RUNTIME_SETTINGS.rsc_python
                if rsc_python is None:
                    raise RuntimeError("RSC Python is not configured for CUDA mode.")
                job_store.update(
                    job_id,
                    status="preprocessing",
                    message="Running RAPIDS PCA preprocessing.",
                    progress=0.08,
                )
                try:
                    _run_python_module(
                        python_executable=rsc_python,
                        module_name="backend.rsc_pca_preprocess_cli",
                        input_h5ad=input_h5ad,
                        spec_path=spec_path,
                        output_dir=Path(job.job_dir),
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                    preprocess_info = _read_json(
                        Path(job.job_dir) / "pca_preprocess.json"
                    )
                    input_h5ad = Path(preprocess_info["result_h5ad"])
                except Exception as exc:
                    _set_runtime_degraded(
                        f"RAPIDS PCA preprocessing failed: {exc}"
                    )
                    job_store.update(
                        job_id,
                        status="running",
                        message="RAPIDS PCA failed; falling back to Scanpy PCA.",
                        progress=0.2,
                    )
                    with stdout_path.open("a") as stdout_file:
                        stdout_file.write(
                            "RAPIDS PCA preprocessing failed; "
                            f"falling back to Scanpy PCA: {exc}\n"
                        )

            job_store.update(
                job_id,
                status="running",
                message="Running LassoARE analysis.",
                progress=0.25,
            )
            _run_python_module(
                python_executable=RUNTIME_SETTINGS.main_python,
                module_name="backend.analysis_cli",
                input_h5ad=input_h5ad,
                spec_path=spec_path,
                output_dir=Path(job.job_dir),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )

            result_info = _read_json(Path(job.job_dir) / "result.json")
            if result_info.get("needs_postprocess"):
                intermediate_path = Path(result_info["result_h5ad"])
                if policy.postprocess_backend == "rapids":
                    rsc_python = RUNTIME_SETTINGS.rsc_python
                    if rsc_python is None:
                        raise RuntimeError(
                            "RSC Python is not configured for CUDA mode."
                        )
                    job_store.update(
                        job_id,
                        status="postprocessing",
                        message="Running RAPIDS postprocessing.",
                        progress=0.7,
                    )
                    try:
                        _run_python_module(
                            python_executable=rsc_python,
                            module_name="backend.rsc_postprocess_cli",
                            input_h5ad=intermediate_path,
                            spec_path=spec_path,
                            output_dir=Path(job.job_dir),
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                    except Exception as exc:
                        spec["execution_backend"] = "scanpy-fallback"
                        _write_json(spec_path, spec)
                        _set_runtime_degraded(
                            f"RAPIDS postprocessing failed: {exc}"
                        )
                        with stdout_path.open("a") as stdout_file:
                            stdout_file.write(
                                "RAPIDS postprocessing failed; "
                                f"falling back to Scanpy: {exc}\n"
                            )
                        job_store.update(
                            job_id,
                            status="postprocessing",
                            message=(
                                "RAPIDS postprocessing failed; "
                                "running Scanpy fallback."
                            ),
                            progress=0.8,
                        )
                        _run_python_module(
                            python_executable=RUNTIME_SETTINGS.main_python,
                            module_name="backend.scanpy_postprocess_cli",
                            input_h5ad=intermediate_path,
                            spec_path=spec_path,
                            output_dir=Path(job.job_dir),
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                else:
                    _run_python_module(
                        python_executable=RUNTIME_SETTINGS.main_python,
                        module_name="backend.scanpy_postprocess_cli",
                        input_h5ad=intermediate_path,
                        spec_path=spec_path,
                        output_dir=Path(job.job_dir),
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                result_info = _read_json(Path(job.job_dir) / "result.json")

        _complete_job(job_id, result_info)
    except Exception as exc:
        _fail_job(job_id, f"Analysis failed: {exc}")


def _validate_selection_ids(adata: ad.AnnData, ids: list[int]) -> list[int]:
    validated: list[int] = []
    for item in ids:
        numeric = int(item)
        if numeric < 0 or numeric >= adata.n_obs:
            raise HTTPException(status_code=400, detail=f"Cell id {numeric} is out of range.")
        validated.append(numeric)
    return sorted(set(validated))


def _validate_selection_groups(adata: ad.AnnData, groups: list[list[int]]) -> list[list[int]]:
    return [_validate_selection_ids(adata, group) for group in groups if group]


def _resolved_job_spec(dataset_id: str, request: AnalysisJobRequest) -> dict[str, Any]:
    adata = store.get(dataset_id)
    analysis_type = request.analysis_type
    if analysis_type not in ANALYSIS_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type '{analysis_type}'.")

    embedding_key = _resolve_embedding_key(adata, request.embedding_key)
    color_by = _resolve_color_column(adata, request.color_by)
    obs_col = request.obs_col or color_by or _default_color_column(adata)
    cluster_key = request.cluster_key or ("leiden" if "leiden" in adata.obs.columns else color_by)

    selected_ids = _validate_selection_ids(adata, request.selected_ids)
    selected_groups = _validate_selection_groups(adata, request.selected_groups)

    if analysis_type == "lasso_view" and not selected_ids:
        raise HTTPException(status_code=400, detail="Lasso-View requires one active selection.")
    if analysis_type in RECONSTRUCTION_TYPES and not selected_groups:
        raise HTTPException(status_code=400, detail="Reconstruction requires at least one confirmed selection group.")
    if analysis_type == "downsample" and embedding_key is None:
        raise HTTPException(status_code=400, detail="Downsample requires a usable embedding.")
    if analysis_type == "reconstruct_embedding" and embedding_key is None:
        raise HTTPException(status_code=400, detail="Embedding reconstruction requires an embedding key.")
    if analysis_type == "lassoare" and request.lassoare_mode == "reconstruct_embedding" and embedding_key is None:
        raise HTTPException(status_code=400, detail="Embedding reconstruction requires an embedding key.")

    if analysis_type == "lasso_view" and obs_col is None:
        raise HTTPException(status_code=400, detail="Lasso-View requires an obs column for label propagation.")

    return {
        **request.model_dump(),
        "dataset_id": dataset_id,
        "dataset_name": store.get_name(dataset_id),
        "selected_ids": selected_ids,
        "selected_groups": selected_groups,
        "embedding_key": embedding_key,
        "color_by": color_by,
        "obs_col": obs_col,
        "cluster_key": cluster_key,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    degraded_reason = _runtime_degradation()
    return {
        "status": "ok",
        "profile": RUNTIME_SETTINGS.profile,
        "torch_device": "cuda" if RUNTIME_SETTINGS.profile == "cuda" else "cpu",
        "rsc": "configured" if RUNTIME_SETTINGS.rsc_python else "disabled",
        "degraded": degraded_reason is not None,
        "degraded_reason": degraded_reason,
    }


@app.post("/api/upload")
async def upload_h5ad(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".h5ad"):
        raise HTTPException(status_code=400, detail="Please upload a .h5ad file.")

    target_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    content = await file.read()
    target_path.write_bytes(content)

    try:
        adata = ad.read_h5ad(target_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read h5ad file: {exc}") from exc

    dataset_id = store.add(file.filename, adata)
    summary = _dataset_summary(dataset_id, adata)
    plot = None
    if summary["default_embedding"] is not None:
        plot = _build_plot_payload(adata, summary["default_embedding"], summary["default_color_by"])

    return {"summary": summary, "plot": plot}


@app.get("/api/samples")
def list_samples() -> dict[str, list[dict[str, object]]]:
    return {"samples": sample_manager.statuses()}


@app.post("/api/load-sample")
def load_sample(name: str = Query("sc_sampled.h5ad")) -> dict[str, Any]:
    try:
        sample_path = sample_manager.prepare(name)
    except SampleError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        adata = ad.read_h5ad(sample_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read sample h5ad file: {exc}",
        ) from exc

    dataset_id = store.add(sample_path.name, adata)
    summary = _dataset_summary(dataset_id, adata)
    plot = None
    if summary["default_embedding"] is not None:
        plot = _build_plot_payload(
            adata, summary["default_embedding"], summary["default_color_by"]
        )

    return {"summary": summary, "plot": plot}


@app.post("/api/datasets/{dataset_id}/plot")
def update_plot(dataset_id: str, request: PlotRequest) -> dict[str, Any]:
    adata = store.get(dataset_id)
    return _build_plot_payload(adata, request.embedding_key, request.color_by)


@app.post("/api/datasets/{dataset_id}/marker-plot")
def marker_plot(dataset_id: str, request: MarkerPlotRequest) -> dict[str, Any]:
    adata = store.get(dataset_id)
    return _build_marker_plot_payload(adata, request)


@app.post("/api/datasets/{dataset_id}/gene-expression-plot")
def gene_expression_plot(dataset_id: str, request: GeneExpressionPlotRequest) -> dict[str, Any]:
    adata = store.get(dataset_id)
    return _build_gene_expression_plot_payload(adata, request)


@app.post("/api/datasets/{dataset_id}/compute-umap")
def compute_umap(dataset_id: str) -> dict[str, Any]:
    adata = store.get(dataset_id)
    try:
        _compute_umap(adata)
        store.persist(dataset_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute UMAP: {exc}") from exc

    summary = _dataset_summary(dataset_id, adata)
    plot = _build_plot_payload(adata, "X_umap", summary["default_color_by"])
    return {"summary": summary, "plot": plot}


@app.post("/api/datasets/{dataset_id}/analysis-jobs")
def create_analysis_job(dataset_id: str, request: AnalysisJobRequest) -> dict[str, Any]:
    store.get(dataset_id)
    spec = _resolved_job_spec(dataset_id, request)

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job_dir / "spec.json", spec)
    (job_dir / "stdout.log").write_text("")
    (job_dir / "stderr.log").write_text("")

    job = AnalysisJob(
        job_id=job_id,
        dataset_id=dataset_id,
        analysis_type=request.analysis_type,
        status="queued",
        message="Job queued.",
        progress=0.0,
        job_dir=str(job_dir),
    )
    job_store.create(job)

    thread = threading.Thread(target=_run_analysis_job, args=(job_id,), daemon=True)
    thread.start()

    return _job_snapshot(job_id)


@app.get("/api/analysis-jobs/{job_id}")
def get_analysis_job(job_id: str) -> dict[str, Any]:
    return _job_snapshot(job_id)


@app.get("/api/analysis-jobs/{job_id}/download/{artifact}")
def download_analysis_artifact(job_id: str, artifact: str) -> FileResponse:
    job = job_store.get(job_id)
    if job.status != "completed" or not job.result_info:
        raise HTTPException(status_code=400, detail="This job has not completed yet.")

    artifact_map = {
        "result_h5ad": job.result_info.get("result_h5ad"),
        "mapping": job.result_info.get("mapping_path"),
    }
    target_path = artifact_map.get(artifact)
    if not target_path:
        raise HTTPException(status_code=404, detail=f"No downloadable artifact '{artifact}' is available for this job.")

    path = Path(target_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact}' was not found on disk.")
    return FileResponse(path, filename=path.name)


@app.post("/api/analysis-jobs/{job_id}/recover-selection")
def recover_selection(job_id: str, request: RecoverSelectionRequest) -> dict[str, Any]:
    job = job_store.get(job_id)
    if job.analysis_type != "downsample":
        raise HTTPException(status_code=400, detail="Selection recovery is only available for downsample jobs.")
    if job.status != "completed" or not job.result_info:
        raise HTTPException(status_code=400, detail="Downsample job is not completed yet.")

    mapping_path = job.result_info.get("mapping_path")
    if not mapping_path:
        raise HTTPException(status_code=400, detail="No mapping file is available for this job.")

    mapping = _read_json(Path(mapping_path))
    nearest_ids = mapping.get("nearest_downsampled_local_id", [])
    selected_set = {int(item) for item in request.ids}
    recovered_ids = [index for index, sampled_id in enumerate(nearest_ids) if sampled_id in selected_set]
    return {"ids": recovered_ids, "count": len(recovered_ids)}


@app.get("/")
def index() -> FileResponse:
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/datasets/{dataset_id}/summary")
def dataset_summary(dataset_id: str) -> dict[str, Any]:
    adata = store.get(dataset_id)
    return _dataset_summary(dataset_id, adata)


@app.get("/api/datasets/{dataset_id}/export-selection")
def export_selection(
    dataset_id: str,
    ids: str = Query(..., description="JSON array or comma separated numeric ids."),
) -> dict[str, Any]:
    store.get(dataset_id)
    try:
        parsed = json.loads(ids)
        if not isinstance(parsed, list):
            raise ValueError("ids must be a list")
        numeric_ids = [int(item) for item in parsed]
    except Exception:
        try:
            numeric_ids = [int(item) for item in ids.split(",") if item.strip()]
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Failed to parse ids.") from exc
    return {"ids": numeric_ids}
