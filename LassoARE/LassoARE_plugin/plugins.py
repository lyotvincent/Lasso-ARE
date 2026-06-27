from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


MetricValue = Union[float, int, Sequence[float], np.ndarray, torch.Tensor]


def ensure_metric_tensor(metric: MetricValue, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert user metric outputs to a 1D tensor on the target device."""
    if isinstance(metric, torch.Tensor):
        out = metric.to(device=device, dtype=dtype)
    else:
        out = torch.as_tensor(metric, dtype=dtype, device=device)
    if out.ndim == 0:
        out = out.unsqueeze(0)
    return out.reshape(-1)


def _match_length(value: Any, n_samples: int) -> bool:
    try:
        return len(value) == n_samples
    except TypeError:
        return False


def slice_requirements(requirements: Any, indices: Sequence[int], n_samples: int) -> Any:
    """
    Slice sample-aligned requirements for batch-safe plugins.

    Square matrices are sliced on both axes, while sample-first arrays/tensors are
    sliced on the first axis. Non sample-aligned objects are returned as-is.
    """
    if requirements is None:
        return None

    if isinstance(requirements, dict):
        return {key: slice_requirements(value, indices, n_samples) for key, value in requirements.items()}

    if isinstance(requirements, torch.Tensor):
        if requirements.ndim >= 2 and requirements.shape[0] == n_samples and requirements.shape[1] == n_samples:
            idx = torch.as_tensor(indices, device=requirements.device, dtype=torch.long)
            return requirements.index_select(0, idx).index_select(1, idx)
        if requirements.ndim >= 1 and requirements.shape[0] == n_samples:
            idx = torch.as_tensor(indices, device=requirements.device, dtype=torch.long)
            return requirements.index_select(0, idx)
        return requirements

    if scipy.sparse.issparse(requirements):
        if requirements.ndim == 2 and requirements.shape[0] == n_samples and requirements.shape[1] == n_samples:
            return requirements[indices][:, indices]
        if requirements.shape[0] == n_samples:
            return requirements[indices]
        return requirements

    if isinstance(requirements, np.ndarray):
        if requirements.ndim >= 2 and requirements.shape[0] == n_samples and requirements.shape[1] == n_samples:
            return requirements[np.ix_(indices, indices)]
        if requirements.ndim >= 1 and requirements.shape[0] == n_samples:
            return requirements[indices]
        return requirements

    if isinstance(requirements, (list, tuple)) and _match_length(requirements, n_samples):
        sliced = [requirements[i] for i in indices]
        return type(requirements)(sliced) if isinstance(requirements, tuple) else sliced

    return requirements


def _resize_vector(vector: torch.Tensor, out_dim: int) -> torch.Tensor:
    vector = vector.reshape(1, 1, -1)
    if vector.shape[-1] == out_dim:
        return vector.reshape(-1)
    resized = F.interpolate(vector, size=out_dim, mode="linear", align_corners=False)
    return resized.reshape(-1)


def summarize_representation(representation: torch.Tensor, summary_dim: int = 64) -> torch.Tensor:
    """
    Pool a representation matrix into a fixed-width vector for surrogate scoring.
    """
    rep = representation
    if rep.ndim == 1:
        rep = rep.unsqueeze(1)
    elif rep.ndim > 2:
        rep = rep.reshape(rep.shape[0], -1)

    rep = rep.float()
    mean_vec = rep.mean(dim=0)
    std_vec = rep.std(dim=0, unbiased=False) if rep.shape[0] > 1 else torch.zeros_like(mean_vec)
    max_vec = rep.max(dim=0).values

    pooled = torch.cat(
        [
            _resize_vector(mean_vec, summary_dim),
            _resize_vector(std_vec, summary_dim),
            _resize_vector(max_vec, summary_dim),
        ],
        dim=0,
    )
    return pooled.unsqueeze(0)


class RunningMetricNormalizer:
    """Stream normalizer for plugin metrics."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None

    def update(self, value: torch.Tensor) -> None:
        vector = value.detach().float().cpu().reshape(-1)
        if self.count == 0:
            self.mean = vector.clone()
            self.m2 = torch.zeros_like(vector)
            self.count = 1
            return

        self.count += 1
        delta = vector - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = vector - self.mean
        self.m2 = self.m2 + delta * delta2

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return value
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        if self.count < 2 or self.m2 is None:
            std = torch.ones_like(mean)
        else:
            std = torch.sqrt(self.m2.to(device=value.device, dtype=value.dtype) / max(self.count - 1, 1) + self.eps)
        return (value - mean) / std.clamp_min(self.eps)


class SurrogateScorer(nn.Module):
    """Small MLP used to fit non-differentiable metrics."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class MetricPluginConfig:
    name: str
    metric_fn: Callable[[Any, Any], MetricValue]
    requirements: Any = None
    apply_on: str = "cluster_probs"
    mode: str = "differentiable"
    goal: str = "maximize"
    weight: float = 1.0
    frequency: int = 5
    warmup_epochs: int = 0
    ramp_epochs: int = 10
    reduction: str = "mean"
    target_value: Optional[MetricValue] = None
    full_dataset_only: bool = False
    standardize: bool = True
    surrogate_summary_dim: int = 64
    surrogate_hidden_dims: Sequence[int] = (128, 64)
    surrogate_train_steps: int = 20
    surrogate_lr: float = 1e-3
    surrogate_buffer_size: int = 128
    extra: Dict[str, Any] = field(default_factory=dict)

    normalizer: RunningMetricNormalizer = field(default_factory=RunningMetricNormalizer, init=False, repr=False)
    scorer: Optional[SurrogateScorer] = field(default=None, init=False, repr=False)
    scorer_optimizer: Optional[optim.Optimizer] = field(default=None, init=False, repr=False)
    output_dim: Optional[int] = field(default=None, init=False)
    surrogate_inputs: List[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    surrogate_targets: List[torch.Tensor] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        valid_modes = {"differentiable", "surrogate", "hybrid"}
        valid_goals = {"maximize", "minimize", "target"}
        valid_apply = {"latent", "reconstruction", "cluster_probs", "custom"}
        valid_reduction = {"mean", "sum", "none"}

        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported plugin mode '{self.mode}'")
        if self.goal not in valid_goals:
            raise ValueError(f"Unsupported plugin goal '{self.goal}'")
        if self.apply_on not in valid_apply:
            raise ValueError(f"Unsupported apply_on '{self.apply_on}'")
        if self.reduction not in valid_reduction:
            raise ValueError(f"Unsupported reduction '{self.reduction}'")
        if self.goal == "target" and self.target_value is None:
            raise ValueError(f"Plugin '{self.name}' uses goal='target' but target_value is missing")
        if self.frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")

    def is_active(self, epoch: int) -> bool:
        return self.weight > 0 and epoch >= self.warmup_epochs

    def current_weight(self, epoch: int) -> float:
        if not self.is_active(epoch):
            return 0.0
        if self.ramp_epochs <= 0:
            return float(self.weight)
        progress = min(1.0, (epoch - self.warmup_epochs + 1) / float(self.ramp_epochs))
        return float(self.weight) * progress

    def should_refresh(self, epoch: int) -> bool:
        if not self.is_active(epoch):
            return False
        return (epoch - self.warmup_epochs) % self.frequency == 0

    def reduce_metric(self, metric: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return metric.mean()
        if self.reduction == "sum":
            return metric.sum()
        if metric.numel() != 1:
            raise ValueError(
                f"Plugin '{self.name}' uses reduction='none' but produced a vector metric of shape {tuple(metric.shape)}"
            )
        return metric.reshape(())

    def normalize_metric(self, metric: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return metric
        return self.normalizer.normalize(metric)

    def update_metric_stats(self, metric: torch.Tensor) -> None:
        if self.standardize:
            self.normalizer.update(metric)

    def _target_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        target = ensure_metric_tensor(self.target_value, device=device, dtype=dtype)
        return self.normalize_metric(target) if self.standardize else target

    def metric_to_loss(self, metric: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        metric_vec = metric.reshape(-1)
        metric_vec = metric_vec if normalized else self.normalize_metric(metric_vec)

        if self.goal == "maximize":
            return -self.reduce_metric(metric_vec)
        if self.goal == "minimize":
            return self.reduce_metric(metric_vec)

        target = self._target_tensor(metric_vec.device, metric_vec.dtype)
        return F.mse_loss(metric_vec, target)

    def ensure_scorer(self, device: torch.device, output_dim: int) -> None:
        input_dim = 3 * self.surrogate_summary_dim
        if self.scorer is not None and self.output_dim == output_dim:
            return
        self.output_dim = output_dim
        self.scorer = SurrogateScorer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.surrogate_hidden_dims,
        ).to(device)
        self.scorer_optimizer = optim.Adam(self.scorer.parameters(), lr=self.surrogate_lr, amsgrad=True)

    def has_scorer(self) -> bool:
        return self.scorer is not None and self.scorer_optimizer is not None

    def add_surrogate_example(self, representation: torch.Tensor, target_metric: torch.Tensor) -> None:
        summary = summarize_representation(representation.detach().cpu(), self.surrogate_summary_dim).squeeze(0)
        target = target_metric.detach().cpu().reshape(-1)
        self.surrogate_inputs.append(summary)
        self.surrogate_targets.append(target)
        if len(self.surrogate_inputs) > self.surrogate_buffer_size:
            self.surrogate_inputs = self.surrogate_inputs[-self.surrogate_buffer_size :]
            self.surrogate_targets = self.surrogate_targets[-self.surrogate_buffer_size :]

    def train_surrogate(self, device: torch.device) -> Optional[float]:
        if not self.has_scorer() or not self.surrogate_inputs:
            return None

        x = torch.stack(self.surrogate_inputs).to(device=device, dtype=torch.float32)
        y = torch.stack(self.surrogate_targets).to(device=device, dtype=torch.float32)

        self.scorer.train()
        loss_value = None
        for _ in range(self.surrogate_train_steps):
            pred = self.scorer(x)
            loss = F.huber_loss(pred, y)
            self.scorer_optimizer.zero_grad()
            loss.backward()
            self.scorer_optimizer.step()
            loss_value = float(loss.item())
        return loss_value


def coerce_plugins(plugins: Optional[Sequence[Union[MetricPluginConfig, Dict[str, Any]]]]) -> List[MetricPluginConfig]:
    if plugins is None:
        return []

    resolved: List[MetricPluginConfig] = []
    for idx, plugin in enumerate(plugins):
        if isinstance(plugin, MetricPluginConfig):
            resolved.append(plugin)
        elif isinstance(plugin, dict):
            resolved.append(MetricPluginConfig(**plugin))
        else:
            raise TypeError(f"Unsupported plugin at position {idx}: {type(plugin)!r}")
    return resolved


def _get_weight_matrix(requirements: Any, device: torch.device, dtype: torch.dtype) -> Union[torch.Tensor, torch.sparse.Tensor]:
    if isinstance(requirements, dict):
        for key in ("adjacency", "weight_matrix", "weights", "W"):
            if key in requirements:
                requirements = requirements[key]
                break

    if isinstance(requirements, torch.Tensor):
        return requirements.to(device=device, dtype=dtype)

    if scipy.sparse.issparse(requirements):
        coo = requirements.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
        values = torch.tensor(coo.data, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()

    if isinstance(requirements, np.ndarray):
        return torch.as_tensor(requirements, dtype=dtype, device=device)

    raise TypeError("Moran's I requires a dense or sparse adjacency/weight matrix in requirements")


def moran_i_soft(representation: torch.Tensor, requirements: Any, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable Moran's I on continuous representations.

    The input should be [n_samples, n_features] or [n_samples].
    """
    if representation.ndim == 1:
        x = representation.unsqueeze(1)
    elif representation.ndim > 2:
        x = representation.reshape(representation.shape[0], -1)
    else:
        x = representation

    x = x.float()
    n_samples = x.shape[0]
    weights = _get_weight_matrix(requirements, device=x.device, dtype=x.dtype)

    if weights.shape[0] != n_samples or weights.shape[1] != n_samples:
        raise ValueError(
            f"Moran's I expected a square weight matrix aligned with {n_samples} samples, "
            f"got shape {tuple(weights.shape)}"
        )

    x_centered = x - x.mean(dim=0, keepdim=True)
    if weights.is_sparse:
        wx = torch.sparse.mm(weights, x_centered)
        s0 = weights.values().sum()
    else:
        wx = weights @ x_centered
        s0 = weights.sum()

    numerator = (x_centered * wx).sum(dim=0)
    denominator = x_centered.pow(2).sum(dim=0).clamp_min(eps)
    moran = (float(n_samples) / s0.clamp_min(eps)) * (numerator / denominator)
    return moran.reshape(-1)


def create_moran_i_plugin(
    requirements: Any,
    name: str = "moran_i_soft",
    apply_on: str = "cluster_probs",
    mode: str = "differentiable",
    goal: str = "maximize",
    weight: float = 1.0,
    frequency: int = 2,
    warmup_epochs: int = 0,
    ramp_epochs: int = 5,
    reduction: str = "mean",
    full_dataset_only: bool = True,
    **kwargs: Any,
) -> MetricPluginConfig:
    return MetricPluginConfig(
        name=name,
        metric_fn=moran_i_soft,
        requirements=requirements,
        apply_on=apply_on,
        mode=mode,
        goal=goal,
        weight=weight,
        frequency=frequency,
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        reduction=reduction,
        full_dataset_only=full_dataset_only,
        **kwargs,
    )
