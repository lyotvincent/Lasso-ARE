"""
Plugin-enabled training entrypoints for LassoARE.
"""

from .plugins import MetricPluginConfig, SurrogateScorer, create_moran_i_plugin, moran_i_soft
from .reconstruction_plugin import reconstruction_with_lasso_are_plugins, reconstruction_with_plugins

__all__ = [
    "MetricPluginConfig",
    "SurrogateScorer",
    "create_moran_i_plugin",
    "moran_i_soft",
    "reconstruction_with_lasso_are_plugins",
    "reconstruction_with_plugins",
]
