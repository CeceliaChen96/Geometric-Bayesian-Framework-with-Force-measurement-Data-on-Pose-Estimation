"""Shared plotting functions extracted from the notebooks."""

from .posterior_core import plot_refinement_history
from .sampling_core import (
    plot_unit_sphere,
    visualize_mf_sampling,
    plot_sphere_scatter,
    visualize_single_case_v2,
    visualize_recovered_axes,
    plot_family_results,
    plot_family_piball_partB,
    plot_partB_angle_boxplot,
    plot_single_axis_heatmap,
    visualize_axis_density_heatmaps,
    compare_re3_density_heatmaps_across_family,
    plot_tangent_space_3d,
    plot_tangent_pairwise,
    visualize_tangent_space_for_results,
    compare_tangent_space_across_family,
    plot_family_angle_density_curves,
)

__all__ = [
    "plot_refinement_history",
    "plot_unit_sphere",
    "visualize_mf_sampling",
    "plot_sphere_scatter",
    "visualize_single_case_v2",
    "visualize_recovered_axes",
    "plot_family_results",
    "plot_family_piball_partB",
    "plot_partB_angle_boxplot",
    "plot_single_axis_heatmap",
    "visualize_axis_density_heatmaps",
    "compare_re3_density_heatmaps_across_family",
    "plot_tangent_space_3d",
    "plot_tangent_pairwise",
    "visualize_tangent_space_for_results",
    "compare_tangent_space_across_family",
    "plot_family_angle_density_curves",
]
