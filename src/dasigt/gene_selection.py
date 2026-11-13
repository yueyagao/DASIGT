"""Consistent gene selection rules for module-level enrichment.

`pathway_top_genes == -1` means: use every gene assigned to the module by the
maximum soft-membership score (row-wise argmax), rather than ranking all genes
for every module. This produces a deterministic, non-overlapping hard assignment
for enrichment only; the model's learned soft assignments remain unchanged.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def select_module_genes(
    assignment: pd.DataFrame,
    module: str,
    top_n: int,
) -> Tuple[List[str], str]:
    """Select genes for one module.

    Args:
        assignment: DataFrame indexed by gene symbol; each column is a module's
            soft assignment/membership score.
        module: Target module column name.
        top_n: Positive integer selects top-N by the target column. ``-1``
            selects all genes whose highest membership score is this module.

    Returns:
        (gene_symbols, selection_mode)

    Ties in row-wise argmax are assigned to the first column, making the rule
    deterministic. Missing membership values are treated as zero.
    """
    if assignment is None or assignment.empty or assignment.shape[1] == 0:
        return [], "all_hard_assigned_to_module" if int(top_n) == -1 else "top_n_by_soft_membership"
    top_n = int(top_n)
    if top_n != -1 and top_n < 1:
        raise ValueError("pathway_top_genes/top_n must be -1 (all genes assigned to each module) or a positive integer")
    columns = list(assignment.columns)
    if module not in columns:
        raise KeyError(f"Module {module!r} is not a column in the assignment matrix")

    values = assignment.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    module_idx = columns.index(module)
    if top_n == -1:
        # One module per gene for enrichment-list construction. This avoids
        # sending the full ~5,000-gene matrix as every module's query list.
        winner_idx = np.argmax(values, axis=1)
        selected_indices = np.flatnonzero(winner_idx == module_idx)
        if selected_indices.size:
            selected_indices = selected_indices[np.argsort(values[selected_indices, module_idx])[::-1]]
        mode = "all_hard_assigned_to_module"
    else:
        scores = values[:, module_idx]
        order = np.argsort(scores)[::-1]
        selected_indices = order[:top_n]
        mode = "top_n_by_soft_membership"

    genes: List[str] = []
    seen = set()
    for idx in selected_indices:
        gene = str(assignment.index[int(idx)]).strip()
        if not gene or gene.lower() == "nan" or gene in seen:
            continue
        genes.append(gene)
        seen.add(gene)
    return genes, mode
