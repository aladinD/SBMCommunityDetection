import torch
from typing import List, Tuple, Any, Dict

import utils


def driver(assignments: torch.Tensor, k: int, edges: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ties together functions of this module to calculate the log probs. Returns
    the log probs together with an MLE of the block matrix.
    Args:
        assignments: Propensity values, probability of nodes having an edge of
            a specific color. Has shape (S, |V|, k).
        edges: Edge List of graph with node indices of tail and head, and
            the number of edges between the groups. Has shape (|E|, 3).
        k: Number of groups.

    Returns:
        log_probs: Log probs of each samle, has shape (S,).
        eta: MLE of eta for each sample, has shape (S, k, k).
    """

    # propensities have shape (S, |E|, k)
    propensities_tail = assignments[:, edges[:, 0], :]
    propensities_head = assignments[:, edges[:, 1], :]
    # Shape (1, |E|)
    num_edges = edges[:, -1].float().unsqueeze(0)
    # Shape (S, |E|)
    tmp = torch.log(torch.clamp_min(torch.sum(propensities_tail * propensities_head, dim=-1), 1e-12))
    # Shape (S,)
    part1 = torch.sum(num_edges * tmp, dim=-1)

    # Shape (S, k, k)
    # tmp = torch.matmul(torch.transpose(assignments, -2, -1), assignments)
    # Shape (S)
    # part2 = torch.sum(torch.sum(tmp, dim=-1), dim=-1)

    # (S, |V|, |V|, k) = (S, |V|, 1, k) * (S, 1, |V|, k)
    tmp = assignments.unsqueeze(2) * assignments.unsqueeze(1)
    part2 = torch.sum(torch.sum(torch.sum(tmp, -1), -1), -1)
    return part1 - part2, {'etas': torch.zeros(assignments.shape[0])}
