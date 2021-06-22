import torch
import utils
from typing import List, Tuple, Any, Dict


def num_nodes_in_groups(assignments: torch.Tensor) -> torch.Tensor:
    """
    Calculate the number of nodes that are in each group.

    Args:
        assignments: Node to group assignments of shape (S, |V|).

    Returns:
        num_nodes_in_groups: Has shape (S, k)
    """
    num_nodes_in_groups = torch.sum(torch.nn.functional.one_hot(assignments), dim=-2)
    return num_nodes_in_groups.float()


def num_edges_between_groups(edges: torch.Tensor, assignments: torch.Tensor,
                             k: int) -> torch.Tensor:
    """
    Calculate the number of edges that ly between each pair of groups.

    Args:
        edges: Tensor of shape (|E|, 3).
        assignments: Tensor of shape (S, |V|)
        k: Number of groups.

    Returns:
        num_edges_btw_groups: Tensor of shape (S, k, k).
    """
    z_tail = assignments[:, edges[:, 0]]  # Replace node id of tail with its group: (S, |E|)
    z_head = assignments[:, edges[:, 1]]  # Replace node id of head with its group: (S, |E|)
    # Shape (S, |E|, k)
    z_tail_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_tail, k), -1).float()
    # Shape (S, |E|, k)
    z_head_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_head, k), -2).float()
    # Shape (S, |E|, k, k)
    num_edges_btw_groups = torch.sum(
        torch.matmul(z_tail_one_hot, z_head_one_hot) \
        * utils.multiunsqueeze(edges[:, -1].float(), [-1, -1]),
        dim=-3
    )
    return num_edges_btw_groups


def estimate_eta(edges_btw_groups: torch.Tensor, nodes_in_groups: torch.Tensor) -> torch.Tensor:
    """
    Maximum Likelihood estimate of block matrix from assignments. Returns one
    MLE estimate per sample.

    Args:
        edges_btw_groups: Number of edges between each pair of groups, has shape (S, k, k).
        nodes_in_groups: Number of nodes in each group has shape (S, k).

    Returns:
        eta: Shape (S, k, k)
    """
    num_interactions = torch.matmul(
        nodes_in_groups.unsqueeze(-1),
        nodes_in_groups.unsqueeze(-2)
    )

    # num_interactions can be zero if no nodes are assigned to a group. In this
    # case, the corresponding entry in eta is also zero. Thus, clamp the values
    # in num_interactions to a small value to avoid division by zero
    eta = edges_btw_groups / torch.clamp_min(num_interactions, 1e-6)
    return eta


def logprob(edges_btw_groups: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Calculates the log prob for each assignment based on the poisson SBM
    given the MLE of eta.

    Args:
        edges_btw_groups: (S, k, k), Number of edges between groups.
        eta: (S, k, k), MLE of eta.

    Returns:
        log_probs: Shape(S, )
    """
    log_probs = torch.sum(
        torch.sum(
            edges_btw_groups * torch.log(torch.clamp_min(eta, 1e-9)),
            dim=-1,
            ),
        dim=-1
    )
    return log_probs


def driver(assignments: torch.Tensor, k: int, edges: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ties together functions of this module to calculate the log probs. Returns
    the log probs together with an MLE of the block matrix.
    Args:
        edges: Edge List of graph with node indices of tail and head, and
            the number of edges between the groups. Has shape (|E|, 3).
        assignments: Node to group assignments, has shape (S, |V|).
        k: Number of groups.

    Returns:
        log_probs: Log probs of each samle, has shape (S,).
        eta: MLE of eta for each sample, has shape (S, k, k).
    """
    nodes_in_groups = num_nodes_in_groups(assignments)
    edges_btw_groups = num_edges_between_groups(edges, assignments, k)
    etas = estimate_eta(edges_btw_groups, nodes_in_groups)
    log_probs = logprob(edges_btw_groups, etas)
    log_probs[torch.isnan(log_probs)] = -1. * 1e9
    return log_probs, {'etas': etas}
