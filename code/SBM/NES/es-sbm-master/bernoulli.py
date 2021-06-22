import torch
from typing import List, Tuple, Any, Dict


def estimate_eta(edges: torch.Tensor, assignments: torch.Tensor, k: int) -> torch.Tensor:
    """
    Maximum Likelihood estimate of block matrix from assignments. Returns one
    MLE estimate per sample.

    Args:
        edges: Shape (|V|, 2).
        assignments: Shape (S, |V|).
        k: Number of groups.

    Returns:
        eta: Shape (S, k, k)
    """
    z_tail = assignments[:, edges[:, 0]]  # Replace node id of tail with its group: (S, |E|)
    z_head = assignments[:, edges[:, 1]]  # Replace node id of head with its group: (S, |E|)

    # Shape (S, |E|, k)
    z_tail_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_tail, k), -1).float()
    # Shape (S, |E|, k)
    z_head_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_head, k), -2).float()

    # Count the number of edges that run between the different groups.
    # matmul has shape (S, |E|, k, k), reduce sum makes to (S, k, k).
    edges_btw_groups = torch.sum(torch.matmul(z_tail_one_hot, z_head_one_hot), dim=1)

    # One hot has shape (S, |V|, k), the sum then has shape (S, k), i.e., the
    # number of nodes in each group.
    num_members = torch.sum(torch.nn.functional.one_hot(assignments, k), 1).float()
    # Has shape (S, k, k), each entry in the matrix contains all possible edges
    # running in the group, including self-edges.
    all_edges_btw_groups = torch.matmul(
        torch.unsqueeze(num_members, -1),
        torch.unsqueeze(num_members, -2)
    )
    # Calculate the number of self-edges, returns a diagonal matrix of shape
    # (S, k, k), with the number of nodes in each group on the main diagonal.
    num_self_edges = torch.unsqueeze(torch.eye(k), 0) * torch.unsqueeze(num_members, 1)
    # Correct for self-edges.
    all_edges_btw_groups = all_edges_btw_groups - num_self_edges
    # Calculate MLE of eta.
    eta = edges_btw_groups / (torch.clamp_min(all_edges_btw_groups.float(), 1e-9))
    return eta


def bernoulli_logprob(edges: torch.Tensor, assignments: torch.Tensor,
                      eta: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculate the log prob for each assignment, given the MLE of eta.

    Args:
        edges: (|V|, 2)
        assignments (S, |V|)
        eta: (S, k, k)
        k: int

    Returns:
        log_prob: Shape (S,)
    """
    z_tail = assignments[:, edges[:, 0]]  # Replace node id of tail with its group: (S, |E|)
    z_head = assignments[:, edges[:, 1]]  # Replace node id of head with its group: (S, |E|)

    # Shape (S, |E|, k)
    z_tail_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_tail, k), -2).float()
    # Shape (S, |E|, k)
    z_head_one_hot = torch.unsqueeze(torch.nn.functional.one_hot(z_head, k), -1).float()

    ps = torch.matmul(torch.matmul(z_tail_one_hot, torch.unsqueeze(eta, -3)), z_head_one_hot)
    log_prob = torch.sum(torch.log(ps[:, :, 0, 0] + 1e-9), dim=-1)
    return log_prob


def driver(assignments: torch.Tensor, k: int, edges: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ties together functions of this module to calculate the log probs. Returns
    the log probs together with an MLE of the block matrix.
    Args:
        edges: Edge List of graph with node indices of tail and head, has shape (|E|, 2).
        assignments: Node to group assignments, has shape (S, |V|).
        k: Number of groups.

    Returns:
        log_probs: Log probs of each samle, has shape (S,).
        eta: MLE of eta for each sample, has shape (S, k, k).
    """
    etas = estimate_eta(edges, assignments, k)
    log_probs = bernoulli_logprob(edges, assignments, etas, k)
    log_probs[torch.isnan(log_probs)] = -1. * 1e9
    return log_probs, {'etas': etas}
