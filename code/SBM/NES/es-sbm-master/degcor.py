"""
Implements estimation methods for the degree corrected SBM.
"""
import torch
from typing import List, Any, Tuple, Dict

import poisson
import sbm
import utils


def estimate_kappas(edges_btw_groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate theta parameters of degree corrected sbm. Those correspond to the
    probability that an edge lands on the group itself.

    Args:
        edges_btw_groups: The number of edges between groups, shape (S, k, k).

    Returns:
        kappa_in: The number of edges that land on the groups itself (S, k).
        kappa_out: The number of edges that originate in each group (S, k).
    """
    kappa_in = torch.sum(edges_btw_groups, -2)
    kappa_out = torch.sum(edges_btw_groups, -1)
    return kappa_in, kappa_out


def estimate_theta(kappas: torch.Tensor, degrees: torch.Tensor,
                   assignment: torch.Tensor) -> torch.Tensor:
    """
    Estimate the theta paramaeters.

    Args:
        kappas: Has shape (S, k).
        degrees:  Has shape (|V|,)
        assignment: Has shape (S, |V|)

    Returns:
        thetas: (S, |V|)
    """
    # Has shape (S, |V|, k)
    assignment_one_hot = torch.nn.functional.one_hot(assignment)
    # Has shape (1, |V|, 1)
    tmp = utils.multiunsqueeze(degrees, [0, -1])
    # Has shape (S, |V|, k), Map degree of node on the correct group index.
    tmp = tmp * assignment_one_hot
    # (S, |V|, k) / (S, 1, k) = (S, |V|, k). Divide degree of each node by sum
    # of degrees of the group.
    theta = tmp / kappas.unsqueeze(-2)
    # (S, |V|) - Sum out the group dimension, not needed.
    theta = torch.sum(theta, -1)
    return theta


def logprob(theta_in: torch.Tensor, theta_out: torch.Tensor, eta: torch.Tensor,
            deg_in: torch.Tensor, deg_out: torch.Tensor, edges_btw_groups: torch.Tensor) -> torch.Tensor:
    """

    Args:
        theta_in: (S, |V|)
        theta_out: (S, |V|)
        eta: (S, k, k)
        deg_in: (S, |V|)
        deg_out: (S, |V|)
        edges_btw_groups: (S, k, k)

    Returns:
        log_probs: (S,)
    """
    in_deg_part = torch.sum(deg_in * torch.log(torch.clamp_min(theta_in, 1e-12)), dim=-1)
    out_deg_part = torch.sum(deg_out * torch.log(torch.clamp_min(theta_out, 1e-12)), dim=-1)
    eta_part = torch.sum(
        torch.sum(
            edges_btw_groups * torch.log(torch.clamp_min(eta, 1e-6)) - eta,
             dim=-1
         ),
        dim=-1
    )
    log_prob = 2. * in_deg_part + 2. * out_deg_part + eta_part
    return log_prob


def driver(assignments: torch.Tensor, k: int, edges: torch.Tensor, in_degrees: torch.Tensor,
           out_degrees: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Ties together functions of this module to calculate the log probs. Returns
    the log probs together with an MLE of the block matrix.
    Args:
        assignments: Node to group assignments, has shape (S, |V|).
        edges: Edge List of graph with node indices of tail and head, and
            the number of edges between the groups. Has shape (|E|, 3).
        in_degrees: In degrees of nodes, has shape (|V|).
        out_degrees: Out degrees of nodes, has shape (|V|).
        k: Number of groups.

    Returns:
        log_probs: Log probs of each samle, has shape (S,).
        eta: MLE of eta for each sample, has shape (S, k, k).
    """
    # nodes_in_groups = poisson.num_nodes_in_groups(assignments)
    edges_btw_groups = poisson.num_edges_between_groups(edges, assignments, k)
    kappa_in, kappa_out = estimate_kappas(edges_btw_groups)
    theta_in = estimate_theta(kappa_in, in_degrees, assignments)
    theta_out = estimate_theta(kappa_out, out_degrees, assignments)
    # etas = poisson.estimate_eta(edges_btw_groups, nodes_in_groups)
    etas = edges_btw_groups
    log_probs = logprob(
        theta_in=theta_in,
        theta_out=theta_out,
        eta=etas,
        deg_in=in_degrees,
        deg_out=out_degrees,
        edges_btw_groups=edges_btw_groups
    )
    log_probs[torch.isnan(log_probs)] = -1. * 1e9
    return log_probs, {"etas": etas, 'theta_in': theta_in, 'theta_out': theta_out}

