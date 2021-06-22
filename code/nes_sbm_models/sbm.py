import time

import torch
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Any, Union

import vis
import degcor
import overlapping
import bernoulli
import poisson

DEV = 'cuda'
DEV_NUM = 0


def make_utilities(population_size: int) -> torch.Tensor:
    """
    Make utility values to weight samples with.

    Args:
        population_size:

    Returns:
        utility_values: Shape (S,).
    """
    utility_values = np.log(population_size / 2. + 1) - \
                     np.log(np.arange(1, population_size + 1))
    utility_values[utility_values < 0] = 0.
    utility_values /= np.sum(utility_values)
    utility_values -= 1. / population_size
    utility_values = utility_values.astype(np.float32)
    return torch.tensor(utility_values, device=DEV)


def scale_noise(noise: torch.Tensor, fitness: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """
    Scale noise with the fitness values. Fitness values are assigned to noise
    vectors based on the `log_probs` vector. The better the `log_probs` vector,
    the higher the fitness and thus the more the gradient should point into
    that direction.

    Args:
        noise: Noise used to perturb the parameters (S, |V|, k).
        fitness: Vector of scalars, first element `fitness[0]` is the fitness corresponding
            to the best sample, the last element `fitness[-1] is the fitness
            for the worst sample. Has shape (S).
        log_probs: Log prob resulting from the sampled placements. Has shape (S,).

    Returns:
        scaled_noise: Noise scaled with fitness values. Has shape (S, |V|, d).
    """
    sorted_values, indices = torch.sort(-1. * log_probs)
    scaled_noise = fitness.unsqueeze(1).unsqueeze(2) * noise[indices, :, :]
    return scaled_noise


def apply_gradient(params: torch.Tensor, grads: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Apply the gradient to the parameters.

    Args:
        params: Tensor with parameters of shape (|V|, d).
        grads: Tensor with gradients of shape (|V|, d).

    Returns:
        params_prime: Tensor of shape (|V|, d).
    """
    params_prime = params + lr * grads
    return params_prime


def calculate_gradient(scaled_noise: torch.Tensor) -> torch.Tensor:
    """
    Calcualte the gradient from the scaled noise.

    Args:
        scaled_noise: Tensor of shape (S, |V|, d).

    Returns:
        grad: Tensor of shape (|V|, d).
    """
    grad = torch.sum(scaled_noise, 0)
    return grad


def sample_placements(mutated_params: torch.Tensor) -> torch.Tensor:
    """
    Sample indices from mutated paramters.

    Args:
        mutated_params: Shape (S, |V|, k)

    Returns:
        memberships: Shape (S, |V|).
    """
    # cumsum = torch.cumsum(torch.softmax(mutated_params, dim=-1), dim=-1)
    # noise = torch.rand(mutated_params.shape[0], mutated_params.shape[1], 1, requires_grad=False)
    # indices = torch.argmin((cumsum < noise).int(), dim=-1)
    indices = torch.distributions.Categorical(logits=mutated_params).sample()
    return indices


def sample_propensities(mutated_params: torch.Tensor) -> torch.Tensor:
    """
    Create propensities for parameters, i.e., soft assignment.

    Args:
        mutated_params: Shape (S, |V|, k)

    Returns:
        memberships: Shape (S, |V|, k).
    """
    return torch.softmax(mutated_params, -1)


def mutate_params(params: torch.Tensor, num_samples: int, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create `num_samples` mutation of parameters.

    Args:
        params: Shape (|V|, k)

    Returns:
        mutated_params: Shape (S, |V|, k)
    """
    noise = torch.randn([num_samples, params.shape[0], params.shape[1]], requires_grad=False, device=DEV)
    mutated_params = torch.unsqueeze(params, 0) + noise
    return noise, mutated_params


def calc_scale_decrease(num_iter: int, fraction: float, target_scale: float) -> float:
    stop = num_iter * fraction
    m = (target_scale - 1.) / stop
    return m


def learned_embeddings(constants: Dict[str, torch.Tensor], population_size: int, k: int,
                       num_nodes: int, driver: Union[str, callable], make_membership: callable,
                       lr: float, num_iter=10000, params: torch.Tensor=None,
                       patience: int=None, population_batch_size=1000000000) -> Dict[str, torch.Tensor]:
    """

    Args:
        constants: Dictionary containing constant arguments to driver functions.
            For example, edges, in_degrees, out_degrees. Check the signature call
            of a specific driver function to determine the fields of this dict.
        population_size: Number of samples for ES.
        k: Number of groups.
        num_nodes: Number of nodes in graph.
        driver: Callable that implements the calculation of the log-prob and
            estimates parameters of a model.
        lr: Learning rate that should be used.
        num_iter: Number of iterations that should be executed.
        params: Initial guess of parameters, if None start from uniform distribution
            over groups.
        patience: Extension of learning after new best log prob has been found.
        population_batch_size: In case tensors get too large for GPU memory, use
            this parameter to evaluate the population samples in batches.

    Returns:
        Dict
    """
    num_groups = k
    if params is None:
        print("Make params")
        # Start from a uniform distribution over group memberships.
        params = torch.ones(num_nodes, k, requires_grad=False, device=DEV)
    print("Make utilities")
    fitness = make_utilities(population_size)

    if type(driver) == str:
        driver = {
            'degcor': degcor.driver,
            'overlapping': overlapping.driver,
            'bernoulli': bernoulli.driver,
            'poisson': poisson.driver
        }[driver]

    data_points = []

    target_scale = 1e-3
    fraction = 0.8
    ascend = calc_scale_decrease(num_iter, fraction, target_scale)
    all_losses = []
    best_loss = 1e9
    best_params = None
    best_eta = None
    best_assignment = None

    patience = int(num_iter * 0.1) if patience is None else patience
    iter = 0
    tmp_size = np.min([population_size, population_batch_size])
    t_start = time.perf_counter()
    print("Start optimizing")
    while iter < num_iter:
        print(f"------------------ Iteration {iter}")
        t_population = time.perf_counter()
        scale = float(np.clip(iter * ascend + 1, target_scale, 1.))
        # scale = 1.
        iter += 1
        all_noise = []

        batch_counter = 0
        batch_end = tmp_size
        all_log_probs = []
        while batch_end <= population_size:
            noise, mutations = mutate_params(params, tmp_size, scale)
            all_noise.append(noise)
            assignments = make_membership(mutations)

            log_probs, estimates = driver(assignments, num_groups, **constants)
            all_log_probs.append(log_probs)

            bl = torch.min(-1. * log_probs)
            if bl <= best_loss:
                idx = torch.argmax(log_probs)
                best_params = mutations[idx, :, :].cpu()
                best_eta = estimates['etas'][idx].cpu()
                best_assignment = assignments[idx].cpu()
                if iter + patience > num_iter and bl < best_loss:
                    num_iter = iter + patience
                best_loss = bl.cpu().detach().item()

            batch_end += population_batch_size
            batch_counter += 1

            del mutations
            del assignments
            torch.cuda.empty_cache()

        noise = torch.cat(all_noise)
        log_probs = torch.cat(all_log_probs)
        all_losses.append(log_probs.cpu().detach().numpy())
        avg_loss = -1. * torch.mean(log_probs).item()
        if iter % 10 == 0:
            print(iter, scale,  best_loss, avg_loss)

        scaled_noise = scale_noise(noise, fitness, log_probs)
        grads = calculate_gradient(scaled_noise)
        params = apply_gradient(params, grads, lr)
        data_points.append({
            'type': 'iteration',
            'value': time.perf_counter() - t_population,
            'num': iter,
            't_ges': time.perf_counter() - t_start,
            'memory': -1. if DEV == 'cpu' else float(torch.cuda.memory_allocated(DEV_NUM)),
            'scale': scale,
            'best_loss': float(best_loss),
            'avg_loss_population': float(avg_loss)
        })
        del all_noise
        del noise
        torch.cuda.empty_cache()

    return {
        "best_eta": best_eta.cpu().numpy(),
        "best_assignment": best_assignment.cpu().numpy(),
        "best_params": best_params.cpu().numpy(),
        "best_log_prob": -1. * best_loss,
        "params": params.cpu().numpy(),
        "data_points": data_points
    }


if __name__ == '__main__':
    edges = []
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            if i < 5 and j >= 5:
                continue
            if i >= 5 and j < 5:
                continue
            edges.append([i, j])
    edges.append([5, 4])
    edges.append([4, 5])
    g = nx.DiGraph()
    g.add_edges_from(edges)
    g = nx.karate_club_graph()
    vis.plot_soft_community_graph(g, torch.softmax(torch.randn(g.number_of_nodes(), 2), -1).numpy())
    edges = list(g.to_directed().edges())
    edges = [(u, v, 1) for u, v in edges]
    edges = torch.tensor(edges, requires_grad=False)
    in_degrees = torch.tensor([float(g.to_directed().in_degree[v]) for v in g.nodes()])
    out_degrees = torch.tensor([float(g.to_directed().out_degree[v]) for v in g.nodes()])
    constants = {
        'edges': edges,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees
    }
    constants = {
        'edges': edges
    }
    embeddings = learned_embeddings(
        constants=constants,
        population_size=10000,
        k=2,
        num_nodes=g.number_of_nodes(),
        driver=bernoulli.driver, #overlapping.driver,#degcor.driver,
        make_membership=sample_placements,#sample_propensities,#sample_placements,
        lr=1,
        num_iter=700
    )
    print(embeddings)
    print(torch.softmax(embeddings['best_params'], -1))
    print(torch.softmax(embeddings['params'], -1))

    members = torch.clamp(torch.softmax(embeddings['best_params'], -1), 0.001, 0.999).numpy()
    for i in range(members.shape[0]):
        for j in range(members.shape[1]):
            print('{:.4f}'.format(members[i, j]), end='\t')
        print()
    vis.plot_soft_community_graph(g, members)

    # colors = ['red' if members[i] == 1 else 'blue' for i in range(members.shape[0])]
    # pos = nx.spring_layout(g)
    # nx.draw_networkx_edges(g, pos=pos)
    # nx.draw_networkx_nodes(g, pos=pos, node_color=colors)
    # plt.savefig("000-graph-es.png")
    # plt.close()
