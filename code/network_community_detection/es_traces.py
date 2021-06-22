import sys
sys.path.append('/home/djuhera/es-sbm-master')
import vis
import degcor
import overlapping
import bernoulli
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import time
from contextlib import contextmanager
import utils_compare as utils

DEV = 'cpu'

# Suppress Console Output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            

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
    return torch.tensor(utility_values)


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
    noise = torch.randn([num_samples, params.shape[0], params.shape[1]], requires_grad=False)
    mutated_params = torch.unsqueeze(params, 0) + noise
    return noise, mutated_params


def calc_scale_decrease(num_iter: int, fraction: float, target_scale: float) -> float:
    stop = num_iter * fraction
    m = (target_scale - 1.) / stop
    return m


def learned_embeddings(constants: Dict[str, torch.Tensor], population_size: int, k: int,
                       num_nodes: int, driver: callable, make_membership: callable,
                       lr: float, num_iter=10000, params: torch.Tensor=None,
                       patience: int=None, population_batch_size=200000000) -> Dict[str, torch.Tensor]:
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
    if params is None:
        # Start from a uniform distribution over group memberships.
        params = torch.ones(num_nodes, k, requires_grad=False, device=DEV)
    fitness = make_utilities(population_size)

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
    while iter < num_iter:
        scale = float(np.clip(iter * ascend + 1, target_scale, 1.))
        # scale = 1.
        iter += 1
        noise, mutations = mutate_params(params, population_size, scale)
        assignments = make_membership(mutations)

        batch_start = 0
        batch_end = np.min([population_size, population_batch_size])
        all_log_probs = []
        all_estimates = []
        while batch_end <= population_size:
            log_probs, estimates = driver(assignments, k, **constants)# driver(assignments[batch_start:batch_end], k, **constants)
            all_log_probs.append(log_probs)
            all_estimates.append(estimates)  # TODO: tensors in estimates must be concatenated as well, leave this for you aladin
            batch_start = batch_end
            batch_end += population_batch_size
            #print("LOOP")
        log_probs = torch.cat(all_log_probs)
        #print("OUT OF LOOP")

        # TODO Estimates
        combined = {}
        for d in all_estimates:
            for k in d.keys():
                if k not in combined:
                    combined[k] = []
                combined[k].append(d[k])
        combined = {k: torch.cat(v) for k, v in combined.items()}
        estimates = combined
        #print(all_estimates)
        #print(estimates)

        all_losses.append(log_probs.cpu().detach().numpy())
        bl = torch.min(-1. * log_probs)
        if bl <= best_loss:
            best_loss = bl.cpu().detach()
            idx = torch.argmax(log_probs)
            best_params = mutations[idx, :, :]
            best_eta = estimates['etas'][idx]
            best_assignment = assignments[idx]
            if iter + patience > num_iter and bl < best_loss:
                num_iter = iter + patience
        if iter % 10 == 0:
            print(iter, scale,  best_loss, torch.mean(log_probs))

        scaled_noise = scale_noise(noise, fitness, log_probs)
        grads = calculate_gradient(scaled_noise)
        params = apply_gradient(params, grads, lr)

    return {
        "best_eta": best_eta.cpu(),
        "best_assignment": best_assignment.cpu(),
        "best_params": best_params.cpu(),
        "best_log_prob": -1. * bl,
        "params": params.cpu()
    }

# Load Graph
dir = '/home/djuhera/graph/traces_graph.json'
g = utils.import_graph_json(dir)
g = nx.convert_node_labels_to_integers(g)
g = nx.DiGraph(g)

edges = list(g.edges())
edges = [(u, v, 1) for u, v in edges]
edges = torch.tensor(edges, requires_grad=False)

print("Done Importing Traces Graph.")

num_edges =  g.number_of_edges()
num_nodes = g.number_of_nodes()

print("Directed: ", g.is_directed())
print("Nodes: ", num_nodes)
print("Edges: ", num_edges)

constants = {
    'edges': edges
}

start = time.time()
embeddings = learned_embeddings(
    constants=constants,
    population_size=10000,
    k=2,
    num_nodes=g.number_of_nodes(),
    #driver=overlapping.driver,#degcor.driver,
    #make_membership=sample_propensities,#sample_placements,
    driver=bernoulli.driver, #overlapping.driver,#degcor.driver,
    make_membership=sample_placements, #sample_propensities,#sample_placements,
    lr=1,
    num_iter=1
)
end = time.time()

duration = end - start
print("Duration:", duration)