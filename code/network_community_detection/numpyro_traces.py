import sys
import networkx as nx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import DiscreteHMCGibbs, MCMC, NUTS
import time
sys.path.append('/home/djuhera/notebooks')
import utils


# Load Graph
dir = '/home/djuhera/graph/traces_graph.json'
g = utils.import_graph_json(dir)
g = nx.convert_node_labels_to_integers(g)
g = nx.DiGraph(g)

A = jnp.array(nx.to_numpy_matrix(g))
K = 3

print("Graph imported")


# NumPyro MCMC Model
def model(A, K):
    # Network Nodes
    n = A.shape[0]

    # Block Matrix Eta
    with numpyro.plate("eta1", K):
        with numpyro.plate("eta2", K):
            eta = numpyro.sample("eta", dist.Beta(1., 1.))

    # Group Memberships (Using a Multinomial Model)
    membership_probs = numpyro.sample("z_probs", dist.Dirichlet(concentration=jnp.ones(K)))
    with numpyro.plate("membership", n):
       sampled_memberships = numpyro.sample("sampled_z", dist.Categorical(probs=membership_probs))
       sampled_memberships = jax.nn.one_hot(sampled_memberships, K)

    # Adjacency Matrix
    p = jnp.matmul(jnp.matmul(sampled_memberships, eta), sampled_memberships.T)
    with numpyro.plate("rows", n):
      with numpyro.plate("cols", n):
            A_hat = numpyro.sample("A_hat", dist.Bernoulli(p), obs=A)


# Runtime Comparison: General & Iterative
def get_mean_runtime(num_iters):
    durs = []
    for i in range(num_iters):
        kernel = DiscreteHMCGibbs(NUTS(model))
        mcmc = MCMC(kernel, num_warmup=0, num_samples=1)
        start = time.time()
        mcmc.run(jax.random.PRNGKey(0), A, K)
        end = time.time()
        duration = end - start
        durs.append(duration)

    mean_dur = sum(durs)/num_iters
    
    return mean_dur, durs

mean_dur, durs = get_mean_runtime(1)
print("\nMean Duration: ", mean_dur)
print("Durs", durs)
print("\nMean Iters/s: ", 447)