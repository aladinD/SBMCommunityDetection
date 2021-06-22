import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import DiscreteHMCGibbs, MCMC, NUTS
import networkx as nx   # Version 1.9.1
from observations import karate
import matplotlib.pyplot as plt


def model(A, K):
    # Network Nodes
    n = A.shape[0]

    # Block Matrix Eta
    with numpyro.plate("eta1", K):
        with numpyro.plate("eta2", K):
            # NB: using c0=c1=1. for simplicity, because it is unnecessary
            # to define "locally" hyperpriors
            # c0 = numpyro.sample("c0", dist.HalfNormal(1.))
            # c1 = numpyro.sample("c1", dist.HalfNormal(1.))
            eta = numpyro.sample("eta", dist.Beta(1., 1.))

            # conc_param = jnp.array([])
            # for i in range(K):
            #     conc_param = jnp.append(conc_param, 1.)
            # eta = numpyro.sample("eta", dist.Beta(conc_param, conc_param))

    # Group Memberships (Using a Multinomial Model)
    # TODO: address label-switching problem (e.g. by fixing the membership of the first entry?
    # or using more informative prior?)
    # FIXME: should we move z_probs outside of the plate? it seems to be a global variable,
    # rather than a local variable
    membership_probs = numpyro.sample("z_probs", dist.Dirichlet(concentration=jnp.ones(K)))
    with numpyro.plate("membership", n):
        sampled_memberships = numpyro.sample("sampled_z", dist.Categorical(probs=membership_probs))
        sampled_memberships = jax.nn.one_hot(sampled_memberships, K)

    # Adjacency Matrix
    p = jnp.matmul(jnp.matmul(sampled_memberships, eta), sampled_memberships.T)
    with numpyro.plate("rows", n):
        with numpyro.plate("cols", n):
            A_hat = numpyro.sample("A_hat", dist.Bernoulli(p), obs=A)

# Simple Toy Data
A = jnp.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])
Z = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
K = 2

# Karate Club Data
A, Z = karate("~~data")
A = A.astype(jnp.float32)
K = 2

# American Football Network
# G = nx.read_gml("/Users/aladindjuhera/aladin_fp/code/Pyro/SBM/scripts/football/football.gml")
# A = nx.to_numpy_matrix(G)
# A = A.astype(jnp.float32)
# K = 12

# Inference
kernel = DiscreteHMCGibbs(NUTS(model))
mcmc = MCMC(kernel, num_warmup=1500, num_samples=2500)
mcmc.run(jax.random.PRNGKey(0), A, K)
mcmc.print_summary()
Z_infer = mcmc.get_samples()["sampled_z"]
Z_infer = Z_infer[-1]
print("Inference Result", Z_infer)
print("Original Memberships", Z)


# Visualization
G = nx.from_numpy_matrix(A)
plt.figure(figsize=(10,4))
plt.subplot(121)
nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z])
plt.title("Original Network")

plt.subplot(122)
nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z_infer])
plt.title("Inferred Network")
plt.show()


# Label Switching Investigation
# n_iters = 20
# for i in range(n_iters):
#     kernel = DiscreteHMCGibbs(NUTS(model))
#     seed = jax.random.PRNGKey(i)
#     mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
#     mcmc.run(seed, A)
#     # mcmc.print_summary()
#
#     final_label = mcmc.get_samples()["sampled_z"]
#     final_label = final_label[-1]
#     print("For seed {} labels are {}".format(seed, final_label))