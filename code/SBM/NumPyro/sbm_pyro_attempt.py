# -------------------------------------------------------------------------------------------------------------------- #
# The script follows the Gaussian Mixture Model example. First, global variables (Stochastic Block Matrix Eta) are     #
# learned. Second, the globals are used to learn/infer the locals (Group Membership Associations z_probs).             #
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex
import networkx as nx
import matplotlib.pyplot as plt


# Pyro Environment Settings
pyro.enable_validation(True)
pyro.set_rng_seed(5)
pyro.clear_param_store()


# 1) Learning Eta

# InitLocFn
class InitLocFn(object):
    def __init__(self, adj: torch.Tensor, k: int, n: int):
        self.adj = adj
        self.k = k
        self.n = n
        self.prior = None
        self.memberships = None
        self.memberships_probs = None

    def _make_members(self):
        self.memberships_probs = dist.Dirichlet(concentration=torch.ones(self.k)).sample([self.n])
        self.memberships = torch.argmax(self.memberships_probs, dim=-1)

    def _init_membership_probs(self):
        init = self.memberships_probs
        return init

    def _init_eta(self):
        self._make_members()
        nom = torch.zeros([self.k, self.k], dtype=torch.float)
        denom = torch.zeros([self.k, self.k], dtype=torch.float)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                z_i = self.memberships[i]
                z_j = self.memberships[j]
                nom[z_i, z_j] += self.adj[i, j]
                nom[z_j, z_i] += self.adj[j, i]
                denom[z_i, z_j] += 1.
                denom[z_j, z_i] += 1.
        init = nom / torch.clip(denom, 1., 1e9)
        print("eta init", init)
        return init

    def _init_c0(self):
        init = torch.ones(2, 2)
        return init

    def _init_c1(self):
        init = torch.ones(2, 2)
        return init

    def __call__(self, site):
        print(site['name'])
        return {
            'z_probs': self._init_membership_probs,
            'eta': self._init_eta,
            'c0': self._init_c0,
            'c1': self._init_c1
        }[site['name']]()


# Model
def model5_auto(A):
    print("\nModel5\n=========")

    # Network Nodes
    n = A.shape[0]

    # Block Matrix Eta
    with pyro.plate("eta1", 2):
        with pyro.plate("eta2", 2):
            c0 = torch.clip(pyro.sample("c0", dist.Normal(loc=torch.tensor([[0., 0.], [0., 0.]]), scale=torch.tensor([1.]))), 0.01, 50)
            c1 = torch.clip(pyro.sample("c1", dist.Normal(loc=torch.tensor([[0., 0.], [0., 0.]]), scale=torch.tensor([1.]))), 0.01, 50)
            d = dist.Beta(concentration0=c0, concentration1=c1)
            eta = pyro.sample("eta", d)

    # Group Memberships (Using a Multinomial Model)
    memberships = torch.ones([n, 2])
    d = dist.Dirichlet(concentration=memberships)
    with pyro.plate("membership", n):
        membership_probs = pyro.sample("z_probs", d)
        d = dist.Multinomial(probs=membership_probs)
        # d = dist.RelaxedOneHotCategoricalStraightThrough(temperature=torch.tensor(0.6), logits=membership_probs)
        sampled_memberships = pyro.sample("sampled_z", d)

    # Adjacency Matrix
    p = torch.matmul(torch.matmul(sampled_memberships, eta), sampled_memberships.T)
    with pyro.plate("rows", n):
        with pyro.plate("cols", n):
            A_hat = pyro.sample("A_hat", dist.Bernoulli(p), obs=A)


# AutoGuide + Learning Process
def _learn_autoguide(A: torch.Tensor):
    # Initializer
    def initialize(seed, init_loc_fn):
        global global_guide, svi
        print("-------------------------------------> INITIALIZE ", seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        global_guide = AutoDelta(poutine.block(model5_auto, expose=['eta']), init_loc_fn=init_loc_fn)
        svi = SVI(model5_auto, global_guide, optim, loss=elbo)
        return svi.loss(model5_auto, global_guide, A)

    # Initializing Process
    init_fn = InitLocFn(A, 2, A.shape[0])
    elbo = pyro.infer.TraceGraph_ELBO(num_particles=25)
    optim = pyro.optim.SGD({"lr": 0.001, "momentum": 0.1})
    loss, seed = min((initialize(seed, init_fn), seed) for seed in range(100))
    initialize(seed, init_loc_fn=init_fn)

    # Learning Process
    print("\nStart Optimizing\n================")
    losses = []
    for i in range(1000):
        print(f"------------------------------------------------------Step {i}")
        loss = svi.step(A)
        losses.append(loss)

    # Learning Results
    plt.plot(losses)
    plt.savefig("0000-loss-auto.png")
    plt.close()

    map_estimates = global_guide(A)
    eta = map_estimates['eta']
    print('eta = {}'.format(eta.data.numpy()))


# 2) Learning/Infering z_probs

# Model
def model5(A):
    print("\nModel5\n=========")

    # Network Nodes
    n = A.shape[0]

    # Good Guess for Eta Concentrations
    # Note: If Eta was learned correctly in Step 1), we do not need to specify c0 and c1 here and would
    # simply use the learned eta value to infer the group memberships. However, eta is still not learned properly. This
    # is a toy example here.
    c0 = torch.tensor([[1., 10.], [10., 1.]])
    c1 = torch.tensor([[10., 1.], [1., 10.]])

    with pyro.plate("eta1", 2):
        with pyro.plate("eta2", 2):
            d = dist.Beta(concentration0=c0, concentration1=c1)
            eta = pyro.sample("eta", d)

    # Group Memberships
    memberships = torch.ones([n, 2])
    d = dist.Dirichlet(concentration=memberships)
    with pyro.plate("membership", n):
        membership_probs = pyro.sample("z_probs", d)
        d = dist.Multinomial(probs=membership_probs)
        # d = dist.RelaxedOneHotCategoricalStraightThrough(temperature=torch.tensor(0.0001), logits=membership_probs)
        sampled_memberships = pyro.sample("sampled_z", d)

    # Adjacency Matrix
    p = torch.matmul(torch.matmul(sampled_memberships, eta), sampled_memberships.T)
    with pyro.plate("rows", n):
        with pyro.plate("cols", n):
            A_hat = pyro.sample("A_hat", dist.Bernoulli(p), obs=A)


# Guide
def guide5(A):
    print("\nGuide5\n=========")

    # Network Nodes
    n = A.shape[0]

    # Block Matrix Eta
    c0 = pyro.param("c0", torch.tensor([[1., 2.], [2., 1.]]))
    c1 = pyro.param("c1", torch.tensor([[2., 1.], [1., 2.]]))
    d = dist.Beta(concentration0=c0, concentration1=c1)
    print("Model: Beta: ", d.event_shape, d.batch_shape)
    with pyro.plate("eta1", 2):
        with pyro.plate("eta2", 2):
            eta = pyro.sample("eta", d)

    # Group Memberships
    ps_z = torch.randn(n, 2)  # n x 1
    with pyro.plate("membership", n):  # n -> n - 1
        print("Guide: ps_z.shape = ", ps_z.shape)
        memberships = pyro.param("param_z", ps_z)  # n x 1
        # memberships = torch.cat([memberships, torch.zeros_like(memberships], -1)
        print("Guide: made prior memberships")
        sampled_memberships = pyro.sample("sampled_z", dist.Multinomial(logits=memberships))
        # sampled_memberships = torch.cat([sampled_memberships, torch.zeros((1,) +  sampled_memberships.shape[1:]))])
        # 1 0
        # 1 0
        # 0 1
        # 0 1
        # 0 0 0
        # memberships = torch.cat([memberships, torch.ones_like(memberships) - memberships], -1)
        # 1 0  1-1  0
        # 1 0  1-1  0
        # 0 0  1-0  1
        # 0 0  1-0  1

        # d = dist.RelaxedOneHotCategoricalStraightThrough(temperature=torch.tensor(0.0001), logits=memberships)
        sampled_memberships = pyro.sample("sampled_z", d)
    print("Guide: sampled_memberships.shape = ", sampled_memberships.shape)

    # Adjacency Matrix Bernoulli Probability
    # p = torch.matmul(torch.matmul(sampled_memberships, eta), sampled_memberships.T)
    # print("Guide: p.shape = ", p.shape)


# Inference Process
def _learn_membership(A: torch.Tensor):
    svi = pyro.infer.SVI(model=model5,
                         guide=guide5,
                         optim=pyro.optim.SGD({"lr": 0.001, "momentum": 0.1}),
                         loss=pyro.infer.TraceGraph_ELBO(num_particles=25))
    losses = []
    num_steps = 1500
    for t in range(num_steps):
        print(f"------------------------------------------------------Step {t}")
        losses.append(svi.step(A))

    plt.plot(losses)
    plt.savefig("0000-loss-simple.png")
    plt.close()

    # Examining Results
    print(torch.softmax(pyro.param("param_z"), dim=-1))
    print(pyro.param("c0"))
    print(pyro.param("c1"))

    # Plot
    g = nx.from_numpy_array(A.numpy())
    members = torch.argmax(torch.softmax(pyro.param("param_z"), -1), -1)
    colors = ['red' if members[i] == 1 else 'blue' for i in range(members.shape[0])]
    pos = nx.spring_layout(g)
    nx.draw_networkx_edges(g, pos=pos)
    nx.draw_networkx_nodes(g, pos=pos, node_color=colors)
    plt.savefig("000-graph-simple.png")
    plt.close()


# Main
if __name__ == '__main__':
    # Data Set
    A = torch.tensor(np.array([
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    ], dtype=np.float64))

    g = nx.karate_club_graph()
    A_k = torch.tensor(nx.to_numpy_matrix(g), dtype=torch.double)

    # Specify Learning Process (Eta-Learning with Autoguide or Membership-Learning with SVI)
    # _learn_autoguide(A)
    _learn_membership(A)
