"""
Stochastic Block Model (SBM) Realization for Zachary's Karate Club Example

The Karate Club consists of 34 members, where two explicit groups (Mr. Hi vs. Officer) are formed during a dispute:
    a) Members which support Mr. Hi
    b) Members which support the Officer

The respective data set contains two information sets:
    1) The general friendship connections between the members
    2) The association/grouping for Mr. Hi or the Officer

The data describes an undirected, simple graph for which a basic standard SBM is implemented. Notation follows Murphy.

Dictionary:
A = Adjacency Matrix
Eta = Stochastic Block Matrix
K = Number of Clusters (Groups)
N = Number of Vertices (Members)
pi = Group Membership Prior ~ Dirichlet Distribution
Z = Group Memberships ~ Categorical Distribution
"""

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

import networkx as nx
import matplotlib.pyplot as plt

# Seeding
ed.set_seed(42)

# Data
A_observed, Z_true = karate("~/data")
N = A_observed.shape[0]

# Model
K = 2
pi = Dirichlet(concentration=tf.ones([K])) #gamma
Eta = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K])) #Pi
Z = Multinomial(total_count=1.0, probs=pi, sample_shape=N)
A = Bernoulli(probs=tf.matmul(Z, tf.matmul(Eta, tf.transpose(Z)))) #X

# Inference (EM using MAP)

# 1) Define PointMass Random Variable realizations of the Latent Variables which are to be optimized during variational
# inference. Initialize these randomly using the normal distribution
q_pi = PointMass(params=tf.nn.softmax(tf.get_variable("q_Pi_PM", initializer=tf.random_normal([K]))))
q_Eta = PointMass(params=tf.nn.sigmoid(tf.get_variable("q_Eta_PM", initializer=tf.random_normal([K, K]))))
q_Z = PointMass(params=tf.nn.softmax(tf.get_variable("q_Z_PM", initializer=tf.random_normal([N, K]))))

# 2) Define the Inference approach: MAP
# Arguments: a) LATENT_VARS: A list or dictionary of Random Variables on which the inference shall be performed
#            b) Observed Data Set
inference = ed.MAP({pi: q_pi, Eta: q_Eta, Z: q_Z}, data={A: A_observed})

# 3) Initialization and Run
inference.initialize(n_iter=250)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

# Criticism
Z_pred = q_Z.mean().eval().argmax(axis=1)
print("Result (label flip,can happen):")
print("Predicted")
print(Z_pred)
print("True")
print(Z_true)
print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))

# Visualization
G = nx.from_numpy_matrix(A_observed)
nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z_pred])
plt.show()
nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z_true])
plt.show()