"""Stochastic block model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass, Categorical, Empirical
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

import networkx as nx
import matplotlib.pyplot as plt


def main(_):
  ed.set_seed(42)

  # DATA
  X_data, Z_true = karate("~/data")
  N = X_data.shape[0]  # number of vertices
  K = 2  # number of clusters

  # MODEL
  gamma = Dirichlet(concentration=tf.ones([K]))  # Parameter fuer Multinom. Distribution (Group Membership prior)
  Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K])) # BLOCKMATRIX
  Z = Multinomial(total_count=1.0, probs=gamma, sample_shape=N) # Zugehoerigkeitswsl . -> MATRIX (N X K)
  X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z)))) # Adjazenzmatrix

  # INFERENCE (EM algorithm)
  #qgamma = PointMass(params=tf.nn.softmax(tf.get_variable("qgamma/params", [K])))
  #qPi = PointMass(params=tf.nn.sigmoid(tf.get_variable("qPi/params", [K, K])))
  #qZ = PointMass(params=tf.nn.softmax(tf.get_variable("qZ/params", [N, K])))

  # Alternatively
  # This is the better realization as no "random normal" initialization is done above! This, however, is exactly what
  # MAP expects us to do as it does so itself when using a list.
  qgamma = PointMass(params=tf.nn.softmax(tf.get_variable("qgamma/params", initializer=tf.random_normal([K]))))
  qPi = PointMass(params=tf.nn.sigmoid(tf.get_variable("qPi/params", initializer=tf.random_normal([K, K]))))
  qZ = PointMass(params=tf.nn.softmax(tf.get_variable("qZ/params", initializer=tf.random_normal([N, K]))))

  # edward MAP is a Variational MAP Approach taking as arguments:
  # a) LATENT_VARS: A list or dictionary of Random Variables on which inference shall be performed on
  # Note that for a list, MAP automatically transforms each random variable into a Point Mass and initializes it using
  # standard normal draws. In case of a dictionary, MAP expects PointMass random variables as input.
  # b) Data Set
  inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: X_data})
  inference.initialize(n_iter=250)

  tf.global_variables_initializer().run()

  for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

  # CRITICISM

  # mean returns self.params => params val in PointMass construction
  # eval --> Evaluates the Matrix Expression
  # ARGMAX, gives the index of the maximum element for each row
  Z_pred = qZ.mean().eval().argmax(axis=1)
  print("Result (label flip can happen):")
  print("Predicted")
  print(Z_pred)
  print("True")
  print(Z_true)
  print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))

  # Graph Visualization
  G = nx.from_numpy_matrix(X_data)
  nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z_pred])
  plt.show()
  nx.draw(G, pos=nx.spring_layout(G), node_color=["red" if x==1 else "blue" for x in Z_true])
  plt.show()


if __name__ == "__main__":
  tf.app.run()