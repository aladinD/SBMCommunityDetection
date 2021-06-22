# Community Detection in Large-Scale CommunicationNetworks using Stochastic Block Modeling

## Motivation
In community detection, the main goal is to identify distinct groups, communities or clusters within a specified distributed network. Examples for such networks can be of different nature and thus community detection methods have been introduced in various fields of research. Yet, the core detection goal remains the same which is to most accurately infer clustering structures that might allow for a further network analysis, identification of group relationships and other tasks specific to the respective field of application. This research internship investigates several Stochastic Block Model (SBM) approaches for community identification in large-scale communication networks. Implementation is done using MCMC in NumPyro and Natural Evolution Strategies (NES).

## Repo Structure
- **literature:** contains relevant theoretical material 
- **data:** contains the generated MultiDigraph from the network traces
- **code:** contains relevant scripts and notebooks
    * **data_analysis:** detailed and general analysis of the network traces (.ipynb)
    * **data_processing:** pcap conversion, graph generation, host name resolve (ipynb, .py)
    * **model_comparison:** MCMC and NES model comparison for several benchmark data (.ipynb)
    * **model_evaluation:** analysis and further investigation of the obtained results using NES (.ipynb)
    * **network_community_detection:** MCMC and NES approaches to community detection for the generated network graph (.py)
    * **nes_sbm_models:** contains different variants of SBM models implemented using NES
    * **SBM:** edward1, NumPyro and NES realizations of the SBM
