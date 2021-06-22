import sys
import vis
import degcor
import overlapping
import bernoulli
import sbm
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import time
from contextlib import contextmanager
import gc 
import utils_compare as utils


# Enable GPU Computation & Clear Cache
if torch.cuda.is_available():  
    DEV = "cuda:0" 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Device set to GPU.")
else:  
    DEV = "cpu" 
    print("Device set to CPU.")

gc.collect()
torch.cuda.empty_cache()


# Load Graph
dir = '/home/djuhera/graph/traces_graph.json'
g = utils.import_graph_json(dir)
g = nx.convert_node_labels_to_integers(g)
g = nx.DiGraph(g)

edges = list(g.edges())
edges = [(u, v, 1) for u, v in edges]
edges = torch.tensor(edges, requires_grad=False)

print("Done Importing Traces Graph.")