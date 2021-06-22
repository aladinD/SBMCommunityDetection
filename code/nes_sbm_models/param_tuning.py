import time
import os
import uuid

import json
import torch
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Any

import utils
import sbm

DEV = 'cpu'
DEV_NUM = 0


class Config(object):

    def __init__(self, population_size: int, num_groups: int, driver: str, lr: float,
                 num_iter: int, patience: int, population_batch_size: int):
        self.population_size = population_size
        self.num_groups = num_groups
        self.driver = driver
        self.lr = lr
        self.num_iter = num_iter
        self.patience = patience
        self.population_batch_size = population_batch_size
        self.best_log_prob = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "num_groups": self.num_groups,
            "driver": self.driver,
            "lr": self.lr,
            "num_iter": self.num_iter,
            "patience": self.patience,
            "population_batch_size": self.population_batch_size,
            "best_log_prob": self.best_log_prob
        }


BASE_PATH = '/opt/project/data/results/'
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)


def run_config(config: Config, g: nx.DiGraph) -> None:
    exp_id = uuid.uuid4().hex
    exp_folder = os.path.join(BASE_PATH, exp_id)
    os.mkdir(exp_folder)

    with open(os.path.join(exp_folder, 'config.json'), 'w') as fh:
        json.dump(config.to_dict(), fh)

    print(f"Integerize graph {g.number_of_nodes()}, {g.number_of_edges()}")
    node2idx, idx2node, g = utils.integerize_graph(g)
    print(f"Integerized graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    # edges = list(g.edges())
    print("Create constants")
    edges = [(u, v, 1) for u, v in g.edges()]
    edges = torch.tensor(edges, requires_grad=False)

    if config.driver == 'degcor':
        tmp = g.to_directed()
        in_degrees = torch.tensor([float(tmp.in_degree[v]) for v in g.nodes()])
        out_degrees = torch.tensor([float(tmp.out_degree[v]) for v in g.nodes()])
        constants = {
            'edges': edges,
            'in_degrees': in_degrees,
            'out_degrees': out_degrees
        }
    else:
        constants = {
            'edges': edges
        }

    sampler = sbm.sample_propensities if config.driver == 'overlapping' \
        else sbm.sample_placements

    print("Learn embeddings")
    embeddings = sbm.learned_embeddings(
        constants=constants,
        population_size=config.population_size,
        k=config.num_groups,
        num_nodes=g.number_of_nodes(),
        driver=config.driver,
        make_membership=sampler,
        lr=config.lr,
        num_iter=config.num_iter,
        population_batch_size=config.population_batch_size
    )

    index = [idx2node[i] for i in range(g.number_of_nodes())]
    df_path = os.path.join(exp_folder, 'model_out.h5')
    best_params = pd.DataFrame(embeddings['best_params'], index=index)
    final_params = pd.DataFrame(embeddings['params'], index=index)
    best_params.to_hdf(df_path, key='best_params')
    final_params.to_hdf(df_path, key='final_params')

    config.best_log_prob = embeddings['best_log_prob']
    with open(os.path.join(exp_folder, 'config.json'), 'w') as fh:
        json.dump(config.to_dict(), fh)
    with open(os.path.join(exp_folder, 'log_out.json'), 'w') as fh:
        json.dump(embeddings['data_points'], fh)


if __name__ == '__main__':
    # g = nx.karate_club_graph().to_directed()
    # node2uuid = {n: uuid.uuid4().hex for n in g.nodes()}
    # g2 = nx.DiGraph()
    # for u, v, d in g.edges(data=True):
    #     g2.add_edge(node2uuid[u], node2uuid[v], **d)
    print("Read Graph")
    g2 = utils.read_graph('/opt/project/data/traces_graph.json').to_directed()
    print("Simplify Graph")
    g2 = utils.simplify_graph(g2)
    configs = []
    for i in range(10):
        for x in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(
                Config(
                    population_size=10000,
                    num_groups=x,
                    driver='bernoulli',
                    lr=1,
                    num_iter=1000,
                    patience=100,
                    population_batch_size=5
                )
            )
    for i, config in enumerate(configs):
        print(f"Do config {i} of {len(configs)}")
        run_config(config, g2)
