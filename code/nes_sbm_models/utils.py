import os

import pandas as pd
import torch
import json
import networkx as nx
from typing import List, Tuple, Any, Dict


def multiunsqueeze(t: torch.Tensor, dims: List[int]) -> torch.Tensor:
    """
    Recurseively unsqueeze the tensor along the dimensions in the list. Dimensions
    are unsqueezed from beginning to end of dims.
    """
    if len(dims) == 0:
        return t
    else:
        return multiunsqueeze(t.unsqueeze(dims[0]), dims[1:])


def read_graph(p: str) -> nx.MultiDiGraph:
    with open(p, "r") as fh:
        d = json.load(fh)
    return nx.readwrite.node_link_graph(d)


def simplify_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for u, v in graph.edges():
        g.add_edge(u, v)
    return g


def integerize_graph(graph: nx.Graph) -> Tuple[Dict[Any, int], Dict[int, Any], nx.Graph]:
    id2node = {i: n for i, n in enumerate(graph.nodes())}
    node2id = {n: i for i, n in enumerate(graph.nodes())}
    g_int = type(graph)()
    for u, v, d in graph.edges(data=True):
        g_int.add_edge(node2id[u], node2id[v], **d)
    return node2id, id2node, g_int


def get_overview_df(exp_folder: str) -> pd.DataFrame:
    """
    Construct a dataframe with the configurations, result and path of the trials.
    The column `trial_dir` returns the path to the folder in which the result
    files of the trial from the configuration of that row are stored. Results to
    a specific configuration can thus be obtained by selecting a configuration
    and then load the required results with the `trial_dir`.

    Args:
        exp_folder: Folder that contains the trials of a run (cryptic folder
            names in the form of a 128 bit uuid).

    Returns:
        dataframe with configuration, log prob and trial dir for each completed trial.
    """
    def load_best_log_prob():
        log = os.path.join(trial_dir, 'log_out.json')
        if os.path.exists(log):
            with open(log, 'r') as fh:
                log_prob = -1. * json.load(fh)[-1]['best_loss']
        else:
            log_prob = None
        return log_prob

    configs = []
    for f in os.listdir(exp_folder):
        trial_dir = os.path.join(exp_folder, f)
        if not os.path.isdir(trial_dir):
            continue
        config_path = os.path.join(trial_dir, 'config.json')
        if not os.path.exists(config_path):
            continue
        with open(config_path, 'r') as fh:
            config = json.load(fh)
        config['trial_dir'] = trial_dir
        if 'best_log_prob' not in config:
            config['best_log_prob'] = load_best_log_prob()
        elif config['best_log_prob'] is None:
            config['best_log_prob'] = load_best_log_prob()
        else:
            pass
        configs.append(config)
    return pd.DataFrame(configs)


def get_time_series(trial_dir: str) -> pd.DataFrame:
    """
    Load a time series that stores stats for each iteration of the ES algorithm.
    The points in the time series have the following attributes:
        - type: The type of time series, for now only `iteration` is available.
            Thus, the field can be ignored.
        - value: In case of iteration, its the time in seconds the iteration
            took. One iteration includes the mutation of parameters, evaluation
            of samples, calculating of gradients and updating of parameters.
        - t_ges: The total time this trial is executed, strictly monotonically
            increasing number.
        - memory: The memory that is still allocated at the *end* of the iteration.
        - scale: The scale (variance) at this iteration of the noise that is sampled.
        - best_loss: The best loss value that has been found so far. Monotonically
            decreasing value (lower is better).
        - avg_population_loss: The average loss over all samples of the population
             of the current iteration (lower is better).

    Args:
        trial_dir: Path to the folder in which the result files of a trial
            are stored.

    Returns:
        ts: List of dictionaries. Each dictionary in the list represents one
            time series.
    """
    with open(os.path.join(trial_dir, 'log_out.json')) as fh:
        ts = pd.DataFrame(json.load(fh))
    return ts


def get_best_params(trial_dir: str) -> pd.DataFrame:
    """
    Get the parameters associated with the best sample.

    Args:
        trial_dir: Path to the folder in which the result files of a trial
            are stored.

    Returns:
        params: DataFrame with best params. The index is the names of the nodes
            in the original graph.
    """
    params = pd.read_hdf(os.path.join(trial_dir, 'model_out.h5'), key='best_params')
    return params


def get_final_params(trial_dir: str) -> pd.DataFrame:
    """
    Get the final parameters of a trial. This are *not* necessarily the parameters
    with the best log-likelihood. Rather, its the parameters after the last
    update has been performed..

    Args:
        trial_dir: Path to the folder in which the result files of a trial
            are stored.

    Returns:
        params: DataFrame with best params. The index is the names of the nodes
            in the original graph.
    """
    params = pd.read_hdf(os.path.join(trial_dir, 'model_out.h5'), key='final_params')
    return params


def get_hard_assignment(params: pd.DataFrame) -> pd.Series:
    """
    Get the hard group assignment from the parameters.

    Args:
        params: Parameters in the form of a dataframe, for example the return
            value of `get_best_params` or `get_final_params`.

    Returns:
        assignment: Series that maps each node (original node labels from the
            graph) to specific group.
    """
    x = torch.softmax(torch.tensor(params.values), dim=-1)
    x = torch.argmax(x, dim=-1).numpy().flatten()
    assignment = pd.Series(x, index=params.index)
    return assignment


if __name__ == '__main__':
    df = get_overview_df('/opt/project/data/results')
    print(df)
    print(df.loc[0])
    print(df.columns)
    get_time_series(df.loc[0, 'trial_dir'])
    best_params = get_best_params(df.loc[0, 'trial_dir'])
    final_params = get_final_params(df.loc[0, 'trial_dir'])
    hard_assignment = get_hard_assignment(best_params)
    print(hard_assignment.value_counts().sort_index())

