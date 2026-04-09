"""
Docstring for embedding_benchmark.py

    Compared to the other embedding script in this project (embedding_bm) 
    which is focused on benhmark of random graphs with controlled density -
    THIS SCRIPT is  focused on embedding benchmark over a pre-existing graph datasets.


About 'pre-existing graph dataset':
    this term refers to a system path that points you
    to a root directory with one or more graph files.
    these files represent graphs under the format implied by Gset/MQLib format/QUBO [See format rules]

    The directory with source graphs, can refer to:
    1. MQLib dataset (5300+) graphs (which is the default dataset)
    2. Gset dataset (~80) graphs
    3. ANY group of imported/generated graphs that had been 
       transformed and written to disk memory under the agreed format.
    
    Note: CLI and defaults:
        This script is opened to be run as a CLI tool, though it has a complete set
        of default values and can be run merely via [python_venv_path] -m [this_file_name].
        The default arguments are defined and initialized at the head of this file : see 'script defaults' section.
        All defaults can be overriden by CLI arguments.

    Note:
    This file exposes the core logic of the benchmark process, however, to actually execute it
    one must have the entire project & virtual environment installed and set up.
"""

# external & dwave packages
import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import networkx as nx
import dwave_networkx as dnx
import statistics as stats
import argparse
from typing import  Tuple, List, Any, Dict,Iterator
from minorminer import find_embedding
from dwave_networkx.generators import (
    chimera_graph,
    pegasus_graph,
    zephyr_graph,   
)


# internal packages:
from max_cut.qubo_io import QUBO # object for read/write/reduce graphs and quadratics
from max_cut import internal_paths 
from max_cut.graph_io import GraphShelf # object for managing graph instances and metrics
from max_cut.heuristic_presets import (
    find_embedding_default_params,
    find_embedding_quick_preset,
    find_embedding_balanced_preset,
    find_embedding_robust_preset,
    find_embedding_high_beta_preset,
    find_embedding_deep_search_preset
) # plug & play heuristics parameters


#----------------script defaults:
# target-graph defaults:
DEFAULT_ARCHITECTURE: str = "chimera"

CHIMERA_ROW_CELLS: int = 25
CHIMERA_COL_CELLS: int = 25
CHIMERA_TILE_SIZE: int = 4

PEGASUS_M: int = 16
PEGASUS_OFFSETS_INDEX: int = 0
PEGASUS_FABRIC_ONLY: bool = True

ZEPHYR_M: int = 12
ZEPHYR_T: int = 4

# source-graphs defaults:
"""
we allow two broad modes of input: [SOURCE_MODE::generator/dataset]

1. generator:
    [GENERATOR_TYPE, GENERATOR_NUM_GRAPHS]
    internal graph generator (currently not active)

2. dataset:
    [DATASET_ROOT]
    user provides path to directory with input graphs (graph.txt - mqlib/gset format).
    in addition, provides a reading mode:
    [DATASET_ROOT::random/sorted]

    2.1 'DATASET_READ_METHOD' = random
        reading 'DATASET_NUM_GRAPHS' from  'DATASET_ROOT'

    2.2 'DATASET_READ_METHOD' = sorted
        reading 'DATASET_NUM_GRAPHS' from  'DATASET_ROOT'
        sorted by the chosen available metric 'DATASET_SORT_METRIC'
        currently available: Dimensionality (node count)

"""
SOURCE_MODE: str = "dataset"   # or "generator" later

DATASET_ROOT: str  = internal_paths.MQLIB_INSTANCES_DIR      # mdefaults to mqlib
DATASET_READ_METHOD: str = "random"  # "random" | "sorted"
DATASET_NUM_GRAPHS: int = 10         # how many graphs to load

DATASET_SORT_METRIC: str | None = None # TODO: Implement reading method and complete options

GENERATOR_TYPE: str = "random" # currently not available
GENERATOR_NUM_GRAPHS: int = 10 # currently not available

# embedding defaults:
EMBEDDING_PER_GRAPH:int = 5
DEFAULT_EMBEDDING_PRESET:str = "default" 

#--------------------arg parse:
def add_architecture_script_arguments(parser:argparse.ArgumentParser):

    parser.add_argument(
    '--architecture',
    type=str,
    choices=['chimera', 'pegasus', 'zephyr'],
    default=DEFAULT_ARCHITECTURE,
    help='Target D-Wave architecture',
    metavar="TARGET_LAYOUR"
)

    #arch-params-chimera:
    parser.add_argument(
        '--row_cells',
        type=int,
        default=CHIMERA_ROW_CELLS,
        metavar='CHIMERA_ROW_SIZE',
        help='Number of tile rows in Chimera graph (default: 25)'
    )

    parser.add_argument(
        '--col_cells',
        type=int,
        default=CHIMERA_COL_CELLS,
        metavar='CHIMERA_COL_SIZE',
        help='Number of tile columns in Chimera graph (default: 25)'
    )

    parser.add_argument(
        '--tile_size',
        type=int,
        default=CHIMERA_TILE_SIZE,
        metavar='CHIMERA_TILE_SIZE',
        help='Tile size for Chimera architecture (default: 4)'
    )

    #arch-params-pegasus:
    parser.add_argument(
        '--pegasus_m',
        type=int,
        default=PEGASUS_M,
        metavar='PEGASUS_M',
        help='Pegasus size parameter m (default: 16 → P16)'
    )

    parser.add_argument(
        '--pegasus_offsets_index',
        type=int,
        default=PEGASUS_OFFSETS_INDEX,
        metavar='PEGASUS_OFFSETS_INDEX',
        help='Pegasus offsets index (default: 0)'
    )

    parser.add_argument(
        '--no_pegasus_fabric_only',
        dest='pegasus_fabric_only',
        action='store_false',
        default=PEGASUS_FABRIC_ONLY,
        help='Disable fabric-only Pegasus graph'
    )

    #arch-params-zephyr:
    parser.add_argument(
        '--zephyr_m',
        type=int,
        default=ZEPHYR_M,
        metavar='ZEPHYR_M',
        help='Zephyr size parameter m (default: 12)'
    )

    parser.add_argument(
        '--zephyr_t',
        type=int,
        default=ZEPHYR_T,
        metavar='ZEPHYR_T',
        help='Zephyr tile parameter t (default: 4)'
    )

    return parser

def add_input_mode_script_arguments(parser:argparse.ArgumentParser):

    # source graphs: input mode
    parser.add_argument(
        '--source_mode',
        type=str,
        choices=['dataset', 'generator'],
        default=SOURCE_MODE,
        help='Source of input graphs',
        metavar="SOURCE_MODE"
    )

    # dataset mode arguments
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=DATASET_ROOT,
        metavar='DATASET_DIR',
        help='Root directory containing input graphs (MQLib/GSet format)'
    )

    parser.add_argument(
        '--dataset_read_method',
        type=str,
        choices=['random', 'sorted'],
        default=DATASET_READ_METHOD,
        help='Method for selecting graphs from dataset',
        metavar="READING_METHOD"
    )

    parser.add_argument(
        '--dataset_num_graphs',
        type=int,
        default=DATASET_NUM_GRAPHS,
        metavar='N',
        help='Number of graphs to read from dataset'
    )

    parser.add_argument(
        '--dataset_sort_metric',
        type=str,
        default=DATASET_SORT_METRIC,
        metavar='METRIC',
        help='Metric name used to sort graphs (only for sorted read method)'
    )

    #  generator mode arguments (placeholder)
    parser.add_argument(
        '--generator_type',
        type=str,
        default=GENERATOR_TYPE,
        metavar='GEN_TYPE',
        help='Internal graph generator type (currently inactive)'
    )

    parser.add_argument(
        '--generator_num_graphs',
        type=int,
        default=GENERATOR_NUM_GRAPHS,
        metavar='N',
        help='Number of graphs to generate (currently inactive)'
    )

def add_embedding_script_arguments(parser:argparse.ArgumentParser):
    parser.add_argument(
        '--embedding_per_graph',
        type=int,
        default=EMBEDDING_PER_GRAPH,
        help="Number of embedding to perform on each instance (graph)",
        metavar="EMBD_COUNT"
    )

    parser.add_argument(
        '--embedding_preset',
        type=str,
        default=DEFAULT_EMBEDDING_PRESET,
        help="find_embedding pre-defined arguments set",
        metavar="PRESET",
        choices=['default', 'quick', 'balanced', 'robust', 'high_beta', 'deep_search']
        
    )

def get_script_arguments() -> argparse.ArgumentParser:
    """
    Returns argument parser loaded with all pre-defined arguments for this script.
    Uses module-level defaults for any user-omited instruction (see defaults in uppder section) 
    :rtype: ArgumentParser
    """
    program = "Graph Embedding Benchmark"

    description = (
        "Benchmark and analyze minor-embedding behavior for combinatorial problem graphs.\n\n"
        "This script orchestrates:\n"
        "selection of source graphs (datasets or generators)\n"
        "construction of target hardware graphs (Chimera / Pegasus / Zephyr)\n"
        "configuration of minor-embedding heuristics via predefined presets\n\n"
        "All parameters are optional — omitting an argument activates its canonical default."
        )

    epilog = (
        "Design philosophy:\n"
        "Defaults encode expert-recommended behavior.\n"
        "Command-line arguments selectively override these defaults.\n\n"
        "Typical workflow:\n"
        "1. Choose a source graph stream\n"
        "2. Choose a target hardware architecture\n"
        "3. Select an embedding heuristic preset\n"
        "4. Run repeated embeddings for statistical insight\n\n"
        "This tool is intended for experimentation, benchmarking, and research — "
        "not for one-off embedding runs."
    )

    parser = argparse.ArgumentParser(prog=program, description=description, epilog=epilog)
    add_architecture_script_arguments(parser)
    add_input_mode_script_arguments(parser)
    add_embedding_script_arguments(parser)
    # add more argument families here if needed

    return parser



#-----------target architecture construction (ideal physical annealers):
def get_chimera_graph(
    rows: int = 25,
    cols: int = 25,
    tile_size: int = 4,
) -> Tuple[nx.Graph, int]:
    """
    Default corresponds to ~5,000-qubit Chimera (25x25, tile size 4).
    """
    G = chimera_graph(rows, cols, tile_size)

    if not G.nodes or not G.edges:
        raise RuntimeError("Chimera target graph is empty")

    return G, len(G.nodes)

def get_pegasus_graph(
    m: int = 16,
    offsets_index: int = 0,
    fabric_only: bool = True,
) -> Tuple[nx.Graph, int]:
    """
    Defaults correspond to D-Wave Advantage (Pegasus P16).
    """
    G = pegasus_graph(
        m=m,
        offsets_index=offsets_index,
        fabric_only=fabric_only,
    )

    if not G.nodes or not G.edges:
        raise RuntimeError("Pegasus target graph is empty")

    return G, len(G.nodes)

def get_zephyr_graph(
    m: int = 12,
    t: int = 4,
) -> Tuple[nx.Graph, int]:
    """
    Defaults produce ~4.4K qubits with max degree ~20
    (Advantage2-scale Zephyr).
    """
    G = zephyr_graph(m=m, t=t)

    if not G.nodes or not G.edges:
        raise RuntimeError("Zephyr target graph is empty")

    return G, len(G.nodes)


#-----------argument-resolvers:
def resolve_target_graph(args:argparse.Namespace)->nx.Graph:
    # obtaining chosen target graph:
    if args.architecture == 'chimera':
        return get_chimera_graph(
            rows=args.row_cells,
            cols=args.col_cells,
            tile_size=args.tile_size
        )

    elif args.architecture == 'pegasus':
        return get_pegasus_graph(
            m=args.pegasus_m,
            offsets_index=args.pegasus_offsets_index,
            fabric_only=args.pegasus_fabric_only
        )

    elif args.architecture == 'zephyr':
        return get_zephyr_graph(
            m=args.zephyr_m,
            t=args.zephyr_t
        )
    else:
        raise ValueError(f"Ilegal architecure input: {args.architecture}")

def resolve_source_graphs(args: argparse.Namespace)->Iterator[nx.Graph]:
    """
    Resolve and return an iterator over source graphs according to CLI arguments.

    This function performs:
    - semantic validation of input arguments
    - dispatch to pre-existing graph iterators

    Returns
    -------
    Iterator[...]
        An iterator over source graph objects (as defined by existing code).
    """

    # dataset mode:
    if args.source_mode == 'dataset':

        if args.dataset_root is None:
            m="DATASET_ROOT must be provided when source_mode='dataset'"
            raise ValueError(m)

        if args.dataset_read_method == 'random': # return dataset::read n randoms
            return GraphShelf.random_sample(dataset_root=args.dataset_root, sample_size=args.dataset_num_graphs)

        elif args.dataset_read_method == 'sorted':  # dataset::read n sorted by [ metric ]

            if args.dataset_sort_metric is None:
                m="DATASET_SORT_METRIC must be provided when dataset_read_method='sorted' "
                raise ValueError(m)
            
            root=args.dataset_root,
            num_graphs=args.dataset_num_graphs,
            sort_metric=args.dataset_sort_metric
            return GraphShelf.read_sorted_by_metric(dataset_root=root, n=num_graphs,by_metric=sort_metric)

        else:
            m=f"Unknown dataset_read_method: {args.dataset_read_method}"
            raise ValueError(m)

    elif args.source_mode == 'generator': # generator mode: (not implemented)

        # TODO: fill this section when generators are supported args.dataset_root

        raise NotImplementedError(
            "Source mode 'generator' is not implemented yet"
        )

    else:# unknown source mode 
        raise ValueError(
            f"Unknown source_mode: {args.source_mode}"
        )
    
def resolve_embedding_preset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    returns copy of preset for find_embedding() heuristic
    :rtype: Dict[str, Any]
    """

    # mapping: arg-name -> preset
    preset_registry = {
        "default": find_embedding_default_params,
        "quick": find_embedding_quick_preset,
        "balanced": find_embedding_balanced_preset,
        "robust": find_embedding_robust_preset,
        "high_beta": find_embedding_high_beta_preset,
        "deep_search": find_embedding_deep_search_preset,
    }

    preset_name = args.embedding_preset

    if preset_name not in preset_registry: 
        m=f"Unknown embedding preset '{preset_name}'. "
        m+=f"Available presets: {list(preset_registry.keys())}"
        raise ValueError(m)
    
    user_preset = preset_registry[preset_name] # heuristic_preset original dict
    preset_copy = dict(user_preset)     
    return preset_copy 


#----------------helper functions:
# embdding
def embedding_info(emb: Dict[int, List[int]]) -> Dict[str, float]:
    """
    Given an embedding object, this function 
    returns a dict with useful metrics about the embedding itself:

    1. legnths of all chain
    2. average chain length 
    3. maximum chain lenght 
    4. total qubits used in the embedding
    5. total number of chain used in the embedding

     what an embedding object is:
     A dictionary of type "Dict[int, List[int]]" that maps
     logical variable to physical qubits (Graph to Graph mapping).

     for example:
     S = {1, 2, 3} U {(1, 2), (1, 3), (2, 3)} ::  A trinagular Source graph
     T = {A, B, C, D} U {(A, B), (B, C), (C, D), (D, A)} ::  A rectagular Target graph

     The result of  "emb = find_embedding( S, T)"  CAN  be:
     
        emb = 
            {
                1: [A],
                2: [B],
                3: [C, D]
            }

    This creates projection of S on T.
    Note:
    1. embdding does not always exist
    2. embdding is an NP-Hard problem
    3. qubits sometimes are forced to be considered as a single entity (like: [C, D])
    4. the embedding search is a stochastic process
    5. embedding can exist yet to not be found

    Summary:
    In this funciton we return useful info abuot the characteristics of the given
    mapping/embedding results.
    
    """

    if not isinstance(emb, dict):
        raise TypeError("embedding_info expects a dict mapping logical nodes to lists of qubits")

    if not emb:
        return {
            "avg_chain_len": 0.0,
            "max_chain_len": 0,
            "num_chains": 0,
            "total_used_qubits": 0,
        }

    chains = emb.values()
    lengths = [len(chain) for chain in chains if chain]  # ignore empty chains
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    total_used = sum(lengths)
    num_chains = len(emb)
    #what more pbservations can we derive here ? chain_lengths historgram ?

    return {
        "avg_chain_len": float(avg_len),
        "max_chain_len": int(max_len),
        "num_chains": int(num_chains),
        "total_used_qubits": int(total_used),
    }

def get_info_row(G:nx.Graph, emb:dict, run_time:float)->dict:
    """
    This method simply extracts all existing properties
    that were previously inserted into Grap-G and from the 
    embedding dictionary - into a single dict.
    this dict is then returned, and usually used as a dataframe row.
    
    """
    row = {}
    for k,v in G.graph.items():
        row[k] = v
    
    emb_info = embedding_info(emb)
    for k,v in emb_info.items():
        row[k] = v
    
    row['embedding_time_sec'] = run_time

    return row

# print current run
def _fmt_value(name: str, value, default) -> str:
    tag = "default" if value == default else "custom"
    return f"{name:<30}: {value!r} [{tag}]"

def render_run_config(args) -> str:
    """
    Returns a string representing the current run complete configuration
    """

    lines = []
    # Run metadata
    now = datetime.datetime.now()
    lines.append("  RUN CONFIGURATION")
    lines.append(f"Run date                : {now.strftime('%A, %Y-%m-%d')}")
    lines.append(f"Run time                : {now.strftime('%H:%M:%S')}")
    lines.append("")


    # Source graphs
    lines.append("  SOURCE GRAPHS ")
    lines.append(_fmt_value("source_mode", args.source_mode, SOURCE_MODE))

    if args.source_mode == "dataset":
        lines.append(_fmt_value("dataset_root", args.dataset_root, DATASET_ROOT))
        lines.append(_fmt_value("dataset_read_method", args.dataset_read_method, DATASET_READ_METHOD))
        lines.append(_fmt_value("dataset_num_graphs", args.dataset_num_graphs, DATASET_NUM_GRAPHS))

        if args.dataset_read_method == "sorted":
            lines.append(_fmt_value("dataset_sort_metric", args.dataset_sort_metric, DATASET_SORT_METRIC))

    elif args.source_mode == "generator":
        lines.append(_fmt_value("generator_type", args.generator_type, GENERATOR_TYPE))
        lines.append(_fmt_value("generator_num_graphs", args.generator_num_graphs, GENERATOR_NUM_GRAPHS))

    lines.append("")


    # Target architecture
    lines.append("  TARGET ARCHITECTURE")
    lines.append(_fmt_value("architecture", args.architecture, DEFAULT_ARCHITECTURE))

    if args.architecture == "chimera":
        lines.append(_fmt_value("row_cells", args.row_cells, CHIMERA_ROW_CELLS))
        lines.append(_fmt_value("col_cells", args.col_cells, CHIMERA_COL_CELLS))
        lines.append(_fmt_value("tile_size", args.tile_size, CHIMERA_TILE_SIZE))

    elif args.architecture == "pegasus":
        lines.append(_fmt_value("pegasus_m", args.pegasus_m, PEGASUS_M))
        lines.append(_fmt_value("pegasus_offsets_index", args.pegasus_offsets_index, PEGASUS_OFFSETS_INDEX))
        lines.append(_fmt_value("pegasus_fabric_only", args.pegasus_fabric_only, PEGASUS_FABRIC_ONLY))

    elif args.architecture == "zephyr":
        lines.append(_fmt_value("zephyr_m", args.zephyr_m, ZEPHYR_M))
        lines.append(_fmt_value("zephyr_t", args.zephyr_t, ZEPHYR_T))

    lines.append("")


    # Embedding configuration
    lines.append("  EMBEDDING CONFIGURATION")
    lines.append(_fmt_value("embedding_preset", args.embedding_preset, DEFAULT_EMBEDDING_PRESET))
    lines.append(_fmt_value("embedding_per_graph", args.embedding_per_graph, EMBEDDING_PER_GRAPH))

    lines.append("")
    lines.append("  END RUN CONFIGURATION")

    return "\n".join(lines)


def get_run_id(args) -> str:
    """
    Generate a run identifier  based on current configuration.
    """

    def last_directory(path):
        path = os.path.normpath(path)
        parent = os.path.dirname(path)
        return os.path.basename(parent)


    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H_%M") 

    arch = args.architecture
    num_graphs = args.dataset_num_graphs
    preset = args.embedding_preset
    

    run_id = f"{date_str}_{time_str}_{arch}_G{num_graphs}_prst_{preset}"

    if args.dataset_root:
        source_ds = args.dataset_root
        source_dir_name = last_directory(source_ds)
        run_id+=f"_src_dir_{source_dir_name}"

    return run_id




# output csv 
def _get_output_csv_path(output_dir: str, _id: str) -> str:
    """
    Create an output CSV path of the form:
    {output_dir}/{YYYYMMDD}_{HHMMSS}_{_id}.csv
    """

    # ---- validate directory
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    filename = f"{_id}.csv"

    return os.path.join(output_dir, filename)

#----------------main:
if __name__ == '__main__':

  
    # current run set-up:
    parser = get_script_arguments()
    args = parser.parse_args()
    architecture, num_qubits = resolve_target_graph(args)
    """
    Obtaining Graph Iterator:
    At this point, we resolve the reading method and source.
    while constructing the graphs container - we implicitly
    call the MQLib CPP BACKEND which calculates set of 70 metrics for each
    loaded graphs.

    in effect, this line encapsulates the reading, analyzing, ordering and
    making iterable - the given dataset of graphs.
    
    """
    graph_iter = resolve_source_graphs(args)  # constructing GraphShelf object with initialized graphs & their metric measures
    embedding_params = resolve_embedding_preset(args)
    config_str = render_run_config(args)

    print_config = True
    if print_config:
        print(config_str)

    # varibales for capturing results:
    info_rows:List[Any]= []


    #  embedding loop:
    start_time = time.perf_counter()
    total_run_count:int = args.embedding_per_graph*args.dataset_num_graphs # Note: this will err on generator mode
    total_graphs:int = args.dataset_num_graphs # Note: this will err on generator mode
    curr_graph_count:int = 0
    curr_embd_cout:int = 0
    print(f"\nTotal embedding runs:{total_run_count}")

    for graph in graph_iter:

        curr_graph_count+=1
        curr_graph_start = time.perf_counter()
        for embd_run in range(args.embedding_per_graph):
            start = time.perf_counter()
            try:
                emb = find_embedding(S=graph, T=architecture, **embedding_params)
            except Exception as e: # can err due to timeout or internal init failure!
                print(f"Error occured while runnig minor-embedding for graph G: {graph}")
                print(e)
                continue

            end = time.perf_counter()
            embd_time = end-start
            graph_row = get_info_row(G=graph, emb=emb, run_time=embd_time)
            info_rows.append(graph_row)
            curr_embd_cout+=1
        total_graph_time = time.perf_counter() - curr_graph_start
        print(f"Done--[ {curr_graph_count}/{total_graphs} ] graphs | [ {curr_embd_cout}/{total_run_count} Embeddings] | graph_time: {total_graph_time} seconds.")

    out_dir_temp = internal_paths.TEMPORARY_OUTPUT_BOX_DIR
    run_id = get_run_id(args)
    temp_ouput_csv = _get_output_csv_path(output_dir=out_dir_temp, _id=run_id)
    df = pd.DataFrame(info_rows)
    df.to_csv(temp_ouput_csv)



    total_run_time = time.perf_counter() - start_time
    print(f"Total time: {total_run_time}")
