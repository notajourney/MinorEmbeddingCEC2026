
####-------------------------------------MODULE-DOC-STRING--------------------------------####
"""
Purpose:
This file is the main hub for all Graphs-Related operations, which can be
devided into three main categories:
1. Readin Graphs (from file system) in specific format (see graph.txt format)
2. Generating Graphs (this involved graph theory)
2. Writing Graphs (to file system) in various forms

"""

####-------------------------------------IMPORTS------------------------------------------####
from __future__ import annotations
# general imports:
import os
import re
import subprocess
import math, random
import numpy as np
import pandas as pd
import  networkx as nx
from sympy import symbols, expand, S 

# typing imports:
from numpy.typing import NDArray
from typing import List, Dict, Any, Optional, Callable, Union, Iterator

# internal imports:
from max_cut.project_types  import  Point, Segment, Distance
from max_cut.utils import (
    get_nx_graph_metrics,
    is_undirected_g,
    is_symetric_mat,
    is_upper_triangular_mat,
    is_lower_triangular_mat
) 
####---------------------------------GRAPH-SHELF-OBJECT-START-----------------------------####

class GraphShelf():
    """
    Object to manage reading and iterating over graph-dataset.
    Inputs:
    dataset_root: root graph dataset root directory
    b_path: benchmark optimum csv file if exists

    Note:
    This object returns the graphs as NetworkX.Graph, with the benchmark info
    ingrained into the graph-object's metadata (NetworkX.Graph.grap['property_name'])
    """
    def __init__(self, dataset_root:str, benchmark_file:str=None, instances:List[nx.Graph]=None):
    
        #TODO: graphshelf init must be rethought
        #initializing root of dataset and empty fields:
        self.dataset_root = dataset_root
        self.graphs:List[nx.Graph] = None
        self.g_paths:list[str] = None
        self.max_iter:int = None
        self.iter_count:int = 0


        if not instances: # no instances given, meaning- we load the entire dataset
            self.graphs = self._read_all_instance(root_path=dataset_root)
            self.g_paths = self.get_all_paths(dataset_root=self.dataset_root)
            self.max_iter = len(self.graphs)
       
        else: # we create from given instances
            self.graphs = instances
            self.g_paths = None # this is meaningless when initializing from instances
            self.max_iter = len(instances)
   
        if benchmark_file:
            self.b_path = os.path.abspath(benchmark_file)
            self.benchmark_table:pd.DataFrame = pd.read_csv(benchmark_file).set_index('id')


##-----------------------------instance methods: bulk readin
    def _read_all_instance(self, root_path:str)->List[nx.Graph]:
        """
        Takes path to root of graphs dataset - reads and parses all graphs
        under root_path.
        TODO _read_gset::Fix bug of reverse reading and a different file traverse
        """
        graphs:List[nx.Graph] = []
        paths:List[str] = self._get_paths(ds_root=root_path)

        for g_path in paths:
            G = self._parse_graph_file(g_path)
            # self.add_bench_info(G) # TODO:  fix benchmarking graph info, it is not longer suppoerted the same waY!
            graphs.append(G)

        return graphs

    @classmethod
    def get_all_paths(cls, dataset_root:str)->List[str]:
        """
        TL;DR ::  given root, returns all paths to graphs.

        This method takes a path to the root directory of the dataset,
        traverses this path and  its sub-directories, and returns a list
        of paths to the various graphs that exist under the root directory.

        Note:
        graph files are expected to end with *.txt, therefor any other file
        extension is ignored.

        input:
        ds_root - system path to a directory that contains dataset of graphs.

        output:
        list of strings, each is a path to a graph under root.
        """
        g_paths:List[str] = []
        for dirpath, _ , filenames in os.walk(dataset_root):
            for filename in filenames:
                g_path = os.path.join(dirpath, filename)

                if not filename.lower().endswith(".txt"):
                    continue
                else: 
                    g_paths.append(g_path)

        message = (
            f"Error while resolving dataset paths:\n"
            f"Dataset root directory is illegal or empty\n"
            f"root: {dataset_root}"
        )
        assert len(g_paths) > 0, message
        return g_paths

##-----------------------------instance methods: previous benchmark info
#TODO: rewrite 'add_bench_info' (not relevant more, not including mqlib)
    def add_bench_info(self, G:nx.Graph)->Dict[str, Any]:
        """
        TODO: add_bench_info method is horrible and not consistent with project anymore - should be rethink and re implemented
        Given an instance of a graph G,
        this mathod adds benchmark info to G.metadata (if exists) 
        """
        g_id = G.graph['name'] # TODO: when is name initialized ?

        id_exists = True

        try:
            b_mark_info = self.benchmark_table.loc[g_id]
        except KeyError:
            id_exists = False
            message = f"Graph \'{g_id} \' does not exist in benchmark file.\n"
            message += f"All metadata of \'{g_id}\'is set to None"
            # TODO: what is happening with message/exception ??

        # These properties are taken from the known optimum table and graph itself:
        G.graph['name'] = g_id # Already populated but whatever
        G.graph['Nodes_Num'] = len(G.nodes)
        G.graph['Edges_Num'] = len(G.edges)   
        if id_exists:
            G.graph['Known_Opt'] = b_mark_info.loc['target']
            G.graph['G_Class'] = b_mark_info.loc['type']
        else:
            G.graph['Known_Opt'] = 0 #TODO: debug this, what happens when NAN ?
            G.graph['G_Class'] = None

        # These properties are filled in post-solving the instance (creating empty properties):
        G.graph['Solver'] = None
        G.graph['Solver_Type'] = None
        G.graph['Total_Time'] = None
        G.graph['Opt_Diff'] = None
        G.graph['Solver_Opt'] = None 
        G.graph['Glob_Opt'] = None 


##-----------------------------instance methods: iterator
    def __iter__(self):
        return self
    
    def __next__(self):

        if self.iter_count == self.max_iter:
            self.iter_count = 0
            raise StopIteration()
        else:
            item = self.graphs[self.iter_count]
            self.iter_count+=1
            return item

    def __len__(self):
        return len(self.graphs)
##-----------------------------class methods: parsing
    @classmethod
    def parse_graph_file(cls, graph_path:str, fall_back:Optional[Callable[ [str],  nx.Graph]]=None)->nx.Graph:
        """
        Given a path to graph.txt file under the MQLib/Gser format(!)
        this function reads and returns the graph as NetworkX.Graph object

        fall_back:: A callable for future functionality of parsing graph-files-
        currently not in use.

        #------------------------------------------------------------------#
        Supported format (pattern):
        "
        # comment1
        # comment2
        Line_1: node_cont edges_count
        Line_2: u_1 v_1 weight_1
        .
        .
        .
        Line_(n+1): u_n v_n weight_n
        "
        #------------------------------------------------------------------#


        #------------------------------------------------------------------#
        Fomat notes:
        1. comments start with #
        2. first data line is N M (nodes, datalines)
        3. each data line is a weighted edge: x_i x_j w_ij, means: (x_i)---w_ij--->(x_j)
        4. x_i sequence must appear in ascending order (not reinforced!)
        5. x_i < x_j for all  i,j (not reinforced!)
        6. if (x_i,  x_j) appear --> (x_j, x_i) must not appear (not reinforced!)
        7. w_ij : can be any real-number except zero (zero weight is defined as non-edge)
        #------------------------------------------------------------------#


        TODO: _parse_graph_file: Understand the note of chat and possibly fix it.
        
        Issues:
        1.  Actual number of nodes:Every graph file starts with a meta-data line "N M" that indicates the number of nodes(N)
            and and the number of datalines (edges).
             after constructing the corresponding graph object-it can happen that the actual number of nodes
             is different from the number N.
             This phenomena has been observed directly through assertions and debug.

        """
        adj_list = []
        with open(graph_path) as f:

            for line in f:
                if len(line.split()) != 3 or line.startswith("#"):
                    continue

                line = line.strip('\n')
                adj_list.append(line)
            G = nx.read_weighted_edgelist(adj_list, nodetype=int)
            G.graph['source_file_name'] =  os.path.basename(graph_path).split(".")[0]
        return  G


##-----------------------------class methods: delicate reading
    @classmethod
    def random_sample(cls, dataset_root:str, sample_size:int=10):
        """
        Docstring for random_sample
        
        :param cls: GraphShelf
        :param dataset_root: path to root dir of graph-dataset in file system
        :type dataset_root: str
        :param sample_size: Numer of graph to sample randomly from dataset
        :type sample_size: int

        returns an instance of GraphShelf() initialized with 'sample_size' graph instances
        drawn from 'dataset_root'
        """
  
        paths:List[str] = cls.get_all_paths(dataset_root)
        chosen = random.sample(paths, sample_size)
        graphs:List[nx.Graph] = []
        for path in chosen:
            graph = cls.parse_graph_file(path)
            cls.add_mqlib_graph_metrics(G=graph, graph_path=path) # currently adding metrics usin the MQLib library
            graphs.append(graph)
        return GraphShelf(dataset_root=dataset_root, instances=graphs)

    @classmethod
    def read_n_smallest_graphs(cls, dataset_root:str, n:int,  by_nodes:bool=True)->GraphShelf:
        """
        Since we defined a singular format in which the graphs appear
        in their txt file, we can read graph in a sroted manner.

        for instance:
        the first line of the graph shows 'N' 'M':
        N - being the number of nodes,
        M - being the number of datalines (edges)

        So, Here we read the n graphs starting from the smallest graph in terms of node count (N) 
        and forward.

        params:
        @ cls : GraphShelf
        @ dataset_root : str, |path to root directory of graphs.txt dataset
        @ n : int  |number of graphs to read
        @ by_nodes : bool=True | take the n smallest with respect to node count, if false - by datalines count
        
        """
        NM_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
        def read_nm_line(path:str)->Optional[tuple[int, int]]:
            """
                Given a path to graph.txt file:
                this method reads the "N M" line
                and returns the two integers.
                
                if line does not exist - returns None
            """
            with open(path, "r", encoding='utf-8') as f:
                for line in f:
                    _match = NM_PATTERN.match(line)
                    if _match:
                        n, m = map(int, _match.groups())
                        return n, m
                return None

        all_path:List[str] = cls.get_all_paths(dataset_root=dataset_root)
        entry:tuple[str, int, int] = None
        sorted_paths:List[tuple[str, int, int]]=[]

        for path in all_path:
            dataline = read_nm_line(path)
            if dataline is None:
                continue
            else:
                N = dataline[0]
                M = dataline[1]
                entry= (path, N, M)
                sorted_paths.append(entry)

        #sorting list:
        if by_nodes:
            f = lambda x:x[1] # N 
        else:
            f = lambda x:x[2] # M 
        sorted_paths.sort(key=f)

        #building GraphShelf based on request:
        requested_amount:int = n
        graphs:List[nx.Graphs] = []
        for i in range(requested_amount):
            path:str = sorted_paths[i][0]
            G:nx.Graph = cls.parse_graph_file(path)
            # TODO : Adding metrics potentially occurs here
            # TODO : currently it is impossible to add metrics in a simple manner
            graphs.append(G)
            #print(f"{i+1} | Nodes: {len(G.nodes)} | Name: {G.graph['source_file_name']}")

        #  running assetion to spot a mysterious bug:

        def is_sorted_path_array(sorted_paths)->bool:
            prev = -1
            for entry in sorted_paths:
                current = entry[1]
                if current < prev:
                    return False
                else:
                    prev=current
                    continue
            return True
       
        def is_sorted_graph_array (graphs):
            prev = -1
            for i, g in enumerate(graphs):
                next = len(g.nodes)
                if  next >= prev:
                    prev = next
                    continue
                else:
                    print(f"found wrong entry at index: {i}")
                    print(f"Associated file : {sorted_paths[i]}")
                    print(f"Associated file : {sorted_paths[i+1]}")
                    return False
            return True

        def is_node_count_eq_N()->bool:
            missmatches = 0
            for i in range(requested_amount):
                N = sorted_paths[i][1]
                node_count = len(graphs[i].nodes)
                if node_count != N:
                    missmatches+=1
            return missmatches == 0


        test = False
        if test:
            assert is_sorted_path_array(sorted_paths), "sorted_paths  are not in ascending order !"
            assert is_node_count_eq_N(), "Mismatch between metadata node count and actual node count !" # Fails
            assert is_sorted_graph_array(graphs), "Graph array (based on sorted paths) is not in ascending order !" # Fails
            
        return GraphShelf(dataset_root=dataset_root, instances=graphs)


#-----------------------------class methods: graph metrics
    @classmethod
    def get_mqlib_graph_metrics(cls, graph_path:str)->Dict[str, Any]:
        """
        Docstring for get_mqlib_graph_metrics
        
        :param cls: GraphShelf class
        :param graph_path: system path to graph.txt file - representing a max-cut problem under the convensional format [COMPLETE REFERENCE]
        :type graph_path: str
        :return: A dictionary with string keys and Numerical value: keys are the name of the metric propery - and value is its measure.
        :rtype: Dict[str, Any]

        Important notes (contract)
        1. This method does not work if MQLib project is not cloned and compiled in the running machine
        2. This method does not work if MQLib executable does not have a run permission
        3. This method runs a bash script to perform external library (C++) metrics calculations
        4. This method checks the exit status of  of the bash script and fails (!) if it fails (throws exeption)
        5. This method reads the bash csv output  file of metrics
        6. TODO: currently, the executable and output paths are hardcoded
        6. TODO: resolve and remove the hard coded paths in this script !


        Script: /home/username/project_dir/MQLib/scripts/run_metrics.sh
        Exec:  /home/username/project_dir/MQLib/bin/MQLib
        Output : /home/username/project_dir/metrics_output_tmp

        """
        def _get_mqlib_metrics_script_path(script_name)->str:
            """
            Docstring for _get_mqlib_metrics_script_path
            
            :param script_name: name of the bash script file that extracts graph metrics

            :return: Full absolute path to the script
            :rtype: str (path)

            Note:
            This method assumes the fixed-realtive location of the script, it searches the script there
            and raises an error if script is not found.

            about location:
            The target scrip should be located under  the project root directory (Q_Opt),
            under the 'bash_scripts' directory - and should have the name 'script_name' !

            Typical location:
            /home/user/Q_Opt/max_cut/bash_scripts/script_name.sh
            """
            # building path starting from __file__:
            bash_scripts_dir_name:str = "bash_scripts"
            current_file = __file__
            module_dir = os.path.join(current_file, os.pardir)
            bash_scripts = os.path.join(module_dir, bash_scripts_dir_name)
            script_path = os.path.join(bash_scripts, script_name)
            return os.path.abspath(script_path)
            
        # Hard coded paths for mqlib_metrics execution:
        bash_script_name:str = 'run_mqlib_metrics.sh'
        script_exec_path:str = _get_mqlib_metrics_script_path(script_name=bash_script_name)
        mqlib_exec_path:str = r'/home/ronl/cloned_project/MQLib/bin/MQLib'
        output_path:str = r'/home/ronl/Q_Opt/max_cut/metrics_output_tmp'
        arguments:List[str] = [script_exec_path, mqlib_exec_path, graph_path, output_path]

        #checking if the bash script has permission to run:
        permission = os.access(script_exec_path, os.X_OK)
        if not permission:
            m=f"Bash script is not executable: {script_exec_path}"
            m+=f"Run  \'chmod +x {script_exec_path}\'"
            raise RuntimeError(m)
        
        #executing with arguments:
        mqlib_metrics:Dict[str, Any] = {}
        try:
            result = subprocess.run(
                arguments,
                capture_output=True, # redirect outputs away from terminal (to here)
                text=True, # strings and not bytes
                check=True # if script exist status != 0 we raise an exeption
            )
        except subprocess.CalledProcessError as e:
            print("Unable to fetch graph metrics info:: bash script has failed")
            print(f"Failed to fetch metrics of: {graph_path}")
            print("Original error:\n")
            print(e.stderr)


        metrics_csv_path = result.stdout.strip()
        if not os.path.isfile(metrics_csv_path):
            m="Error: unable to locate mqlib metrics output file!"
            raise RuntimeError(m)
        else:
            graph_metrics = pd.read_csv(metrics_csv_path)
            mqlib_metrics = graph_metrics.iloc[0].to_dict()   
        return mqlib_metrics

    @classmethod
    def add_mqlib_graph_metrics(cls, G:nx.Graph, graph_path:str)->None:
        """
        Docstring for add_mqlib_graph_metrics
        TL;DR ::  fills G with MQLib metrics inplace
        This methods takes a graph instance (togther with its file-system-path),
        runs MQLib library to extract varaiaty of graph metrics for the given instance,
        and populates the nx.Graph instance with the metrics results.

        important notes:
        1. MQLib should be cloned and compiled
        2. Bash scrip should have the permission to run
        3. The path 'g_path' should point to a graph.txt file
        4. The graph file should be aligned with the MQLib/Gset format convensions (important!)
        5. This method may throw an exeption if things go wrong with external lib and/or bash script exit status/permissions
        
        :param cls: GraphShelf
        :param G: nx.Graph initialized graph object which is associated with  this file 'g_path'
        :type G: nx.Graph
        :param graph_path: system path to a text file containing graph (in MQLib/Gset formats), associated with 'G' (important!)
        :type graph_path: str (path)

        Returns None 
        (filles G.graph dict inplace)
        """
        try:
            metrics_dict:Dict[str, Any] = GraphShelf.get_mqlib_graph_metrics(graph_path)
        except Exception as e:
            print(e) # TODO: decide what to do with bash script exceptions
            raise
        G.graph.update(metrics_dict)

    @classmethod
    def add_nx_graph_metrics(cls, G:nx.Graph)->None:
        """
        Fills (inplace!) the G.graph dictionary with pre-defined NetworkX graph metrics
        (for more info see  max_cut.utils.get_nx_graph_metrics)
        
        :param G: nx.Graph object to fill with metrics info
        :type G: nx.Graph

        Returns none
        """
        metrics:Dict[str, Any] = get_nx_graph_metrics(G)
        G.graph.update(metrics)


##-----------------------------class methods: graphs output
# TODO: finlize&test graph output logic
    def graph_to_text_format(G:Union[nx.Graph, NDArray], comments:Optional[List[str]]=None )->str:

        """
        Docstring for graph_to_text_format
        
        :param G: A graph object - represented by NetworkX or Numpy matrix
        :type G: Union[nx.Graph, NDArray]
        :return: string, representing the graph in MQLib/Gset convensional format
        :rtype: str

        About the format (which is stricly reindforced throughout the entire project):
        1. The format is: a textual representation of a graph
        2. Has header and comments
        3. Defines the graph via edges & weights
        4. Has rules about the order of edges (see discussion)
        5. Concerned with UNDIRECTED graphs
        6. Allows weighted graphs

        Format example-definition & Rules:

        1. Example
            Input matrix:
                0    1     2     3
            0   0----7-----0-----9-

            1   -----0-----1-----1-

            2   -----------0-----6-

            3   ------------------0

            
            Output text:
            # optional comment ...
            4 5
            1 2 7
            1 4 9
            2 3 1
            2 4 1
            3 4 6


        2. definition & Rules:
            2.1 lines starting with '#' are treated as comments
            2.2 first non-comment line:
                conatains two numbers:
                N - number of nodes
                M - number of edges (data lines)
            2.3 each data-line (edge) consists of:
                2.3.1 a - source node (integer 1 ... (N-1) ) 
                2.3.2 b - target node (integer 2 ... (N) ) 
                2.3.3 w - float/int (the weight of the (a,b) edge)

            2.4 if (a, b, w) exist, then (b, a, w) must not exist
            2.5 a < b
            2.6  "a a w" self loops are forbidden TODO - verify this 
            2.7 a_i appears in ascending order (not necessarily true for b_i!)

        Note:
        1. Input matrix shuod represent an undirected graph, meaning:
            - it is either upper/lower triangular
            - or symetric
        2. Isolated node are omminted and ignored
        """
        graph_matrix:NDArray = None
        graph_str:str =""

        # input type validity:
        if isinstance(G, np.ndarray):
            graph_matrix = G.copy()

        elif isinstance(G, nx.Graph):
            """
            Here we need to be very careful !
            when converting nx.Graph to numpy array,
            we might encounter cases in which the nodes of G
            are not what we expect them to be, for example:
            #----------------example--------------#
                G = nx.Graph()
                G.add_edge(3, 1, weight=1)
                G.add_edge(4, 5, weight=1)
                G.add_edge(10, 4, weight=1)
     
                M = nx.to_numpy_array(G)
                print(M)
                print(G.nodes)


                [[ 0.  1.  0.  0.  0.]
                [  1.  0.  0.  0.  0.]
                [  0.  0.  0.  1.  1.]
                [  0.  0.  1.  0.  0.]
                [  0.  0. 1.  0.  0.]]
            
                [3, 1, 4, 5, 10]

            in the matrix representation,
            the index i is mapped to a node-lable, in this case:
            [0, 1, 2, 3, 4] --> [3, 1, 4, 5, 10]

            but:
            who does the mapping ?
            what index does node 3 correspond to?
            what node does index 0 correspond to?
            can we safely determing that: M[0] = node 1?

            answer:
            We DO NOT KNOW ANY of these.

            So, our goal is to make a clear order and mapping between matrix[i] to nodes[i].
            To achieve this we sort the nodes, and treat nodes[0] as "1", nodes[1] as "2" ..etc.


            Notes:
              1.This function DOES NOT preserve node lables !
                input nodes: [2, 7, 8]
                output nodes [1, 2, 3]
                meaning:
                we change the lables, but not the structure.
            """
            
            def is_positive_integer_labled_graph(G):
                for n in G.nodes():
                    if not isinstance(n, int):
                        return False
                    elif isinstance(n, int) and n <=0:
                        return False
                    elif isinstance(n, bool):
                        return False  
                return True
            assert is_positive_integer_labled_graph(G), "Error: The given nx.Graph must be integer-based lables"

            # Now, sorting nodes befor conversion ( i still dont understand why is this a solution? who cares what M[0] represents if it is all isomorphic?)
            #graph_matrix = nx.to_numpy_array(G)  # previous wrong code, why wrong ?
        
            # #test deleting isolated nodes to pass test
            # G = G.copy()
            # G.remove_nodes_from(list(nx.isolates(G)))


            orderd_nodes = sorted(G.nodes())
            graph_matrix = nx.to_numpy_array(G, nodelist=orderd_nodes)
              
        else:
            m="Encountred unexpected type while writing graph to text format"
            m+=f"Unsupported type: {type(G)}"
            raise TypeError(m)


        # matrix shape validity:
        if graph_matrix.ndim != 2:
            m= "Error: while writing Graph object to string, \n"
            m+="Expected 2-dimensional matrix, encountered ndim={graph_matrix.ndim}"
            raise ValueError(m)
        
        if any(np.diag(graph_matrix)!= 0):
            m="Error while converting graph to text format:\n"
            m+="Self loop are forbidden."
            raise ValueError(m)
        
        if is_lower_triangular_mat(graph_matrix):
            m="Error: while converting Graph to string,"
            m+="Lower triangular reprsentations are currently not supported "
            raise ValueError(m)

        if not is_undirected_g(graph_matrix):
            """
            Here we verify that the matrix represents an un-directed graph,
            which is either upper-triangular/symetric matrix
            """
            m="Error: while converting Graph to string,"
            m+="encountred a directed graph - \n {graph_matrix}"
            raise ValueError(m)
        

        # building the edges output lines: "a b w":
        edge_entries:str=""
        edges_count:int=0
    
        for i in range(graph_matrix.shape[0]):
            ith_entries = graph_matrix[i, i+1:] # for example i=3: row=3, columns: 4...(N-1)
            for index, value in enumerate(ith_entries):
                if value == 0: # this is a non-edge by convension
                    pass
                else:
                    edges_count+=1
                    current_node = i+1
                    target_node = current_node + (index+1)
                    weight = value 
                    edge_entries+= f"{current_node} {target_node} {weight}\n"

        # Adding comments:
        if comments:
            for comment in comments:
                if comment.startswith("#"):
                    graph_str+=comment+"\n"
                else:
                    graph_str+= "#"+comment+"\n"

        # Adding N M line:
        node_num = graph_matrix.shape[0] 
        graph_str+= f"{node_num} {edges_count}" +"\n"

        # Adding edges:
        graph_str+= edge_entries

        return graph_str
####---------------------------------GRAPH-SHELF-OBJECT-END-------------------------------####


####--------------------------------------------------------------------------------------####
####--------------------------------------GENERATORS--------------------------------------####
"""
TL;DR : build_graph_with_density() generates a random graph with n nodes and specified density
"""
#TODO: wrap density graph generator into a class
#TODO: come up with more graph generators (metric controlled generators)

def generate_spaced_points(n: int, x: Segment, y: Segment, rng = None) -> List[Point]:

    rng = rng or random # if seeded random object is give - we use it to generate points
    xmin, xmax = x
    ymin, ymax = y
    points = [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(n)]
    return points

def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def pairwise_dist_list(points:List[Point])->List[float]:
    """
    returns a list of distances between each pair of points
    """
    n = len(points)
    edges_dist = []
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean_distance(points[i], points[j])
            edges_dist.append(d)
    return edges_dist

def get_quantile_radius(pw_distance:list[float], q:float)->float:

    """
    returns a radius from which q*100% of distances are less than, or eual to.
    (that is used to controll the number of edges entring the final graph)
    """
    return np.quantile(pw_distance, q)

def build_graph_from_radius(points:List[Point], distance:Distance, raduis:float)->np.ndarray:
    '''
    M[i][j] == 1 iff distance between point_i, point_j < = radius

    '''
    assert isinstance(raduis,(int, float)) and raduis >= 0, f"wrong value for radius {raduis}"
    dim = len(points)
    graph_matrix = np.zeros(shape=(dim, dim), dtype=np.uint8)
    for i in  range(dim):
        for j in range(i+1,dim):
            if distance(points[i], points[j]) <= raduis:
                #graph_matrix[i][j] = graph_matrix[j][i] = 1
                graph_matrix[i][j] =  1
    return graph_matrix

def build_graph_with_density(n: int, density: float, rng:random.Random=None)->nx.Graph:
    """
    This fucntion generates a random nx.Graph (undirecred)
    with a specified  number of nodes and a specifies edges-density.

    params:
    n: int, number of nodes in the result graph
    density: float, (0,1] the requested percentage of edges ("1"  means full graph)
    rng: random.Random, random number generator. None means random run, else is user controlled run
        'generate_spaced_points' uses either random or  user-defined random.Random generator

    How it works:
    1. defining space segment
    2. randomizing n 2-d points in that segment
    3. calculating distance between each pair of point
    4. calculatin a distance (radius) that include x% (or in fact density%) of the points'
    5. setting up an edge bertween two points iff they are 'close enough' for that radius

    """
    x_seg = (0, 10)
    y_seg = (0, 10)
    points = generate_spaced_points(n, x_seg, y_seg, rng=rng)
    dist_list = pairwise_dist_list(points)
    target_r = get_quantile_radius (dist_list, q=density)
    M = build_graph_from_radius(points, euclidean_distance, target_r)
    G = nx.from_numpy_array(M)
    return G

class DensityGraphGenerator:
    """
    Class wrapper around the existing functions to generate graphs
    with given density, and to iterate over a sequence of densities.
    """

    def __init__(self, rng: random.Random | None = None):
        """
        rng: optional random.Random instance (for reproducibility).
        If None, uses the global 'random' module.
        """
        self.rng = rng

    def generate_graph(self, n: int, density: float) -> nx.Graph:

        return build_graph_with_density(n=n, density=density, rng=self.rng)

    def iter_graphs(self, graph_count: int, node_count: int) -> Iterator[nx.Graph]:

        if graph_count <= 0:
            raise ValueError("graph_count must be a positive integer")

        densities = np.linspace(0.05, 1.0, graph_count)
        for d in densities:
            yield self.generate_graph(n=node_count, density=float(d))
####---------------------------------------GRAPH_IO-UTILS----------------------------------####

####----------------------------------------TESTS------------------------------------------####
def test_mqlib_metrics():
    graph_path:str = r'/home/ronl/Q_Opt/max_cut/datasets/mqlib_graphs/g002732/g002732.txt'
    result = GraphShelf.get_mqlib_graph_metrics(graph_path)
    print(result)

def test_graph_to_str():

    def test_simple_graph_to_text():
        G = nx.Graph()
        G.add_edge(1, 2, weight=3)
        text = GraphShelf.graph_to_text_format(G)
        lines = text.strip().splitlines()
        # header
        assert lines[0] == "2 1"
        # edge line
        assert lines[1] == "1 2 3.0"

    def test_non_consecutive_labels():
        G = nx.Graph()
        G.add_edge(10, 5, weight=7)
        G.add_edge(7, 3, weight=2)

        text = GraphShelf.graph_to_text_format(G)
        lines = text.strip().splitlines()

        # N = 4 nodes, M = 2 edges
        assert lines[0] == "4 2"
        edges = set(lines[1:])
        assert len(edges) == 2

    # isomorpism:
    def parse_text_graph(text):
        lines = text.strip().splitlines()
        n, m = map(int, lines[0].split())

        edges = []
        for line in lines[1:]:
            a, b, w = line.split()
            edges.append((int(a), int(b), float(w)))

        return n, edges

    def test_graph_structure_preserved():
        G = nx.Graph()
        G.add_edge(10, 30, weight=5)
        G.add_edge(5, 7, weight=9)
        G.add_edge(7, 4, weight=10)

        text = GraphShelf.graph_to_text_format(G)
        n, edges = parse_text_graph(text)

        assert n == G.number_of_nodes()
        assert len(edges) == G.number_of_edges()

  
        H = nx.Graph()
        for a, b, w in edges:
            H.add_edge(a, b, weight=w)

        assert nx.is_isomorphic(
            G,
            H,
            edge_match=lambda e1, e2: e1["weight"] == e2["weight"]
        )

    def test_self_loop_forbidden():
        G = nx.Graph()
        G.add_edge(1, 1, weight=5)
        try:
            GraphShelf. graph_to_text_format(G)
        except Exception as e:
            print(e)
            pass

    def test_non_integer_labels():
        G = nx.Graph()
        G.add_edge("a", "b", weight=1)
        try:
            GraphShelf.graph_to_text_format(G)
        except Exception as e:
            print(e)
            pass

    
    def print_example_output():
        G = nx.Graph()
        # intentionally ugly, non-consecutive, non-sorted labels
        G.add_edge(42, 7, weight=1)
        G.add_edge(7, 100, weight=2)
        G.add_edge(42, 300, weight=3)
        G.add_edge(5, 100, weight=4)
        G.add_edge(5, 42, weight=5)
        text = GraphShelf.graph_to_text_format(G)
        print("Output for graph G:")
        print(f"with nodes:{G.nodes()}")
        print(f"with edges:{G.edges()}")
        print(text)

    def test_random_invariants(trails:int=10):

        def graph_invariants(G):
            return {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "degree_sequence": sorted(dict(G.degree()).values()),
                "connected_components": sorted(len(c) for c in nx.connected_components(G)),
                "total_weight": sum(d["weight"] for _,_,d in G.edges(data=True))
            }

        def random_graph_test():
            G = nx.gnm_random_graph(20, 40)
            for u, v in G.edges():
                G[u][v]["weight"] = random.randint(1, 10)

            # relabel randomly
            labels = random.sample(range(1000, 10000), G.number_of_nodes())
            mapping = dict(zip(G.nodes(), labels))
            G = nx.relabel_nodes(G, mapping)

            return G

        def build_graph_from_text(text:str):
            # parse text back to graph H
            lines = text.strip().splitlines()
            n, m = map(int, lines[0].split())

            H = nx.Graph()
            for line in lines[1:]:
                a, b, w = line.split()
                H.add_edge(int(a), int(b), weight=float(w))
            return H
            
        def is_isomorphic(G,H):
            
            return nx.is_isomorphic(
                G,
                H,
                edge_match=lambda e1, e2: e1["weight"] == e2["weight"]
            )

        for _ in range(trails):
            G = random_graph_test()

            G.remove_nodes_from(list(nx.isolates(G)))
            text = GraphShelf.graph_to_text_format(G)
            H = build_graph_from_text(text)

            inv_G = graph_invariants(G)
            inv_H = graph_invariants(H)
            assert inv_G == inv_H
            assert is_isomorphic(G, H)


    # running all tests:
    test_simple_graph_to_text()
    test_non_consecutive_labels()
    test_graph_structure_preserved()
    test_self_loop_forbidden()
    test_non_integer_labels()
    test_random_invariants(trails=10000)
    print_example_output()

def test_n_smallest_graphs():

    def is_sorted_array(graphs)->bool:
        prev = -1
        for graph in graphs:
            current = len(graph.nodes)
            if current < prev:
                return False
            else:
                prev=current
                continue
        return True

    dataset_path:str = r'/home/ronl/Q_Opt/max_cut/datasets/mqlib_graphs'
    read_amount:int = 1000
    n_smallest_g = GraphShelf.read_n_smallest_graphs(dataset_root=dataset_path, n=read_amount, by_nodes=True)

    assert len(n_smallest_g) == read_amount, "wrong number of reads"
    assert is_sorted_array(n_smallest_g), "TEST FAILED: the result is not in ascending order"
    # testing if list is not sorted:


####-----------------------------------------MAIN--------------------------------------####
if __name__ == '__main__':
    test_graph_to_str()
    print()
####-----------------------------------------END---------------------------------------####
