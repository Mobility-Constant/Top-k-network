"""
Simulation with Synthetic Urban Mobility Networks
------------------------------------------------
This module provides tools for simulating and analyzing urban mobility networks 
using various random network models and finding percolation thresholds.

"""

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os


def generate_random_network(flow_matrix):
    """
    Generate a network of popularity where each node has a probability to connect
    to other nodes based on its out-degree in the empirical network.
    
    Parameters:
    -----------
    flow_matrix : numpy.ndarray
        Empirical flow matrix to base connection probabilities on
        
    Returns:
    --------
    new_adj_matrix : numpy.ndarray
        Adjacency matrix of the generated network
    """
    # Ensure input is numpy array
    flow_matrix = np.array(flow_matrix)
    
    # Get dimensions
    num_nodes = flow_matrix.shape[0]
    
    # Calculate out-degrees for each node in the empirical network
    out_degrees = np.sum(flow_matrix > 0, axis=1)
    
    # Calculate connection probability for each node
    # pm = Eout_m/(n-1) where Eout_m is the out-degree of node m
    node_probs = out_degrees / (num_nodes - 1)
    
    # Initialize adjacency matrix
    new_adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # For each node, generate outgoing edges based on its probability
    for i in range(num_nodes):
        # Skip if node has no outgoing edges in empirical network
        if out_degrees[i] == 0:
            continue
            
        # Generate random connections with probability p_i
        for j in range(num_nodes):
            if i != j:  # No self-loops
                if np.random.random() < node_probs[i]:
                    new_adj_matrix[i, j] = np.random.uniform(1.0, 10000.0)
    
    return new_adj_matrix


def configuration_model_fixed_out_degree(flow_matrix, weight_scale=1000):
    """
    Generate a directed configuration model preserving out-degree distribution.
    
    Parameters:
    -----------
    flow_matrix : numpy.ndarray
        Mobility flow matrix
    weight_scale : int
        Maximum value for random edge weights
        
    Returns:
    --------
    new_adj_matrix : numpy.ndarray
        Flow matrix for the generated network
    """
    # Ensure input is numpy array
    adj_matrix = np.array(flow_matrix)
    
    # Calculate out-degrees (preserve these)
    out_degrees = np.sum(adj_matrix > 0, axis=1)
    num_nodes = len(out_degrees)
    
    # Randomize in-degrees, ensuring total equals total out-degrees
    in_degrees = np.random.multinomial(out_degrees.sum(), [1/num_nodes]*num_nodes)
    
    # Create directed configuration model
    G = nx.directed_configuration_model(in_degrees.tolist(), out_degrees.tolist(), create_using=nx.DiGraph())
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    
    # Create new adjacency matrix
    new_adj_matrix = np.zeros_like(adj_matrix)
    
    # Preserve weight distribution and assign to new edges
    # For each node, collect original weights, shuffle, and assign to new edges
    for node in range(num_nodes):
        # Get original outgoing edge weights
        original_weights = adj_matrix[node][adj_matrix[node] > 0]
        
        if len(original_weights) == 0:
            continue
            
        # Get new outgoing edges
        new_edges = list(G.out_edges(node))
        
        if len(new_edges) == 0:
            continue
        
        # Adjust weights list based on number of new edges
        if len(new_edges) > len(original_weights):
            # If more new edges, repeat original weights
            weights = np.tile(original_weights, (len(new_edges) // len(original_weights)) + 1)[:len(new_edges)]
        else:
            # If fewer new edges, select a random subset of original weights
            weights = np.random.choice(original_weights, size=len(new_edges), replace=False)
        
        # Assign weights to new edges
        for (u, v), weight in zip(new_edges, weights):
            new_adj_matrix[u, v] = weight
        
    return new_adj_matrix


def configuration_model_degree_preserving(flow_matrix):
    """
    Generate a directed configuration model preserving both in-degree and out-degree distributions.
    
    Parameters:
    -----------
    flow_matrix : numpy.ndarray
        Mobility flow matrix
        
    Returns:
    --------
    new_adj_matrix : numpy.ndarray
        Flow matrix for the generated network
    """
    # Ensure input is numpy array
    adj_matrix = np.array(flow_matrix)
    
    # Calculate in-degrees and out-degrees (preserve both)
    in_degrees = np.sum(adj_matrix > 0, axis=0)
    out_degrees = np.sum(adj_matrix > 0, axis=1)
    
    # Create directed configuration model
    G = nx.directed_configuration_model(in_degrees.tolist(), out_degrees.tolist(), create_using=nx.DiGraph())
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    
    # Create new adjacency matrix
    new_adj_matrix = np.zeros_like(adj_matrix)
    
    # Assign weights using the same approach as the provided function
    for node in range(len(adj_matrix)):
        # Get original outgoing edge weights
        original_weights = adj_matrix[node][adj_matrix[node] > 0]
        if len(original_weights) == 0:
            continue
            
        # Get new outgoing edges
        new_edges = list(G.out_edges(node))
        if len(new_edges) == 0:
            continue
            
        # If new edges count differs from original weights count, adjust
        if len(new_edges) > len(original_weights):
            # If more new edges, repeat original weights
            weights = np.tile(original_weights, (len(new_edges) // len(original_weights)) + 1)[:len(new_edges)]
        else:
            # If fewer new edges, select a random subset of original weights
            weights = np.random.choice(original_weights, size=len(new_edges), replace=False)
            
        # Shuffle weights
        np.random.shuffle(weights)
        
        # Assign weights to new edges
        for (u, v), weight in zip(new_edges, weights):
            new_adj_matrix[u, v] = weight
    
    return new_adj_matrix


def gravity_model(dist_matrix, sum_in, sum_out, beta=2.0):
    """
    Generate a gravity model network.
    
    Parameters:
    -----------
    dist_matrix : numpy.ndarray
        Distance matrix between nodes
    sum_in : numpy.ndarray
        Sum of inflows for each node
    sum_out : numpy.ndarray
        Sum of outflows for each node
    beta : float
        Distance decay exponent
        
    Returns:
    --------
    flow_matrix : numpy.ndarray
        Flow matrix generated by the gravity model
    """
    # Make a copy of distance matrix to avoid modifying original
    dist_matrix = dist_matrix.copy()
    
    # Avoid division by zero
    np.fill_diagonal(dist_matrix, 1)
    
    # Compute flow matrix using gravity model formula
    flow_matrix = np.outer(sum_out, sum_in) / (dist_matrix ** beta)
    
    # Set diagonal elements to zero (no self-loops)
    np.fill_diagonal(flow_matrix, 0)
    
    return flow_matrix

def weighted_random_percolation_threshold(flow_matrix, percentile=2, num_trials=10):
    """
    Calculate the percolation threshold based on weighted random sampling of edges.
    
    Parameters:
    -----------
    flow_matrix : numpy.ndarray
        Flow matrix to analyze
    percentile : int
        Percentile of nodes to remove based on degree
    num_trials : int
        Number of random trials to perform and average the results
        
    Returns:
    --------
    float
        Average number of edges needed for strong connectivity
    """
    # Set diagonal elements to zero
    np.fill_diagonal(flow_matrix, 0)
    
    # Create directed graph from flow matrix
    G = nx.from_numpy_array(flow_matrix, create_using=nx.DiGraph)
    
    # Convert graph to numpy adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    numpy_array = adj_matrix.toarray()
    
    # Convert original flows to binary
    flows_binary = (numpy_array > 0).astype(int)
    
    # Calculate degrees (sum of in and out connections)
    degrees_all = np.sum(np.logical_or(flows_binary, flows_binary.T), axis=1)
    
    # Determine percentile threshold
    percentile_threshold = np.percentile(degrees_all, percentile)
    
    # Find nodes to remove (those with degree below threshold)
    nodes_to_remove = np.where(degrees_all <= percentile_threshold)[0]
    print(f"Removing {len(nodes_to_remove)} nodes ({percentile}% with lowest degree)")
    
    # Remove these nodes from the matrix
    reduced_matrix = np.delete(numpy_array, nodes_to_remove, axis=0)
    reduced_matrix = np.delete(reduced_matrix, nodes_to_remove, axis=1)
    
    # Create directed graph from reduced matrix
    reduced_G = nx.from_numpy_array(reduced_matrix, create_using=nx.DiGraph)
    
    # Find the largest strongly connected component
    scc = list(nx.strongly_connected_components(reduced_G))
    largest_scc = max(scc, key=len)
    strongly_connected_G = reduced_G.subgraph(largest_scc).copy()
    
    print(f"Largest strongly connected component has {len(largest_scc)} nodes")
    
    # Calculate total outgoing weight for each node
    node_total_weights = {}
    for node in strongly_connected_G.nodes():
        out_edges = strongly_connected_G.out_edges(node, data='weight')
        node_total_weights[node] = sum(weight for _, _, weight in out_edges)
    
    # Run multiple trials and average the results
    thresholds = []
    
    for trial in range(num_trials):
        # For each node, create cumulative probability distribution of edges
        node_edge_probabilities = {}
        for node in strongly_connected_G.nodes():
            out_edges = list(strongly_connected_G.out_edges(node, data='weight'))
            if not out_edges or node_total_weights[node] == 0:
                node_edge_probabilities[node] = []
                continue
                
            # Calculate probability for each edge
            probs = [weight/node_total_weights[node] for _, _, weight in out_edges]
            edges = [(u, v) for u, v, _ in out_edges]
            node_edge_probabilities[node] = list(zip(edges, probs))
        
        # Initialize empty graph
        H = nx.DiGraph()
        H.add_nodes_from(strongly_connected_G.nodes())
        
        # Edge count
        total_edges = 0
        max_possible_edges = strongly_connected_G.number_of_edges()
        edge_count = 0
            
        # Add edges until graph becomes strongly connected
        while not nx.is_strongly_connected(H) and total_edges < max_possible_edges:
            edges_to_add = []
            
            # For each node, randomly select an edge based on weight probability
            for node in H.nodes():
                if not node_edge_probabilities[node]:
                    continue
                    
                # Select edge based on probability distribution
                remaining_edges = [(e, p) for e, p in node_edge_probabilities[node] 
                                 if not H.has_edge(e[0], e[1])]
                
                if remaining_edges:
                    edges, probs = zip(*remaining_edges)
                    # Normalize probabilities
                    probs = np.array(probs) / sum(probs)
                    chosen_edge = np.random.choice(len(edges), p=probs)
                    edges_to_add.append(edges[chosen_edge])
            
            # Add selected edges
            for u, v in edges_to_add:
                if not H.has_edge(u, v):
                    H.add_edge(u, v, weight=strongly_connected_G[u][v]['weight'])
                    total_edges += 1
            
            # Exit if no more edges can be added
            if not edges_to_add:
                break
                
            edge_count += 1
        
        print(f"Trial {trial+1}: {edge_count} edges needed for strong connectivity")
        thresholds.append(edge_count)
    
    # Return average threshold
    avg_threshold = sum(thresholds) / len(thresholds)
    print(f"Average threshold across {num_trials} trials: {avg_threshold}")
    return avg_threshold

def find_percolation_threshold(flow_matrix, percentile=2):
    """
    Find the percolation threshold k* - the minimum k needed for the network to be strongly connected.
    
    Parameters:
    -----------
    flow_matrix : numpy.ndarray
        Flow matrix to analyze
    percentile : int
        Percentile of nodes to remove based on degree (default: 2%)
        
    Returns:
    --------
    min_k : int
        Percolation threshold (minimum k for strong connectivity)
    """
    # Ensure matrix is numpy array
    numpy_array = np.array(flow_matrix)
    
    # Set the diagonal elements to zero
    np.fill_diagonal(numpy_array, 0)
    
    # Convert the original flows to binary
    flows_binary = (numpy_array > 0).astype(int)
    
    # Calculate the degrees (sum of in and out connections)
    degrees_all = np.sum(np.logical_or(flows_binary, flows_binary.T), axis=1)
    
    # Determine the percentile threshold
    percentile_threshold = np.percentile(degrees_all, percentile)
    
    # Find nodes to remove (those with degree below threshold)
    nodes_to_remove = np.where(degrees_all <= percentile_threshold)[0]
    
    # Remove these nodes from the matrix
    reduced_matrix = np.delete(numpy_array, nodes_to_remove, axis=0)
    reduced_matrix = np.delete(reduced_matrix, nodes_to_remove, axis=1)
    
    # Create directed graph
    reduced_G = nx.from_numpy_array(reduced_matrix, create_using=nx.DiGraph)
    
    # Find the largest strongly connected component
    scc = list(nx.strongly_connected_components(reduced_G))
    largest_scc = max(scc, key=len)
    strongly_connected_G = reduced_G.subgraph(largest_scc).copy()
    
    num_units = len(strongly_connected_G.nodes())
    
    # Binary search to find minimum k for strong connectivity
    low = 1
    high = num_units
    
    while low < high:
        mid = low + (high - low) // 2
        
        if check_strongly_connected_k(strongly_connected_G, mid):
            high = mid
        else:
            low = mid + 1
            
    # Final verification
    if check_strongly_connected_k(strongly_connected_G, low):
        return low
    else:
        return -1


def check_strongly_connected_k(graph, k):
    """
    Check if the graph remains strongly connected when only top k outgoing edges are kept for each node.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        Directed graph to check
    k : int
        Number of most frequent destinations to keep for each node
        
    Returns:
    --------
    bool
        True if the resulting graph is strongly connected, False otherwise
    """
    # Create a new graph keeping only top k edges per node
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(graph.nodes())
    
    for node in graph.nodes():
        # Get outgoing edges and their weights
        edges = sorted(graph.out_edges(node, data=True), key=lambda x: x[2]['weight'], reverse=True)
        
        # Keep only top k edges (or all if fewer than k)
        k_actual = min(k, len(edges))
        for i in range(k_actual):
            u, v, data = edges[i]
            new_graph.add_edge(u, v, weight=data['weight'])
            
    # Check if new graph is strongly connected
    return nx.is_strongly_connected(new_graph)


def run_simulation(model_type, flow_matrix=None, dist_matrix=None, num_nodes=None,output_file=None, **model_params):
    """
    Run simulation with specified model and parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model to use: 'random', 'heterogeneous', 'config_in', 'config_out', 'config_both', 'gravity'
    flow_matrix : numpy.ndarray, optional
        Flow matrix for reference (required for configuration models)
    dist_matrix : numpy.ndarray, optional
        Distance matrix (required for gravity model)
    num_nodes : int, optional
        Number of nodes (required for random models if flow_matrix not provided)
    output_file : str, optional
        File to save results
    **model_params : dict
        Additional parameters for the specific model
        
    Returns:
    --------
    results : pandas.DataFrame
        DataFrame with simulation results
    """
    results = []
    
    # Determine number of nodes
    if flow_matrix is not None:
        num_nodes = flow_matrix.shape[0]
    elif num_nodes is None:
        raise ValueError("Either flow_matrix or num_nodes must be provided")
    
    # Get inflow and outflow sums for gravity model
    if model_type == 'gravity' and flow_matrix is not None:
        sum_in = flow_matrix.sum(axis=0)
        sum_out = flow_matrix.sum(axis=1)

    if model_type =="probabilistic_percolation":
        min_k = weighted_random_percolation_threshold(flow_matrix)
        result = {
            'model_type': model_type,
            'num_nodes': num_nodes,
            'min_k': min_k,
            **model_params
        }
        result_df = pd.DataFrame(result)
        return result_df
    
    # Generate network based on model type
    if model_type == 'random':
        if flow_matrix is None:
            raise ValueError("flow_matrix is required for popularity model")
        adj_matrix = generate_random_network(flow_matrix)
    
    elif model_type == 'config_out':
        if flow_matrix is None:
            raise ValueError("flow_matrix is required for configuration model")
        adj_matrix = configuration_model_fixed_out_degree(flow_matrix)
    
    elif model_type == 'config_both':
        if flow_matrix is None:
            raise ValueError("flow_matrix is required for configuration model")
        adj_matrix = configuration_model_degree_preserving(flow_matrix)
    
    elif model_type == 'gravity':
        if dist_matrix is None or flow_matrix is None:
            raise ValueError("dist_matrix and flow_matrix are required for gravity model")
        beta = model_params.get('beta', 2.0)
        adj_matrix = gravity_model(dist_matrix, sum_in, sum_out, beta)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Find percolation threshold
    min_k = find_percolation_threshold(adj_matrix)
    
    # Store result
    result = {
        'model_type': model_type,
        'num_nodes': num_nodes,
        'min_k': min_k,
        **model_params
    }
    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    
    return result_df


def run_city_simulations(city_list, year_range, month_range, model_type, 
                        data_dir=None, output_dir=None, 
                        **model_params):
    """
    Run simulations for multiple cities across different time periods.
    
    Parameters:
    -----------
    city_list : list
        List of city IDs to process
    year_range : list or range
        Years to process
    month_range : list or range
        Months to process
    model_type : str
        Type of model to use
    data_dir : str
        Directory containing flow matrix data
    output_dir : str
        Directory to save results
    **model_params : dict
        Additional parameters for the specific model
        
    Returns:
    --------
    all_results : pandas.DataFrame
        Combined DataFrame with all simulation results
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified")
        
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'simulation_results')
        
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for city in city_list:
        for year in year_range:
            for month in month_range:
                # Create formatted month string (zero-padded)
                month_str = str(month).zfill(2)
                
                # Construct file paths
                flow_file = os.path.join(data_dir, f"cbg_visit_{year}-{month_str}_{city}.npy")
                
                if not os.path.exists(flow_file):
                    print(f"Flow file not found: {flow_file}")
                    continue
                
                # Load flow matrix
                flow_matrix = np.load(flow_file)
                
                # Load distance matrix if using gravity model
                if model_type == 'gravity':
                    dist_file = os.path.join(data_dir, f"precomputed_distances_{city}.npy")
                    if not os.path.exists(dist_file):
                        print(f"Distance file not found: {dist_file}")
                        continue
                    dist_matrix = np.load(dist_file)
                else:
                    dist_matrix = None
                
                print(f"Processing: City {city}, Year {year}, Month {month}")
                
                # Run simulation
                try:
                    results = run_simulation(
                        model_type=model_type,
                        flow_matrix=flow_matrix,
                        dist_matrix=dist_matrix,
                        **model_params
                    )
                    
                    # Add city, year, and month information
                    results['city'] = city
                    results['year'] = year
                    results['month'] = month
                    
                    all_results.append(results)
                except Exception as e:
                    print(f"Error processing {city}, {year}, {month}: {e}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_file = os.path.join(output_dir, f"{model_type}_results.csv")
        combined_results.to_csv(combined_file, index=False)
        
        return combined_results
    else:
        print("No results generated")
        return None


if __name__ == "__main__":
    # Example usage
    
    # List of models to run
    models = [
        # {'type': 'random', 'params': {}},
        {'type': 'config_out', 'params': {}},
        #{'type': 'config_both', 'params': {}},
        # {'type': 'gravity', 'params': {'beta': 1.6}}
    ]
    
    # City IDs, years, and months to process
    city_list = range(1,9)  # Adjust based on your data
    year_range = range(2018, 2022)
    month_range = range(1, 13)
    
    # Base directory containing flow data
    data_dir = "./data0/Mobility"
    
    # Run simulations for each model
    for model_config in models:
        model_type = model_config['type']
        model_params = model_config['params']
        
        print(f"\nRunning simulations for model: {model_type}")
        
        results = run_city_simulations(
            city_list=city_list,
            year_range=year_range,
            month_range=month_range,
            model_type=model_type,
            data_dir=data_dir,
            output_dir=f"simulation_results_{model_type}",
            **model_params
        )
        
        print(f"Completed simulations for {model_type}")