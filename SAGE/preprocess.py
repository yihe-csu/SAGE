import os
import torch
import random
import numpy as np
import scanpy as sc
import scipy
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import pandas as pd
from torch.backends import cudnn
from sklearn.neighbors import NearestNeighbors 
from sklearn.cluster import KMeans
from tqdm import tqdm
from . import utils
from .utils import mclust_R_f, refine_label
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg


def construct_interaction_KNN(adata, n_neighbors=7):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # Compute K nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)  
    _, indices = nbrs.kneighbors(position)

    # Build adjacency matrix coordinate indices
    row_idx = np.repeat(np.arange(n_spot), n_neighbors)  # Source node indices
    col_idx = indices[:, 1:].flatten()  # Target node indices (excluding self)

    # Build adjacency matrix (initially all 1)
    data = np.ones_like(row_idx)
    interaction_sparse = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_spot, n_spot))

    # Ensure symmetry (undirected graph)
    interaction_sparse = interaction_sparse.maximum(interaction_sparse.T)

    # Store in adata.obsm['adj']
    adata.obsm['graph_neigh'] = interaction_sparse.toarray()  # Store as numpy.ndarray
    adata.obsm['adj'] = interaction_sparse

    print('>>> Graph_coord constructed!')


def compute_cosine_similarity_matrix(adata, use_rep='X_pca'):
    embedding = adata.obsm[use_rep]
    cos_sim_mat = cosine_similarity(embedding)
    np.fill_diagonal(cos_sim_mat, 0) 
    adata.obsm["cos_sim_mat"] = cos_sim_mat
    return cos_sim_mat

def optimize_cut_adj_matrix(adata, threshold=0.2, min_neighbors=1):
    """
    Optimize adjacency matrix by removing cross-domain edges with low consensus probability,
    ensuring each node retains at least min_neighbors neighbors.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Initial adjacency matrix (N, N).
        pairwise_probabilities (numpy.ndarray): Consensus probability matrix (N, N).
        threshold (float): Edges with probability below this value will be removed.
        min_neighbors (int): Minimum number of neighbors to retain for each node.

    Returns:
        scipy.sparse.csr_matrix: Optimized adjacency matrix (N, N).
        dict: Optimization info, including number of removed edges and neighbors per node.
    """
    adj_matrix = adata.obsm["adj"].copy()
    pairwise_probabilities = adata.obsm["consensus_freq"].copy()
    n = adj_matrix.shape[0]

    cut_spots = np.zeros(n, dtype=int)

    # Convert adjacency matrix to COO format for easy access
    coo_adj = adj_matrix.tocoo()

    # Track number of removed edges
    removed_edges = 0

    # Iterate over each node
    for i in range(n):
        # Get neighbors of current node
        neighbors = coo_adj.col[coo_adj.row == i]
        if len(neighbors) <= min_neighbors:
            continue  # Ensure basic connectivity, avoid removing too many neighbors

        # Get consensus probabilities with neighbors
        neighbor_probs = pairwise_probabilities[i, neighbors]

        # Find neighbors with probability below threshold
        low_prob_neighbors = neighbors[neighbor_probs < threshold]

        # Ensure at least min_neighbors remain after removal
        if len(neighbors) - len(low_prob_neighbors) < min_neighbors:
            keep_count = min_neighbors - (len(neighbors) - len(low_prob_neighbors))
            low_prob_neighbors = low_prob_neighbors[:-keep_count]

        if len(low_prob_neighbors) > 0:
            cut_spots[i] = 1

        # Remove these edges
        for j in low_prob_neighbors:
            adj_matrix[i, j] = 0
            adj_matrix[j, i] = 0  # Ensure symmetry
            removed_edges += 1

    # Ensure adjacency matrix remains sparse
    adj_matrix.eliminate_zeros()

    # Record optimization info
    optimization_info = {
        "removed_edges": removed_edges,
        "final_neighbors_per_node": np.array(adj_matrix.sum(axis=1)).flatten()
    }

    return adj_matrix, optimization_info, cut_spots


################# feature graph #################
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

def construct_interaction_by_feat(adata, n_neighbors=7):
    adj_matrix = adata.obsm["adj"].copy()
    pairwise_probabilities = adata.obsm["consensus_freq"].copy()
    cosine_sim_matrix = adata.obsm["cos_sim_mat"].copy()

    optimized_add_adj_matrix, optimization_info = optimize_add_adj_matrix(adj_matrix, 
                                                                          pairwise_probabilities, 
                                                                          cosine_sim_matrix, 
                                                                          max_edges_per_spot=n_neighbors)
    # Store the sparse matrix without converting to dense to save memory
   
    print('>>> Graph_feat constructed!')

    return optimized_add_adj_matrix , optimization_info
    



def optimize_add_adj_matrix(adj_matrix, pairwise_probabilities, cosine_sim_matrix, max_edges_per_spot):
    """
    Add edges based on consensus probability and cosine similarity.
    For each node, select top 7 neighbors from both matrices and take the intersection,
    while limiting the maximum number of added edges per node.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Original adjacency matrix (N, N).
        pairwise_probabilities (numpy.ndarray): Consensus probability matrix (N, N).
        cosine_sim_matrix (numpy.ndarray): Cosine similarity matrix (N, N).
        max_edges_per_spot (int): Maximum number of edges to add per node.

    Returns:
        scipy.sparse.csr_matrix: Optimized adjacency matrix (N, N).
        dict: Optimization info, including number of added edges per node and total added edges.
    """

    adj_matrix = adj_matrix.copy().tolil()  # Use LIL format for efficient edge addition
    n = adj_matrix.shape[0]
    
    # Track number of added edges per node
    added_edges_per_spot = np.zeros(n, dtype=int)
    added_edges = 0

    # Iterate over each node
    for i in range(n):
        # 1. Select neighbors with consensus probability == 1
        full_prob_neighbors = np.where(pairwise_probabilities[i] == 1)[0]

        # 2. If less than 7, fill up with highest consensus probability neighbors
        if len(full_prob_neighbors) < 7:
            remaining_neighbors = np.where(pairwise_probabilities[i] < 1)[0]
            sorted_remaining_neighbors = remaining_neighbors[np.argsort(pairwise_probabilities[i][remaining_neighbors])[::-1]]
            prob_neighbors = np.concatenate((full_prob_neighbors, sorted_remaining_neighbors))[:7]
        else:
            prob_neighbors = full_prob_neighbors
            
        sim_neighbors = np.argsort(cosine_sim_matrix[i])[::-1][:7]  # Top 7 by cosine similarity

        # Intersection of both neighbor sets
        common_neighbors = set(prob_neighbors) & set(sim_neighbors)
        
        # Sort by consensus probability, descending
        sorted_neighbors = sorted(common_neighbors, key=lambda j: pairwise_probabilities[i, j], reverse=True)

        for j in sorted_neighbors:
            # Ensure i != j and edge does not already exist
            if i != j and adj_matrix[i, j] == 0:
                # Check if both nodes have not exceeded max added edges
                if added_edges_per_spot[i] < max_edges_per_spot and added_edges_per_spot[j] < max_edges_per_spot:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Ensure symmetry
                    added_edges += 1
                    added_edges_per_spot[i] += 1
                    added_edges_per_spot[j] += 1

    # Ensure result is CSR format
    adj_matrix = adj_matrix.tocsr()
    adj_matrix.eliminate_zeros()

    # Record optimization info
    optimization_info = {
        "added_edges": added_edges,
        "final_added_edges_per_node": added_edges_per_spot
    }

    return adj_matrix, optimization_info



def pre_cluster(adata, method="mclust", n_clusters=7, res=1, dims=25, radius=20, refinement=False):
    if 'highly_variable' not in adata.var.keys():
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        
    if 'X_pca' not in adata.obsm.keys():
        hvg_bool = adata.var['highly_variable']
        hvg_genes = adata.var_names[hvg_bool].tolist()

        pca = PCA(n_components=dims, random_state=42) 
        embedding = pca.fit_transform(adata[:, hvg_genes].X)
        adata.obsm['X_pca'] = embedding

    adata = utils.clustering(adata, 
                            data = adata.obsm['X_pca'][:,:dims], 
                            method=method, 
                            n_clusters=n_clusters, 
                            radius=radius, 
                            res = res,
                            refinement=refinement
                            )
    
    adata.obs['domain']=adata.obs['domain'].astype("int")
    adata.obs['domain']=adata.obs['domain'].astype("category")
    adata.obs['pre_domain']=adata.obs['domain']

    print(">>> Pre-clustering finished successfully!")
    print(f">>> Clustering labels stored in `adata.obs['pre_domain'] and ['{method}']`.")
    return adata

def topics_selection(adata, n_topics=10):

    if 'genes_mri' not in adata.var.keys():
        print(">>> Start the genes_mri calculation ...")
        # Start timing
        start_time = time.time()

        sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=7)
        genes_MRI = sc.metrics.morans_i(adata)
        adata.var["genes_mri"] = genes_MRI

        # End timing
        end_time = time.time()
        # Print calculation time
        print(f">>> Finish genes_mri calculation! Time taken: {end_time - start_time:.2f} seconds")
        print(f">>> adata.var['genes_mri'] generated!")

    print(">>> Step1: Starting NMF calculation...")
    adata = utils.perform_nmf(adata, n_topics=n_topics, random_state=42, max_iter=500)

    print(">>> Step2: Filtering based on Topics Moran's I values...")
    adata, topic_mri_list = utils.calculate_MRI_for_topics(adata)
    adata = utils.filter_and_update_nmf_topics(adata, topic_mri_list, threshold=0.2)
    
    print(">>> Step3: Filtering based on Random Forest importances...")
    sorted_indices, sorted_importance = utils.fit_rf_and_extract_importance(adata)
    final_topics_list = utils.filter_top_topics_by_importance(adata, threshold=0.1)

    print(f">>> {len(final_topics_list)} topics selected !")

    print(">>> adata.obsm['W_nmf'], adata.varm['H_nmf'] generate!")
    print(">>> adata.uns['topic_mri_list'] generate!")
    print(">>> adata.uns['sorted_indices_imp'] generate!")
    print(">>> adata.uns['final_topics'] generate!")



def genes_selection(adata, n_genes=3000, lower_p=15, upper_p=95):
    adata = utils.get_top_n_genes_for_topics(adata, n=100)
    svgs_set, no_svgs_set, sorted_rank_matrix = utils.select_genes_from_hvgs(
        adata, total_genes=n_genes, lower_p=lower_p, upper_p=upper_p
    )
    adata = utils.create_genes_add_column(adata, n_max=n_genes // 4)
    adata = utils.create_genes_del_column(adata, n_max=n_genes // 4)
    adata = utils.create_HSGs_column(adata)
    selected_genes = adata.var.loc[(adata.var["HSG"] == True)].index
    print(">>> HSGs selected!")
    print(">>> adata.uns['HSG'] generate!")

    ##### Generate marker_genes_dict (by the way) #########
    # 1. Compute gene-topic correlation matrix
    utils.compute_gene_topic_corr(adata, method="pearson")
    print(">>> adata.varm['gene_topic_corr'] generate! (method='pearson')")
    # 2. Generate marker genes dictionary
    utils.generate_marker_genes_dict(adata, corr_threshold=0.2)
    print(">>> adata.uns['marker_genes_all_dict'] generate! (corr_threshold=0.2)")

    # return selected_genes, svgs_set, no_svgs_set, sorted_rank_matrix


def get_feature(adata):
    """Extracts feature matrix from selected genes and generates augmented features."""

    HSG_bool = adata.var['HSG']
    HSGs = adata.var_names[HSG_bool].tolist()

    # Extract feature matrix, keep sparse format if possible to save memory
    feat = adata[:, HSGs].X

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=256, random_state=42) 
    # embedding = pca.fit_transform(feat)
    # feat = embedding

    if scipy.sparse.issparse(feat):
        feat = feat.toarray()  # Convert to dense matrix only if needed

    # Store features
    adata.obsm['feat'] = feat

    # print(">>>Feature extraction completed! ")


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    
    # Handle zero-degree nodes safely
    d_inv_sqrt = np.zeros_like(rowsum, dtype=float)
    nonzero_mask = rowsum > 0
    d_inv_sqrt[nonzero_mask] = np.power(rowsum[nonzero_mask], -0.5)

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return adj_normalized.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    

def consensus_clustering(adata, method='leiden', resolution_range=(1.0, 5.0, 0.1), n_neighbors=7, use_rep='X_pca', dims=25, radius=20, refinement=True):

    all_clusters = []         # Store all clustering results
    cluster_labels = []       # Store labels for DataFrame
    param_list = []           # Store each tested parameter value

    start_time = time.time()
    
    # Ensure neighbor graph is computed (Leiden only)
    if method == 'leiden':
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    print(f">>> Starting {method} clustering...")

    # Iterate through resolution or cluster number range
    for param in tqdm(np.arange(*resolution_range), desc=f"Running {method} clustering"):
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=param, random_state=0)
            labels = adata.obs['leiden'].astype(int).values
        
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=int(param), random_state=0, n_init=10)
            labels = kmeans.fit_predict(adata.obsm[use_rep])
        
        elif method == 'mclust': 
            labels = mclust_R_f(data=adata.obsm[use_rep][:, :dims], n_clusters=int(param), random_seed=42)
        else:
            raise ValueError("Method must be 'leiden', 'kmeans', or 'mclust'.")

        all_clusters.append(labels)
        cluster_labels.append(labels)
        param_list.append(param)

    print(f">>> {method} clustering finished. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Store clustering results in a DataFrame
    clusters_df = pd.DataFrame(np.array(cluster_labels).T, index=adata.obs.index,
                               columns=[f"{method}_{p:.2f}" for p in param_list])
    
    # Compute consensus matrix
    print(">>> Computing consensus matrix...")
    start_time = time.time()
    num_objects = len(all_clusters[0])
    similarity_matrix = np.zeros((num_objects, num_objects), dtype=int)

    for labels in all_clusters:
        labels = np.array(labels)
        bool_matrix = labels[:, None] == labels
        similarity_matrix += bool_matrix

    pairwise_probabilities = similarity_matrix / len(all_clusters)
    np.fill_diagonal(pairwise_probabilities, 0)

    print(f">>> Consensus matrix computed. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Perform Leiden clustering on the consensus matrix
    print(">>> Performing final Leiden clustering on consensus matrix...")
    G = ig.Graph.Adjacency((pairwise_probabilities > 0).tolist())
    G.es['weight'] = pairwise_probabilities[pairwise_probabilities.nonzero()]
    partition = leidenalg.find_partition(G, leidenalg.SurpriseVertexPartition, weights='weight')
    print(">>> Consensus clustering completed.")

    con_domain = np.array(partition.membership)
    adata.obs["pre_domain"] = con_domain
    if refinement:
        new_type = refine_label(adata, radius, key='pre_domain')
        adata.obs['pre_domain'] = new_type
    clusters_df[f'{method}_con'] = con_domain

    # Merge with existing clustering results if present
    if "clusters_results" in adata.obsm:
        adata.obsm["clusters_results"] = pd.concat([adata.obsm["clusters_results"], clusters_df], axis=1)
    else:
        adata.obsm["clusters_results"] = clusters_df

    adata.obs["pre_domain"] = adata.obs["pre_domain"].astype("category")
    adata.obsm["consensus_freq"] = pairwise_probabilities

    print(">>> adata.obs['pre_domain'] generated!")
    print(">>> adata.obsm['consensus_freq'] generated!")
    print(">>> adata.obsm['clusters_results'] generated!")


def calculate_add_side_accuracy(adj_matrix_before, adj_matrix_after, ground_truth):
    """
    Calculate the accuracy of added edges, i.e., the proportion of newly added edges that connect nodes of the same class.

    Args:
        adj_matrix_before (scipy.sparse.csr_matrix): Original adjacency matrix (N, N).
        adj_matrix_after (scipy.sparse.csr_matrix): Optimized adjacency matrix (N, N).
        ground_truth (numpy.ndarray): Class labels for each node (N,).

    Returns:
        float: Accuracy of the added edges.
    """
    # Convert adjacency matrices to COO format for easy access
    coo_before = adj_matrix_before.tocoo()
    coo_after = adj_matrix_after.tocoo()

    # Extract newly added edges
    added_edges = []
    for i, j in zip(coo_after.row, coo_after.col):
        if adj_matrix_before[i, j] == 0:  # Only consider newly added edges
            added_edges.append((i, j))

    # Count how many added edges connect nodes of the same class
    correct_additions = 0
    for i, j in added_edges:
        if ground_truth[i] == ground_truth[j]:
            correct_additions += 1

    # Calculate accuracy
    if len(added_edges) > 0:
        accuracy = correct_additions / len(added_edges)
    else:
        accuracy = 0.0  # If no new edges, accuracy is 0

    return accuracy

def calculate_cut_side_accuracy(adj_matrix_before, adj_matrix_after, ground_truth):
    """
    Calculate the accuracy of cut edges, i.e., the proportion of removed edges that connect nodes of different classes.

    Args:
        adj_matrix_before (np.ndarray or scipy.sparse.csr_matrix): Original adjacency matrix (N, N)
        adj_matrix_after (np.ndarray or scipy.sparse.csr_matrix): Optimized adjacency matrix (N, N)
        ground_truth (np.ndarray): Ground truth class labels for nodes (N,)

    Returns:
        float: Accuracy of cut-side operation
    """

    # Get removed edges
    edges_before = set(tuple(sorted((i, j))) for i, j in zip(*adj_matrix_before.nonzero()) if i < j)
    edges_after = set(tuple(sorted((i, j))) for i, j in zip(*adj_matrix_after.nonzero()) if i < j)
    removed_edges = edges_before - edges_after  # Find edges that were removed

    # Count how many removed edges connect nodes of different classes
    correct_removed = 0
    total_removed = len(removed_edges)

    for i, j in removed_edges:
        if ground_truth[i] != ground_truth[j]:
            correct_removed += 1  # If nodes are of different classes, consider as correct removal

    # Calculate accuracy
    if total_removed > 0:
        accuracy = correct_removed / total_removed
    else:
        accuracy = 0.0  # If no edges removed, accuracy is 0

    return accuracy


def optimize_graph_topology(
    adata,
    n_neig_coord=6,
    cut_side_thr=0.1,
    min_neighbor=2,
    n_neig_feat=20,
    verbose=False,
    use_rep=None
):

    # ----- Step 1: Construct initial coord-based graph -----
    construct_interaction_KNN(adata, n_neighbors=n_neig_coord)
    
    adj_opt, opt_info_1, cut_spots = optimize_cut_adj_matrix(
        adata,
        threshold=cut_side_thr,
        min_neighbors=min_neighbor
    )
    adata.obsm["adj_opt"] = adj_opt
    adata.obs["cut_spots"] = cut_spots
    adata.obs["cut_spots"] = adata.obs["cut_spots"].astype("category")
    print(">>> adata.obsm['adj_opt'] generate!")

    if verbose:
        if use_rep is None:
            raise ValueError("`use_rep` must be specified when verbose=True.")
        print(f"Removed edges (cut-side): {opt_info_1['removed_edges']}")
        adj = adata.obsm["adj"]
        ground_truth = adata.obs[use_rep].values
        acc_cut = calculate_cut_side_accuracy(adj, adj_opt, ground_truth)
        print(f"Cut-side accuracy: {acc_cut:.3f}")

    # ----- Step 2: Construct feature-based graph -----
    compute_cosine_similarity_matrix(adata)
    adj_feat, opt_info_2 = construct_interaction_by_feat(
        adata,
        n_neighbors=n_neig_feat
    )
    adata.obsm["adj_feat"] = adj_feat
    print(">>> adata.obsm['adj_feat'] generate!")

    if verbose:
        print(f"Added edges (add-side): {opt_info_2['added_edges']}")
        adj = adata.obsm["adj"]
        ground_truth = adata.obs[use_rep].values
        acc_add = calculate_add_side_accuracy(adj, adj_feat, ground_truth)
        print(f"Add-side accuracy: {acc_add:.3f}")
