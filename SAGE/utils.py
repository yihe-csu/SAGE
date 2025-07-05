import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import umap
import scanpy as sc
import ot  # optimal transport
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import time
import os
import re

def mclust_R(adata,data, num_cluster, modelNames='EEE', random_seed=42):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def mclust_R_f(data, n_clusters, random_seed=42):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # import os
    # os.environ['R_HOME'] = 'E:\\R-4.4.1'
    modelNames = 'EEE'

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), n_clusters, modelNames)
    mclust_res = np.array(res[-2])
    
    return mclust_res

def clustering(adata, data, method='mclust', n_clusters=7, radius=50,  res = 1, refinement=False):

    if method == 'mclust':
       adata = mclust_R(adata, data = data, num_cluster=n_clusters,random_seed=42)
       adata.obs['domain'] = adata.obs['mclust']
       print(f">>> Clustering completed using mclust with {n_clusters} clusters.")
       print(f">>> adata.obsm['domain'] & ['mclust'] generate!")

    elif method == 'kmeans':
        Kmeans  = KMeans(n_clusters = n_clusters , random_state = 42, n_init = 10)
        cluster_labels = Kmeans.fit_predict(data)
        adata.obs['Kmeans'] = cluster_labels
        adata.obs['domain'] = adata.obs['Kmeans']
        print(f">>> Clustering completed using Kmeans with {n_clusters} clusters.")
        print(f">>> adata.obsm['domain'] & ['Kmeans'] generate!")

    elif method == 'leiden':
       if 'neighbors' not in adata.uns:
           sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=15)
       sc.tl.leiden(adata, random_state=42, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
       print(f">>> Clustering completed using leiden with res : {res}")
       print(f">>> adata.obsm['domain'] & ['leiden'] generate!")
       
    elif method == 'louvain':
       sc.tl.louvain(adata, random_state=42, resolution=res)
       adata.obs['domain'] = adata.obs['louvain']
       print(f">>> Clustering completed using louvain with res : {res}")
       print(f">>> adata.obsm['domain'] & ['louvain'] generate!")

    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    # adata.obs['label_refined'] = np.array(new_type)
    
    return new_type



def perform_nmf(adata, n_topics=20, random_state=42, max_iter=500):
    """
    Perform Non-negative Matrix Factorization (NMF) on the gene expression data in `adata`.
    
    Parameters:
    adata : AnnData
        The AnnData object containing gene expression data.
    n_topics : int, optional (default=20)
        The number of topics/components for NMF.
    random_state : int, optional (default=42)
        The random state for reproducibility.
    max_iter : int, optional (default=500)
        The maximum number of iterations for the NMF algorithm.
    
    Returns:
    adata : AnnData
        The updated AnnData object with the NMF results stored in `obsm` and `varm`.
    """

    # Start timing
    start_time = time.time()

    # Convert to dense matrix if it's sparse
    X = adata.X.A if hasattr(adata.X, "A") else adata.X
    
    # Define NMF model
    nmf_model = NMF(n_components=n_topics, init='nndsvd', random_state=random_state, max_iter=max_iter)
    
    
    # Perform matrix factorization
    W_nmf = nmf_model.fit_transform(X)  # Spot × Topic matrix
    H_nmf = nmf_model.components_       # Topic × Gene matrix
    
    # Save the results into AnnData object
    adata.obsm["W_nmf"] = W_nmf  # Each spot's topic weights
    adata.varm["H_nmf"] = H_nmf.T  # Each gene's topic contributions

    adata.obsm["W_nmf_norm"] = (W_nmf - np.min(W_nmf, axis=0)) / (np.ptp(W_nmf, axis=0))  # ptp() calculates max - min
    # End timing
    end_time = time.time()

    print(f">>> NMF completed. {n_topics} topics identified. Time taken: {end_time - start_time:.2f} seconds")
    
    return adata


def calculate_MRI_for_topics(adata):
    """
    Calculate Moran's I for each topic extracted via NMF on the expression matrix in the AnnData object.
    
    Parameters:
    adata : AnnData
        The AnnData object containing NMF results and the expression data.
        
    Returns:
    adata : AnnData
        The updated AnnData object with additional `obs` columns for each topic's Moran's I value.
    """


    # Extract the topic matrix H from NMF
    W = adata.obsm["W_nmf"]  # Topic matrix (spots × topics)
    
    # Build adjacency matrix
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=7)

    # Calculate Moran's I for each topic
    topics_MRI = sc.metrics.morans_i(adata, obsm="W_nmf")

    # Store Moran's I value for each topic
    topic_mri_list = []

    # Calculate Moran's I for each topic
    for i in range(W.shape[1]):  # Iterate over all topics
        adata.obs[f"Topic_{i}"] = W[:, i]
        topic_mri_list.append((i , round(topics_MRI[i], 4)))  # Store topic index and Moran's I value
    
    # Store Moran's I list in adata.uns
    adata.uns["topic_mri_list"] = topic_mri_list
    

    return adata, topic_mri_list


def filter_and_update_nmf_topics(adata, topic_mri_list, threshold=0.2):

    filtered_topics = []
    filtered_MRI = []
    
    # Filter topics that meet the condition
    for idx, mri_idx in enumerate(topic_mri_list):
        mri_value = mri_idx[1]
        if mri_value > threshold:
            filtered_topics.append(idx)  # Record column index that meets the condition
            filtered_MRI.append(mri_value)  # Save corresponding Moran's I value

    # Store results as tuples
    filtered_topic_mri_list = [(filtered_topics[i], filtered_MRI[i]) for i in range(len(filtered_topics))]
    
    # Reconstruct W based on filtered columns
    W = adata.obsm["W_nmf"]
    filtered_W = W[:, filtered_topics]

    # Update results in adata
    adata.obsm["W_nmf_filtered"] = filtered_W

    adata.uns["filtered_topic_mri_list"] = filtered_topic_mri_list

    # Convert to DataFrame
    filtered_topic_mri_df = pd.DataFrame(filtered_topic_mri_list, columns=["Topic", "Moran's I"])

    filtered_topic_mri_df = filtered_topic_mri_df.sort_values(by="Moran's I", ascending=False)

    return adata

def fit_rf_and_extract_importance(adata):

    # Get filtered NMF topic matrix
    W_filtered = adata.obsm["W_nmf_filtered"]
    
    # Check if domain label (pre-cluster label) exists
    if "pre_domain" not in adata.obs:
        raise ValueError("The 'pre_domain' column is missing in adata.obs. Please ensure it is computed before calling this function.")
    
    # Get domain label (pre-cluster label)
    y = adata.obs["pre_domain"]
    
    # Fit random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(W_filtered, y)
    
    # Get feature importance
    topic_importance = model.feature_importances_
    
    # Get original topics index
    filtered_topic_mri_list = adata.uns["filtered_topic_mri_list"]
    filtered_topics = [item[0] for item in filtered_topic_mri_list]

    # Sort by feature importance
    sorted_indices = np.argsort(topic_importance)[::-1]
    sorted_importance = topic_importance[sorted_indices]  # Sorted importance scores

    # Save information
    sorted_indices_imp = [(sorted_indices[i],sorted_importance[i]) for i in range(len(sorted_importance))]
    adata.uns["sorted_indices_imp"] = sorted_indices_imp

    return sorted_indices, sorted_importance


def filter_top_topics_by_importance(adata, threshold=0.1):

    # Get sorted_indices_imp information
    sorted_indices_imp = adata.uns.get("sorted_indices_imp", None)
    if sorted_indices_imp is None:
        raise ValueError("Cannot find 'sorted_indices_imp' in adata.uns, please make sure it is computed and stored before calling this function.")

    # Get sorted topic indices and importance values
    sorted_indices = [item[0] for item in sorted_indices_imp]  # Sorted indices
    sorted_importance = [item[1] for item in sorted_indices_imp]  # Sorted importance

    # Calculate the threshold to remove the last 10%
    num_topics_to_remove = int(len(sorted_importance) * threshold)
    
    # Calculate the number of remaining topics
    num_topics_to_keep = len(sorted_importance) - num_topics_to_remove

    # Get indices of remaining topics
    filtered_indices = sorted_indices[:num_topics_to_keep]
    
    # Get filtered_topic_mri_list and get topic names in original order
    filtered_topic_mri_list = adata.uns.get("filtered_topic_mri_list", [])
    if not filtered_topic_mri_list:
        raise ValueError("Cannot find 'filtered_topic_mri_list' in adata.uns, please make sure it is computed and stored before calling this function.")
    
    filtered_topics = [filtered_topic_mri_list[i][0] for i in filtered_indices]

    # Update adata.uns, save filtered topics
    adata.uns["final_topics"] = filtered_topics
    
    # Return remaining topics
    return filtered_topics


def get_top_n_genes_for_topics(adata, n=500, use_rep="H_nmf"):

    # Select matrix to use
    if use_rep not in adata.varm:
        raise ValueError(f"Cannot find '{use_rep}' matrix in adata.varm. Options: 'H_nmf', 'gene_topic_corr'")
    
    rep_matrix = adata.varm[use_rep]  # Selected gene importance matrix (gene × topic)

    # Get gene names
    gene_names = adata.var_names

    # Store top n genes for each topic
    top_genes_all_dict = {}
    for topic_idx in range(rep_matrix.shape[1]):  # Iterate over all topics
        # Get gene importance for current topic
        topic_gene_scores = rep_matrix[:, topic_idx]

        # Select top n genes
        sorted_indices = np.argsort(topic_gene_scores)[::-1]  # Descending order
        top_gene_indices = sorted_indices[:n]  # Top n indices

        # Get gene names
        top_genes = gene_names[top_gene_indices]

        # Store in dictionary
        top_genes_all_dict[str(topic_idx)] = top_genes.tolist()

    # Store in adata.uns
    if use_rep == "H_nmf":
        adata.uns["top_genes_all_dict"] = top_genes_all_dict
    if use_rep == "gene_topic_corr":
        adata.uns["corr_top_genes_all_dict"] = top_genes_all_dict
    return adata    


def select_genes_from_hvgs(adata, total_genes=3000, lower_p=15, upper_p=95):

    topics_final_list = adata.uns["final_topics"].copy()
    n_topics=len(topics_final_list)

    sorted_indices_imp = adata.uns["sorted_indices_imp"]
    sorted_indices, sorted_importance = zip(*sorted_indices_imp)
    sorted_indices = list(sorted_indices)
    sorted_importance = list(sorted_importance)

    # Get topic indices with highest feature importance
    sorted_indices = sorted_indices[:n_topics]  # Get top n_topics topic indices
    top_indices_weight = sorted_importance[:n_topics]  # Get corresponding weights
    top_indices_weight = np.array(top_indices_weight)

    H_nmf_filtered = adata.varm["H_nmf"][:,topics_final_list]

    # Calculate sum of each column
    col_sums = H_nmf_filtered.sum(axis=0, keepdims=True)  # axis=0 means sum by column, keepdims=True keeps matrix shape
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    # Normalize each column
    H_nmf_filtered_norm = H_nmf_filtered / col_sums  # Normalize each column

    # Calculate rank matrix
    sorted_rank_matrix = get_sorted_rank_matrix(H_nmf_filtered_norm)
    
    genes_counts_list = np.full(len(top_indices_weight), total_genes, dtype=int) 

    percentile_genes = get_genes_at_percentiles(adata, H_nmf_filtered_norm, sorted_indices, genes_counts_list, sorted_rank_matrix, lower_percentile=lower_p, upper_percentile=upper_p)
    
    svgs_set = select_SVGs(adata, percentile_genes, sorted_rank_matrix)
    no_svgs_set = select_no_SVGs(adata, percentile_genes,sorted_rank_matrix)

    adata.var["Svgs"] = adata.var.index.isin(svgs_set)
    adata.var["no_Svgs"] = adata.var.index.isin(no_svgs_set)
    adata.varm["sorted_rank_matrix"] = sorted_rank_matrix

    print(f"number_svgs:{len(svgs_set)}")
    print(f"number_no_svgs:{len(no_svgs_set)}")

    return svgs_set , no_svgs_set,  sorted_rank_matrix



def create_genes_add_column(adata, n_max):

    # Ensure svgs_rank and highly_variable columns exist
    if "svgs_rank" not in adata.var or "highly_variable" not in adata.var:
        raise ValueError("adata.var must contain 'svgs_rank' and 'highly_variable' columns")

    # Get the number of genes with Svgs column True
    svgs_true_count = adata.var["Svgs"].sum()

     # Compare the number of genes with Svgs True and n
    if svgs_true_count < n_max:
        # If the number of genes with Svgs True is less than n, select all genes with Svgs True
        top_n_genes_idx = adata.var.loc[adata.var["Svgs"]].index
    else:
        # If the number of genes with Svgs True is greater than or equal to n, select the top n genes
        top_n_genes_idx = adata.var.loc[adata.var["Svgs"]].sort_values(by="svgs_rank", ascending=True).index[:n_max]

    # Initialize genes_add column as False
    adata.var["genes_add"] = False

    # Update highly_variable and top n svgs_rank genes boolean values
    adata.var.loc[adata.var["highly_variable"], "genes_add"] = True
    adata.var.loc[top_n_genes_idx, "genes_add"] = True

    return adata

def create_genes_del_column(adata, n_max):

    # Ensure no_svgs_rank and highly_variable columns exist
    if "no_svgs_rank" not in adata.var or "highly_variable" not in adata.var:
        raise ValueError("adata.var must contain 'no_svgs_rank' and 'highly_variable' columns")

    # Get the number of genes with no_Svgs column True
    no_svgs_true_count = adata.var["no_Svgs"].sum()

    # Compare the number of genes with no_Svgs True and n
    if no_svgs_true_count < n_max:
        # If the number of genes with no_Svgs True is less than n, select all genes with no_Svgs True
        top_n_genes_idx = adata.var.loc[adata.var["no_Svgs"]].index
    else:
        # If the number of genes with no_Svgs True is greater than or equal to n, select the top n genes
        top_n_genes_idx = adata.var.loc[adata.var["no_Svgs"]].sort_values(by="no_svgs_rank", ascending=False).index[:n_max]

    # Initialize genes_del column as False
    adata.var["genes_del"] = False

    # Update highly_variable and top n svgs_rank genes boolean values
    adata.var.loc[adata.var["highly_variable"], "genes_del"] = True
    adata.var.loc[top_n_genes_idx, "genes_del"] = False

    return adata

def create_HSGs_column(adata):

    if "genes_del" not in adata.var or "highly_variable" not in adata.var or "genes_add" not in adata.var or "no_svgs_rank" not in adata.var:
        raise ValueError("adata.var must contain 'genes_del', 'highly_variable', 'genes_add', 'no_svgs_rank' columns")
    # Get genes with highly_variable True and genes_del False
    genes_del = adata.var.loc[(adata.var["highly_variable"] == True) & (adata.var["genes_del"] == False)].index

    # Sort by no_svgs_rank descending
    genes_del_sorted = adata.var.loc[genes_del].sort_values(by="no_svgs_rank", ascending=False).index

    n_del_max = len(adata.var[adata.var["genes_add"]==True]) - len(adata.var[adata.var["highly_variable"]==True])

    # Initialize counter
    removed_count = 0
    adata.var["HSG"] = adata.var["genes_add"] 

    # Remove genes from genes_sets step by step
    for gene in genes_del_sorted:
        if removed_count >= n_del_max:
            break

        adata.var.loc[gene, "HSG"] = False
        removed_count += 1
    return adata
    

def get_sorted_rank_matrix(H_nmf_filtered_norm):


    # Get shape of H_nmf_filtered_norm (genes, topics)
    num_genes, num_topics = H_nmf_filtered_norm.shape
    
    # Create sorting matrix
    sorted_rank_matrix = np.zeros_like(H_nmf_filtered_norm, dtype=int)
    
    # Sort gene expression values for each topic in descending order
    for topic_idx in range(num_topics):
        # Get gene expression values for current topic
        topic_gene_expression = H_nmf_filtered_norm[:, topic_idx]
        
        # Get indices sorted by gene expression value in descending order
        sorted_indices = np.argsort(topic_gene_expression)[::-1]  # Descending order
        
        # Get gene ranking
        sorted_rank_matrix[sorted_indices, topic_idx] = np.arange(1, num_genes + 1)
        
    
    return sorted_rank_matrix


def filter_genes_by_mri(adata, Svgs_dict, mri_percentile=25,reverse=True):


    # Get MRI values for all genes, corresponding to gene names
    genes_mri = adata.var["genes_mri"]
    
    # Combine genes_mri and gene names in Svgs_dict
    genes_mri_dict = {gene: genes_mri[gene] for gene in Svgs_dict.keys()}
    
    # Sort genes by MRI value, from low to high
    sorted_genes_by_mri = sorted(genes_mri_dict.items(), key=lambda x: x[1], reverse=reverse)
    
    # Calculate the number of genes to remove, remove the lowest 25% MRI genes
    num_to_remove = int(len(sorted_genes_by_mri) * mri_percentile / 100)
    
    # Get genes to keep (exclude the lowest 25% MRI genes)
    genes_to_keep = {gene: rank_sum for gene, rank_sum in Svgs_dict.items() if gene in dict(sorted_genes_by_mri[num_to_remove:])}
    
    return genes_to_keep


def get_genes_at_percentiles(adata, H_nmf_filtered_norm, sorted_indices, genes_counts_list, sorted_rank_matrix, lower_percentile=15, upper_percentile =95):

    num_topics = len(sorted_indices)
    percentile_genes = {}
    
    # Pre-build gene index dictionary for faster lookup
    gene_index_dict = {gene: idx for idx, gene in enumerate(adata.var_names)}

    for topic_idx in range(num_topics):
        # Get gene expression values for current topic
        n_genes = genes_counts_list[topic_idx]
        topic_gene_Imp = H_nmf_filtered_norm[:, topic_idx]

        # Get indices sorted by gene expression value in descending order
        sorted_gene_indices = np.argsort(topic_gene_Imp)[::-1][:n_genes]

        # Get sorted gene names
        sorted_gene_names = np.array(adata.var_names)[sorted_gene_indices]

        sorted_gene_Imp = topic_gene_Imp[sorted_gene_indices]

        total_sum = np.sum(sorted_gene_Imp)
        target_sum_lower = total_sum * (lower_percentile/100)
        target_sum_upper = total_sum * (upper_percentile/100)

        cumulative_sum = 0
        # Find index where cumulative sum reaches or exceeds target_sum_lower
        for idx, value in enumerate(sorted_gene_Imp):
            cumulative_sum += value
            if cumulative_sum >= target_sum_lower:
                idx_lower = idx
                break

        cumulative_sum = 0
        for idx, value in enumerate(sorted_gene_Imp):
            cumulative_sum += value
            if cumulative_sum >= target_sum_upper:
                idx_upper = idx
                break

        gene_lower_name = sorted_gene_names[idx_lower]
        gene_upper_name = sorted_gene_names[idx_upper]

        # Find gene rank in sorted_rank_matrix (using dictionary index)
        rank_lower = sorted_rank_matrix[gene_index_dict[gene_lower_name], topic_idx]
        rank_upper = sorted_rank_matrix[gene_index_dict[gene_upper_name], topic_idx]

        # Save gene and its rank at lower and upper percentiles for current topic
        percentile_genes[topic_idx] = {
            'gene_lower': gene_lower_name,
            'rank_lower': rank_lower,
            'gene_upper': gene_upper_name,
            'rank_upper': rank_upper
        }

    percentile_genes_df = pd.DataFrame.from_dict(percentile_genes, orient='index')
    percentile_genes_df.reset_index(inplace= True)
    percentile_genes_df.rename(columns={'index':'Topic'}, inplace=True)

    print("\n percentile_genes")
    print(percentile_genes_df)

    return percentile_genes



def select_SVGs(adata, percentile_genes, sorted_rank_matrix):

    Svgs_set = set()

    selected_topics = np.array(adata.uns["final_topics"])  # Selected topics
    H_nmf = adata.varm["H_nmf"][:, selected_topics]  # gene × selected topics
    z_threshold=1.96

    H_zscore = zscore(H_nmf, axis=0)  # Calculate Z-score

    for topic_idx, gene_info in percentile_genes.items():
        rank_lower = gene_info['rank_lower']
        
        # Get genes ranked in the top 25% by contribution
        genes_before_25 = np.where(sorted_rank_matrix[:, topic_idx] <= rank_lower)[0]
        
        # Select genes with Z-score > 1.96
        selected_genes = genes_before_25[H_zscore[genes_before_25, topic_idx] > z_threshold]
        
        Svgs_set.update(adata.var_names[selected_genes])

    Svgs_dict = compute_gene_rank_sums(adata, Svgs_set, sorted_rank_matrix, reverse=False)
    Svgs_dict = filter_genes_by_mri(adata, Svgs_dict, mri_percentile=25, reverse=False)

    # Store in adata.var
    adata.var["svgs_rank"] = [Svgs_dict.get(gene, float('nan')) for gene in adata.var_names]
    # Store H_zscore in adata.varm
    adata.varm['H_zscore'] = H_zscore

    return Svgs_set


def select_no_SVGs(adata,  percentile_genes, sorted_rank_matrix):

    # Convert adata.var_names to dictionary for faster lookup
    gene_name_to_idx = {gene_name: idx for idx, gene_name in enumerate(adata.var_names)}

    no_svgs = set()  # Store genes that meet the condition

    # Extract boolean marker for highly variable genes
    hvg_bool = adata.var['highly_variable']

    # Get list of highly variable gene names
    hvg_genes = adata.var_names[hvg_bool].tolist()

    # Iterate over each gene in the gene list
    for gene in hvg_genes:
        # Skip gene if not in adata.var_names
        if gene not in gene_name_to_idx:
            continue
        
        # Record whether the gene meets the condition in all topics
        is_valid = True
        
        # Iterate over each topic
        for topic_idx, gene_info in percentile_genes.items():
            rank_upper = gene_info['rank_upper']

            gene_idx = gene_name_to_idx[gene]  # Lookup gene position using dictionary
            # Get gene's rank in this topic
            gene_rank = sorted_rank_matrix[gene_idx, topic_idx].item()
            
            # If gene's rank in any topic is less than or equal to rank_upper, it does not meet the condition
            if gene_rank <= rank_upper:
                is_valid = False
                break
        
        # If the gene meets the condition in all topics, add it to valid_genes set
        if is_valid:
            no_svgs.add(gene)
    
    no_Svgs_dict = compute_gene_rank_sums(adata, no_svgs, sorted_rank_matrix, reverse=True)

    no_Svgs_dict = filter_genes_by_mri(adata, no_Svgs_dict, mri_percentile=20,reverse=True)

    # Create a default value list (NaN for genes not in no_Svgs_dict) corresponding to adata.var.index
    no_svgs_rank_column = [no_Svgs_dict.get(gene, float('nan')) for gene in adata.var_names]

    # Add to new column in adata.var
    adata.var["no_svgs_rank"] = no_svgs_rank_column

    return no_svgs


def compute_gene_rank_sums(adata, combined_genes, sorted_rank_matrix, reverse=True):

    gene_rank_sums = {}  # Initialize dictionary to store sum of rankings

    # Convert adata.var_names to dictionary for faster lookup
    gene_name_to_idx = {gene_name: idx for idx, gene_name in enumerate(adata.var_names)}

    # Iterate over each gene, calculate sum of rankings across all topics
    for gene in combined_genes:
        rank_sum = 0  # Initialize sum of rankings

        # Get gene's ranking in current topic
        gene_idx = gene_name_to_idx[gene]

        # Iterate over each topic, get gene's ranking in each topic
        for topic_idx in range(sorted_rank_matrix.shape[1]):

            gene_rank = sorted_rank_matrix[gene_idx, topic_idx]
            
            # Add gene's ranking in current topic
            rank_sum += gene_rank
        
        # Add gene's sum of rankings to dictionary
        gene_rank_sums[gene] = rank_sum
    
    # gene_rank_sums = sorted(gene_rank_sums.items(), key=lambda x: x[1], reverse=reverse)

    return gene_rank_sums



def plot_loss(epochs_t,loss_list,loss_name):
    plt.figure(figsize=(10, 2)) 
    plt.plot(np.arange(epochs_t), loss_list)
    plt.ylabel(loss_name)
    plt.xlabel('iteration')
    plt.tight_layout()
    plt.show()


def get_H_zscore_value(adata, origin_topics_index, genes):
    """
    Get H_zscore value for specified gene under specified topic.

    Parameters:
    - adata: AnnData object
    - origin_topics_index: selected original topic index (e.g. 1, 2, 3...)
    - genes: target gene name (string)

    Returns:
    - H_zscore_value: H_zscore value for specified gene under specified topic
    """
    # Ensure final_topics exists
    if "final_topics" not in adata.uns:
        raise ValueError("adata.uns['final_topics'] does not exist!")

    # Ensure H_zscore exists
    if "H_zscore" not in adata.varm:
        raise ValueError("adata.varm['H_zscore'] does not exist!")

    # Get sorted topic array
    selected_topics = np.array(adata.uns["final_topics"])

    # Check if origin_topics_index is valid
    if origin_topics_index not in selected_topics:
        raise ValueError(f"origin_topics_index {origin_topics_index} not in final_topics!")

    # Get topic index
    filtered_index = np.where(selected_topics == origin_topics_index)[0][0]

    # Check if gene is in var_names
    if genes not in adata.var_names:
        raise ValueError(f"Gene {genes} not in adata.var_names!")

    # Get gene index
    genes_index = np.where(adata.var_names == genes)[0][0]

    # Get H_zscore value
    H_zscore_value = adata.varm['H_zscore'][genes_index, filtered_index]

    return H_zscore_value


def export_H_zscore_to_csv(adata, out_dir):
    """
    Export H_zscore for each gene and topic to CSV file.
    
    Parameters:
    - adata: AnnData object
    - out_dir: output directory path
    """
    
    # Get H_zscore
    H_zscore = adata.varm["H_zscore"]
    
    # Get gene names
    genes = adata.var_names
    
    # Get original topic indices
    original_topics_index = np.array(adata.uns["final_topics"])
    
    # Convert data to DataFrame, ensure columns use original_topics_index
    H_zscore_df = pd.DataFrame(H_zscore, 
                               index=genes, 
                               columns=original_topics_index)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_result = os.path.join(out_dir, "Zscore_genes_topics.csv")
    # Write DataFrame to CSV file
    H_zscore_df.to_csv(output_result)
    print(f"Zscore CSV file has been saved to {output_result}")


def export_Corr_to_csv(adata, out_dir):
    """
    Export corr for each gene and topic to CSV file.
    Parameters:
    - adata: AnnData object
    - out_dir: output directory path
    """
    
    # Get H_zscore
    Corr = adata.varm["gene_topic_corr"]
    
    # Get gene names
    genes = adata.var_names
    
    # Get original topic indices
    original_topics_index = np.array(adata.uns["final_topics"])

    # Only keep topics in original_topics_index
    Corr_filtered = Corr[:, original_topics_index]  # Select corresponding columns
    
    # Convert data to DataFrame, ensure columns use original_topics_index
    Corr_df = pd.DataFrame(Corr_filtered, 
                               index=genes, 
                               columns=original_topics_index)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        output_result = os.path.join(out_dir, "Corr_genes_topics.csv")
    # Write DataFrame to CSV file
    Corr_df.to_csv(output_result)
    print(f"Corr CSV file has been saved to {output_result}")


############################## select marker genes for each topics (without multigenes)###############################

def compute_gene_topic_corr(adata, method="pearson"):
    """
    Compute correlation between gene expression (spots × genes) and topic probabilities (spots × topics) using matrix operations.

    The result is stored in adata.varm["gene_topic_corr"], shape (genes × topics).

    Parameters:
    - adata: AnnData object, containing:
        - adata.X (spots × genes): gene expression data
        - adata.obsm["W_nmf"] (spots × topics): topic probability data
    - method: str, correlation method, options are "pearson" or "cosine" (default "pearson")

    Returns:
    - None, result is stored in adata.varm["gene_topic_corr"]
    """
    # Get spots × genes matrix (gene expression)
    gene_expr_matrix = adata.X

    # If gene_expr_matrix is sparse, convert to dense
    if sp.issparse(gene_expr_matrix):
        gene_expr_matrix = gene_expr_matrix.toarray()
    
    # Get spots × topics matrix (topic probabilities)
    topic_matrix = adata.obsm["W_nmf"]

    # Ensure data is float type to avoid integer calculation errors
    gene_expr_matrix = gene_expr_matrix.astype(np.float64)
    topic_matrix = topic_matrix.astype(np.float64)

    if method == "pearson":
        # Compute Pearson correlation coefficient
        gene_mean = np.mean(gene_expr_matrix, axis=0, keepdims=True)  # Mean for each gene
        gene_std = np.std(gene_expr_matrix, axis=0, keepdims=True)  # Std for each gene
        topic_mean = np.mean(topic_matrix, axis=0, keepdims=True)  # Mean for each topic
        topic_std = np.std(topic_matrix, axis=0, keepdims=True)  # Std for each topic

        # Standardize matrix (Z-score)
        gene_expr_norm = (gene_expr_matrix - gene_mean) / gene_std  # spots × genes
        topic_norm = (topic_matrix - topic_mean) / topic_std  # spots × topics

        # Compute Pearson correlation coefficient matrix (genes × topics)
        correlation_matrix = np.dot(gene_expr_norm.T, topic_norm) / gene_expr_matrix.shape[0]

    elif method == "cosine":
        # Compute cosine similarity
        gene_norm = np.linalg.norm(gene_expr_matrix, axis=0, keepdims=True)  # L2 norm for each gene
        topic_norm = np.linalg.norm(topic_matrix, axis=0, keepdims=True)  # L2 norm for each topic
        
        # Avoid division by 0
        gene_norm[gene_norm == 0] = 1e-10
        topic_norm[topic_norm == 0] = 1e-10
        
        # Normalize
        gene_expr_norm = gene_expr_matrix / gene_norm
        topic_norm = topic_matrix / topic_norm

        # Compute cosine similarity (genes × topics)
        correlation_matrix = np.dot(gene_expr_norm.T, topic_norm)

    else:
        raise ValueError("method only supports 'pearson' or 'cosine'")

    # Store in adata.varm
    adata.varm["gene_topic_corr"] = correlation_matrix


def generate_marker_genes_dict(adata, corr_threshold=0.2):
    """
    Generate marker_genes_all_dict and store in adata.uns, ensuring each gene belongs only to the most correlated topic.
    
    Parameters:
    - adata: AnnData object
        Contains adata.varm["gene_topic_corr"] (genes × topics) correlation matrix.
    - corr_threshold: float, default 0.2
        Only keep genes with correlation greater than this threshold.

    Returns:
    - None, result is stored in adata.uns['marker_genes_all_dict'].
    """

    # Get gene-topic correlation matrix (genes × topics)
    gene_topic_corr = adata.varm.get("gene_topic_corr", None)
    if gene_topic_corr is None:
        raise ValueError("Cannot find 'gene_topic_corr' matrix in adata.varm.")

    genes, topics = gene_topic_corr.shape  # Matrix size
    rank_matrix = np.zeros((genes, topics), dtype=int)  # Store sorting matrix

    # 1. Compute sorting matrix (gene ranking for each topic, starting from 1)
    for topic_idx in range(topics):
        sorted_indices = np.argsort(-gene_topic_corr[:, topic_idx])  # Descending order
        rank_matrix[sorted_indices, topic_idx] = np.arange(1, genes + 1)  # 1 to genes

    # 2. Compute mask matrix (each gene only corresponds to the best topic)
    mask_matrix = np.zeros((genes, topics), dtype=bool)  # Mask matrix
    best_topics = np.argmin(rank_matrix, axis=1)  # Each gene selects best topic
    mask_matrix[np.arange(genes), best_topics] = True  # Only best topic is True

    # 3. Iterate over each topic, filter genes that meet the condition
    marker_genes_all_dict = {}
    for topic_idx in range(topics):
        # Find genes related to this topic (row indices where mask matrix is True)
        topic_genes_idx = np.where(mask_matrix[:, topic_idx])[0]
        topic_gene_corrs = gene_topic_corr[topic_genes_idx, topic_idx]  # Extract correlation

        # Only keep genes with correlation greater than threshold
        valid_genes_mask = topic_gene_corrs > corr_threshold
        valid_genes_idx = topic_genes_idx[valid_genes_mask]  # Filter gene indices
        valid_gene_corrs = topic_gene_corrs[valid_genes_mask]  # Filter correlation

        # Sort by correlation in descending order
        sorted_indices = np.argsort(-valid_gene_corrs)  # Descending order
        sorted_genes = adata.var_names[valid_genes_idx[sorted_indices]].tolist()

        # Store in dictionary
        marker_genes_all_dict[str(topic_idx)] = sorted_genes

    # Store in adata.uns
    adata.uns['marker_genes_all_dict'] = marker_genes_all_dict



def cal_topics_dist_mat(adata, method="pearson"):
    """
    Compute similarity clustering for topics (based on spot probabilities) and plot heatmap with similarity annotation.

    Parameters:
    - adata: AnnData object
    - method: str, similarity calculation method, options ["cosine", "pearson"]
    - annot: bool, whether to show similarity values on heatmap

    Returns:
    - Clustered heatmap
    """
    # Get W_nmf matrix (spots × topics) and transpose to (topics × spots)
    W_nmf = adata.obsm.get("W_nmf", None)
    if W_nmf is None:
        raise ValueError("Cannot find 'W_nmf' matrix in adata.obsm.")
    
    W_nmf_T = W_nmf.T  # Topics × Spots matrix

    # **Compute similarity between topics**
    if method == "cosine":
        similarity_matrix = cosine_similarity(W_nmf_T)  # Cosine similarity
        distance_matrix = 1 - similarity_matrix  # Convert cosine similarity to distance
    elif method == "pearson":
        similarity_matrix = np.corrcoef(W_nmf_T)  # Pearson correlation
        distance_matrix = 1 - similarity_matrix  # Convert to distance
    else:
        raise ValueError("Unsupported similarity calculation method, choose 'cosine' or 'pearson'.")
    return distance_matrix

def select_max_min_order(adata, distance_matrix):
    """Generate topic selection order by greedy strategy maximizing minimum distance, only selecting topics in filtered_topics"""
    
    n_topics = distance_matrix.shape[0]
    filtered_topics = set(adata.uns["final_topics"])  # Convert to set for lookup
    candidates = list(filtered_topics & set(range(n_topics)))  # Only keep topics in filtered_topics

    if len(candidates) < 2:
        return [str(t) for t in candidates]  # If less than 2 topics, return them directly
    
    # Compute maximum distance within filtered_topics
    max_dist = -np.inf
    pair = None
    
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if distance_matrix[candidates[i], candidates[j]] > max_dist:
                max_dist = distance_matrix[candidates[i], candidates[j]]
                pair = (candidates[i], candidates[j])

    if pair is None:
        return []  # Should not happen, but just in case

    selected = list(pair)
    candidates = set(candidates) - set(pair)

    while candidates:
        best_topic = None
        best_min_dist = -np.inf

        for t in candidates:
            temp_selected = selected + [t]
            sub_matrix = distance_matrix[np.ix_(temp_selected, temp_selected)]
            np.fill_diagonal(sub_matrix, 0)  # Ignore diagonal
            non_zero_dists = sub_matrix[sub_matrix > 0]
            current_min_dist = np.min(non_zero_dists) if non_zero_dists.size > 0 else 0

            if current_min_dist > best_min_dist:
                best_min_dist = current_min_dist
                best_topic = t

        if best_topic is None:
            break  # No topics to select, stop

        selected.append(best_topic)
        candidates.remove(best_topic)

    selected_filtered = [str(topic) for topic in selected]
    # Ensure only topics in filtered_topics are returned, as strings
    return selected_filtered

def compute_umap_marker_genes(
    adata, 
    n_pca_components=50, 
    n_neighbors=15,
    method="umap",
    min_dist=0.1, 
    random_state=42, 
    n_genes=None,  # Number of genes to select for each topic, default None
    topics_list=None  # Topics to plot, default None (all topics)
):
    """
    Compute PCA + UMAP embedding for marker genes.

    Parameters:
    - adata: AnnData object
    - n_pca_components: PCA dimension (default 50)
    - n_neighbors: UMAP neighbors (default 15)
    - min_dist: UMAP min_dist parameter (default 0.1)
    - random_state: random seed (default 42)
    - n_genes: number of genes to select for each topic (default None, all)
    - topics_list: topics to visualize (default None, all)

    Returns:
    - DataFrame with UMAP coordinates (UMAP1, UMAP2), gene name (Gene), and topic (Topic)
    """

    # Get marker genes dictionary
    marker_genes_dict = adata.uns.get("marker_genes_all_dict", None)
    if marker_genes_dict is None:
        raise ValueError("Cannot find 'marker_genes_all_dict' in adata.uns, please compute marker genes first.")

    # Filter topics
    if topics_list is not None:
        selected_topics = {topic: marker_genes_dict[topic] for topic in topics_list if topic in marker_genes_dict}
    else:
        selected_topics = marker_genes_dict  # All topics
    # Get all genes for selected topics
    selected_genes = []
    for topic, genes in selected_topics.items():
        if n_genes is not None:
            selected_genes.extend(genes[:n_genes])  # Only top n_genes
        else:
            selected_genes.extend(genes)  # All genes

    # Unique set of selected genes
    selected_genes = list(set(selected_genes))
    # Ensure genes are in adata.var_names
    valid_genes = [gene for gene in selected_genes if gene in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("Selected genes not found in adata.var_names, please check gene names.")

    # Get gene expression matrix (spots × selected_genes)
    gene_expr_matrix = adata[:, valid_genes].X.toarray() if hasattr(adata[:, valid_genes].X, "toarray") else adata[:, valid_genes].X

    gene_expr_matrix = gene_expr_matrix.T  # (selected_genes × spots)

    # **PCA reduction**
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    gene_expr_pca = pca.fit_transform(gene_expr_matrix)

    # **Select embedding method**
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    else:
        raise ValueError("method only supports 'umap' or 'tsne'")

    embedding = reducer.fit_transform(gene_expr_pca)  # (spots × 2)

    # Build DataFrame
    df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df["Gene"] = valid_genes  # Record gene names

    # Build Gene → Topic mapping
    gene_to_topic = {gene: topic for topic, genes in selected_topics.items() for gene in genes if gene in valid_genes}
    df["Topic"] = df["Gene"].map(gene_to_topic)

    return df

def analyze_topics_and_umap(
    adata, 
    method="pearson",  
    corr_threshold=0.2,   
    n_pca_components=50, 
    n_neighbors=15,
    embedding_method="umap",  # "umap" or "tsne"
    min_dist=0.1, 
    random_state=42, 
    n_genes=None,  
    topics_list=None  # int or List
):
    """
    Compute topic correlation, select most dissimilar topics, and perform UMAP visualization.

    Parameters:
    - adata: AnnData object
    - method: str, correlation method, default "pearson"
    - corr_threshold: float, marker gene correlation threshold
    - n_pca_components: int, number of PCA components
    - n_neighbors: int, UMAP/T-SNE neighbors
    - embedding_method: str, "umap" or "tsne"
    - min_dist: float, UMAP min_dist parameter
    - random_state: int, random seed
    - n_genes: int, number of genes to select for each topic, default None (all marker genes)
    - topics_list: int or List[int], topics to visualize
      - If int, automatically select the most dissimilar `n` topics
      - If list, use the topics in the list

    Returns:
    - df: DataFrame for UMAP visualization
    - selected_order: List[int], selected topic order
    """

    # 1. **Compute gene-topic correlation matrix**
    compute_gene_topic_corr(adata, method=method)

    # 2. **Generate marker genes dictionary**
    generate_marker_genes_dict(adata, corr_threshold=corr_threshold)

    # 3. **Compute topic distance matrix**
    distance_matrix = cal_topics_dist_mat(adata, method=method)

    # 4. **Select most dissimilar topic order**
    selected_order = select_max_min_order(adata, distance_matrix)

    # 5. **Determine topics to plot**
    if isinstance(topics_list, int):  # If topics_list is int, select top topics_list most dissimilar topics
        topics_list = selected_order[:topics_list]
    elif topics_list is None:  # If None, use all topics
        topics_list = selected_order

    # 6. **Compute DataFrame for UMAP visualization**
    marker_genes_umap_df = compute_umap_marker_genes(
        adata, 
        n_pca_components=n_pca_components, 
        n_neighbors=n_neighbors,
        method=embedding_method,  # "umap" or "tsne"
        min_dist=min_dist, 
        random_state=random_state, 
        n_genes=n_genes,  
        topics_list=topics_list  
    )

    return marker_genes_umap_df, topics_list


############################## select marker genes for each topics (multigenes)########################################

def generate_multi_marker_genes(adata, corr_threshold=0.2):

    gene_topic_corr = adata.varm.get("gene_topic_corr", None)
    if gene_topic_corr is None:
        raise ValueError("Cannot find 'gene_topic_corr' matrix in adata.varm.")

    genes, topics = gene_topic_corr.shape
    marker_genes_dict = {}

    for topic_idx in range(topics):
        # All gene correlations for current topic
        topic_corrs = gene_topic_corr[:, topic_idx]
        
        # Genes with correlation > threshold
        valid_idx = np.where(topic_corrs > corr_threshold)[0]
        valid_corrs = topic_corrs[valid_idx]
        
        # Sort by correlation in descending order
        sorted_indices = np.argsort(-valid_corrs)
        sorted_valid_idx = valid_idx[sorted_indices]
        sorted_gene_names = adata.var_names[sorted_valid_idx]
        sorted_corrs = valid_corrs[sorted_indices]


        # Build (gene_name, corr_value) list
        gene_corr_pairs = [
            (gene, float(corr)) for gene, corr in zip(sorted_gene_names, sorted_corrs)
        ]

        # Store in dictionary
        marker_genes_dict[str(topic_idx)] = gene_corr_pairs

    adata.uns["marker_genes_multi_dict"] = marker_genes_dict


def match_domain_to_topic(adata, useref="domain", topic_key="W_nmf"):

    if useref not in adata.obs:
        raise ValueError(f"{useref} not in adata.obs")
    if topic_key not in adata.obsm:
        raise ValueError(f"{topic_key} not in adata.obsm")

    domains = adata.obs[useref].astype(str).values
    unique_domains = np.unique(domains)
    topic_probs = adata.obsm[topic_key]  # shape: (n_spots, n_topics)
    n_topics = topic_probs.shape[1]

    domain_mapping_topic = {}

    for domain in unique_domains:
        # Build one-hot vector for this domain
        domain_one_hot = (domains == domain).astype(float)  # shape: (n_spots,)

        # Compute Pearson correlation with each topic probability column
        corrs = []
        for i in range(n_topics):
            topic_prob_vec = topic_probs[:, i]
            corr, _ = pearsonr(domain_one_hot, topic_prob_vec)
            corrs.append(corr)

        best_topic_idx = int(np.argmax(corrs))
        domain_mapping_topic[domain] = f"Topic_{best_topic_idx}"

    # Ensure assignment happens even if unique_domains is empty
    adata.uns["domain_mapping_topic"] = domain_mapping_topic

    return domain_mapping_topic


def generate_domain_to_genes(adata, 
                             domain_topic_key="domain_mapping_topic", 
                             topic_gene_key="marker_genes_multi_dict", 
                             output_key="domain_to_genes"):

    # Check if required keys exist
    if domain_topic_key not in adata.uns:
        raise KeyError(f"{domain_topic_key} not found in adata.uns, please run domain-topic matching step first.")
    if topic_gene_key not in adata.uns:
        raise KeyError(f"{topic_gene_key} not found in adata.uns, please run marker gene identification step first.")

    domain_to_best_topic = adata.uns[domain_topic_key]
    marker_genes_multi_dict = adata.uns[topic_gene_key]

    # Build domain → genes mapping
    domain_to_genes = {}
    for domain, topic in domain_to_best_topic.items():
        topic_id_match = re.search(r"\d+", topic)
        topic_id = topic_id_match.group(0)
        gene_corr_pairs  = marker_genes_multi_dict.get(topic_id, [])
        domain_to_genes[domain] = gene_corr_pairs 

    # Store in adata.uns
    adata.uns[output_key] = domain_to_genes
    print(f"generate adata.uns['{output_key}'], contain {len(domain_to_genes)} domain.")

    return domain_to_genes

def run_domain_gene_mapping_pipeline(
    adata,
    useref="domain",
    topic_key="W_nmf",
    corr_threshold=0.2,
    topic_gene_corr_key="gene_topic_corr",
    domain_mapping_topic_key="domain_mapping_topic",
    marker_genes_dict_key="marker_genes_multi_dict",
    domain_to_genes_key="domain_to_genes"
):

    # Raise an error if the required topic-gene correlation matrix is not found in adata.varm
    if topic_gene_corr_key not in adata.varm:
        raise KeyError(f"{topic_gene_corr_key} not found in adata.varm.")
    
    print(" Step 1: Generating marker genes for topics...")
    generate_multi_marker_genes(adata, corr_threshold=corr_threshold)

    print(" Step 2: Matching domains to best topics...")
    match_domain_to_topic(adata, useref=useref, topic_key=topic_key)

    print(" Step 3: Generating domain → genes dictionary...")
    generate_domain_to_genes(
        adata,
        domain_topic_key=domain_mapping_topic_key,
        topic_gene_key=marker_genes_dict_key,
        output_key=domain_to_genes_key
    )

    print(" Pipeline completed.")
