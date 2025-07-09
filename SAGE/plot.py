import scanpy as sc
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import seaborn as sns

def visualize_topics(adata, 
                     n_top_topics=None,
                     spot_size=150, 
                     out_dir=None,
                     figsize=(3, 3),
                     fontsize=7,
                     frameon=None, 
                     legend_loc="right margin", 
                     colorbar_loc="right",
                     show_title=True
                     ):

    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial
    
    adata_plot = adata.copy()

    W = adata_plot.obsm["W_nmf"] 

    for i in range(W.shape[1]):  # Iterate all topics
        adata_plot.obs[f"Topic_{i}"] = W[:, i]

    if "topic_mri_list" not in adata_plot.uns:
        raise ValueError("topic_mri_list not found in adata.uns. Please run `calculate_MRI_for_topics` first.")

    topic_mri_list = adata_plot.uns["topic_mri_list"]
    # Select top n_top_topics topics
    if n_top_topics is not None:
        topic_mri_list = topic_mri_list[:n_top_topics]

    # Visualize topics
    for topic in topic_mri_list:

        topic_title = [f"Topic {int(topic[0])}, MRI: {topic[1]:.2f}"] if show_title else ""

        if 'spatial' in adata.uns:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{int(topic[0])}"], 
                img_key=None,
                spot_size=spot_size, 
                title=topic_title, 
                cmap="inferno",
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc,
                show=False
            )
        else:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{int(topic[0])}"], 
                title=topic_title, 
                cmap="inferno", 
                spot_size=spot_size,
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )

        if out_dir:
            output_result_dir = os.path.join(out_dir, "select_topics")
            os.makedirs(output_result_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_result_dir, f"Topic_{int(topic[0])}_MRI_ {topic[1]:.2f}.png"),
                dpi=600, bbox_inches='tight', pad_inches=0
            )
            plt.close()


def visualize_filtered_topics_by_mri(adata, n_top_topics=None, spot_size=150, out_dir=None):

    adata_plot = adata.copy()
    filtered_W = adata_plot.obsm["W_nmf_filtered"]

    if "filtered_topic_mri_list" not in adata_plot.uns:
        raise ValueError("filtered_topic_mri_list not found in adata.uns. Please run `filter_and_update_nmf_topics` first.")
    filtered_topic_mri_list = adata.uns["filtered_topic_mri_list"]

    filtered_topics = [item[0] for item in filtered_topic_mri_list]
    filtered_MRI = [item[1] for item in filtered_topic_mri_list]
    
    for i in range(filtered_W.shape[1]):  # Iterate all topics
        adata_plot.obs[f"Topic_{filtered_topics[i]}"] = filtered_W[:, i]
        
    # Get and sort Moran's I list
    filtered_topic_mri_sorted = sorted(filtered_topic_mri_list, key=lambda x: x[1], reverse=True)

    # Select top n_top_topics topics
    if n_top_topics is not None:
        filtered_topic_mri_sorted = filtered_topic_mri_sorted[:n_top_topics]
    for topic in filtered_topic_mri_sorted:
        if 'spatial' in adata.uns:
            # Visualize topic
            sc.pl.spatial(
                adata_plot,
                color=[f"Topic_{int(topic[0])}"],
                title=[f"Topic {int(topic[0])}, MRI: {topic[1]:.2f}"],
                frameon=None,
                spot_size=spot_size,
                cmap="inferno",
                show=True
            )
            if out_dir:
                # Ensure output directory exists
                os.makedirs(out_dir, exist_ok=True)
                output_result_dir = out_dir + "\\select_topics_by_mri"
                if not os.path.exists(output_result_dir):
                    os.makedirs(output_result_dir)
        
                plt.savefig(
                    f"{output_result_dir}\\Topic_{int(topic[0])}_MRI_ {topic[1]}.png",
                    dpi=600, bbox_inches='tight'
                )
                plt.close()  # Close current figure to avoid overlap

        else:
            # Visualize topic
            sc.pl.spatial(
                adata_plot,
                color=[f"Topic_{int(topic[0])}"],
                title=[f"Topic {int(topic[0])}, MRI: {topic[1]:.2f}"],
                frameon=None,   
                spot_size=spot_size,
                cmap="inferno",
                show=True
            )
            if out_dir:
                # Ensure output directory exists
                os.makedirs(out_dir, exist_ok=True)
                output_result_dir = out_dir + "\\select_topics_by_mri"
                if not os.path.exists(output_result_dir):
                    os.makedirs(output_result_dir)
        
                plt.savefig(
                    f"{output_result_dir}\\Topic_{int(topic[0])}_MRI_ {topic[1]:.2f}.png",
                    dpi=600, bbox_inches='tight'
                )
                plt.close()  # Close current figure to avoid overlap


def visualize_topics_imp(adata, figsize=(10, 6)):

    if "sorted_indices_imp" not in adata.uns:
        raise ValueError("sorted_indices_imp not found in adata.uns. Please run `fit_rf_and_extract_importance` first.")
    
    sorted_indices_imp = adata.uns["sorted_indices_imp"]
    sorted_indices = [item[0] for item in sorted_indices_imp]
    sorted_importance = [item[1] for item in sorted_indices_imp]

    # Get original topic indices
    filtered_topic_mri_list = adata.uns["filtered_topic_mri_list"]
    filtered_topics = [item[0] for item in filtered_topic_mri_list]
    
    sorted_topics = [f"Topic {filtered_topics[i]}" for i in sorted_indices]  # Sorted topic names

    # Create plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(sorted_importance)), sorted_importance, tick_label=sorted_topics)
    plt.xlabel("Topics")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance by Topic (Random Forest)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def visualize_filtered_topics_by_RF(
    adata, 
    spot_size=150, 
    out_dir=None,
    figsize=(3, 3),
    fontsize=7, 
    frameon=None, 
    legend_loc="right margin", 
    colorbar_loc="right",
    show_title=True
):
    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial
    
    adata_plot = adata.copy()

    sorted_indices_imp = adata.uns["sorted_indices_imp"]
    sorted_indices = [item[0] for item in sorted_indices_imp]
    sorted_importance = [item[1] for item in sorted_indices_imp]

    # Get original topic indices
    filtered_topic_mri_list = adata.uns["filtered_topic_mri_list"]
    filtered_topics = [item[0] for item in filtered_topic_mri_list]

    filtered_W = adata_plot.obsm["W_nmf_filtered"] 
    
    # Update Scanpy visualization
    top_n = len(adata.uns['final_topics'])  # Select top_n topics to visualize
    top_indices = sorted_indices[:top_n]

    # Add most important topics to adata.obs (if not already added)
    for i in top_indices:
        adata_plot.obs[f"Topic_{int(filtered_topics[int(i)])}"] = filtered_W[:, int(i)]

    # Visualize spatial distribution of top_n topics
    for idx, top_idx in enumerate(top_indices):
        topic_title = [f"Topic {int(filtered_topics[int(top_idx)])}, Imp : {sorted_importance[idx]:.2f}"] if show_title else ""

        if 'spatial' in adata.uns:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{int(filtered_topics[int(top_idx)])}"], 
                title=topic_title, 
                img_key=None,
                spot_size=spot_size, 
                cmap="inferno",
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc,
                show=False
            )
        else:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{int(filtered_topics[int(top_idx)])}"], 
                title=topic_title, 
                cmap="inferno", 
                spot_size=spot_size,
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )

        if out_dir:
            output_result_dir = os.path.join(out_dir, "select_topics_by_RF")
            os.makedirs(output_result_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_result_dir, f"Topic_{int(filtered_topics[int(top_idx)])}_Imp_ {sorted_importance[idx]:.2f}.png"),
                dpi=600, bbox_inches='tight', pad_inches=0
            )
            plt.close()


def plot_svgs_and_no_svgs_rank_distribution(adata):

    # Ensure 'svgs_rank' and 'no_svgs_rank' columns exist
    if "svgs_rank" not in adata.var or "no_svgs_rank" not in adata.var:
        raise ValueError("adata.var must contain 'svgs_rank' and 'no_svgs_rank' columns")

    # Extract svgs_rank and no_svgs_rank column data
    svgs_rank_data = adata.var["svgs_rank"]
    no_svgs_rank_data = adata.var["no_svgs_rank"]

    # Set figure size
    plt.figure(figsize=(10, 6))

    # Plot histogram for svgs_rank
    sns.histplot(svgs_rank_data, kde=True, bins=30, color='skyblue', edgecolor='black', label='svgs_rank', alpha=0.6)

    # Plot histogram for no_svgs_rank
    sns.histplot(no_svgs_rank_data, kde=True, bins=30, color='lightcoral', edgecolor='black', label='no_svgs_rank', alpha=0.6)

    # Set title and labels
    plt.title("Distribution of svgs_rank and no_svgs_rank", fontsize=16)
    plt.xlabel("Rank Values", fontsize=14)
    plt.ylabel("Frequency / Density", fontsize=14)

    # Show legend
    plt.legend()

    # Show figure
    plt.show()

def plot_scatter_with_regression(adata, x_column, y_column, plot_title, xlabel, ylabel):

    # Ensure x and y columns exist
    if x_column not in adata.var or y_column not in adata.var:
        raise ValueError(f"adata.var must contain '{x_column}' and '{y_column}' columns")

    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=adata.var[x_column], y=adata.var[y_column])

    # Add regression line
    sns.regplot(x=adata.var[x_column], y=adata.var[y_column], scatter=False, color="red")

    # Set title and labels
    plt.title(plot_title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Show figure
    plt.show()


def visualize_top_genes_with_zscore(
    adata, 
    n=25, 
    spot_size=150, 
    out_dir=None,
    figsize=(6, 6), 
    fontsize=7,
    frameon=None, 
    legend_loc="right margin", 
    colorbar_loc="right",
    show_title=True
):

    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial
    # 1. Get topic-gene info
    top_genes_all_dict = adata.uns.get("top_genes_all_dict", None)
    if top_genes_all_dict is None:
        raise ValueError("Could not find 'top_genes_all_dict' in adata.uns!")
    
    if "W_nmf" not in adata.obsm:
        raise ValueError("adata.obsm['W_nmf'] does not exist, cannot get NMF topic info!")

    W = adata.obsm["W_nmf"]

    final_topics = adata.uns.get("final_topics", [])
    final_topics = [str(i) for i in final_topics]
    filter_top_genes_dict = {key: top_genes_all_dict[key] for key in final_topics if key in top_genes_all_dict}

    # 2. Iterate each topic for visualization
    for idx, (topic_idx, genes) in enumerate(filter_top_genes_dict.items()):
        # 2.1 Ensure topic_idx can be converted to int
        try:
            topic_idx = int(topic_idx)
        except ValueError:
            raise ValueError(f"Cannot convert {topic_idx} to int!")

        # 2.2 Ensure top n genes
        genes_to_plot = genes[:n]

        # 2.3 Assign topic info to `adata.obs`
        adata.obs[f"Topic_{topic_idx}"] = W[:, topic_idx]

        # 2.4 Visualize topic
        topic_title = [f"Topic {topic_idx}"] if show_title else ""
        if 'spatial' in adata.uns:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{topic_idx}"], 
                title=topic_title, 
                img_key=None,
                spot_size=spot_size, 
                cmap="inferno",
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc,
                show=False
            )
        else:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{topic_idx}"], 
                title=topic_title, 
                cmap="inferno", 
                spot_size=spot_size,
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )

        if out_dir:
            output_result_dir = os.path.join(out_dir, "Z_socre_sorted", f"Topic_{topic_idx}_genes")
            os.makedirs(output_result_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_result_dir, f"Topic_{topic_idx}.png"),
                dpi=600, bbox_inches='tight', pad_inches=0
            )
            plt.close()

        # 2.5 Visualize each gene
        if "H_zscore" not in adata.varm or adata.varm["H_zscore"] is None:
            raise ValueError("adata.varm['H_zscore'] does not exist or is None!")

        for top_n, gene in enumerate(genes_to_plot, start=1):
            if gene not in adata.var_names:
                print(f"Warning: gene {gene} not in adata.var_names, skipping!")
                continue

            # Get z_score for gene
            gene_idx = np.where(adata.var_names == gene)[0][0]
            gene_zscore = adata.varm["H_zscore"][gene_idx, idx]

            if np.isnan(gene_zscore):
                gene_zscore = 0.0  # Handle NaN

            # Generate file path (replace illegal characters)
            gene_safe = re.sub(r'[<>:"/\\|?*]', "_", gene)
            file_path = os.path.join(
                output_result_dir,
                f"Topic_{topic_idx}_No_{top_n}_Gene_{gene_safe}_z_score_{gene_zscore:.2f}.png"
            )

            # Visualization
            gene_title = [f"{gene} \n (z_score: {gene_zscore:.2f})"] if show_title else ""
            if 'spatial' in adata.uns:
                sc.pl.spatial(
                    adata, 
                    color=[gene], 
                    img_key=None,
                    spot_size=spot_size, 
                    cmap="inferno", 
                    title=gene_title, 
                    frameon=frameon, 
                    legend_loc=legend_loc, 
                    colorbar_loc=colorbar_loc, 
                    show=False
                )
            else:
                sc.pl.spatial(
                    adata, 
                    color=[gene], 
                    cmap="inferno", 
                    spot_size=spot_size, 
                    title=gene_title, 
                    frameon=frameon, 
                    legend_loc=legend_loc, 
                    colorbar_loc=colorbar_loc, 
                    show=False
                )

            if out_dir:
                plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)
                plt.close()


def visualize_top_genes_with_corr(
    adata, 
    n=25, 
    spot_size=150, 
    out_dir=None,
    figsize=(6, 6), 
    fontsize=7,
    frameon=None, 
    legend_loc="right margin", 
    colorbar_loc="right",
    show_title=True
):

    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial
    # 1. Get topic-gene info
    marker_genes_all_dict = adata.uns.get("marker_genes_all_dict", None)
    if marker_genes_all_dict is None:
        raise ValueError("Could not find 'marker_genes_all_dict' in adata.uns!")
    
    if "W_nmf" not in adata.obsm:
        raise ValueError("adata.obsm['W_nmf'] does not exist, cannot get NMF topic info!")

    W = adata.obsm["W_nmf"]

    final_topics = adata.uns.get("final_topics", [])
    gene_topic_corr = adata.varm["gene_topic_corr"][:, final_topics]  
    final_topics = [str(i) for i in final_topics]
    filter_top_genes_dict = {key: marker_genes_all_dict[key] for key in final_topics if key in marker_genes_all_dict}

    # 2. Iterate each topic for visualization
    for idx, (topic_idx, genes) in enumerate(filter_top_genes_dict.items()):
        # 2.1 Ensure topic_idx can be converted to int
        try:
            topic_idx = int(topic_idx)
        except ValueError:
            raise ValueError(f"Cannot convert {topic_idx} to int!")

        # 2.2 Ensure top n genes
        genes_to_plot = genes[:n]

        # 2.3 Assign topic info to `adata.obs`
        adata.obs[f"Topic_{topic_idx}"] = W[:, topic_idx]
        
        # 2.4 Visualize topic
        topic_title = [f"Topic {topic_idx}"] if show_title else ""
        if 'spatial' in adata.uns:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{topic_idx}"], 
                title=topic_title, 
                img_key=None,
                spot_size=spot_size, 
                cmap="inferno", 
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )
        else:
            sc.pl.spatial(
                adata, 
                color=[f"Topic_{topic_idx}"], 
                title=topic_title, 
                cmap="inferno", 
                spot_size=spot_size, 
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )

        if out_dir:
            output_result_dir = os.path.join(out_dir, "Corr_sorted", f"Topic_{topic_idx}_genes")
            os.makedirs(output_result_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_result_dir, f"Topic_{topic_idx}.png"),
                dpi=600, bbox_inches='tight', pad_inches=0
            )
            plt.close()

        # 2.5 Visualize each gene
        if "gene_topic_corr" not in adata.varm or adata.varm["gene_topic_corr"] is None:
            raise ValueError("adata.varm['gene_topic_corr'] does not exist or is None!")

        for top_n, gene in enumerate(genes_to_plot, start=1):
            if gene not in adata.var_names:
                print(f"Warning: gene {gene} not in adata.var_names, skipping!")
                continue

            # Get correlation for gene
            gene_idx = np.where(adata.var_names == gene)[0][0]
            gene_corr = gene_topic_corr[gene_idx, idx]
            
            if np.isnan(gene_corr):
                gene_corr = 0.0  # Handle NaN
            
            # Generate file path (replace illegal characters)
            gene_safe = re.sub(r'[<>:"/\\|?*]', "_", gene)
            file_path = os.path.join(
                output_result_dir,
                f"Topic_{topic_idx}_No_{top_n}_Gene_{gene_safe}_corr_{gene_corr:.2f}.png"
            )
                
            # Visualization
            gene_title = [f"{gene}  \n (corr: {gene_corr:.2f})"] if show_title else ""
            if 'spatial' in adata.uns:
                sc.pl.spatial(
                    adata, 
                    color=[gene], 
                    cmap="inferno", 
                    img_key=None,
                    spot_size=spot_size, 
                    title=gene_title, 
                    frameon=frameon, 
                    legend_loc=legend_loc, 
                    colorbar_loc=colorbar_loc, 
                    show=False
                )
            else:
                sc.pl.spatial(
                    adata, 
                    color=[gene], 
                    cmap="inferno", 
                    spot_size=spot_size, 
                    title=gene_title, 
                    frameon=frameon, 
                    legend_loc=legend_loc, 
                    colorbar_loc=colorbar_loc, 
                    show=False
                )

            if out_dir:
                plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)
                plt.close()

def plot_umap(df):

    palette = [
        '#F4E8B8',  '#EEC30E',  '#8F0100',  '#058187',  '#0C4DA7',  
        '#B44541',  '#632621',  '#92C7A3',  '#D98882',  '#6A93CB',  
        '#F0C94A',  '#AD6448',  '#4F6A9C',  '#CCB9A1',  '#0B3434',  
        '#3C4F76',  '#C1D354',  '#7D5BA6',  '#F28522',  '#4A9586',
        '#FF6F61',  '#D32F2F',  '#1976D2',  '#388E3C',  '#FBC02D', 
        '#8E24AA',  '#0288D1',  '#7B1FA2',  '#F57C00',  '#C2185B',
        '#1B4F72',  '#117864',  '#D4AC0D',  '#922B21',  '#6C3483',
        '#1F618D',  '#A04000',  '#196F3D',  '#2C3E50',  '#F39C12',
        '#7D6608',  '#4A235A',  '#D68910',  '#B03A2E',  '#7B241C',
        '#2471A3',  '#148F77',  '#9C640C',  '#6E2C00',  '#512E5F',
        '#154360',  '#145A32'
    ]

    unique_topics = sorted(set(df["Topic"]))  # Get unique topics and sort
    topic_palette = sns.color_palette(palette, len(unique_topics))  # Generate palette
    topic_color_map = {topic: color for topic, color in zip(unique_topics, topic_palette)}  # Color mapping

    # Plot UMAP
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=df, x="UMAP1", y="UMAP2", hue="Topic", 
        palette=topic_color_map, alpha=0.8, s=100, edgecolor="k"
    )
    plt.title('')
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def visualize_genes_with_zscore(
    adata, 
    genes, 
    spot_size=450, 
    out_dir=None,
    figsize=(6, 6), 
    fontsize=7,
    frameon=None, 
    legend_loc="right margin", 
    colorbar_loc="right",
    show_title=True
):
    
    # Set global plotting parameters
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = "Arial"

    # Check if H_zscore exists
    if "H_zscore" not in adata.varm or adata.varm["H_zscore"] is None:
        raise ValueError("adata.varm['H_zscore'] does not exist or is None, cannot visualize!")

    # Determine genes type
    if isinstance(genes, list):
        genes_to_plot = {"all_genes": genes}
    elif isinstance(genes, dict):
        genes_to_plot = genes
    else:
        raise ValueError("Parameter genes should be list or dict!")

    # Iterate groups and gene lists
    for group_name, gene_list in genes_to_plot.items():
        sub_out_dir = os.path.join(out_dir, group_name) if out_dir else None
        if sub_out_dir:
            os.makedirs(sub_out_dir, exist_ok=True)

        for gene in gene_list:
            if gene not in adata.var_names:
                print(f"Warning: gene {gene} not in adata.var_names, skipping.")
                continue

            # Get z-score
            gene_idx = np.where(adata.var_names == gene)[0][0]
            gene_zscores = adata.varm["H_zscore"][gene_idx]
            gene_zscore = np.nanmax(gene_zscores) if isinstance(gene_zscores, (np.ndarray, list)) else gene_zscores
            if np.isnan(gene_zscore):
                gene_zscore = 0.0

            gene_title = [f"{gene} \n (z_score: {gene_zscore:.2f})"] if show_title else ""

            # Visualization
            sc.pl.spatial(
                adata, 
                color=[gene], 
                img_key=None,
                spot_size=spot_size, 
                cmap="inferno", 
                title=gene_title, 
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )

            # Save image
            if sub_out_dir:
                gene_safe = re.sub(r'[<>:"/\\|?*]', "_", gene)
                file_path = os.path.join(sub_out_dir, f"{gene_safe}_z_score_{gene_zscore:.2f}.png")
                plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)
                plt.close()


def plot_marker_genes(
    adata,
    userep="domain",
    domain=None,
    n_genes=8, 
    ncols=4,
    spot_size=1.3, 
    out_dir=None,
    figsize=(4, 4), 
    fontsize=10,
    frameon=None, 
    legend_loc="right margin", 
    colorbar_loc="right",
    show_title=True,
    palette_dict=None
):
    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial

    # Ensure domain is int and convert to categorical
    adata.obs[userep] = adata.obs[userep].astype(int)
    adata.obs[userep] = adata.obs[userep].astype("category")

    # Get all unique domains
    unique_domains = adata.obs[userep].cat.categories
    if palette_dict is None:
        palette = [
            '#F4E8B8',  '#EEC30E',  '#8F0100',  '#058187',  '#0C4DA7',  
            '#B44541',  '#632621',  '#92C7A3',  '#D98882',  '#6A93CB',  
            '#F0C94A',  '#AD6448',  '#4F6A9C',  '#CCB9A1',  '#0B3434',  
            '#3C4F76',  '#C1D354',  '#7D5BA6',  '#F28522',  '#4A9586',
            '#FF6F61',  '#D32F2F',  '#1976D2',  '#388E3C',  '#FBC02D', 
            '#8E24AA',  '#0288D1',  '#7B1FA2',  '#F57C00',  '#C2185B',
            '#1B4F72',  '#117864',  '#D4AC0D',  '#922B21',  '#6C3483',
            '#1F618D',  '#A04000',  '#196F3D',  '#2C3E50',  '#F39C12',
            '#7D6608',  '#4A235A',  '#D68910',  '#B03A2E',  '#7B241C',
            '#2471A3',  '#148F77',  '#9C640C',  '#6E2C00',  '#512E5F',
            '#154360',  '#145A32'
        ]

        palette_dict = {label: palette[i] for i, label in enumerate(unique_domains)}

    # Check required fields
    if "domain_mapping_topic" not in adata.uns:
        raise KeyError("Could not find adata.uns['domain_mapping_topic']")
    if "domain_to_genes" not in adata.uns:
        raise KeyError("Could not find adata.uns['domain_to_genes']")
    if userep not in adata.obs.columns:
        raise KeyError(f"{userep} not found in adata.obs")

    domain = str(domain)
    topic = adata.uns["domain_mapping_topic"].get(domain, None)
    if topic is None:
        raise ValueError(f"Could not find domain {domain} in adata.uns['domain_to_best_topic']")
    
    gene_corr_pairs = adata.uns["domain_to_genes"].get(domain, [])
    gene_list = [gene for gene, _ in gene_corr_pairs][:n_genes]
    gene_corr = [corr for _, corr in gene_corr_pairs][:n_genes]

    print(f" Domain {domain} → {topic}")
    print(f" Top {n_genes} genes: {gene_list}")
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
        # ---------- Figure 1: domain highlight ----------
        palette = {label: "lightgray" for label in unique_domains}
        palette[int(domain)] = palette_dict[int(domain)]
        domain_title = [f"Domain {domain}"] if show_title else ""
        sc.pl.spatial(
            adata,
            color=[userep],
            palette=palette,
            title=domain_title,
            img_key=None,
            spot_size=spot_size,
            frameon=False, 
            legend_loc=None, 
            wspace=0.1,
            colorbar_loc=None, 
            show=False
        )
        plt.savefig(os.path.join(out_dir, f"{domain}_highlight.png"), dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()
    
        # ---------- Figure 2: topic expression ----------
        if topic not in adata.obs.columns:
            raise ValueError(f"Topic {topic} not found in adata.obs.columns")
        topic_title = [f"Domain {domain} → {topic}"] if show_title else ""    
        sc.pl.spatial(
            adata,
            color=topic,
            title=topic_title,
            img_key=None,
            spot_size=spot_size,
            frameon=frameon, 
            legend_loc=legend_loc, 
            colorbar_loc=colorbar_loc, 
            show=False
        )
        plt.savefig(os.path.join(out_dir, f"{domain}_{topic}.png"), dpi=600, pad_inches=0)
        plt.close()
    
        # ---------- Figure 3: top n_genes ----------
        for top_n, gene in enumerate(gene_list):
            if gene not in adata.var_names:
                print(f" gene {gene} not found in adata.var_names, skipping.")
                continue
            gene_title = [f"{gene} (Corr:{gene_corr[top_n]:.3f})"] if show_title else ""  
            sc.pl.spatial(
                adata,
                color=gene,
                title=gene_title,
                spot_size=spot_size,
                frameon=frameon, 
                legend_loc=legend_loc, 
                colorbar_loc=colorbar_loc, 
                show=False
            )
            plt.savefig(os.path.join(out_dir, f"{domain}_{gene}.png"), dpi=600, pad_inches=0)
            plt.close()

    # ---------- display: domain and topics ----------    
    fig, axs = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]))  # Keep original size for each subplot

    sc.pl.spatial(
        adata,
        img_key=None,
        color=[userep],
        title=[userep],
        spot_size=spot_size,
        frameon=False,
        legend_loc=legend_loc,
        wspace=-1,
        colorbar_loc=None,
        palette=palette_dict,
        ax=axs[0],
        show=False
    )
    
    palette = {label: "lightgray" for label in unique_domains}
    palette[int(domain)] = palette_dict[int(domain)]
    domain_title = [f"Domain {domain}"] if show_title else ""
    sc.pl.spatial(
        adata,
        color=[userep],
        palette=palette,
        title=domain_title,
        img_key=None,
        spot_size=spot_size,
        frameon=False, 
        legend_loc=None, 
        wspace=0.1,
        colorbar_loc=None, 
        ax=axs[1],
        show=False
    )
    
    if topic not in adata.obs.columns:
        raise ValueError(f"Topic {topic} not found in adata.obs.columns")
    topic_title = [f"Domain {domain} → {topic}"] if show_title else ""   
    sc.pl.spatial(
        adata,
        color=topic,
        title=topic_title,
        img_key=None,
        spot_size=spot_size,
        frameon=False, 
        legend_loc=None, 
        wspace=0.1,
        colorbar_loc=None, 
        ax=axs[2],
        show=False
    )
        
    # ---------- display: top n_genes ----------
    title_list = []
    for top_n, gene in enumerate(gene_list):
        if gene not in adata.var_names:
            print(f" gene {gene} not found in adata.var_names, skipping.")
            continue
        title_list.append(f"{gene} (Corr:{gene_corr[top_n]:.3f})")
    
    sc.pl.spatial(
        adata,
        color=gene_list,
        title=title_list,
        spot_size=spot_size,
        frameon=False, 
        legend_loc=None, 
        ncols=ncols,
        colorbar_loc=None, 
        show=False
    )


def plot_topics(
    adata, 
    uns_key="final_topics", 
    img_key=None, 
    spot_size=1.3, 
    ncols=5,
    figsize=(4, 4),
    fontsize=10,
    frameon=None, 
    cmap="inferno",
    legend_loc="right margin", 
    colorbar_loc="right",
    show=True
):
    plt.rcParams["figure.figsize"] = figsize  # Set global figure size
    plt.rcParams["font.size"] = fontsize      # Set global font size
    plt.rcParams["font.family"] = "Arial"     # Set global font to Arial
    
    # Get and copy final_topics
    final_topics = adata.uns[uns_key].copy()
    
    # Add prefix for visualization labels
    labeled_topics = [f"Topic_{idx}" for idx in final_topics]
    
    # Plot
    sc.pl.spatial(
        adata,
        img_key=img_key,
        color=labeled_topics,
        title=labeled_topics,
        spot_size=spot_size,
        frameon=frameon, 
        cmap=cmap,
        wspace=-0.05,
        legend_loc=legend_loc, 
        colorbar_loc=colorbar_loc, 
        ncols=ncols,
        show=show
    )

def plot_neighbors_cut(
    adata,
    img_key=None,
    spot_size=1.3,
    figsize=(4, 4),
    frameon=False,
    legend_loc=None,
    colorbar_loc=None,
    show=True
):
    # Extract original and optimized adjacency matrices
    adj_orig = adata.obsm["adj"]
    adj_opt = adata.obsm["adj_opt"]

    # Convert to sparse format if not already
    adj_orig = adj_orig.tocsr()
    adj_opt = adj_opt.tocsr()

    # Difference in number of neighbors per spot
    deg_orig = np.array(adj_orig.sum(axis=1)).flatten()
    deg_opt = np.array(adj_opt.sum(axis=1)).flatten()
    reduced_neighbors = deg_orig - deg_opt

    # Write to adata.obs
    adata.obs["neighbors_cut"] = reduced_neighbors

    # Set figure size and plot spatial
    plt.rcParams["figure.figsize"] = figsize
    sc.pl.spatial(
        adata,
        img_key=img_key,
        color=["neighbors_cut"],
        title=[" "],
        spot_size=spot_size,
        frameon=frameon,
        legend_loc=legend_loc,
        colorbar_loc=colorbar_loc,
        show=show
    )
