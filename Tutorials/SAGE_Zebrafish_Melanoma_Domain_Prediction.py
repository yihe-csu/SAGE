import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import importlib
import SAGE
sys.path.append(os.path.abspath("C://Users//heyi//Desktop/code_iteration/SAGE-main/"))
print(SAGE.__version__)
print(SAGE.__author__)
print(SAGE.__email__)

# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'E:\\R-4.4.1'

base_path = "C:/Users/heyi/Desktop/code_iteration/SAGE-main/"
Dataset = "Zebrafish_melanomas/Sample_B"

file_fold = f'{base_path}/Dataset/{Dataset}/'
# Set directory (If you want to use additional data, please change the file path)
output_dir=f"{base_path}/Result/{Dataset}"
output_process_dir = output_dir + "/process_data"
output_result_dir = output_dir + "/result_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_process_dir):
    os.makedirs(output_process_dir)
if not os.path.exists(output_result_dir):
    os.makedirs(output_result_dir)
    
feature_bc_matrix_path = file_fold + f"\\filtered_feature_bc_matrix.h5"
pos_path = file_fold + f"\\tissue_positions_list.csv.gz"
img_path =  file_fold + f"\\image.tif.gz"

adata = sc.read_10x_h5(feature_bc_matrix_path)
positions = pd.read_csv(pos_path, header=None)
positions.columns = ["barcode", "in_tissue", "array_x", "array_y", "image_x", "image_y"]
positions.set_index("barcode", inplace=True)
max_x = positions['image_x'].max()
positions["image_x"] = max_x - positions["image_x"]
positions = positions.loc[adata.obs.index]
adata.obs = adata.obs.join(positions)
adata.obsm["spatial"] = adata.obs[["image_x", "image_y"]].values
adata.var_names_make_unique()

# Expression data preprocessing
adata = adata.copy()
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.filter_genes(adata, min_counts=5)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50, mask_var="highly_variable", svd_solver='arpack')

# Multi-resolution Consensus Clustering
con_method = "leiden"  #["leiden","mclust"]
con_range = ( 0.1, 3 , 0.1)
con_use_rep = 'X_pca'
n_neig_coord=6
n_neig_feat =6
con_dim =25
con_radius=20
con_refine=True

SAGE.consensus_clustering(adata, 
                          method=con_method, 
                          resolution_range=con_range, 
                          n_neighbors=n_neig_feat, 
                          use_rep=con_use_rep, 
                          dims=con_dim, 
                          radius=con_radius, 
                          refinement=con_refine)

# Topic Selection via Supervised Learning
SAGE.preprocess.topics_selection(adata, n_topics=30)

# Draw Topics detected by RF
SAGE.plot.plot_topics(adata, 
                    uns_key="final_topics", 
                    img_key=None, 
                    spot_size=500, 
                    ncols=9,
                    figsize=(4, 4),
                    fontsize=20,
                    frameon=False, 
                    legend_loc=False, 
                    colorbar_loc=None,
                    show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/ZM_topics_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Construction of high-specificity genes (HSGs)
SAGE.preprocess.genes_selection(adata, n_genes=3000, lower_p=10, upper_p=99)

# Consensus-driven Graph Construction
SAGE.preprocess.optimize_graph_topology(
    adata,
    n_neig_coord=6,
    cut_side_thr=0.3,
    min_neighbor=2,
    n_neig_feat=15,
)

# Draw SAG edge
SAGE.plot.plot_neighbors_cut(adata,
                   img_key=None,
                   spot_size=500,
                   figsize=(5, 5),
                   frameon=False,
                   legend_loc=None,
                   colorbar_loc=None,
                   show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/ZM_cut_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Run SAGE
model = SAGE.SAGE(adata, device=device, epochs=800)
adata = model.train()

# Domain clustering
n_clusters = 20
SAGE.utils.clustering(adata, 
                     data= adata.obsm["emb_latent_att"], 
                     method='mclust', 
                     n_clusters=n_clusters, 
                     res = 0.3, 
                     radius=30,  
                     refinement=False)

# Save result
adata.write_h5ad(output_result_dir+"/result.h5")
clusters = adata.obs["domain"] 
clusters.to_csv(output_result_dir+f"/clusters.csv",header = False)
embedding = adata.obsm["emb_latent_att"]
np.savetxt(output_result_dir+"/embedding.txt",embedding)
SAGE.utils.export_H_zscore_to_csv(adata, out_dir=output_process_dir)
SAGE.utils.export_Corr_to_csv(adata, out_dir=output_process_dir)

# Plot clustering results
adata = sc.read(output_result_dir+'/result.h5') # Read data
custom_palette = [
    '#F4E8B8',  '#EEC30E',  '#8F0100',  '#058187',  '#0C4DA7',  
    '#B44541',  '#632621',  '#92C7A3',  '#D98882',  '#6A93CB',  
    '#F0C94A',  '#AD6448',  '#4F6A9C',  '#CCB9A1',  '#0B3434',  
    '#3C4F76',  '#C1D354',  '#7D5BA6',  '#F28522',  '#4A9586',
    '#FF6F61',  '#D32F2F',  '#1976D2',  '#388E3C',  '#FBC02D', 
    '#8E24AA',  '#0288D1',  '#7B1FA2',  '#F57C00',  '#C2185B'
]
cluster_palette = {int(cluster): custom_palette[i % len(custom_palette)] 
                   for i, cluster in enumerate(adata.obs['domain'].unique())}

plt.rcParams["figure.figsize"] = (4,4)
plt.rcParams['font.size'] = 10
labels_pred = adata.obs["domain"]
X = adata.obsm["emb_latent_att"]
SC = metrics.silhouette_score(X, labels_pred)
DB = metrics.davies_bouldin_score(X, labels_pred)

sc.pl.spatial(adata, 
            img_key = None, 
            spot_size = 500,
            palette=cluster_palette,
            color = ["domain"],
            title = [f'(SC={SC:.3f}  DB={DB:.3f})'], 
            na_in_legend = False,
            frameon=False,
            show= False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/ZM_Domain.png",bbox_inches='tight', dpi=1200,pad_inches =0)

# Marker genes detection
SAGE.utils.run_domain_gene_mapping_pipeline(
    adata,
    useref="domain",
    topic_key="W_nmf",
    corr_threshold=0.2,
    topic_gene_corr_key="gene_topic_corr",
    domain_mapping_topic_key="domain_mapping_topic",
    marker_genes_dict_key="marker_genes_multi_dict",
    domain_to_genes_key="domain_to_genes"
)

SAGE.plot.plot_marker_genes(
    adata,
    userep="domain",
    domain='13',
    n_genes=4, 
    ncols=4,
    spot_size=500, 
    out_dir=None,
    figsize=(4, 4), 
    fontsize=10,
    frameon=None, 
    legend_loc='on data', 
    colorbar_loc="right",
    show_title=True,
    palette_dict=cluster_palette
) 
plt.savefig(f"{output_result_dir}/Domain_13_marker_genes.png",bbox_inches='tight', dpi=1200,pad_inches =0)

# Plot UMAP and PAGA graph
adata = sc.read(output_result_dir+'/result.h5')
adata.var_names_make_unique()
SAGE_embed = pd.read_csv(output_result_dir+"/embedding.txt",header = None, delim_whitespace=True)
labels_true = adata.obs["domain"].copy()
SAGE_embed.index = labels_true.index
adata.obsm["SAGE"] = SAGE_embed
adata.obsm["SAGE"].columns = adata.obsm["SAGE"].columns.astype(str)
# Plot
plt.rcParams["figure.figsize"] = (6,5)
sc.pp.neighbors(adata, use_rep="emb_latent_att")
sc.tl.umap(adata)
sc.tl.paga(adata,groups='domain')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc.pl.umap(adata,
           color='domain',
           palette=cluster_palette,
           legend_loc='on data',
           legend_fontoutline=3,
           add_outline=True, s=30,
           outline_width=(0.3, 0.05),
           legend_fontsize=10,
           frameon=False,
           ax=axes[0],   
           show=False)  

axes[0].set_title('UMAP', fontsize=20)

umap_coords = pd.DataFrame(adata.obsm['X_umap'],
                           index=adata.obs.index,
                           columns=['UMAP1', 'UMAP2'])

clusters = adata.obs['domain'].astype(int)
cluster_centers = umap_coords.groupby(clusters).mean()

sc.pl.paga(adata,
           color='domain',
           pos=cluster_centers.values,
           node_size_scale=10,
           fontoutline=3,
           frameon=False,
           edge_width_scale=1,
           fontsize=10,
           # fontweight='bold',
           ax=axes[1], 
           show=False)

axes[1].set_title('PAGA', fontsize=20)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/ZM_UMAP_PAGA.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Plot Topic-gene UMAP
topics_list = ['1','2','3','6','18','26']
marker_genes_umap_df, selected_order = SAGE.utils.analyze_topics_and_umap(
                        adata, 
                        method="pearson",  
                        corr_threshold=0.2,   
                        n_pca_components=50, 
                        n_neighbors=15,
                        embedding_method="umap",  
                        min_dist=0.1, 
                        random_state=42, 
                        n_genes=20,  
                        topics_list=topics_list 
                    )
SAGE.plot.plot_umap(marker_genes_umap_df)