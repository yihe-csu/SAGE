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

base_path = "C:/Users/heyi/Desktop/code_iteration/SAGE-main"
Dataset = "DLPFC_151673"

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

# Read data from input_dir
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()
adata.raw =adata.copy()

# add ground_truth
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
df_meta_layer = df_meta['layer_guess']
adata.obs['ground_truth'] = df_meta_layer.values
adata = adata[~pd.isnull(adata.obs['ground_truth'])] # filter out NA nodes

# Expression data preprocessing
adata = adata.copy()
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.filter_genes(adata, min_counts=5)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50, mask_var="highly_variable", svd_solver='arpack')

# Multi-resolution Consensus Clustering
con_method = "mclust"  #["leiden","mclust"]
con_range = ( 2, 9 , 1)
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
                    spot_size=180, 
                    ncols=5,
                    figsize=(4, 4),
                    fontsize=10,
                    frameon=False, 
                    legend_loc=False, 
                    colorbar_loc=None,
                    show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/DLPFC_151673_topics_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Construction of high-specificity genes (HSGs)
SAGE.preprocess.genes_selection(adata, n_genes=3000)

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
                   spot_size=180,
                   figsize=(5, 5),
                   frameon=False,
                   legend_loc=None,
                   colorbar_loc=None,
                   show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/DLPFC_151673_cut_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Run SAGE
model = SAGE.SAGE(adata, device=device, epochs=800)
adata = model.train()

# Domain clustering
n_clusters = len(adata.obs["ground_truth"].unique())
SAGE.utils.clustering(adata, 
                     data= adata.obsm["emb_latent_att"], 
                     method='mclust', 
                     n_clusters=n_clusters, 
                     res = 0.3, 
                     radius=30,  
                     refinement=True)

# Save result
adata.write_h5ad(output_result_dir+"/result.h5")
clusters = adata.obs["domain"] 
clusters.to_csv(output_result_dir+f"/clusters.csv",header = False)
embedding = adata.obsm["emb_latent_att"]
np.savetxt(output_result_dir+"/embedding.txt",embedding)
SAGE.utils.export_H_zscore_to_csv(adata, out_dir=output_process_dir)
SAGE.utils.export_Corr_to_csv(adata, out_dir=output_process_dir)

# Plot clustering results
adata = sc.read(output_result_dir+'/result.h5')
combine_palette = {
    '0': '#F4E8B8',   '1': '#058187',   '2': '#632621',   '3': '#F4E8B8',
    '4': '#B44541',   '5': '#0C4DA7',   '6': '#EEC30E',   '7': '#8F0100',
    'Layer1': '#EEC30E', 'Layer2': '#0C4DA7', 'Layer3': '#F4E8B8', 'Layer4': '#B44541',
    'Layer5': '#632621', 'Layer6': '#058187', 'WM': '#8F0100'
                  }
plt.rcParams["figure.figsize"] = (4,4)
labels_true = adata.obs["ground_truth"]
labels_pred = adata.obs["domain"]
NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
sc.pl.spatial(adata, 
            img_key = None, 
            spot_size = 180,
            color = ["ground_truth","domain"],
            title = ["Manual annotation",f'151673 (NMI={NMI:.3f})'], 
            palette=combine_palette,  
            na_in_legend = False,
            frameon=False,
            ncols = 2,
            size = 1,
            show= False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/151673_Manual_Domain.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Plot UMAP and PAGA graph
SAGE_embed = pd.read_csv(output_result_dir+"/embedding.txt",header = None, delim_whitespace=True)
labels_true = adata.obs["ground_truth"].copy()
SAGE_embed.index = labels_true.index
adata.obsm["SAGE"] = SAGE_embed
plt.rcParams["figure.figsize"] = (6,5)
sc.pp.neighbors(adata, use_rep="SAGE")
sc.tl.umap(adata)
sc.tl.paga(adata,groups='ground_truth')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc.pl.umap(adata,
           color='ground_truth',
           palette=combine_palette,
           legend_loc='on data',
           legend_fontoutline=5,
           add_outline=True, s=150,
           outline_width=(0.8, 0.05),
           legend_fontsize=25,
           frameon=False,
           ax=axes[0],   
           show=False)  
axes[0].set_title('UMAP', fontsize=25)
umap_coords = pd.DataFrame(adata.obsm['X_umap'],
                           index=adata.obs.index,
                           columns=['UMAP1', 'UMAP2'])
clusters = adata.obs['ground_truth'].astype(str)
cluster_centers = umap_coords.groupby(clusters).mean()
sc.pl.paga(adata,
           color='ground_truth',
           pos=cluster_centers.values,
           node_size_scale=30,
           fontoutline=5,
           frameon=False,
           edge_width_scale=3,
           fontsize=25,
           fontweight='bold',
           ax=axes[1], 
           show=False)
axes[1].set_title('PAGA', fontsize=25)

plt.tight_layout()
plt.savefig(f"{output_result_dir}/151673_UMAP_PAGA.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()
