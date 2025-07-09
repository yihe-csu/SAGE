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
Dataset = "Human_Breast_Cancer"

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
df_meta_layer = df_meta['ground_truth']
adata.obs['ground_truth'] = df_meta_layer.values
# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

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
con_range = ( 0.1, 2 , 0.1)
con_use_rep = 'X_pca'
n_neig_coord=6
n_neig_feat =6
con_dim =25
con_radius=20
con_refine=False
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
                    spot_size=400, 
                    ncols=6,
                    figsize=(4, 4),
                    fontsize=20,
                    frameon=False, 
                    legend_loc=False, 
                    colorbar_loc=None,
                    show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/HBC_topics_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
plt.show()

# Construction of high-specificity genes (HSGs)
SAGE.preprocess.genes_selection(adata, n_genes=3000)

# Consensus-driven Graph Construction
SAGE.preprocess.optimize_graph_topology(
    adata,
    n_neig_coord=6,
    cut_side_thr=0.4,
    min_neighbor=2,
    n_neig_feat=15,
)

# Draw SAG edge
SAGE.plot.plot_neighbors_cut(adata,
                   img_key=None,
                   spot_size=400,
                   figsize=(3, 3),
                   frameon=False,
                   legend_loc=None,
                   colorbar_loc=None,
                   show=False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/HBC_cut_plot.png",bbox_inches='tight', dpi=1200,pad_inches =0)
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
combine_palette ={
    '1': '#632621',            '6': '#92C7A3',           '11': '#EEC30E',           '16': '#CCB9A1',
    '2': '#0B3434',            '7': '#F0C94A',           '12': '#B44541',           '17': '#3C4F76',
    '3': '#0C4DA7',            '8': '#4F6A9C',           '13': '#AD6448',           '18': '#7D5BA6',
    '4': '#058187',            '9': '#8F0100',           '14': '#F4E8B8',           '19': '#F28522',
    '5': '#C1D354',           '10': '#D98882',           '15': '#6A93CB',           '20': '#4A9586',
    
    'DCIS/LCIS_1': '#C1D354',  'IDC_2': '#0C4DA7',       'Tumor_edge_2': '#92C7A3', 'IDC_7': '#0B3434',
    'DCIS/LCIS_2': '#7D5BA6',  'IDC_3': '#058187',       'Tumor_edge_3': '#B44541', 'IDC_8': '#632621',
    'DCIS/LCIS_4': '#CCB9A1',  'IDC_4': '#EEC30E',       'Tumor_edge_4': '#F28522', 'Tumor_edge_6': '#F0C94A',
    'DCIS/LCIS_5': '#D98882',  'IDC_5': '#4F6A9C',       'Tumor_edge_5': '#F4E8B8', 'IDC_6': '#6A93CB',
    'Healthy_1': '#8F0100',    'Healthy_2': '#AD6448',   'IDC_1': '#4A9586',        'Tumor_edge_1': '#3C4F76'}


plt.rcParams["figure.figsize"] = (4,4)
plt.rcParams['font.size'] = 10
labels_true = adata.obs["ground_truth"]
labels_pred = adata.obs["domain"]
NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
ARI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sc.pl.spatial(adata, 
            img_key = None, 
            spot_size = 400,
            color = ["ground_truth"],
            title = ["Manual annotation"], 
            palette=combine_palette,  
            na_in_legend = False,
            frameon=False,
            ncols = 2,
            size = 1,
            ax = axes[0],
            show= False)
sc.pl.spatial(adata, 
            img_key = None, 
            spot_size = 400,
            color = ["domain"],
            title = [f'(ARI={ARI:.3f} NMI={NMI:.3f})'], 
            palette=combine_palette,  
            na_in_legend = False,
            frameon=False,
            ncols = 2,
            size = 1,
            ax = axes[1],
            show= False)
plt.tight_layout()
plt.savefig(f"{output_result_dir}/HBC_Manual_Domain.png",bbox_inches='tight', dpi=1200,pad_inches =0)

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
    domain='4',
    n_genes=4, 
    ncols=4,
    spot_size=400, 
    out_dir=None,
    figsize=(4, 4), 
    fontsize=10,
    frameon=None, 
    legend_loc='on data', 
    colorbar_loc="right",
    show_title=True,
    palette_dict=combine_palette,
)
plt.savefig(f"{output_result_dir}/Domain_4_marker_genes.png",bbox_inches='tight', dpi=1200,pad_inches =0)