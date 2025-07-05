import torch
import numpy as np
import torch.nn.functional as F
import scanpy as sc
from tqdm import tqdm
from .preprocess import preprocess_adj, preprocess_adj_sparse, get_feature, fix_seed,preprocess_adj
from .utils import plot_loss
from .model import SAGE_model,SwAVLoss


class SAGE():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=32,
        random_seed = 42,
        lamda1 = 10,
        lamda2 = 1,
        technology = '10X',
        ):
        
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.technology = technology
        self.n_clusters = len(adata.obs["pre_domain"].unique())

        fix_seed(self.random_seed)

        # Initialize model, loss function, and optimizer
        self.model = None
        self.loss_CSL = None
        self.optimizer = None
        self.swav_loss = None

        if 'X_pca' not in adata.obsm.keys():
            if 'highly_variable' not in self.adata.var.keys():
                sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.pca(self.adata, n_comps=50, mask_var="highly_variable", svd_solver='arpack')

        if 'feat' not in self.adata.obsm.keys():
           get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj_opt']
        self.adj_feat = self.adata.obsm['adj_feat']
        self.adj_combine = self.adj  +  self.adj_feat
        self.adj_edge_index = torch.tensor(self.adj.toarray(), dtype=torch.float32, device=device).nonzero(as_tuple=False).t()
        self.adj_feat_edge_index = torch.tensor(self.adj_feat.toarray(), dtype=torch.float32, device=device).nonzero(as_tuple=False).t()
        self.adj_combine_index = torch.tensor(self.adj_combine.toarray(), dtype=torch.float32, device=device).nonzero(as_tuple=False).t()
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_hid = 64
        self.dim_output = dim_output
        
        if self.technology in ['Stereo', 'Slide']:
           # Using sparse matrix
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # Standard version
           self.adj  = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
           self.adj_feat  = preprocess_adj(self.adj_feat)
           self.adj_feat = torch.FloatTensor(self.adj_feat).to(self.device)


    def train(self, epochs_t = None):
        if epochs_t is None:
            epochs_t = self.epochs
            
        if self.model is None:
            self.model = SAGE_model(self.dim_input, self.dim_hid, self.dim_output, dropout=0.0).to(self.device)
            
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                                weight_decay=self.weight_decay)
            
        if self.swav_loss is None:
            self.swav_loss = SwAVLoss(n_clusters = self.n_clusters)

        print(">>> SAGE model construction completed, training begins.")

        loss_KL_list = []
        loss_recon_coord_list = []
        loss_recon_feat_list = []
        loss_recon_att_list = []
        loss_list = []
        loss_graph_list =[]

        self.model.train() 

        epoch_bar = tqdm(range(epochs_t), desc="Training Progress", dynamic_ncols=True)
        

        for epoch in epoch_bar: 
            results = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)
            
            self.loss_KL = self.swav_loss(results['emb_latent_coord'] , 
                                   results['emb_latent_feat'], 
                                   results['emb_latent_att'], 
                                   epoch=epoch)

            self.loss_att = self.model.calc_attn_reg(results)

            self.graph_loss = self.model.calc_graph_loss(results, self.adj, self.adj_feat)
            
            self.loss_recon_coord = F.mse_loss(self.features, results['emb_recon_coord'])
            self.loss_recon_feat = F.mse_loss(self.features, results['emb_recon_feat'])
            self.loss_recon_att = F.mse_loss(self.features, results['emb_recon_att'])


            # Get dynamic weights
            weights = self.get_dynamic_weights(epoch, epochs_t)
            
            # Loss calculation (with moving average normalization)
            loss = (
                weights['recon'] * ((self.loss_recon_coord + self.loss_recon_feat + self.loss_recon_att)/3) +
                weights['graph'] * self.graph_loss +
                weights['kl'] * self.loss_KL +
                weights['att'] * self.loss_att
            )


            
            # Gradient clipping (prevent KL loss gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

            # Save loss 

            loss_recon_coord_list.append(self.loss_recon_coord.cpu().item())
            loss_recon_feat_list.append(self.loss_recon_feat.cpu().item())
            loss_recon_att_list.append(self.loss_recon_att.cpu().item())
            loss_graph_list.append(self.graph_loss.cpu().item())
            loss_KL_list.append(self.loss_KL.cpu().item())

            loss_list.append(loss.cpu().item())

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Plot
        # plot_loss(epochs_t,loss_recon_coord_list,loss_name='loss_recon_coord')
        # plot_loss(epochs_t,loss_recon_feat_list,loss_name='loss_recon_feat')
        # plot_loss(epochs_t,loss_recon_att_list,loss_name='loss_recon_att')
        # plot_loss(epochs_t,loss_graph_list,loss_name='graph_loss')
        # plot_loss(epochs_t,loss_KL_list,loss_name='loss_KL')
        # plot_loss(epochs_t,loss_list,loss_name='loss')

        print(">>> Optimization finished for ST data!")
    
        with torch.no_grad():
            self.model.eval()

            self.emb_recon_coord = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_recon_coord']
            self.emb_recon_feat = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_recon_feat']
            self.emb_recon_att = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_recon_att']
            
            self.emb_latent_coord = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_latent_coord']
            self.emb_latent_feat = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_latent_feat']
            self.emb_latent_att = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['emb_latent_att']

            self.alpha = self.model(self.features, self.adj_edge_index, self.adj_feat_edge_index, self.adj_combine_index)['alpha']

            self.adata.obsm['emb_rec_coord'] = self.emb_recon_coord.detach().cpu().numpy()
            self.adata.obsm['emb_rec_feat'] = self.emb_recon_feat.detach().cpu().numpy()
            self.adata.obsm['emb_rec_att'] = self.emb_recon_att.detach().cpu().numpy()

            self.adata.obsm['emb_latent_coord'] = self.emb_latent_coord.detach().cpu().numpy()
            self.adata.obsm['emb_latent_feat'] = self.emb_latent_feat.detach().cpu().numpy()
            self.adata.obsm['emb_latent_att'] = self.emb_latent_att.detach().cpu().numpy() 

            self.adata.obsm['alpha'] = self.alpha.detach().cpu().numpy() 

            return self.adata
    
    def get_dynamic_weights(self, epoch, max_epoch=300):
        """Dynamic weight calculation by stage"""
        # Basic weight configuration
        weights = {
            'recon': 10.0,  # Feature reconstruction group
            'graph': 10.0,   # Graph structure
            'kl': 5.0,       # Clustering loss
            'att': 1.0       # Attention regularization
        }
        
        phase = epoch / max_epoch
        
        # Phase 1: 0-30% epochs - Feature learning
        if phase < 0.3:
            weights['recon'] *= 1.0 - 0.2*(phase/0.3)  # Slowly decay
            weights['graph'] *= 0.8 + 0.2*(phase/0.3)  # Slowly increase
            weights['kl'] *= 0.5*(phase/0.3)           # Linear growth from 0
        
        # Phase 2: 30-70% epochs - Structure optimization
        elif phase < 0.7:
            progress = (phase - 0.3) / 0.4
            weights['recon'] *= 0.8*(1 - progress) + 0.6*progress
            weights['graph'] *= 1.2*(1 - progress) + 1.5*progress
            weights['kl'] *= 1.0 + 2.0*progress
        
        # Phase 3: 70-100% epochs - Clustering fine-tuning
        else:
            progress = (phase - 0.7) / 0.3
            weights['recon'] = 5.0 * (1 - progress) + 3.0*progress
            weights['graph'] = 8.0 * (1 - 0.5*progress)
            weights['kl'] = 15.0 * (1 + 0.2*progress)
        
        return weights
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
