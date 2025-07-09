import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

class Encoder_GCN(nn.Module):
    def __init__(self, in_feat, hidden_feat,out_feat, dropout=0.0, act=F.relu):
        super(Encoder_GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

        self.gc1 = GCNConv(in_feat, hidden_feat)
        self.fc1 = nn.Linear(hidden_feat, out_feat)

    def forward(self, x, edge_index_adj):
        x = self.gc1(x, edge_index_adj)
        x = self.fc1(x)
        return x 
    
class Decoder_GCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder_GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

        self.fc1 = nn.Linear(in_feat, hidden_feat)
        self.gc1 = GCNConv(hidden_feat, out_feat)

    def forward(self, x, edge_index_adj):
        x = self.fc1(x)
        x = self.gc1(x, edge_index_adj)
        return x 

class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        adj_reconstructed = torch.sigmoid(torch.matmul(z, z.t()))  # Decode edges by inner product
        return adj_reconstructed


class AttentionLayer(torch.nn.Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6,dim=-1)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha              

    
class SAGE_model(nn.Module):
    
    def __init__(self, dim_in_features,dim_hid_features, dim_out_features, dropout=0.0, act=nn.PReLU()):
        super().__init__()
        self.dim_in_features = dim_in_features
        self.dim_hid_features = dim_hid_features
        self.dim_out_features = dim_out_features
        self.dropout = dropout
        self.act = act
        self.sigm = nn.Sigmoid()

        self.encode_gcn_coord = Encoder_GCN(self.dim_in_features, self.dim_hid_features, self.dim_out_features)
        self.encode_gcn_feat = Encoder_GCN(self.dim_in_features, self.dim_hid_features, self.dim_out_features)

        self.decode_gcn = Decoder_GCN(self.dim_out_features, self.dim_hid_features, self.dim_in_features)

        self.decode_gcn_att = Decoder_GCN(self.dim_out_features, self.dim_hid_features, self.dim_in_features)

        self.attentionlayer = AttentionLayer(self.dim_out_features,self.dim_out_features)
        
    def forward(self, features, adj_coord_ei, adj_feat_ei, adj_combine_ei):

        emb_latent_coord = self.encode_gcn_coord(features,adj_coord_ei)
        emb_latent_feat = self.encode_gcn_feat(features,adj_feat_ei)

        emb_latent_att, alpha = self.attentionlayer(emb_latent_coord, emb_latent_feat)

        emb_recon_coord = self.decode_gcn(emb_latent_coord,adj_coord_ei)

        emb_recon_feat  =  self.decode_gcn(emb_latent_feat,adj_feat_ei)
        
        emb_recon_att   =   self.decode_gcn_att(emb_latent_att,adj_combine_ei)
          

        results = {'emb_latent_att':emb_latent_att,
                  'emb_recon_coord':emb_recon_coord,
                  'emb_recon_feat':emb_recon_feat,
                  "emb_recon_att":emb_recon_att,
                  'emb_latent_coord':emb_latent_coord,
                  'emb_latent_feat':emb_latent_feat,
                  'alpha':alpha,
                  }
        
        return results
    
    def bce_loss(self, adj_reconstructed, adj_original):
  
        # Convert to float type
        adj_original = adj_original.to(torch.float32)
        adj_reconstructed = adj_reconstructed.to(torch.float32)

        # BCE loss, calculate the difference between the reconstructed probability of each node pair and the original adjacency matrix
        loss = F.binary_cross_entropy(adj_reconstructed, adj_original)

        return loss

    def calc_graph_loss(self, results, adj_coord, adj_feat):

        z = results['emb_latent_att']
        sim_matrix = torch.sigmoid(torch.mm(z, z.T))
        
        # Coordinate graph structure preservation
        loss_coord = F.binary_cross_entropy(
            sim_matrix[adj_coord > 0], 
            torch.ones_like(sim_matrix[adj_coord > 0]))
        
        # Feature graph false edge penalty
        loss_feat = F.binary_cross_entropy(
            sim_matrix[adj_feat > 0], 
            torch.ones_like(sim_matrix[adj_feat > 0]))
        
        return 0.3*loss_coord + 0.7*loss_feat
        
    
    def calc_attn_reg(self, results):

        alpha = results['alpha']
        return torch.mean((alpha - 0.5)**2)  # Force balance between two views

def sinkhorn_knopp(
    Q,                  # Input matrix [n, m]
    n_iters=10,         # Maximum number of iterations
    epsilon=1e-3,       # Convergence threshold
):

    assert Q.min() >= 0, "Input matrix must be non-negative"
    
    for _ in range(n_iters):
        # Row normalization (ensure row sum is 1)
        Q_row = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
        
        # Column normalization (ensure column sum is 1)
        Q = Q_row / (Q_row.sum(dim=0, keepdim=True) + 1e-8)
        
        # Calculate error
        row_error = torch.abs(Q.sum(dim=1) - 1).mean()
        col_error = torch.abs(Q.sum(dim=0) - 1).mean()
        error = (row_error + col_error).item()
            
        # Check convergence
        if error < epsilon:
            break

    return Q

class SwAVLoss(torch.nn.Module):
    def __init__(self, n_clusters=10, epsilon=0.05, n_iters=10, update_interval=10):
        super(SwAVLoss, self).__init__()
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.update_interval = update_interval  # Update cluster centers every n epochs
        self.kmeans = None  # Clustering model
        self.current_epoch = 0  # Current training epoch

    def fit_kmeans(self, z):

        # Convert input tensor z to numpy array for K-means
        z_np = z.detach().cpu().numpy()

        # Perform K-means clustering to get cluster centers
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=50, random_state=42)
        self.kmeans.fit(z_np)  # Train K-means clustering
        # print(f"Cluster centers: {self.kmeans.cluster_centers_}")
    
    def _should_update_clusters(self, epoch):
 
        # Update if epoch meets interval or if cluster centers have not been trained
        return (epoch % self.update_interval == 0) or (self.kmeans is None)
    

    def forward(self, z1, z2, z3, epoch=None):
  
        if self.kmeans is None:
            # If K-means clustering has not been trained, train once
            self.fit_kmeans(z3)  

        # Use trained cluster centers to generate cluster assignments
        Q1 = self._get_cluster_assignments(z1)
        Q2 = self._get_cluster_assignments(z2)

        # Normalize cluster assignments
        Q1 = sinkhorn_knopp(Q1, n_iters=self.n_iters, epsilon=self.epsilon)
        Q2 = sinkhorn_knopp(Q2, n_iters=self.n_iters, epsilon=self.epsilon)

        # Swap assignments and compute KL divergence
        loss = F.kl_div(Q1.log(), Q2, reduction='batchmean') + \
               F.kl_div(Q2.log(), Q1, reduction='batchmean')
        
        # Update current epoch
        if epoch is not None:
            self.current_epoch = epoch
        return loss

    def _get_cluster_assignments(self, z):

        # Convert input tensor z to numpy array for distance calculation
        z_np = z.detach().cpu().numpy()

        # Compute distance from each sample to cluster centers
        distances = self.kmeans.transform(z_np)  # (batch_size, n_clusters)

        # Convert distances to probability distribution (softmax)
        Q = torch.tensor(distances, dtype=torch.float32, device=z.device)
        Q = F.softmax(-Q / self.epsilon, dim=-1)  # Softmax probability of negative distance
        return Q
