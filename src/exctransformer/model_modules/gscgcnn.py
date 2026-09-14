import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_



class GraphSAGELayer(nn.Module):
    def __init__(self, 
                 node_in_dim, 
                 nbr_in_dim,
                 out_dim):
        """
        One GraphSAGE layer.
        Parameters:
            node_in_dim: Dimension of the atom features.
            nbr_in_dim: Dimension of the aggregated neighbor features.
            out_dim: Output dimension after this layer.
        """
        super(GraphSAGELayer, self).__init__()
        self.fc = nn.Linear(node_in_dim + nbr_in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, node_features, nbr_features):
        """
        Parameters:
            node_features : Tensor of shape (N, node_in_dim)
            nbr_features  : Tensor of shape (N, M, nbr_in_dim)
        Returns:
            Updated node features of shape (N, out_dim)
        """
        nbr_agg = torch.mean(nbr_features, dim=1)  # (N, nbr_in_dim)
        h_cat = torch.cat([node_features, nbr_agg], dim=1)  # (N, node_in_dim + nbr_in_dim)
        h = self.fc(h_cat)
        h = self.bn(h)
        h = self.relu(h)
        return h

class GraphSAGENet(nn.Module):
    def __init__(self, 
                 atom_fea_len, 
                 nbr_fea_len, 
                 num_layers,
                 hidden_dims=128, 
                 output_dim=128):
      
        super(GraphSAGENet, self).__init__()
        init_nbr_dim = 2 * atom_fea_len + nbr_fea_len
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(GraphSAGELayer(atom_fea_len, init_nbr_dim, hidden_dims))

        for i in range(1, self.num_layers):
            self.layers.append(GraphSAGELayer(hidden_dims, hidden_dims, output_dim))

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Parameters:
            atom_in_fea : Tensor of shape (N, atom_fea_len)
            nbr_fea     : Tensor of shape (N, M, nbr_fea_len)
            nbr_fea_idx : LongTensor of shape (N, M) indicating neighbor indices
            N         : Total number of atoms per batch
            M         : Total Number of neighbors per atom
        Returns:
            Final node embeddings.
        """
        N, M = nbr_fea_idx.shape

        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, atom_in_fea.shape[1]),  
            atom_nbr_fea,                                                   
            nbr_fea                                                         
        ], dim=2)  # (N, M, 2*atom_fea_len + nbr_fea_len)
        h = self.layers[0](atom_in_fea, total_nbr_fea)
        for layer in self.layers[1:]:
            total_h = h[nbr_fea_idx, :]  # (N, M, hidden_dim)
            h = layer(h, total_h)
        return h


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=6, dropout=0.0):
        """
        Multi-head attention mechanism that computes attention
          over the node embedding pairs
        generated from the GraphSAGE network.
        Parameters:
        -----------
        in_dim : int
            Dimension of the input node embeddings from GraphSAGE.
        out_dim : int
            Output dimension per head.
        num_heads : int, optional
            Number of attention heads (default: 6).
        dropout : float, optional
            Dropout rate for attention coefficients.
        """
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.linear_proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        # Attention mechanism: 
        self.attn_linear = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.ln = nn.LayerNorm(num_heads * out_dim)

    def forward(self, node_embeddings, nbr_fea_idx):
        """
        Compute multi-head attention over node neighborhoods
          based on GraphSAGE embeddings.

        Parameters:
        -----------
        node_embeddings : Tensor of shape (N, in_dim)
            Updated node features (embeddings) produced by GraphSAGENet.
        nbr_fea_idx : LongTensor of shape (N, M)
            Neighbor indices for each node.

        Returns:
        --------
        h_prime : Tensor of shape (N, num_heads*out_dim)
            Updated node representations after multi-head attention.
        """
        N, _ = node_embeddings.shape
        M = nbr_fea_idx.shape[1]

        # neighbor_embeddings: (N, M, hidden_dim)
        neighbor_embeddings = node_embeddings[nbr_fea_idx, :]

        # central_proj: (N, num_heads*out_dim)
        central_proj = self.linear_proj(node_embeddings)
        # Reshape to (N, num_heads, out_dim)
        central_proj = central_proj.view(N, self.num_heads, self.out_dim)

        neighbor_proj = self.linear_proj(neighbor_embeddings) #(N, M, num_heads*out_dim)

        # Reshape to (N, M, num_heads, out_dim)
        neighbor_proj = neighbor_proj.view(N, M, self.num_heads, self.out_dim)
        central_proj_expanded = central_proj.unsqueeze(1).expand(-1, M, -1, -1) #(N, M, num_heads, out_dim)

        cat_features = torch.cat([central_proj_expanded, neighbor_proj], dim=-1) #(N, M, num_heads, 2*out_dim)
        e = self.leakyrelu(self.attn_linear(cat_features)).squeeze(-1) #(N, M, num_heads)

        # Apply softmax over neighbors (dim=1) to compute attention coefficients
        attention = F.softmax(e, dim=1)
        attention = self.dropout(attention)

        h_weighted = torch.sum(attention.unsqueeze(-1) * neighbor_proj, dim=1) # (N, num_heads, out_dim)

        h_weighted = h_weighted.view(N, -1)  # (N, num_heads*out_dim)

        h_weighted = self.ln(h_weighted)
        h_prime = F.relu(h_weighted)
        return h_prime


class Crystal_GraphSage(nn.Module):
    """
    Constructs node embeddings using a two-stage process:
      1. GraphSAGE module for message passing and feature aggregation.
      2. Multi-head attention module to reweight node embeddings.
    """
    def __init__(self,
                 atom_fea_len,
                 nbr_fea_len,
                 max_crystal_atoms,
                 num_layers=4, 
                 hidden_dims=128,
                 num_heads=6,
                 out_dim=768):
        """
        Parameters
        ----------
        atom_fea_len : int
            Number of initial atom features.
        nbr_fea_len : int
            Number of extra neighbor (e.g., bond) features.
        nbr_fea_idx : Tensor of shape (N, M)
            Neighbor indices for constructing graph edges.
        hidden_dims : list of int, optional
            Hidden dimensions for the stacked GraphSAGE layers.
        max_crystal_atoms : int
            Maximum number of atoms in a crystal structure. 
        num_layers : int, optional
            Number of GraphSAGE layers to stack.
        num_heads : int, optional
        out_dim : int, optional
            Output dimension of the final node embeddings.
            Note: The `self.project` layer uses `hidden_dims * num_heads` as its input size, 
            which corresponds to the output size of the attention mechanism.
        returns
        -------
        -------
        h_prime : Tuple[Tensor, Tensor]
            A tuple containing:
            - reshaped_atom_fea (Tensor): Shape [B, max_crystal_atoms, out_dim].
            - mask (Tensor): Shape [B, max_crystal_atoms].
            Updated node representations after GraphSAGE and attention.
        """
        super(Crystal_GraphSage, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_crystal_atoms = max_crystal_atoms
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.num_layers = num_layers
        

        # Initialize GraphSAGE network.
        self.graphsage_net = GraphSAGENet(atom_fea_len=self.atom_fea_len,
                                           nbr_fea_len=self.nbr_fea_len,
                                          num_layers=self.num_layers,
                                          hidden_dims=self.hidden_dims,  # Ensure this is an integer
                                          output_dim=self.hidden_dims)
        # Initialize Attention network
        self.attn_net = MultiHeadGraphAttention(in_dim=self.hidden_dims, 
                                                out_dim= self.hidden_dims, 
                                                num_heads=self.num_heads)
        # Project layer
        self.project = nn.Linear(hidden_dims*self.num_heads, self.out_dim)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, num_atoms_per_crystal):
        """
        Forward pass:
          1. Compute node embeddings using GraphSAGE.
          2. Update embeddings using multi-head graph attention.

        Parameters
        ----------
        atom_in_fea : Tensor of shape (N, atom_fea_len)
            Input atom features.
        nbr_fea : Tensor of shape (N, M, nbr_fea_len)
            Extra neighbor features.
        nbr_fea_idx : LongTensor of shape (N, M)
            Neighbor indices.

        Returns
        -------
        h_prime : Tensor of shape (N, num_heads*out_dim)
            Updated node representations.
        """
        h = self.graphsage_net(atom_in_fea, nbr_fea, nbr_fea_idx)  # h: (N, gsage_out_dim)
       
        h_prime = self.attn_net(h, nbr_fea_idx) 

        if h_prime.shape[1] != self.out_dim:      # h_prime as (N, hidden_dims * num_heads = out_dim). 
           out = self.project(h_prime)            # Allows richer embeddings without needing projection
        
        h_prime = (out, h_prime)  if h_prime.shape[1] != self.out_dim else (None, h_prime)
        return rebuild_and_mask_batch(h_prime, num_atoms_per_crystal, self.max_crystal_atoms, self.out_dim)
    

class Additional_feat_Embedding(nn.Module):
    """
    Embeds additional features with CLS token, similar to grid embeddings in VisionTransformer3D
    """
    def __init__(self, emb_dim, add_feat_num):
        super().__init__()
        self.add_feat_num = add_feat_num
        self.emb_dim = emb_dim
        
        # Linear layer for feature embedding
        self.linear = nn.Linear(add_feat_num, emb_dim)
        self.activation = nn.ReLU()
        
        # CLS token 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, added_fea):
        """
        Parameters:
        added_fea: Tensor of shape (batch_size, add_feat_num)
        
        Returns:
        embeddings: Tensor of shape (batch_size, 1 + 1, emb_dim) 
                    (CLS token + feature embedding)
        mask: Tensor of shape (batch_size, 2) (all ones)
        """
        batch_size = added_fea.shape[0]
        
        # Embed the additional features
        emb = self.linear(added_fea.reshape(-1, self.add_feat_num))
        emb = self.activation(emb)  # [batch_size, emb_dim]
        
        # Reshape and add CLS token
        emb = emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        
        # Concatenate CLS token and feature embedding
        embeddings = torch.cat([cls_tokens, emb], dim=1)  # [batch_size, 2, emb_dim]
        
        # Create mask (all ones since both tokens are valid)
        mask = torch.ones(batch_size, 2, device=embeddings.device)
        
        return embeddings, mask.bool()
    
    

def rebuild_and_mask_batch(h_prime, num_atoms_per_crystal, max_crystal_atoms, out_dim):
    """
    Reshapes node embeddings into fixed-size tensors and generates a padding mask.
    
    Parameters:
        -----------
        h_prime (Tensor): Node embeddings, shape [total_atoms, out_dim].
        num_atoms_per_crystal (List[int]): Number of atoms per crystal in the batch.
        max_crystal_atoms (int): Maximum allowed atoms per crystal.
        out_dim (int): Dimension of node embeddings.
    
    Returns:
        reshaped_atom_fea (Tensor): Shape [B, max_crystal_atoms, out_dim].
        mask (Tensor): Shape [B, max_crystal_atoms].
    """
    batch_size = len(num_atoms_per_crystal)

    h_prime = h_prime[0] if h_prime[0] is not None else h_prime[1]
    
    # Initialize output tensors
    reshaped_atom_fea = torch.zeros(
        (batch_size, max_crystal_atoms, out_dim),
        dtype=h_prime.dtype,
        device=h_prime.device
    )
    mask = torch.zeros(
        (batch_size, max_crystal_atoms),
        dtype=torch.float32,
        device=h_prime.device
    )
    
    # Slice h_prime for each crystal
    start = 0
    for i, num_atoms in enumerate(num_atoms_per_crystal):
        end = start + num_atoms
        crystal_atoms = h_prime[start:end]  # [num_atoms, out_dim]
        
        # Pad/truncate to max_crystal_atoms
        num_valid = min(num_atoms, max_crystal_atoms)
        reshaped_atom_fea[i, :num_valid] = crystal_atoms[:num_valid]
        mask[i, :num_valid] = 1.0
        
        start = end
    mask = mask.bool()
    return reshaped_atom_fea, mask