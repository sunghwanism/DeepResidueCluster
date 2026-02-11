import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSageBackbone(nn.Module):
    """
    Step 2 & 3: Backbone Feature Extractor.
    Input: Node Features -> GraphSAGE -> Pooling -> Graph Vector
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSageBackbone, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)           # x: [num_nodes, hidden_channels]
        x = F.relu(x)
        # Layer 2
        x = self.conv2(x, edge_index)           # x: [num_nodes, hidden_channels]
        x = F.relu(x)
        # Layer 3
        x = self.conv3(x, edge_index)           # x: [num_nodes, out_channels]
        
        # Step 3: Pooling (Graph Vector Extraction)
        graph_vec = global_mean_pool(x, batch)  # graph_vec: [batch_size, out_channels]
        
        return graph_vec

class CrossAttentionBlock(nn.Module):
    """
    Step 5: Cross Attention Mechanism.
    Query: Masked Node Features
    Key/Value: Graph Vector (from Backbone)
    """
    def __init__(self, node_in_dim, graph_dim, hidden_dim, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        
        # Projections
        self.query_proj = nn.Linear(node_in_dim, hidden_dim) # Raw features to Hidden
        self.key_proj = nn.Linear(graph_dim, hidden_dim)     # Graph Vector to Hidden
        self.value_proj = nn.Linear(graph_dim, hidden_dim)   # Graph Vector to Hidden
        
        # Multi-head Attention
        # batch_first=False to align with PyG node stacking (Sequence length as nodes)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
        
        # Residual connection & Norm (Optional but recommended for stability)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, masked_x, graph_vec, batch_indices):
        """
        Args:
            masked_x: [num_nodes, node_in_dim] (Masked Raw Features)
            graph_vec: [batch_size, graph_dim] (From Backbone)
            batch_indices: [num_nodes]
        """
        # 1. Prepare Query (From Node Features)
        # unsqueeze(0) -> [1, num_nodes, hidden_dim] (Treating all nodes as one sequence)
        query = self.query_proj(masked_x).unsqueeze(0)
        
        # 2. Prepare Key & Value (From Graph Vector)
        # Broadcast graph vector to corresponding nodes
        expanded_graph_vec = graph_vec[batch_indices]            # [num_nodes, graph_dim]
        key = self.key_proj(expanded_graph_vec).unsqueeze(0)     # [1, num_nodes, hidden_dim]
        value = self.value_proj(expanded_graph_vec).unsqueeze(0) # [1, num_nodes, hidden_dim]
        
        # 3. Attention
        attn_out, _ = self.attn(query, key, value)               # [1, num_nodes, hidden_dim]
        attn_out = attn_out.squeeze(0)                           # [num_nodes, hidden_dim]
        
        # Residual + Norm
        # Note: query is projected x, so we add residual to the projected version if needed
        # Here we return the attention output directly as the "Encoded" representation
        return self.norm(attn_out)

class IntegratedEncoder(nn.Module):
    """
    The "Encoder" defined in the concept.
    Combines Backbone (SAGE) + CrossAttention.
    """
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super(IntegratedEncoder, self).__init__()
        self.backbone = GraphSageBackbone(feature_dim, hidden_dim, out_dim)
        self.cross_attn = CrossAttentionBlock(feature_dim, out_dim, hidden_dim)

    def forward(self, x_bb_masked, x_att_masked, edge_index, batch):
        # 1. Backbone Pass (Step 2 & 3)
        graph_vec = self.backbone(x_bb_masked, edge_index, batch)
        
        # 2. Cross Attention Pass (Step 5)
        # Uses Graph Vector (Key/Value) and Node Features (Query)
        node_embeddings = self.cross_attn(x_att_masked, graph_vec, batch)
        
        return graph_vec, node_embeddings

class AttributeDecoder(nn.Module):
    """
    Step 6: Decoder to predict original attributes.
    """
    def __init__(self, hidden_dim, output_dim):
        super(AttributeDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_embeddings):
        return self.net(node_embeddings)

class ContrastiveGraphModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, 
                 bb_masking_ratio=0.1, att_mask_ratio=0.15, 
                 alpha=1.0, beta=1.0):
        super(ContrastiveGraphModel, self).__init__()
        
        self.bb_masking_ratio = bb_masking_ratio
        self.att_mask_ratio = att_mask_ratio
        self.alpha = alpha
        self.beta = beta
        
        # Encoder (Backbone + CrossAttn)
        self.encoder = IntegratedEncoder(feature_dim, hidden_dim, out_dim)
        
        # Decoder
        self.decoder = AttributeDecoder(hidden_dim, feature_dim)

    def mask_features(self, x, ratio):
        """Randomly masks input features."""
        if ratio <= 0.0 or not self.training:
            return x, None
        mask = torch.empty_like(x).bernoulli_(1 - ratio)
        return x * mask, (mask == 0)

    def contrastive_loss(self, z_i, z_j, label, margin=1.0):
        """Step 4: Contrastive Loss on Graph Vectors."""
        dist = F.pairwise_distance(z_i, z_j)
        loss = label * torch.pow(dist, 2) + \
               (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        return torch.mean(loss)

    def forward_single_branch(self, data):
        """Pipeline for one graph batch."""
        x_orig = data.x
        
        # --- Masking Strategies ---
        # Masking for Backbone (Robustness)
        x_bb, _ = self.mask_features(x_orig, self.bb_masking_ratio)
        # Masking for Attention (Reconstruction Task)
        x_att, mask_indices = self.mask_features(x_orig, self.att_mask_ratio)
        
        # --- Encoder Forward ---
        # graph_vec: Used for Contrastive Loss
        # node_embs: Used for Prediction Loss
        graph_vec, node_embs = self.encoder(x_bb, x_att, data.edge_index, data.batch)
        
        # --- Decoder Forward ---
        pred_x = self.decoder(node_embs)
        
        # --- Prediction Loss Calculation ---
        if mask_indices is not None and mask_indices.any():
            l_pred = F.mse_loss(pred_x[mask_indices], x_orig[mask_indices])
        else:
            l_pred = torch.tensor(0.0, device=x_orig.device)
            
        return graph_vec, l_pred

    def forward(self, batch_a, batch_b, labels):
        # 1. Process Pair
        vec_a, loss_pred_a = self.forward_single_branch(batch_a)
        vec_b, loss_pred_b = self.forward_single_branch(batch_b)
        
        # 2. Contrastive Loss (Step 4)
        loss_con = self.contrastive_loss(vec_a, vec_b, labels)
        
        # 3. Total Prediction Loss (Step 6)
        loss_pred_total = (loss_pred_a + loss_pred_b) / 2
        
        # 4. Total Loss (Step 7)
        # L_total = alpha * L_con + beta * L_pred
        total_loss = (self.alpha * loss_con) + (self.beta * loss_pred_total)
        
        return total_loss, loss_con, loss_pred_total

    def get_embedding(self, data):
        """
        Extracts embeddings for downstream tasks.
        Returns both graph-level and node-level embeddings.
        """
        self.eval()
        with torch.no_grad():
            # No masking during inference
            graph_vec, node_embs = self.encoder(data.x, data.x, data.edge_index, data.batch)
        return graph_vec, node_embs