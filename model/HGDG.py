
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, idx):
    """
    批量索引点云
    
    Args:
        points: (B, N, C)
        idx: (B, M, K)
    Returns:
        indexed_points: (B, M, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class SimplifiedNormalEstimator(nn.Module):

    def __init__(self, k=30):
        super().__init__()
        self.k = k
    
    def forward(self, points):
        """
        Args:
            points: (B, N, 3)
        Returns:
            normals: (B, N, 3)
        """
        B, N, _ = points.shape
        
    
        dist_matrix = torch.cdist(points, points)
        knn_idx = dist_matrix.argsort(dim=2)[:, :, 1:self.k+1]
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbors = torch.gather(
            points.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            knn_idx_expanded
        )  
        centroid = neighbors.mean(dim=2, keepdim=True)
        centered = neighbors - centroid
        
        cov = torch.matmul(
            centered.transpose(2, 3),
            centered
        ) / self.k  # (B, N, 3, 3)
        
        try:
            U, S, V = torch.svd(cov)
            normals = U[:, :, :, -1]  
        except:
            normals = torch.zeros(B, N, 3, device=points.device)
            normals[:, :, 2] = 1.0
        normals = F.normalize(normals, dim=-1)
        
        return normals

class KeypointPPF_EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, use_ppf=True):
        super().__init__()
        self.k = k
        self.use_ppf = use_ppf
        
 
        if self.use_ppf:
            self.ppf_encoder = nn.Sequential(
                nn.Linear(4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )
           
            nn.init.zeros_(self.ppf_encoder[-1].weight)
            nn.init.zeros_(self.ppf_encoder[-1].bias)
            edge_input_dim = in_channels * 2 + 64  # [中心, 相对, PPF]
        else:
            edge_input_dim = in_channels * 2
        
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        nn.init.zeros_(self.pos_encoder[0].weight)
        nn.init.zeros_(self.pos_encoder[0].bias)
        
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(edge_input_dim + 64, 256, 1),  # +64 for pos encoding
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        nn.init.zeros_(self.edge_mlp[3].weight)
        nn.init.zeros_(self.edge_mlp[3].bias)
    
    def forward(self, kpt_feature, kpt_xyz, neighbor_feature, neighbor_xyz, neighbor_normals=None):
        B, kpt_num, C = kpt_feature.shape
        k = neighbor_feature.shape[2]
        
        center_feature = kpt_feature.unsqueeze(2).expand(-1, -1, k, -1)  
        edge_feature = torch.cat([
            center_feature,
            neighbor_feature - center_feature
        ], dim=-1)  
        
        relative_pos = neighbor_xyz - kpt_xyz.unsqueeze(2)  
        relative_pos = relative_pos.permute(0, 3, 1, 2)  
        pos_encoding = self.pos_encoder(relative_pos)  
        pos_encoding = pos_encoding.permute(0, 2, 3, 1)  
        
        if self.use_ppf and neighbor_normals is not None:
            kpt_normals = neighbor_normals.mean(dim=2)  
            kpt_normals = F.normalize(kpt_normals, dim=-1)
            
            knn_idx = torch.arange(k, device=kpt_feature.device).unsqueeze(0).unsqueeze(0)
            knn_idx = knn_idx.expand(B, kpt_num, k)  
            
            ppf = self._compute_ppf_for_keypoints(
                kpt_xyz, kpt_normals, neighbor_xyz, neighbor_normals
            )  
        
            ppf_flat = ppf.reshape(B * kpt_num * k, 4)
            ppf_encoded = self.ppf_encoder(ppf_flat)  
            ppf_encoded = ppf_encoded.reshape(B, kpt_num, k, 64)
            
            edge_feature = torch.cat([edge_feature, ppf_encoded], dim=-1)
        
        edge_feature = torch.cat([edge_feature, pos_encoding], dim=-1)
        
        edge_feature = edge_feature.permute(0, 3, 1, 2)  
        edge_feature = self.edge_mlp(edge_feature)  
        out = edge_feature.max(dim=-1)[0]  
        out = out.transpose(1, 2)  
        
        return out
    
    def _compute_ppf_for_keypoints(self, kpt_xyz, kpt_normals, neighbor_xyz, neighbor_normals):

        B, kpt_num, k = neighbor_xyz.shape[0], neighbor_xyz.shape[1], neighbor_xyz.shape[2]
        
        d = neighbor_xyz - kpt_xyz.unsqueeze(2)  
        d_norm = torch.norm(d, dim=-1, keepdim=True)  
        d = d / (d_norm + 1e-8)  

        n1 = kpt_normals.unsqueeze(2).expand(-1, -1, k, -1)  
        n2 = neighbor_normals  
        
        alpha = torch.sum(n1 * d, dim=-1, keepdim=True)  
        phi = torch.sum(n2 * d, dim=-1, keepdim=True)    
        theta = torch.sum(n1 * n2, dim=-1, keepdim=True) 
        alpha = torch.clamp(alpha, -1.0, 1.0)
        phi = torch.clamp(phi, -1.0, 1.0)
        theta = torch.clamp(theta, -1.0, 1.0)
        
        ppf = torch.cat([d_norm, alpha, phi, theta], dim=-1)  
        return ppf


class VectorizedGeometricConsistencyGraph(nn.Module):
    def __init__(self, d_model, k_graph=8, num_heads=4):
        super().__init__()
        self.k_graph = k_graph
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
 
        self.geom_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.geom_encoder[-1].weight)
        nn.init.zeros_(self.geom_encoder[-1].bias)
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.geom_attention = nn.Linear(7, num_heads)
        
        self.out_proj = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, kpt_feature, kpt_3d):
        """
        Args:
            kpt_feature: (b, kpt_num, d_model)
            kpt_3d: (b, kpt_num, 3)
        Returns:
            geometry_enhanced: (b, kpt_num, d_model)
        """
        b, kpt_num, d = kpt_feature.shape
        
        dist_matrix = torch.cdist(kpt_3d, kpt_3d)
        knn_idx = dist_matrix.argsort(dim=2)[:, :, 1:self.k_graph+1]
        
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        knn_positions = torch.gather(
            kpt_3d.unsqueeze(1).expand(-1, kpt_num, -1, -1),
            2,
            knn_idx_expanded
        )
        
        geometry_features = self.compute_geometry_features_vectorized(
            kpt_3d, knn_positions, knn_idx, dist_matrix
        )

        geom_aggregated = geometry_features.mean(dim=2)
        geom_encoding = self.geom_encoder(geom_aggregated)

        knn_idx_feat = knn_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        knn_features = torch.gather(
            kpt_feature.unsqueeze(1).expand(-1, kpt_num, -1, -1),
            2,
            knn_idx_feat
        )
        
        Q = self.query_proj(kpt_feature).view(b, kpt_num, self.num_heads, self.head_dim)
        K = self.key_proj(knn_features).view(b, kpt_num, self.k_graph, self.num_heads, self.head_dim)
        V = self.value_proj(knn_features).view(b, kpt_num, self.k_graph, self.num_heads, self.head_dim)
        
        Q_expanded = Q.unsqueeze(2)
        attn_scores = torch.sum(Q_expanded * K, dim=-1) / (self.head_dim ** 0.5)
        
        geom_bias = self.geom_attention(geometry_features)
        attn_scores = attn_scores + geom_bias
        
        attn_weights = F.softmax(attn_scores, dim=2)
        
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        aggregated = torch.sum(attn_weights_expanded * V, dim=2)
        
        aggregated = aggregated.reshape(b, kpt_num, d)
        graph_feature = self.out_proj(aggregated)
        
        geometry_enhanced = geom_encoding + graph_feature
        
        return geometry_enhanced
    
    def compute_geometry_features_vectorized(self, kpt_3d, knn_positions, knn_idx, dist_matrix):
        b, kpt_num, k_graph = knn_idx.shape
        
        knn_idx_dist = knn_idx.unsqueeze(-1)
        neighbor_dists = torch.gather(
            dist_matrix.unsqueeze(2).expand(-1, -1, k_graph, -1),
            3,
            knn_idx_dist
        ).squeeze(-1)
        
        mean_dist = neighbor_dists.mean(dim=2, keepdim=True)
        std_dist = neighbor_dists.std(dim=2, keepdim=True)
        edge_dist = neighbor_dists.unsqueeze(-1)
        #min_dist = neighbor_dists.min(dim=2, keepdim=True)[0]
        
        neighbor_centroid = knn_positions.mean(dim=2)
        relative_centroid = kpt_3d - neighbor_centroid

        centered_neighbors = knn_positions - neighbor_centroid.unsqueeze(2)
        cov = torch.matmul(
            centered_neighbors.transpose(2, 3),
            centered_neighbors
        ) / k_graph
        
        eigenvalues = torch.linalg.eigvalsh(cov)
        anisotropy = eigenvalues[:, :, -1] / (eigenvalues[:, :, 0] + 1e-8)
        anisotropy = anisotropy.unsqueeze(-1)
        
        geom_feats = torch.cat([
            edge_dist,
            mean_dist.unsqueeze(2).expand(-1, -1, k_graph, -1),
            std_dist.unsqueeze(2).expand(-1, -1, k_graph, -1),
            relative_centroid.unsqueeze(2).expand(-1, -1, k_graph, -1),
            anisotropy.unsqueeze(2).expand(-1, -1, k_graph, -1)
        ], dim=-1)
        
        return geom_feats


class ImprovedGAFA_PPF_EdgeConv(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.k = k
        self.d_model = d_model
    
        self.normal_estimator = SimplifiedNormalEstimator(k=30)
        
        self.keypoint_edge_conv = KeypointPPF_EdgeConv(
            in_channels=d_model,
            out_channels=d_model,
            k=k,
            use_ppf=True
        )
        
        self.geometric_graph = VectorizedGeometricConsistencyGraph(
            d_model=d_model,
            k_graph=8,
            num_heads=4
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)
        
        self.relu = nn.ReLU()
    
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        with torch.no_grad():
            normals = self.normal_estimator(pts)  
        
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)
        knn_idx = dis_mat.argsort()[:, :, :self.k]  

        neighbor_xyz = index_points(pts, knn_idx)          
        neighbor_feature = index_points(pts_feature, knn_idx)  
        neighbor_normals = index_points(normals, knn_idx)  

        local_feature = self.keypoint_edge_conv(
            kpt_feature=kpt_feature,           
            kpt_xyz=kpt_3d,                    
            neighbor_feature=neighbor_feature, 
            neighbor_xyz=neighbor_xyz,         
            neighbor_normals=neighbor_normals  
        )       
        kpt_feature = self.relu(local_feature + kpt_feature)    
        geom_enhanced = self.geometric_graph(kpt_feature, kpt_3d)        
        kpt_feature = self.relu(geom_enhanced + kpt_feature)  
        out = self.out_mlp(kpt_feature)  
        return self.relu(out + kpt_feature)



class HierarchicalGeometricDualGraph(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.K = cfg.K
        self.d_model = cfg.d_model
        
        assert len(self.K) == self.block_num

        for k in self.K:
            if isinstance(k, list):
                raise ValueError(f"此版本使用单尺度配置，但得到k={k}。")

        self.GAFA_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.GAFA_blocks.append(
                ImprovedGAFA_PPF_EdgeConv(self.K[i], self.d_model)
            )
        
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """
        Args:
            kpt_feature: (b, kpt_num, dim)
            kpt_3d: (b, kpt_num, 3)
            pts_feature: (b, n, dim)
            pts: (b, n, 3)
        Returns:
            kpt_feature: (b, kpt_num, dim)
        """
        for i in range(self.block_num):
            kpt_feature = self.GAFA_blocks[i](kpt_feature, kpt_3d, pts_feature, pts)
        
        return kpt_feature

