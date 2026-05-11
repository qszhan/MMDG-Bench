"""
Multimodal Fusion Module

This module contains various multimodal fusion mechanisms including:
- AttentionFusion: Cross-attention based fusion
- BottleneckFusion: Late fusion with attention bottlenecks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Cross-attention based multimodal fusion.

    Each modality attends to other modalities through cross-attention mechanism,
    followed by a final self-attention for fusion.

    Args:
        projected_dim (int): Dimension of projected features from external projectors
        output_dim (int): Number of output classes
        hidden_dim (int): Hidden dimension for internal processing
        max_modalities (int): Maximum number of modalities to support
    """

    def __init__(self, projected_dim, output_dim=8, hidden_dim=512, max_modalities=3):
        super(AttentionFusion, self).__init__()
        self.projected_dim = projected_dim  # Dimension after projector (args.out_dim)
        self.hidden_dim = hidden_dim
        self.max_modalities = max_modalities

        # Single projection from projector output to hidden_dim
        self.feature_projection = nn.Linear(projected_dim, hidden_dim)

        # Cross-attention layers: each modality attends to others
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
            for _ in range(max_modalities)
        ])

        # Layer normalization for each modality (after residual connection)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(max_modalities)
        ])

        # Final fusion attention to combine all enhanced features
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Final classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, features):
        """
        Forward pass for attention fusion.

        Args:
            features (list): List of projected tensors from external projectors
                           Each tensor is [batch_size, projected_dim]

        Returns:
            tuple: (prediction, fused_features)
                - prediction: [batch_size, output_dim]
                - fused_features: [batch_size, hidden_dim]
        """
        # features is a list of projected tensors from external projectors
        # Each tensor is [batch_size, projected_dim] where projected_dim = args.out_dim
        projected = []
        for feat in features:
            # Project from projector output to hidden_dim
            proj = F.relu(self.feature_projection(feat))
            projected.append(proj)  # projected[i] is now [batch_size, hidden_dim]

        num_modalities = len(projected)

        # Cross-attention: each modality queries others
        enhanced_features = []
        for i in range(num_modalities):
            # Current modality's projected feature as query
            query = projected[i].unsqueeze(0)  # [1, batch_size, hidden_dim]

            # Collect other modalities' projected features to serve as keys and values
            other_features = [projected[j] for j in range(num_modalities) if j != i]

            if other_features:
                # Stack other modalities' projected features for K, V
                keys_values = torch.stack(other_features, dim=0)  # [num_others, batch_size, hidden_dim]

                # Apply cross-attention for the current modality
                attended, _ = self.cross_attentions[i](query, keys_values, keys_values)
                attended = attended.squeeze(0)  # [batch_size, hidden_dim]

                # Residual connection: add attention output to the original projected feature
                # Then apply Layer Normalization
                enhanced = self.layer_norms[i](attended + projected[i])
            else:
                # If only one modality is present, there are no "other_features" to attend to.
                # In this case, just apply LayerNorm to the projected feature,
                # as there's no attention output to add a residual to.
                enhanced = self.layer_norms[i](projected[i])

            enhanced_features.append(enhanced)  # enhanced_features[i] is now [batch_size, hidden_dim]

        # Final fusion attention across all enhanced features
        # Stack all enhanced features to form a sequence for the final self-attention
        stacked_enhanced = torch.stack(enhanced_features, dim=0)  # [num_modalities, batch_size, hidden_dim]

        # Apply self-attention for final fusion
        # query, key, value are all the same stacked_enhanced features
        final_attended, _ = self.fusion_attention(stacked_enhanced, stacked_enhanced, stacked_enhanced)

        # Global average pooling across modalities to get the final fused feature vector
        # This averages across the 'num_modalities' dimension
        fused = final_attended.mean(dim=0)  # [batch_size, hidden_dim]

        # Apply Layer Normalization to the final fused representation
        fused = self.fusion_norm(fused)

        # Final classification head
        output = self.classifier(fused)

        # Return both the classification predictions and the fused feature representation
        return output, fused


class BottleneckFusion(nn.Module):
    """
    Bottleneck fusion mechanism based on "Attention Bottlenecks for Multimodal Fusion".

    Forces cross-modal information to pass through a small set of fusion bottleneck tokens,
    requiring the model to collate and condense relevant information in each modality.

    Args:
        projected_dim (int): Dimension of projected features from external projectors
        output_dim (int): Number of output classes
        hidden_dim (int): Hidden dimension for internal processing
        num_bottlenecks (int): Number of bottleneck tokens
    """

    def __init__(self, input_dims, projected_dim, output_dim=8, hidden_dim=512, num_bottlenecks=4):
        super(BottleneckFusion, self).__init__()
        self.input_dims = input_dims
        self.projected_dim = projected_dim
        self.hidden_dim = hidden_dim
        self.num_bottlenecks = num_bottlenecks

        # Smart projection detection: check if all input dims are the same
        unique_dims = set(input_dims)
        self.use_per_modality_projection = len(unique_dims) > 1

        if self.use_per_modality_projection:
            # Different dims: need separate projection for each modality
            self.feature_projection = nn.ModuleList([
                nn.Linear(dim, hidden_dim) for dim in input_dims
            ])
            print(f"BottleneckFusion: Different input dims {input_dims}, using per-modality projection -> {hidden_dim}")
        else:
            # All dims same: use single shared projection or identity
            single_dim = list(unique_dims)[0]
            if single_dim == hidden_dim:
                # No projection needed
                self.feature_projection = nn.Identity()
                print(f"BottleneckFusion: All input dims = {single_dim} = hidden_dim, skipping projection")
            else:
                # Single shared projection
                self.feature_projection = nn.Linear(single_dim, hidden_dim)
                print(f"BottleneckFusion: All input dims = {single_dim}, using single shared projection -> {hidden_dim}")

        # Fusion bottleneck tokens - learnable parameters
        self.bottleneck_tokens = nn.Parameter(
            torch.empty(1, num_bottlenecks, hidden_dim).normal_(std=0.02)
        )

        # Scaling parameters for modalities (max 3: video, flow, audio)
        self.scale_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(3)
        ])

        # Layer normalization for each modality
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def attention(self, q, k, v):
        """Simplified attention mechanism"""
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    def bottleneck_fusion_step(self, modality_features):
        """
        Apply bottleneck fusion between modalities.

        Args:
            modality_features (list): List of modality features [B, 1, hidden_dim]

        Returns:
            list: Enhanced modality features after bottleneck fusion
        """
        batch_size = modality_features[0].shape[0]

        # Expand bottleneck tokens for the batch
        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)

        # Concatenate all modality features
        all_features = torch.cat(modality_features, dim=1)  # [B, sum(N_i), D]

        # Cross attention: bottlenecks attend to all modality features
        fused_bottlenecks = self.attention(
            q=bottlenecks,
            k=all_features,
            v=all_features
        )

        # Cross attention: each modality attends to fused bottlenecks
        enhanced_modalities = []
        for i, feat in enumerate(modality_features):
            enhanced = feat + self.scale_params[i] * self.attention(
                q=feat,
                k=fused_bottlenecks,
                v=fused_bottlenecks
            )
            enhanced_modalities.append(enhanced)

        return enhanced_modalities

    def forward(self, features):
        """
        Forward pass for bottleneck fusion.

        Args:
            features (list): List of projected tensors from external projectors
                           Each tensor is [batch_size, projected_dim]

        Returns:
            tuple: (prediction, fused_features)
                - prediction: [batch_size, output_dim]
                - fused_features: [batch_size, hidden_dim]
        """
        # features is a list of projected tensors from external projectors
        # Each tensor is [batch_size, projected_dim] where projected_dim = args.out_dim
        projected = []

        for i, feat in enumerate(features):
            # Add sequence dimension and apply projection
            feat = feat.unsqueeze(1)  # [B, 1, input_dim]

            if self.use_per_modality_projection:
                # Different dims: use per-modality projection
                proj = F.relu(self.feature_projection[i](feat))  # [B, 1, hidden_dim]
            else:
                # All dims same: use single shared projection or identity
                proj = F.relu(self.feature_projection(feat))  # [B, 1, hidden_dim]

            projected.append(proj)

        # Apply bottleneck fusion
        enhanced_features = self.bottleneck_fusion_step(projected)

        # Apply layer normalization
        enhanced_features = [
            self.layer_norms[i](feat) for i, feat in enumerate(enhanced_features)
        ]

        # Global average pooling across modalities and sequence dimension
        fused = torch.stack([feat.mean(dim=1) for feat in enhanced_features], dim=1)
        fused = fused.mean(dim=1)  # [B, hidden_dim]

        # Final classification
        output = self.classifier(fused)

        return output, fused

 
 

 

 

# Utility function for easy model creation
def create_fusion_model(fusion_type, **kwargs):
    """
    Factory function to create fusion models.

    Args:
        fusion_type (str): Type of fusion ('attention', 'bottleneck', 'transformer_bottleneck')
        **kwargs: Additional arguments for the specific fusion model

    Returns:
        nn.Module: The requested fusion model

    Example:
        model = create_fusion_model('attention', projected_dim=128, output_dim=8)
        model = create_fusion_model('bottleneck', projected_dim=128, num_bottlenecks=4)
        
    """
    if fusion_type == 'attention':
        return AttentionFusion(**kwargs)
    elif fusion_type == 'bottleneck':
        return BottleneckFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                         f"Supported types: 'attention', 'bottleneck', 'transformer_bottleneck'")


# Export all classes and utility function
__all__ = [
    'AttentionFusion',
    'BottleneckFusion',
    'create_fusion_model'
]
