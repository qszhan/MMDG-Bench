import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTranslationMLP(nn.Module):
    """
    Cross-modal translation module

    Learns to translate features from one modality to another.
    Helps the model to be robust to missing modalities.
    """

    def __init__(self, modalities, feature_dims, hidden_dim=2048):
        """
        Args:
            modalities: List of modality names (e.g., ['rgb', 'flow', 'audio'])
            feature_dims: Dict mapping modality names to feature dimensions
            hidden_dim: Hidden dimension for translation MLPs
        """
        super(CrossModalTranslationMLP, self).__init__()
        self.modalities = modalities
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim

        # Create translation MLPs for all modality pairs
        self.translation_mlps = nn.ModuleDict()

        for source_mod in modalities:
            for target_mod in modalities:
                if source_mod != target_mod:
                    mlp_name = f"{source_mod}2{target_mod}"
                    self.translation_mlps[mlp_name] = nn.Sequential(
                        nn.Linear(feature_dims[source_mod], hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, feature_dims[target_mod])
                    )

    def forward(self, modality_features, normalize=True):
        """
        Perform cross-modal translation and compute translation loss

        Args:
            modality_features: Dict mapping modality names to feature tensors
            normalize: Whether to normalize features before computing loss

        Returns:
            translation_loss: Average translation loss across all pairs
        """
        loss = 0
        count = 0

        available_modalities = list(modality_features.keys())

        for source_mod in available_modalities:
            for target_mod in available_modalities:
                if source_mod != target_mod:
                    mlp_name = f"{source_mod}2{target_mod}"

                    if mlp_name in self.translation_mlps:
                        # Translate from source to target
                        translated = self.translation_mlps[mlp_name](modality_features[source_mod])
                        target = modality_features[target_mod]

                        # Normalize if requested
                        if normalize:
                            translated = F.normalize(translated, dim=1)
                            target = F.normalize(target, dim=1)

                        # L2 distance loss
                        pair_loss = torch.mean(torch.norm(translated - target, dim=1))
                        loss += pair_loss
                        count += 1

        if count > 0:
            loss = loss / count

        return loss

    def translate(self, source_features, source_modality, target_modality):
        """
        Translate features from source modality to target modality

        Args:
            source_features: Features from source modality
            source_modality: Name of source modality
            target_modality: Name of target modality

        Returns:
            Translated features
        """
        mlp_name = f"{source_modality}2{target_modality}"
        if mlp_name not in self.translation_mlps:
            raise ValueError(f"Translation from {source_modality} to {target_modality} not available")

        return self.translation_mlps[mlp_name](source_features)





class CrossModalTranslation(nn.Module):
    """
    Simplified cross-modal translation using projected features (same dimension)

    Since projected features are already in the same space,
    we just compute alignment loss directly.

    Supports adaptive weighting strategy from original implementation:
    - Computes all pairwise translation losses
    - Normalizes each loss by the mean loss
    - Averages the normalized losses
    """
    def __init__(self, normalize=False):
        super(CrossModalTranslation, self).__init__()
        self.normalize = normalize


    def forward(self, modality_projected_features):
        """
        Compute cross-modal alignment loss for projected features

        Args:
            modality_projected_features: Dict mapping modality names to projected tensors
                All tensors should have the same dimension (e.g., 128 or 512)

        Returns:
            Alignment loss
        """
        available_modalities = list(modality_projected_features.keys())

        if len(available_modalities) < 2:
            return torch.tensor(0.0, device=next(iter(modality_projected_features.values())).device)

        # Normalize features if requested
        if self.normalize:
            features = {
                k: F.normalize(v, dim=1)
                for k, v in modality_projected_features.items()
            }
        else:
            features = modality_projected_features

        # Compute pairwise alignment losses
        translation_losses = []
        modality_list = sorted(available_modalities)  # Sort for consistency

        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod_i = modality_list[i]
                mod_j = modality_list[j]


                # L2 distance
                pair_loss = torch.mean(torch.norm(features[mod_i] - features[mod_j], dim=1))
                translation_losses.append(pair_loss)



        if len(translation_losses) == 0:
            return torch.tensor(0.0, device=next(iter(modality_projected_features.values())).device)
        # elif len(translation_losses) == 1:
        #     return  sum(translation_losses) / len(translation_losses)
        # else:
        #     # Normalize each loss by mean loss
        #     mean_loss = sum(translation_losses) / len(translation_losses)
        #     normalized_losses = [tl / (mean_loss + 1e-8) for tl in translation_losses]
        #     loss = sum(normalized_losses) / len(normalized_losses)
        else:
            return sum(translation_losses) / len(translation_losses)


 
def cross_modal_translation(v_emd, f_emd, audio_emd):
    loss = 0
    # Cross-modal Translation
    if args.use_video and args.use_flow and args.use_audio:
        a_emd_t = mlp_v2a(v_emd)
        v_emd_t = mlp_a2v(audio_emd)
        f_emd_t = mlp_v2f(v_emd)
        v_emd_t2 = mlp_f2v(f_emd)
        a_emd_t2 = mlp_f2a(f_emd)
        f_emd_t2 = mlp_a2f(audio_emd)
        a_emd_t = a_emd_t / torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t / torch.norm(v_emd_t, dim=1, keepdim=True)
        f_emd_t = f_emd_t / torch.norm(f_emd_t, dim=1, keepdim=True)
        a_emd_t2 = a_emd_t2 / torch.norm(a_emd_t2, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2 / torch.norm(v_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2 / torch.norm(f_emd_t2, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        v2f_loss = torch.mean(torch.norm(f_emd_t - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2a_loss = torch.mean(torch.norm(a_emd_t2 - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2 - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2 - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans * (v2a_loss + a2v_loss + v2f_loss + f2a_loss + f2v_loss + a2f_loss) / 6
    elif args.use_video and args.use_flow:
        f_emd_t = mlp_v2f(v_emd)
        v_emd_t2 = mlp_f2v(f_emd)
        f_emd_t = f_emd_t / torch.norm(f_emd_t, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2 / torch.norm(v_emd_t2, dim=1, keepdim=True)
        v2f_loss = torch.mean(torch.norm(f_emd_t - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2 - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans * (v2f_loss + f2v_loss) / 2
    elif args.use_video and args.use_audio:
        a_emd_t = mlp_v2a(v_emd)
        v_emd_t = mlp_a2v(audio_emd)
        a_emd_t = a_emd_t / torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t / torch.norm(v_emd_t, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t - v_emd / torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans * (v2a_loss + a2v_loss) / 2
    elif args.use_flow and args.use_audio:
        a_emd_t2 = mlp_f2a(f_emd)
        f_emd_t2 = mlp_a2f(audio_emd)
        a_emd_t2 = a_emd_t2 / torch.norm(a_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2 / torch.norm(f_emd_t2, dim=1, keepdim=True)
        f2a_loss = torch.mean(torch.norm(a_emd_t2 - audio_emd / torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2 - f_emd / torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans * (f2a_loss + a2f_loss) / 2
        return loss