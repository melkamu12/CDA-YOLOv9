# ---- [1] InstanceEmbedding Module ---- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceEmbedding(nn.Module):
    """
    Simple 1x1 conv-based embedding module to extract instance-level features
    from detection feature maps.
    """
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.embedding_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: feature map [B, C, H, W]
        x = self.embedding_conv(x)  # [B, D, H, W]
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        return x


# ---- [2] Extract Instance Embeddings from Multi-Scale Features ---- #

def extract_instance_embeddings(feature_maps, embed_modules):
    """
    Args:
        feature_maps: list of [B, C_i, H_i, W_i] from P3–P6
        embed_modules: list of nn.Modules (1x1 convs)
    Returns:
        [B, N_total, D] tensor of all instance embeddings
    """
    embeddings = []
    for fmap, embedder in zip(feature_maps, embed_modules):
        emb = embedder(fmap)  # [B, D, H, W]
        B, D, H, W = emb.shape
        emb = emb.permute(0, 2, 3, 1).reshape(B, -1, D)  # [B, H*W, D]
        embeddings.append(emb)
    return torch.cat(embeddings, dim=1)  # [B, N_total, D]


# ---- [3] Compute Prototypes from Source Embeddings ---- #

def compute_prototypes(instance_embeddings, labels, num_classes):
    """
    Args:
        instance_embeddings: [N, D]
        labels: [N] class labels
    Returns:
        prototypes: [num_classes, D]
    """
    prototypes = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            prototypes.append(torch.zeros(instance_embeddings.shape[1]).to(instance_embeddings.device))
        else:
            prototypes.append(instance_embeddings[mask].mean(0))
    return torch.stack(prototypes)  # [C, D]


# ---- [4] Supervised Contrastive Loss ---- #

def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    loss = -mean_log_prob_pos.mean()
    return loss


# ---- [5] Cosine Similarity-Based Pseudo-Labeling ---- #

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def filter_pseudo_labels(inst_embeddings, class_prototypes, cls_preds, conf_thresh=0.7, sim_thresh=0.8):
    pseudo_labels = []
    preds = cls_preds.argmax(dim=1)
    confs = cls_preds.max(dim=1).values
    for i, (embed, cls_id, conf) in enumerate(zip(inst_embeddings, preds, confs)):
        if conf < conf_thresh:
            continue
        sim = cosine_similarity(embed, class_prototypes[cls_id.item()])
        if sim >= sim_thresh:
            pseudo_labels.append((i, cls_id.item()))
    return pseudo_labels
