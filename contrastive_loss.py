from itertools import combinations, product
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, num_classes=5, margin=1.0, threshold=0.5, pull_positive=True):
        super(ContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.threshold = threshold
        self.pull_positive = pull_positive

    def forward(self, feats, labels):
        # Normalize the features
        feats = F.normalize(feats, p=2, dim=1)

        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(feats, feats, p=2)

        # Create label matrix
        labels = labels.unsqueeze(1)
        label_matrix = labels == labels.t()

        # Positive pairs
        if self.pull_positive:
            positive_pairs = dist_matrix[label_matrix].pow(2)
        else:
            positive_pairs = torch.tensor(0.0, device=feats.device)

        # Negative pairs
        negative_pairs = torch.clamp(self.margin - dist_matrix[~label_matrix], min=0.0).pow(2)

        # Combine losses
        loss = positive_pairs.mean() + negative_pairs.mean()
        return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, num_classes=9, margin=1.0, threshold=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.num_classes = num_classes
#         self.margin = margin
#         self.threshold = threshold

#     def forward(self, feats, labels):
#         # only use predictions that matches GT labels and exceeds certain threshold
#         max_values, max_indices =  class_prob.max(dim=-1)
#         index_mask_gt = labels == max_indices
#         index_mask_th = max_values >= self.threshold
#         index_mask = index_mask_gt & index_mask_th
#         feats_filtered = nn.functional.normalize(feats[index_mask], dim=1)
#         label_filtered = labels[index_mask]

#         loss = torch.tensor(0, dtype=torch.float32).to(torch.device("cuda"))
#         ind_group = []
#         for ci in range(1, self.num_classes):
#             ci_indices = label_filtered == ci
#             ci_ind_nonzero = ci_indices.nonzero(as_tuple=True)[0].tolist()
#             if len(ci_ind_nonzero) == 0:
#                 continue
#             feats_filtered_class = feats_filtered[ci_indices]
#             ind_group.append(ci_ind_nonzero)
#             pairs = [*combinations(list(range(feats_filtered_class.shape[0])), 2)]
#             if len(pairs) == 0:
#                 continue
#             f1_index_pull, f2_index_pull = zip(*pairs)
#             f1_pull, f2_pull = feats_filtered_class[[f1_index_pull]], feats_filtered_class[[f2_index_pull]]
#             euclidean_distance_pull = nn.functional.pairwise_distance(f1_pull, f2_pull)
#             # pull same classes for FG instances
#             loss = loss + torch.mean(torch.pow(euclidean_distance_pull, 2))
#         # push different classes
#         if len(ind_group) >= 2:
#             group_pairs = [*combinations(ind_group, 2)]
#             for group_pair in group_pairs:
#                 push_idx = [*product(*group_pair)]
#                 f1_index_push, f2_index_push = zip(*push_idx)
#                 f1_push, f2_push = feats_filtered[[f1_index_push]], feats_filtered[[f2_index_push]]
#                 euclidean_distance_push = nn.functional.pairwise_distance(f1_push, f2_push)
#                 loss = loss + torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance_push, min=0.0), 2))
#         return loss


# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, labels, preds, scores):
#         # pull same classes
#         # push different classes
#         positive_distance = nn.functional.pairwise_distance(anchor, positive)
#         negative_distance = nn.functional.pairwise_distance(anchor, negative)
#         loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0.0))
#         return loss


class ContrastiveCosineLoss(nn.Module):
    def __init__(self, num_classes=9, margin=0.0, threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.threshold = threshold
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
    
    def forward(self, labels, feats, scores):
        # instance class prob. of predicted regions
        class_prob = scores.softmax(dim=-1)
        # only use predictions that matches GT labels and exceeds certain threshold
        max_values, max_indices =  class_prob.max(dim=-1)
        index_mask_gt = labels == max_indices
        index_mask_th = max_values >= self.threshold
        index_mask = index_mask_gt & index_mask_th
        feats_filtered = nn.functional.normalize(feats[0][index_mask], dim=1)
        label_filtered = labels[index_mask]

        loss = 0.0
        ind_group = []
        for ci in range(1, self.num_classes):
            ci_indices = label_filtered == ci
            ci_ind_nonzero = ci_indices.nonzero(as_tuple=True)[0].tolist()
            if len(ci_ind_nonzero) == 0:
                continue
            feats_filtered_class = feats_filtered[ci_indices]
            ind_group.append(ci_ind_nonzero)
            pairs = [*combinations(list(range(feats_filtered_class.shape[0])), 2)]
            if len(pairs) == 0:
                continue
            f1_index_pull, f2_index_pull = zip(*pairs)
            f1_pull, f2_pull = feats_filtered_class[[f1_index_pull]], feats_filtered_class[[f2_index_pull]]
            # pull same classes for FG instances
            loss = loss + self.cosine_loss(f1_pull, f2_pull, torch.ones(len(f1_pull), device=f1_pull.device))
        # push different classes
        if len(ind_group) >= 2:
            group_pairs = [*combinations(ind_group, 2)]
            for group_pair in group_pairs:
                push_idx = [*product(*group_pair)]
                f1_index_push, f2_index_push = zip(*push_idx)
                f1_push, f2_push = feats_filtered[[f1_index_push]], feats_filtered[[f2_index_push]]
                loss = loss + self.cosine_loss(f1_push, f2_push, -1 * torch.ones(len(f1_push), device=f1_push.device))
        return loss


class ContrastiveSoftmaxLoss(nn.Module):
    def __init__(self, dist_function=torch.mm, temperature=1.0, threshold=0.5, e=1e-8):
        super().__init__()
        """
        Calculate contrastive loss using softmax function.

        Args:
            features: Tensor of shape (batch_size, feature_dim)
            positive_indices: Tensor with indices of positive samples
            temperature: A scaling factor T

        Returns:
            Tensor representing the contrastive loss.
        """
        self.dist_function = dist_function
        self.temperature = temperature
        self.threshold = threshold
        self.e = e

    def forward(self, labels, features, scores):
        # instance class prob. of predicted regions
        class_prob = scores.softmax(dim=-1)
        # only use predictions that matches GT labels and exceeds certain threshold
        max_values, max_indices =  class_prob.max(dim=-1)
        index_mask_gt = labels == max_indices
        index_mask_th = max_values >= self.threshold
        index_mask = index_mask_gt & index_mask_th
        if index_mask.sum() == 0:
            return torch.tensor(0, dtype=torch.float32, device=labels.device)
        feats_filtered = nn.functional.normalize(features[index_mask], dim=1)
        label_filtered = labels[index_mask]

        # Calculate similarity scores S(v_i; v_m)
        similarities = self.dist_function(feats_filtered)
        # Create mask for positive and negative samples
        label_filtered = label_filtered.unsqueeze(1)
        mask = torch.eq(label_filtered, label_filtered.t()).float().to(feats_filtered.device) - torch.eye(len(label_filtered), device=label_filtered.device)
        mask_pos = torch.triu(mask)
        # Positive similarities
        sim_pos = similarities[mask_pos.to(torch.bool)]
        sim_pos_exp = torch.exp(sim_pos / self.temperature)
        if torch.all(label_filtered == label_filtered[0]):
            # Only return the positive part of the loss
            loss_contrastive = -torch.mean(torch.log(sim_pos_exp) - torch.log(sim_pos_exp.sum()))
            return loss_contrastive
        # Negative similarities
        mask_neg = torch.triu(1 - mask) - torch.eye(len(mask), device=mask.device)
        sim_neg = similarities[mask_neg.to(torch.bool)]
        sim_neg_exp = torch.exp(sim_neg / self.temperature)
        # Final contrastive loss
        denom = sim_pos_exp.sum() + sim_neg_exp.sum()
        loss_contrastive = -torch.mean(torch.log((sim_pos_exp / denom).sum() - (sim_neg_exp / denom).sum()))

        return loss_contrastive


def dist_mm(features):  # cosine similarity
    return torch.mm(features, features.t())

def dist_l2(features):
    diff = features.unsqueeze(1) - features.unsqueeze(0)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-9)
    return -dist

class DistCos(nn.Module):  # cos embedding loss
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineEmbeddingLoss(margin)

    def forward(self, features):
        pass
