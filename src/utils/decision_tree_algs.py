import torch

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Leaf node has value not None, internal node has value None

# Learn a decision tree using CART (binary splits, Gini impurity)
def CART(
    data: torch.Tensor,
    labels,
    depth:int = 0,
    max_depth:int = None,
    ):
    # If all labels are the same decision tree is trivial
    if len(set(labels.tolist())) == 1:
        return TreeNode(value=labels[0].item())

    # If max depth reached or no features left, return majority class
    if max_depth is not None and depth >= max_depth:
        values, counts = torch.unique(labels, return_counts=True)
        return TreeNode(value=values[counts.argmax()].item())

    best_gini = float('inf')
    best_feat, best_thresh = None, None
    best_left_idx, best_right_idx = None, None
    n_features = data.shape[1]
    for feat in range(n_features):
        thresholds = torch.unique(data[:, feat])
        for thresh in thresholds:
            left_idx = data[:, feat] <= thresh
            right_idx = data[:, feat] > thresh
            if left_idx.sum() == 0 or right_idx.sum() == 0:
                continue
            left_labels = labels[left_idx]
            right_labels = labels[right_idx]
            def gini(labels):
                if len(labels) == 0:
                    return 0
                _, counts = torch.unique(labels, return_counts=True)
                probs = counts.float() / counts.sum()
                return 1.0 - (probs ** 2).sum().item()
            gini_split = (len(left_labels) * gini(left_labels) + len(right_labels) * gini(right_labels)) / len(labels)
            if gini_split < best_gini:
                best_gini = gini_split
                best_feat = feat
                best_thresh = thresh
                best_left_idx = left_idx
                best_right_idx = right_idx
    if best_feat is None:
        # No split found, return majority class
        values, counts = torch.unique(labels, return_counts=True)
        return TreeNode(value=values[counts.argmax()].item())
    left = CART(data[best_left_idx], labels[best_left_idx], depth+1, max_depth)
    right = CART(data[best_right_idx], labels[best_right_idx], depth+1, max_depth)
    return TreeNode(feature=best_feat, threshold=best_thresh.item(), left=left, right=right)

def ID3(
    data:torch.Tensor,
    labels:torch.Tensor,
    depth:int = 0,
    max_depth:int = None,
    )->TreeNode:
    if (labels==0).all() or (labels==1).all():
        return TreeNode(value = labels[0].item())
    
    pass
