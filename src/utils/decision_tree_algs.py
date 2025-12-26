import torch
from enum import Enum
from typing import Tuple

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Leaf node has value not None, internal node has value None
    
    def get_depth(self):
        if self.value is not None:
            return 1
        left_depth = self.left.get_depth() if self.left is not None else 0
        right_depth = self.right.get_depth() if self.right is not None else 0
        return 1 + max(left_depth, right_depth)
        
class GainMeasure(Enum):
    train_error = 0
    information_gain = 1
    gini = 2

def gini(labels):
    if len(labels) == 0:
        return 0
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()
    return 1.0 - (probs ** 2).sum().item()
def information_gain(labels):
    if len(labels) == 0:
        return 0
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()
    return - (probs*(probs.log())).sum()
def train_error(labels):
    if len(labels) == 0:
        return 0
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()
    return 1-probs.max().item()

gain_func_map = {
    GainMeasure.gini: gini,
    GainMeasure.information_gain: information_gain,
    GainMeasure.train_error: train_error,
}

def split_data(
    data:torch.Tensor,
    labels:torch.Tensor,
    gain_measure:GainMeasure
    )->Tuple[int,float]:
    # Get the function used to calculate the gain measure
    gain_func = gain_func_map[gain_measure]

    n_features = data.shape[1]
    best_gain,best_feature,best_threshold,best_left_ind,best_right_ind = float("-inf"),None,None,None,None
    for feature in range(n_features):
        thresholds = torch.unique(data[:, feature])
        for threshold in thresholds:
            left_idx = data[:, feature] <= threshold
            right_idx = data[:, feature] > threshold
            if left_idx.sum().item() == 0 or right_idx.sum().item() == 0:
                # If any case results in an empty child node, we do not consider it
                # If all possible splits do this, we do not split
                continue
            left_labels = labels[left_idx]
            right_labels = labels[right_idx]
            gain = gain_func(labels) - (len(left_labels) * gain_func(left_labels) + len(right_labels) * gain_func(right_labels)) / len(labels)
            if gain > best_gain:
                best_gain,best_feature,best_threshold,best_left_ind,best_right_ind = gain,feature,threshold,left_idx,right_idx
    return best_feature,best_threshold,best_left_ind,best_right_ind

    

def general_algorithm(
    data:torch.Tensor,
    labels:torch.Tensor,
    gain_measure:GainMeasure,
    depth:int = 0,
    max_depth:int = None,
    )->TreeNode:
    # If all labels are the same decision tree is trivial
    if torch.unique(labels).numel()==1:
        return TreeNode(value = labels[0].item())

    # If max depth reached or no features left, return majority class
    # If all labels are the same, return that label
    values, counts = torch.unique(labels, return_counts=True)
    if (max_depth is not None and depth >= max_depth) or (counts.max()==counts.sum()):
        return TreeNode(value=values[counts.argmax()].item())
    
    # Feature exhaustion check: if all features are constant, return majority class
    if all(torch.unique(data[:, i]).numel() == 1 for i in range(data.shape[1])):
        return TreeNode(value=values[counts.argmax()].item())
    
    best_feature,best_threshold,left_indices,right_indices = split_data(data,labels,gain_measure)
    if best_feature==None:
        # No split is found when all possible splits result in at least one empty child node
        return TreeNode(value=values[counts.argmax()].item())
    
    left = general_algorithm(data=data[left_indices],labels=labels[left_indices],gain_measure=gain_measure,depth=depth+1,max_depth=max_depth)
    right = general_algorithm(data=data[right_indices],labels=labels[right_indices],gain_measure=gain_measure,depth=depth+1,max_depth=max_depth)
    return TreeNode(feature=best_feature,threshold=best_threshold, left=left,right=right)

def ID3(
    data:torch.Tensor,
    labels:torch.Tensor,
    depth:int = 0,
    max_depth:int = None,
    )->TreeNode:
    return general_algorithm(data=data,labels=labels,gain_measure=GainMeasure.information_gain,depth=depth,max_depth=max_depth)

def CART(
    data:torch.Tensor,
    labels:torch.Tensor,
    depth:int = 0,
    max_depth:int = None,
    )->TreeNode:
    return general_algorithm(data=data,labels=labels,gain_measure=GainMeasure.gini,depth=depth,max_depth=max_depth)


def prune_tree(tree):
    #TODO
    pass