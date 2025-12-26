# Decision tree 
import torch 
from torch.profiler import profile, ProfilerActivity, record_function,tensorboard_trace_handler
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Optional,List
import argparse
from enum import Enum
import time

from src.utils.data import get_data,Dataset
from src.utils.decision_tree_algs import TreeNode,CART,ID3
from src.utils.classification import ClassificationResults

class TreeAlg(Enum):
    ID3 = 0
    C45 = 1
    CART = 2
    CHAID = 3

def learn_tree(
    data: torch.Tensor,
    labels: torch.Tensor,
    max_depth:int = None,
    depth:int = 0,
    algorithm : TreeAlg = TreeAlg.CART,
    log_level : Optional[int] = 0,
    )-> TreeNode:
    if algorithm==TreeAlg.CART:
        return CART(data,labels,depth,max_depth)
    if algorithm==TreeAlg.ID3:
        return ID3(data,labels,depth,max_depth)
    else:
        raise(NotImplementedError(f"Algorithm {algorithm.name} is not implemented"))

# Inference function for decision tree
def infer_tree(tree, x)->torch.Tensor:
    node = tree
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def classify_dataset(tree:TreeNode,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    predictions = torch.Tensor([infer_tree(tree,x) for x in data])
    return ClassificationResults(labels,predictions)


def evaluate_tree(tree:TreeNode,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    eval_results = classify_dataset(tree,data,labels)
    eval_results.print_confusion_matrix()
    print(f"Tree accuracy: {eval_results.accuracy()}")
    visualize_tree(tree)
    
    return eval_results

def visualize_tree(tree:TreeNode,depth=0)->None:
    #depth = tree.get_depth()
    indent = "  " * depth
    if tree.value is not None:
        print(f"{indent}Leaf: value={tree.value}")
    else:
        print(f"{indent}tree: feature={tree.feature}, threshold={tree.threshold}")
        visualize_tree(tree.left, depth+1)
        visualize_tree(tree.right, depth+1)

def visualize_sweep(accuracies:List[float],num_classes:int=None):
    accuracies_np = np.array(accuracies)
    plt.scatter(accuracies_np[:, 0], accuracies_np[:, 1],label="Tree accuracy")
    plt.xlabel('Maximum tree depth')
    plt.ylabel('Accuracy')
    if num_classes is not None and num_classes > 1:
        random_acc = 1.0 / num_classes
        plt.axhline(random_acc, color='red', linestyle='--', label=f'Random Accuracy ({random_acc:.2f})')
    plt.legend()
    plt.show()

def get_profiler(device:str,algorithm:TreeAlg):
    return torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
                ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=tensorboard_trace_handler(f"./log/decision_tree_{algorithm.name}_{device}")
            ) 


if __name__ == "__main__":
    # User input
    parser = argparse.ArgumentParser(description="Decision Tree classification")
    parser.add_argument('--dataset', type=str,choices = [d.name for d in Dataset], default='IRIS', help='Name of dataset to be used')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in TreeAlg], default='CART', help='Which algorithm to learn the decision tree')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum tree depth')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = Dataset[args.dataset]
    algorithm = TreeAlg[args.algorithm]
    to_profile = args.profile
    max_depth = args.max_depth
    sweep = args.sweep
    
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data,labels = get_data(dataset_name,with_labels=True) # N by dim data tensor
    data = data.to(device)
    labels = labels.to(device)

    if sweep:
        trees = [learn_tree(data,labels,max_depth=ind,algorithm=algorithm,log_level=0) for ind in range(max_depth)]
        eval_results = [classify_dataset(tree,data,labels) for tree in trees]
        accuracies = [(i,results.accuracy()) for i,results in enumerate(eval_results)]
        visualize_sweep(accuracies,num_classes=len(labels.unique())) 
    else:
        if to_profile:
                prof = get_profiler(device,algorithm)
                start = time.time()
                prof.start()
                tree = learn_tree(
                    data,
                    labels,
                    max_depth=max_depth,
                    algorithm=algorithm,
                    log_level=args.log,
                )
                prof.stop()
                end = time.time()

                print(f"{algorithm.name} runtime:{end-start} seconds")
                print(prof.key_averages().table())
        else:
            start = time.time()
            tree = learn_tree(
                data,
                labels,
                max_depth=max_depth,
                algorithm=algorithm,
                log_level=args.log,
            )
            end = time.time()

            print(f"{algorithm.name} runtime:{end-start} seconds")

        # Evaluate the learned means 
        results = evaluate_tree(tree,data,labels)

