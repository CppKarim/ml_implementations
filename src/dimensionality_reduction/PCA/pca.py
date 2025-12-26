# Principle component analysis
import torch 
from torch.profiler import profile, ProfilerActivity, record_function,tensorboard_trace_handler
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Optional,List,Tuple
import argparse
from enum import Enum
import time

from src.utils.data import get_data,Dataset


class PCA_Matrix:
    def __init__(self, A:torch.Tensor):
        self.A = A
    
    def compress(self,x:torch.Tensor)->torch.Tensor:
        return x@self.A.T

    #def recover(self,x:torch.Tensor)->torch.Tensor:
        #return self.A.T@x


class PCA_Algorithm(Enum):
    vanilla = 0

def pca(
    data: torch.Tensor,
    n:int,
    algorithm:PCA_Algorithm=PCA_Algorithm.vanilla,
    log_level : Optional[int] = 0,
    )-> Tuple[PCA_Matrix,torch.Tensor]:
    points, features = data.size()
    n = min(n,data.size(0),data.size(1))
    if n<0:
        raise(ValueError(f"n must be non negative, got {n} instead"))
    if n==0:
        return torch.tensor(0),torch.tensor(0)
    if points>features:
        A = data.T @ data
        U,S,Vh = torch.linalg.svd(A,full_matrices=False)
        components =  Vh[:n]
        variance = S[:n].sum()/S.sum() 
        return components,variance
    else:
        B = data @ data.T
        U,S,Vh = torch.linalg.svd(B,full_matrices=False)
        components = Vh[:n]@data
        components = components/torch.norm(components,p=2,dim=1,keepdim=True)
        variance = S[:n].sum()/S.sum() 
        return components,variance

# Inference function for decision tree
def compress_vector(pca_matrix:PCA_Matrix, x)->torch.Tensor:
    return pca_matrix.compress(x)

def classify_dataset(pca_matrix:PCA_Matrix,data:torch.Tensor)->torch.tensor:
    #predictions = torch.Tensor([halfspace.classify(x) for x in data])
    compressed_data = pca_matrix.compress(data)
    return compressed_data

def evaluate_pca(pca_matrix:PCA_Matrix,data:torch.Tensor)->None:
    #eval_results = classify_dataset(tree,data)
    #eval_results.print_confusion_matrix()
    #print(f"Halfspace accuracy: {eval_results.accuracy()}")
    #visualize_tree(tree)
    pass
    
    #return eval_results

def visualize_pca(pca_matrix:PCA_Matrix)->None:
    pass

def visualize_sweep(accuracies:List[float]):
    accuracies_np = np.array(accuracies)
    plt.scatter(accuracies_np[:, 0], accuracies_np[:, 1],label="PCA tradeoff")
    for x, y in accuracies_np:
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center')
    plt.xlabel('Number of components')
    plt.ylabel('Retained variance')
    plt.legend()
    plt.show()

def get_profiler(device:str,algorithm:PCA_Algorithm):
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
    parser.add_argument('--dataset', type=str,choices = [d.name for d in Dataset], default='WINE', help='Name of dataset to be used')
    parser.add_argument('--n', type=int, default=3, help='Number of components to keep')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in PCA_Algorithm], default='vanilla', help='Which algorithm to use to learn the SVM halfspace')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = Dataset[args.dataset]
    algorithm = PCA_Algorithm[args.algorithm]
    n = args.n
    to_profile = args.profile
    sweep = args.sweep
    
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data = get_data(dataset_name) # N by dim data tensor
    data = data.to(device)

    if sweep:
        start = time.time()
        retained_vars = [(i,pca(data,i,algorithm=algorithm,log_level=0)[1].cpu()) for i in tqdm(range(0,n))]
        end = time.time()
        print(f"{algorithm.name} sweep runtime:{end-start} seconds")
        visualize_sweep(retained_vars) 
    else:
        if to_profile:
                prof = get_profiler(device,algorithm)
                start = time.time()
                prof.start()
                pca_matrix, retained_var = pca(
                    data,
                    n,
                    algorithm=algorithm,
                    log_level=args.log,
                )
                prof.stop()
                end = time.time()

                print(f"{algorithm.name} runtime:{end-start} seconds")
                print(prof.key_averages().table())
        else:
            start = time.time()
            pca_matrix, retained_var = pca(
                data,
                n,
                algorithm=algorithm,
                log_level=args.log,
            )
            end = time.time()

            print(f"{algorithm.name} runtime:{end-start} seconds")

        # Evaluate the learned means 
        results = evaluate_pca(pca_matrix,data)

