# Principle component analysis
# Source: Shalev-Shwartz & Ben-David, Chap 23
import torch 
from torch.profiler import profile, ProfilerActivity, record_function,tensorboard_trace_handler
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from sklearn.decomposition import PCA as skPCA

from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Optional,List,Tuple
import argparse
from enum import Enum
import time

from src.utils.data import get_data_dim_reduction,DimReductionDataset

class PCA_Solution:
    def __init__(self, U:torch.Tensor,mu:Optional[torch.Tensor]=None):
        self.U = U
        self.features,self.n = self.U.size()
        self.mu:torch.Tensor=mu if mu is not None else torch.zeros(self.features,dtype=self.U.dtype,device=self.U.device)
    
    def compress(self,x:torch.Tensor)->torch.Tensor:
        """
        x.size() = (feats) or (num_points,feats)
        """
        return (x-self.mu)@self.U

    def recover(self,x:torch.Tensor)->torch.Tensor:
        """
        x.size() = (n) or (num_points,n)
        return \tidle x = 
        """
        return x@self.U.T+self.mu

class PCA_Algorithm(Enum):
    vanilla = 0
    sklearn = 1

def pca(
    data: torch.Tensor,
    n:int,
    centering:bool=True,
    algorithm:PCA_Algorithm=PCA_Algorithm.vanilla,
    log_level : Optional[int] = 0,
    )-> Tuple[PCA_Solution,torch.Tensor]:
    """
    Takes a data tensor and returns the associated PCA matrix for a compression down to n dimensions.
    Returns PCA_Solutions, retrained variance
    """
    
    points, features = data.size()
    n = min(n,points,features)
    if n<0:
        raise(ValueError(f"n must be non negative, got {n} instead"))
    if n==0:
        return PCA_Solution(torch.tensor(0)),torch.tensor(0)
    if algorithm == PCA_Algorithm.vanilla:
        data_rel = data
        mu=None
        if centering:
            mu = data.mean(dim=0)
            data_rel = data - mu 
        if points>features:
            A = data_rel.T @ data_rel 
            # Matrix is PSD, eigendecomp and singular value decomp is the same
            #U,S,Vh = torch.linalg.svd(A,full_matrices=False)
            #components =  Vh[:n]
            eigenvals,eigenvecs = torch.linalg.eigh(A)
            eigenvals,eigenvecs = eigenvals.flip(0),eigenvecs.flip(1)
            components =  eigenvecs[:,:n]
            variance = eigenvals[:n].sum()/eigenvals.sum() 
            return PCA_Solution(components,mu),variance
        else:
            # Gram trick when the number of features is much greater than the num of points
            B = data_rel @ data_rel.T
            # Matrix is PSD, eigendecomp and singular value decomp is the same
            #U,S,Vh = torch.linalg.svd(B,full_matrices=False)
            #components = Vh[:n]@data
            #components = components/torch.norm(components,p=2,dim=1,keepdim=True)
            #variance = S[:n].sum()/S.sum() 
            eigenvals,eigenvecs = torch.linalg.eigh(B)
            eigenvals,eigenvecs = eigenvals.flip(0),eigenvecs.flip(1)
            components = data_rel.T@eigenvecs[:,:n]
            components = components/torch.norm(components,p=2,dim=0,keepdim=True)
            variance = eigenvals[:n].sum()/eigenvals.sum() 
            return PCA_Solution(components,mu),variance
    if algorithm == PCA_Algorithm.sklearn:
        reference_implementation = skPCA(n_components=n).fit(data.numpy())
        return(PCA_Solution(torch.from_numpy(reference_implementation.components_).T,torch.from_numpy(reference_implementation.mean_)),torch.from_numpy(reference_implementation.explained_variance_ratio_.sum()))


# Inference functions for PCA
def compress_vector(pca_matrix:PCA_Solution, x)->torch.Tensor:
    return pca_matrix.compress(x)

def recover_fector(pca_matrix:PCA_Solution, x)->torch.Tensor:
    return pca_matrix.recover(x)

# Eval functions
def check_agreement(pca_matrix_1:PCA_Solution,pca_matrix_2:PCA_Solution)->bool:
    # Check if the projection matrix
    projection_1 = (pca_matrix_1.U @ pca_matrix_1.U.T).numpy()
    projection_2 = (pca_matrix_2.U @ pca_matrix_2.U.T).numpy()
    return np.allclose(projection_1,projection_2,atol=1e-7) 

def compress_dataset(pca_matrix:PCA_Solution,data:torch.Tensor)->torch.tensor:
    compressed_data = pca_matrix.compress(data)
    return compressed_data

def evaluate_pca(pca_sol:PCA_Solution,data:torch.Tensor)->dict:
    reconstruction = pca_sol.recover(pca_sol.compress(data))
    residual = data-reconstruction
    data_rel = data - pca_sol.mu
    
    mse = residual.pow(2).sum(dim=1).mean(0)
    relative_error = residual.pow(2).sum()/data_rel.pow(2).sum()
    
    # Eckart-Young: the best any rank n projection can do is the tail of the spectrum
    S = torch.linalg.svdvals(data_rel)
    optimal = S[pca_sol.n:].pow(2).sum()/S.pow(2).sum()
    # Components must be orthonormal
    identity = torch.eye(pca_sol.n,dtype=data.dtype,device=data.device)
    orthonormality = (pca_sol.U.T@pca_sol.U - identity).abs().max()

    eval_results = {
        "n": pca_sol.n,
        "reconstruction_mse": mse.item(),
        "relative_error": relative_error.item(),
        "retained_variance": 1.0 - relative_error.item(),
        "optimal_relative_error": optimal.item(),
        "excess_over_optimal": relative_error.item() - optimal.item(),
        "orthonormality_error": orthonormality.item(),
    }
    return eval_results

def visualize_pca(pca_matrix:PCA_Solution)->None:
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

def get_profiler(device,algorithm:PCA_Algorithm):
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
    parser = argparse.ArgumentParser(description="PCA dimensionality reduction")
    parser.add_argument('--dataset', type=str,choices = [d.name for d in DimReductionDataset], default='WINE', help='Name of dataset to be used')
    parser.add_argument('--n', type=int, default=3, help='Number of components to keep')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in PCA_Algorithm], default='vanilla', help='Which algorithm to use to learn the SVM halfspace')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = DimReductionDataset[args.dataset]
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
    data = get_data_dim_reduction(dataset_name) # N by dim data tensor
    data = data.to(device)

    if sweep:
        start = time.time()
        retained_vars = [(i,pca(data,i,algorithm=algorithm,log_level=0)[1].cpu()) for i in tqdm(range(0,n))]
        end = time.time()
        print(f"{algorithm.name} sweep runtime:{end-start} seconds")
        #visualize_sweep(retained_vars) 
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
        print(f"PCA to {results['n']} components from {data.size(1)}")
        print(f"  reconstruction mse   : {results['reconstruction_mse']:.6f}")
        print(f"  retained variance    : {results['retained_variance']:.4f}")
        print(f"  relative error       : {results['relative_error']:.6f} (optimal {results['optimal_relative_error']:.6f})")
        print(f"  excess over optimal  : {results['excess_over_optimal']:.2e}, {'less than' if results['excess_over_optimal']<1e-6 else 'more than'} tolerance.")
        print(f"  orthonormality error : {results['orthonormality_error']:.2e}, {'less than' if results['orthonormality_error']<1e-6 else 'more than'} tolerance.")

