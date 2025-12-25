# K-means
import torch 
from torch.profiler import profile, ProfilerActivity, record_function,tensorboard_trace_handler
torch.set_num_threads(16)
import os
os.environ["OMP_NUM_THREADS"] = "16"
import numpy as np
from sklearn.datasets import load_iris
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Optional
import argparse
from enum import Enum
import time
import itertools

from src.utils.data import get_data,Dataset

class InitAlg(Enum):
    RANDOM = "random"
    PLUSPLUS = "plusplus"
    
def init_means(data:torch.Tensor,k:int,alg:InitAlg = InitAlg.RANDOM):
    if alg == InitAlg.PLUSPLUS:
        means = []
        index = torch.randint(0,data.size(0),(1,))
        means.append(data[index].squeeze(0))
        while means.size(0)<k:
            means_tensor = torch.stack(means)
            dists = torch.cdist(data, means_tensor, p=2).min(dim=1).values ** 2
            probs = dists/dists.sum()
            next_index = torch.multinomial(probs,1)
            means.append(data[next_index].squeeze(0))
        means = torch.stack(means)
    else: 
        indices = torch.randperm(data.size(0))[:k]
        means = data[indices]
    return means

def k_means(
    data:torch.tensor,
	k:int,
	max_iterations:Optional[int]=int(1e2),
	epsilon:Optional[float]=1e-6,
	inits:int=1,
	log_level:int=2,
 ) -> torch.tensor:
    dim = data.size(1)
    #mean = data.mean(dim=0,keepdim=True)
    #std = data.std(dim=0,keepdim=True)

    best_means = (None,None)
    for init in tqdm(range(inits),disable=log_level<1):
        # Initialize means
        means = init_means(data,k) 
        #distances = torch.tensor([[ (point - mean).pow(2).sum() for mean in means] for point in data])
        distances = torch.cdist(data, means, p=2)   # (N, k), built in pytorch is more efficient, distance from each points to each mean
                                                    # Done outside loop to find loss
        if init==0:
            best_means = (means,distances.min(dim=1).values.mean())

        stop=False
        iteration = 0
        with tqdm(total=max_iterations,disable=log_level<2) as pbar:
            while not stop and iteration< max_iterations:
                # Assign points to means
                assignment = distances.argmin(dim=1)
                
                # Find new means
                #new_means = torch.stack( [  (data*(assignment==i).unsqueeze(1)).sum(dim=0)/(assignment==i).sum()  for i in range(means.size(0))] )
                counts = torch.bincount(assignment, minlength=k).clamp(min=1).unsqueeze(1)  # (k, 1) tensor, countains the number of points in each group
                sums = torch.zeros(k, dim, device=data.device).scatter_add_(0, assignment.unsqueeze(1).expand(-1, dim), data) # Adds the entries in data 
                                                                                        # according to the indices in assignment to the relevant rows in a torch.zeros tensor
                new_means = sums / counts 
                new_distances = torch.cdist(data,new_means,p=2)
                new_loss = new_distances.min(dim=1).values.mean()
                if new_loss < best_means[1]:
                    best_means = (new_means, new_loss)
                
                # Stopping criterion
                delta = (means-new_means).max() 
                stop = delta < epsilon
                
                # Next iteration
                if iteration%5==0 and log_level>2:
                    print(f"Iteration:{iteration}, max mean delta:{delta}")
                    #pbar.set_description(f"Iteration:{iteration}, max mean delta:{delta}")
                means = new_means
                distances = new_distances
                loss = new_loss
                iteration+=1
                pbar.update(1)
            stopping_reason = "convergence" if delta<epsilon else "iteration backstop"
        if log_level>0:
            print(f"Iteration results\n\tFinal loss:{loss}\n\t Stopping criterion:{stopping_reason}")
    return best_means

def evaluate_means(data,means):
    visualize_means(data,means)

def visualize_means(data, means, num_pairs=4):
    """
    Visualize the data and means along several 2D projections.
    If num_pairs is None, visualize all unique pairs. Otherwise, visualize the first num_pairs pairs.
    """
    data_np = data.cpu().numpy()
    means_np = means.cpu().numpy()
    dim = data_np.shape[1]
    pairs = list(itertools.combinations(range(dim), 2))
    rng = np.random.default_rng()
    rng.shuffle(pairs)
    if num_pairs is not None:
        pairs = pairs[:num_pairs]
    for i, (dim_0, dim_1) in enumerate(pairs):
        plt.figure()
        plt.scatter(data_np[:, dim_0], data_np[:, dim_1], label="Data")
        plt.scatter(means_np[:, dim_0], means_np[:, dim_1], label="Means")
        plt.xlabel(f"Dimension {dim_0}")
        plt.ylabel(f"Dimension {dim_1}")
        plt.title(f"Projection {i+1}: dim {dim_0} vs dim {dim_1}")
        plt.legend()
        plt.show()
    
def visualize_sweep(points):
    points_np = np.array(points)
    plt.plot(points_np[:, 0], points_np[:, 1])
    plt.xlabel('# of means')
    plt.ylabel('Loss')
    plt.show()

def get_profiler():
    return torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
                ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=tensorboard_trace_handler(f"./log/k_means_{device}")
            ) 


if __name__ == "__main__":
    # User input
    parser = argparse.ArgumentParser(description="K-means clustering")
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--N', type=int, default=100, help='Number of data points')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of data')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Max iterations')
    parser.add_argument('--dataset', type=str,choices = [d.name for d in Dataset], default='IRIS', help='Name of dataset to be used')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Convergence threshold')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in InitAlg], default='PLUSPLUS', help='Which algorithm to initialize the means')
    parser.add_argument('--inits', type=int, default=1, help='How many random initializations to perform')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    args = parser.parse_args()
    N = args.N
    k = args.k
    dim = args.dim
    epsilon = args.epsilon
    max_iter = args.max_iterations
    inits = args.inits
    sweep = args.sweep
    to_profile = args.profile
    dataset_name = Dataset[args.dataset]

    # Initialization
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data = get_data(dataset_name,N=N,dim=dim) # N by dim data tensor
    data.to(device)

    # Learn the means from the data
    if sweep:
        losses = [(i,k_means(data,i,max_iter,epsilon,inits,log_level=0)[1]) for i in tqdm(range(1,k))]
        visualize_sweep(losses) 
    else:
        if to_profile:
            prof = get_profiler()
            start = time.time()
            prof.start()
            means,loss = k_means(data,k,max_iter,epsilon,inits,log_level=args.log,profiler=prof)
            prof.stop()
            end = time.time()
            print(f"Average point-mean distance: {loss}")
            print(f"k-means runtime:{end-start} seconds")
            print(prof.key_averages().table())
        else:
            start = time.time()
            means,loss = k_means(data,k,max_iter,epsilon,inits,log_level=0)
            end = time.time()
            print(f"Average point-mean distance: {loss}")
            print(f"k-means runtime:{end-start} seconds")

    # Evaluate the learned means 
    evaluate_means(data,means)


