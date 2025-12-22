# K-means
import torch 
import numpy as np
from tqdm import tqdm

from typing import Optional
import argparse

# Function that returns data to be clustered
def get_data(N,dim):
    return torch.rand((N,dim))

def init_means(data,k, random=True):
    if random:
        means = torch.rand(k,data.size(1))
    else: 
        indices = torch.randperm(data.size(0))[:k]
        means = data[indices]
    return means

def k_means(data_points:torch.tensor,k:int,max_iterations:Optional[int]=int(1e2),epsilon:Optional[float]=1e-6) -> torch.tensor:
    # Initialize means
    means = init_means(data,k,random=False) 
    distances = torch.cdist(data, means, p=2)   # (N, k), built in pytorch is more efficient, distance from each points to each mean
                                                # Done outside loop to find loss
    dim = data_points.size(1)

    stop=False
    iteration = 0
    best_means = (means,distances.min(dim=1).values.sum())
    with tqdm(total=max_iterations) as pbar:
        while not stop and iteration< max_iterations:
            # Assign points to means
            #distances = torch.tensor([[ (point - mean).pow(2).sum() for mean in means] for point in data_points])
            assignment = distances.argmin(dim=1)
            
            # Find new means
            #new_means = torch.stack( [  (data_points*(assignment==i).unsqueeze(1)).sum(dim=0)/(assignment==i).sum()  for i in range(means.size(0))] )
            counts = torch.bincount(assignment, minlength=k).clamp(min=1).unsqueeze(1)  # (k, 1) tensor, countains the number of points in each group
            sums = torch.zeros(k, dim, device=data_points.device).scatter_add_(0, assignment.unsqueeze(1).expand(-1, dim), data_points) # Adds the entries in data_points 
                                                                                    # according to the indices in assignment to the relevant rows in a torch.zeros tensor
            new_means = sums / counts 
            new_distances = torch.cdist(data,new_means,p=2)
            new_loss = new_distances.min(dim=1).values.sum()
            if new_loss < best_means[1]:
                best_means = (new_means, new_loss)
            
            
            # Stopping criterion
            delta = (means-new_means).max() 
            stop = delta < epsilon
            
            # Next iteration
            if iteration%5==0:
                pbar.set_description(f"Iteration:{iteration}, max mean delta:{delta}")
            means = new_means
            distances = new_distances
            iteration+=1
            pbar.update(1)
    return means

def visualize_means(data,means):
    import matplotlib.pyplot as plt

    data_np = data.numpy()
    means_np = means.numpy()
    plt.scatter(data_np[:,0], data_np[:,1],label="Data")
    plt.scatter(means_np[:,0], means_np[:,1],label="Means")
    plt.show()



# User input
parser = argparse.ArgumentParser(description="K-means clustering")
parser.add_argument('--k', type=int, default=10, help='Number of clusters')
parser.add_argument('--N', type=int, default=100, help='Number of data points')
parser.add_argument('--dim', type=int, default=2, help='Dimension of data')
parser.add_argument('--max_iterations', type=int, default=1000, help='Max iterations')
parser.add_argument('--epsilon', type=float, default=1e-6, help='Convergence threshold')
args = parser.parse_args()
N = args.N
k = args.k
epsilon = args.epsilon
dim = args.dim
max_iter = args.max_iterations

# Prepare data
data = get_data(N,dim) # N by dim data tensor

# Find the required means
means = k_means(data,k,epsilon,max_iter)

# Visualize results
visualize_means(data,means)

