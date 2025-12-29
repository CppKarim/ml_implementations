# Linear regression
import torch 
from torch.profiler import profile, ProfilerActivity, record_function,tensorboard_trace_handler
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from tqdm import tqdm
import matplotlib.pyplot as plt
import cvxpy as cp

from typing import Optional,List
import argparse
from enum import Enum
import time

from src.utils.data import get_data,ClassificationDataset
from src.utils.classification import ClassificationResults


class Halfspace:
    def __init__(self, w:torch.Tensor, b:torch.Tensor):
        self.w = w
        self.b = b
        self.device = self.w.device
    
    def classify(self,x:float)->torch.Tensor:
        return (x.to(self.device)@self.w>=self.b).int()

class Halfspace_Algorithm(Enum):
    LP = 0
    BATCH_PERCEPTRON = 1
    SVM = 2
    SVM_SOFT = 2

def learn_halfspace(
    data: torch.Tensor,
    binary_labels: torch.Tensor,
    algorithm : Halfspace_Algorithm = Halfspace_Algorithm.LP,
    lambda_par:torch.Tensor=None,
    log_level : Optional[int] = 0,
    )-> Halfspace:
    if not ((binary_labels==0) + (binary_labels==1)).all() :
        raise(ValueError("Only binary labels (0/1) are accepted."))
    hom_data = torch.cat([data,torch.ones((data.size(0),1),dtype=data.dtype,device=data.device)],dim=1)
    labels = 2*binary_labels-1 # -1,1 labels

    if algorithm==Halfspace_Algorithm.LP:
        # Define optimization variable
        data_points = hom_data.size(0)
        features = hom_data.size(1) # This includes the added bias dimension
        w = cp.Variable(features)
        A = (hom_data*labels.unsqueeze(dim=1)).cpu().numpy()
        v = np.ones(data_points)

        # Define objective function
        objective = cp.Maximize(0)

        # Define quadratic constraints
        constraints = [
            A@w >= v
        ]

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return Halfspace(torch.tensor(w.value[:-1],dtype=torch.float32),torch.tensor(w.value[-1],dtype=torch.float32))

    elif algorithm==Halfspace_Algorithm.BATCH_PERCEPTRON:
        w = torch.zeros_like(hom_data[0])
        
        classification = (hom_data @ w)*labels
        with tqdm(disable=log_level<2) as pbar:
            while (classification<=0).any():
                indices = torch.where(classification<=0)[0] 
                x_index = indices[torch.randint(0,indices.size(0),(1,)).item()]
                w.add_(labels[x_index]*hom_data[x_index])
                classification = (hom_data @ w)*labels
                pbar.update(1)
        
        return Halfspace(w[:-1],w[-1])

    elif algorithm==Halfspace_Algorithm.SVM:
        # Define optimization variable
        data_points = hom_data.size(0)
        features = hom_data.size(1) # This includes the added bias dimension
        w = cp.Variable(features)
        A = (hom_data*labels.unsqueeze(dim=1)).cpu().numpy()
        v = np.ones(data_points)

        # Define objective function
        objective = cp.Minimize(cp.norm(w,p=2)) 
        #objective = cp.Minimize(cp.norm(w[:-1],p=2)) 

        # Define quadratic constraints
        constraints = [
            A@w >= v
        ]

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return Halfspace(torch.tensor(w.value[:-1],dtype=torch.float32),torch.tensor(w.value[-1],dtype=torch.float32))
        
    elif algorithm==Halfspace_Algorithm.SVM_SOFT:
        if lambda_par is None:
            raise(ValueError(f"Expected a value for lambda_par for the implementation of soft SVM, got {lambda_par} instead"))
        # Define optimization variable
        data_points = hom_data.size(0)
        features = hom_data.size(1) # This includes the added bias dimension

        w = cp.Variable(features)
        slack_vector = cp.Variable(features)
        lambda_np = lambda_par.cpu().numpy()
        A = (hom_data*labels.unsqueeze(dim=1)).cpu().numpy()
        v = np.ones(data_points)-slack_vector

        # Define objective function
        objective = cp.Minimize(cp.norm(lambda_np*w+slack_vector.mean(),p=2)) 
        #objective = cp.Minimize(cp.norm(w[:-1],p=2)) 

        # Define quadratic constraints
        constraints = [
            A@w >= v,
            slack_vector>=0
        ]

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return Halfspace(torch.tensor(w.value[:-1],dtype=torch.float32),torch.tensor(w.value[-1],dtype=torch.float32))
 
    else:
        raise(NotImplementedError(f"Algorithm {algorithm.name} is not implemented"))

# Inference function for decision tree
def predict_class(halfspace:Halfspace, x)->torch.Tensor:
    return halfspace.classify(x)

def classify_dataset(halfspace:Halfspace,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    predictions = halfspace.classify(data)
    return ClassificationResults(labels,predictions)

def evaluate_halspace(halfspace:Halfspace,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    eval_results = classify_dataset(halfspace,data,labels)
    eval_results.print_confusion_matrix()
    print(f"Halfspace accuracy: {eval_results.accuracy()}")
    #visualize_halfspace(halfspace)
    
    return eval_results

def visualize_halfspace(halfpace:Halfspace,depth=0)->None:
    pass

def visualize_sweep(accuracies:List[float],num_classes:int=None):
    accuracies_np = np.array(accuracies)
    plt.scatter(accuracies_np[:, 0], accuracies_np[:, 1],label="Halfspace accuracy")
    plt.xlabel('')
    plt.ylabel('Accuracy')
    if num_classes is not None and num_classes > 1:
        random_acc = 1.0 / num_classes
        plt.axhline(random_acc, color='red', linestyle='--', label=f'Random Accuracy ({random_acc:.2f})')
    plt.legend()
    plt.show()

def get_profiler(device:str,algorithm:Halfspace_Algorithm):
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
    parser.add_argument('--dataset', type=str,choices = [d.name for d in ClassificationDataset], default='IRIS', help='Name of dataset to be used')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in Halfspace_Algorithm], default='LP', help='Which algorithm to use to learn the halfspace')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = ClassificationDataset[args.dataset]
    algorithm = Halfspace_Algorithm[args.algorithm]
    to_profile = args.profile
    sweep = args.sweep
    
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data,labels = get_data(dataset_name,with_labels=True,binary=True) # N by dim data tensor
    data = data.to(device)
    labels = labels.to(device)


    if to_profile:
            prof = get_profiler(device,algorithm)
            start = time.time()
            prof.start()
            halfspace = learn_halfspace(
                data,
                labels,
                algorithm=algorithm,
                log_level=args.log,
            )
            prof.stop()
            end = time.time()

            print(f"{algorithm.name} runtime:{end-start} seconds")
            print(prof.key_averages().table())
    else:
        start = time.time()
        halfspace = learn_halfspace(
            data,
            labels,
            algorithm=algorithm,
            log_level=args.log,
        )
        end = time.time()

        print(f"{algorithm.name} runtime:{end-start} seconds")

    # Evaluate the learned means 
    results = evaluate_halspace(halfspace,data,labels)

