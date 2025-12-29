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

from src.utils.data import get_data_regression,RegressionDataset
from src.utils.regression import RegressionResults

class Halfplane:
    def __init__(self, w:torch.Tensor, b:torch.Tensor):
        self.w = w
        self.b = b
        self.device = self.w.device

    def predict(self,x:float)->torch.Tensor:
        return x.to(self.device)@self.w + self.b

class Linear_Regression_Algorithm(Enum):
    LEAST_SQUARES = 0
    RIDGE = 1
    LASSO = 2

class Kernel(Enum):
    IDENTITY = 0
    POLYNOMIAL = 1

def polynomial(row:torch.tensor,n:int=3):
    powers = torch.cat([row ** i for i in range(1, n + 1)])
    return powers

kernel_func_map = {
    Kernel.POLYNOMIAL: polynomial,
}

def learn_halfspace(
    data: torch.Tensor,
    outputs: torch.Tensor,
    algorithm : Linear_Regression_Algorithm = Linear_Regression_Algorithm.LEAST_SQUARES,
    Lambda : float = None,
    log_level : Optional[int] = 0,
    )-> Halfplane:
    hom_data = torch.cat([data,torch.ones((data.size(0),1),dtype=data.dtype,device=data.device)],dim=1)
    
    if algorithm==Linear_Regression_Algorithm.LEAST_SQUARES:
        A = hom_data.T @ hom_data
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        reciprocal = torch.where(eigenvalues != 0, 1.0 / eigenvalues, torch.zeros_like(eigenvalues)) 
        D_plus = torch.diag(reciprocal)
        V = eigenvectors
        A_plus = V @ D_plus @ V.T       
        b = hom_data.T @ outputs
        
        w_hat = A_plus@b
        
        return Halfplane(w_hat[:-1],w_hat[-1])

    if algorithm==Linear_Regression_Algorithm.RIDGE:
        if Lambda==None:
            raise(ValueError(f"Complexity parameter Lambda must be defined for ridge regression, instead got {Lambda}"))
        A = hom_data.T @ hom_data + Lambda*torch.eye(hom_data.size(1),device=hom_data.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        reciprocal = torch.where(eigenvalues != 0, 1.0 / eigenvalues, eigenvalues) 
        D_plus = torch.diag(reciprocal)
        V = eigenvectors
        A_plus = V @ D_plus @ V.T       
        b = hom_data.T @ outputs
        
        w_hat = A_plus@b
        
        return Halfplane(w_hat[:-1],w_hat[-1])


    if algorithm==Linear_Regression_Algorithm.LASSO:
        if Lambda==None:
            raise(ValueError(f"Complexity parameter Lambda must be defined for ridge regression, instead got {Lambda}"))
        # Define optimization variable
        data_points = hom_data.size(0)
        features = hom_data.size(1) # This includes the added bias dimension

        w = cp.Variable(features)

        # Define objective function
        objective = cp.Minimize(0.5*cp.sum_squares(values.cpu().numpy() - hom_data.cpu().numpy()@w)+Lambda*cp.norm1(w))

        # Create and solve the problem
        prob = cp.Problem(objective)
        prob.solve()
        return Halfplane(torch.tensor(w.value[:-1],dtype=torch.float32,device=hom_data.device),torch.tensor(w.value[-1],dtype=torch.float32,device=hom_data.device))
        
    else:
        raise(NotImplementedError(f"Algorithm {algorithm.name} is not implemented"))

# Inference function for the halfplane
def predict_class(halfplane:Halfplane, x)->torch.Tensor:
    return halfplane.predict(x)

def predict_dataset(halfplane:Halfplane,data:torch.Tensor,values:torch.Tensor)->RegressionResults:
    predictions = halfplane.predict(data)
    return RegressionResults(values,predictions,torch.nn.functional.mse_loss)

def evaluate_halfspace(halfplane:Halfplane,data:torch.Tensor,values:torch.Tensor)->RegressionResults:
    eval_results = predict_dataset(halfplane,data,values)
    #eval_results.print_confusion_matrix()
    print(f"Halfplane loss: {eval_results.loss()}")
    #visualize_tree(halfplane)
    
    return eval_results

def visualize_tree(halfplane:Halfplane,depth=0)->None:
    pass

def visualize_sweep(accuracies:List[float],num_classes:int=None):
    accuracies_np = np.array(accuracies)
    plt.scatter(accuracies_np[:, 0], accuracies_np[:, 1],label="SVM accuracy")
    plt.xlabel('Maximum tree depth')
    plt.ylabel('Accuracy')
    if num_classes is not None and num_classes > 1:
        random_acc = 1.0 / num_classes
        plt.axhline(random_acc, color='red', linestyle='--', label=f'Random Accuracy ({random_acc:.2f})')
    plt.legend()
    plt.show()

def get_profiler(device:str,algorithm:Linear_Regression_Algorithm):
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
    parser.add_argument('--dataset', type=str,choices = [d.name for d in RegressionDataset], default='DIABETES', help='Name of dataset to be used')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in Linear_Regression_Algorithm], default='LEAST_SQUARES', help='Which algorithm to use to learn the SVM halfspace')
    parser.add_argument('--Lambda', type=float, default='0.1', help='Complexity parameter for ridge regression (see ESL 3.4.1).')
    parser.add_argument('--kernel', type=str, choices=[a.name for a in Kernel], default='IDENTITY', help='Kernel for data preprocessing')
    parser.add_argument('--n', type=int, default='3', help='Polynomial degree to project data.')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = RegressionDataset[args.dataset]
    algorithm = Linear_Regression_Algorithm[args.algorithm]
    Lambda = args.Lambda
    kernel = Kernel[args.kernel]
    to_profile = args.profile
    sweep = args.sweep
    
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data,values = get_data_regression(dataset_name) # N by dim data tensor
    if kernel != Kernel.IDENTITY:
        kernel_fn = kernel_func_map[kernel]
        data = torch.stack([kernel_fn(row,args.n) for row in data])
    data = data.to(device)
    values = values.to(device)

    if to_profile:
            prof = get_profiler(device,algorithm)
            start = time.time()
            prof.start()
            halfspace = learn_halfspace(
                data,
                values,
                algorithm=algorithm,
                Lambda=Lambda,
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
            values,
            algorithm=algorithm,
            Lambda=Lambda,
            log_level=args.log,
        )
        end = time.time()

        print(f"{algorithm.name} runtime:{end-start} seconds")

    # Evaluate the learned means 
    results = evaluate_halfspace(halfspace,data,values)

