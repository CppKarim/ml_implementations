# Logistic regression
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

class Halfplane:
    def __init__(self, w:torch.Tensor, b:torch.Tensor):
        self.w = w
        self.b = b
        self.device = self.w.device

    def predict(self,x:float)->torch.Tensor:
        return torch.sigmoid(x.to(self.device)@self.w + self.b)

    def classify(self,x:float)->torch.Tensor:
        return torch.sigmoid(x.to(self.device)@self.w + self.b)>=0.5

class Logistic_Regression_Algorithm(Enum):
    CONVEX_PROGRAMMING = 0
    SGD = 1
    ADAM = 2

class Kernel(Enum):
    IDENTITY = 0
    POLYNOMIAL = 1

def polynomial(row:torch.tensor,n:int=-1):
    if n<2:
        raise(ValueError(f"n must be an integer greater than 1, got {n} instead"))
    powers = torch.cat([row ** i for i in range(2, n + 1)])
    return powers

kernel_func_map = {
    Kernel.POLYNOMIAL: polynomial,
}

def learn_halfplane(
    data: torch.Tensor,
    binary_labels: torch.Tensor,
    algorithm : Logistic_Regression_Algorithm = Logistic_Regression_Algorithm.CONVEX_PROGRAMMING,
    log_level : Optional[int] = 0,
    )-> Halfplane:
    if not ((binary_labels==0) + (binary_labels==1)).all() :
        raise(ValueError("Only binary labels (0/1) are accepted."))
    hom_data = torch.cat([data,torch.ones((data.size(0),1),dtype=data.dtype,device=data.device)],dim=1)
    labels = 2*binary_labels-1 # -1,1 labels

    if algorithm==Logistic_Regression_Algorithm.CONVEX_PROGRAMMING:
        # Optimization variable
        w = cp.Variable(hom_data.size(1))
        labels_np = labels.cpu().numpy()
        data_np = hom_data.cpu().numpy()
        
        vec = -cp.multiply(labels_np,(data_np@w))
        
        loss_func = cp.logistic(vec).mean()
        objective = cp.Minimize(loss_func)
        
        problem = cp.Problem(objective=objective)
        problem.solve()
        w = w.value
        
        return Halfplane(w=w[:-1],b=w[-1])
    
    if algorithm==Logistic_Regression_Algorithm.SGD:
        w_hat = torch.randn((hom_data.size(1),),requires_grad=True,device=hom_data.device)
        
        opt_params = {
            'lr': 1e-1,
            #'momentum': 0.9,
            #'weight_decay': 1e-4
        }
        optimizer = torch.optim.SGD([w_hat],**opt_params)
        
        max_epochs = 10000
        pbar =  tqdm(range(max_epochs), desc="SGD Logistic Regression")
        for _ in pbar:
            logits = -labels * (hom_data@w_hat)
            loss = torch.nn.functional.softplus(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
            if loss<1e-3:
                break

        return Halfplane(w=w_hat[:-1].detach(), b=w_hat[-1].detach())
            
    
    if algorithm==Logistic_Regression_Algorithm.ADAM:
        w_hat = torch.randn((hom_data.size(1),),requires_grad=True,device=hom_data.device)
        
        opt_params = {
            'lr': 1e-1,
            #'momentum': 0.9,
            #'weight_decay': 1e-4
        }
        optimizer = torch.optim.Adam([w_hat],**opt_params)
        
        max_epochs = 10000
        pbar =  tqdm(range(max_epochs), desc="SGD Logistic Regression")
        for _ in pbar:
            logits = -labels * (hom_data@w_hat)
            loss = torch.nn.functional.softplus(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
            if loss<1e-3:
                break

        return Halfplane(w=w_hat[:-1].detach(), b=w_hat[-1].detach())
            
        
    else:
        raise(NotImplementedError(f"Algorithm {algorithm.name} is not implemented"))

# Inference function for the halfplane
def classify_class(halfplane:Halfplane, x)->torch.Tensor:
    return halfplane.classify(x)

def classify_dataset(halfplane:Halfplane,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    predictions = halfplane.classify(data)
    return ClassificationResults(labels,predictions)

def evaluate_halfplane(halfplane:Halfplane,data:torch.Tensor,labels:torch.Tensor)->ClassificationResults:
    eval_results = classify_dataset(halfplane,data,labels)
    eval_results.print_confusion_matrix()
    print(f"Halfplane loss: {eval_results.accuracy()}")
    #visualize_tree(halfplane)
    
    return eval_results

def visualize_halfplane(halfplane:Halfplane,depth=0)->None:
    pass

def visualize_sweep(accuracies:List[float],num_classes:int=None):
    accuracies_np = np.array(accuracies)
    plt.scatter(accuracies_np[:, 0], accuracies_np[:, 1],label="Classifier accuracy")
    plt.xlabel('')
    plt.ylabel('Accuracy')
    if num_classes is not None and num_classes > 1:
        random_acc = 1.0 / num_classes
        plt.axhline(random_acc, color='red', linestyle='--', label=f'Random Accuracy ({random_acc:.2f})')
    plt.legend()
    plt.show()

def get_profiler(device:str,algorithm:Logistic_Regression_Algorithm):
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
    parser.add_argument('--dataset', type=str,choices = [d.name for d in ClassificationDataset], default='DIABETES', help='Name of dataset to be used')
    parser.add_argument('--algorithm', type=str, choices=[a.name for a in Logistic_Regression_Algorithm], default='CONVEX_PROGRAMMING', help='Which algorithm to use to learn the SVM halfplane')
    parser.add_argument('--kernel', type=str, choices=[a.name for a in Kernel], default='IDENTITY', help='Kernel for data preprocessing')
    parser.add_argument('--n', type=int, default='-1', help='Polynomial degree to project data.')
    parser.add_argument('--profile', action='store_true', help='Whether to profile the code using the pytorch profiler')
    parser.add_argument('--log', type=int, default=1, choices=[0,1,2,3], help='Level of output logs')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda'] , help='cpu or cuda')
    parser.add_argument('--sweep', action='store_true', help='Sweep through values of k')
    args = parser.parse_args()
    dataset_name = ClassificationDataset[args.dataset]
    algorithm = Logistic_Regression_Algorithm[args.algorithm]
    kernel = Kernel[args.kernel]
    to_profile = args.profile
    sweep = args.sweep
    
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=="cuda" else 'cpu')

    # Prepare data
    data,labels = get_data(dataset_name,with_labels=True,binary=True) # N by dim data tensor
    if kernel != Kernel.IDENTITY:
        kernel_fn = kernel_func_map[kernel]
        data = torch.stack([kernel_fn(row,args.n) for row in data])
    data = data.to(device)
    labels = labels.to(device)

    if to_profile:
            prof = get_profiler(device,algorithm)
            start = time.time()
            prof.start()
            halfplane = learn_halfplane(
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
        halfplane = learn_halfplane(
            data,
            labels,
            algorithm=algorithm,
            log_level=args.log,
        )
        end = time.time()

        print(f"{algorithm.name} runtime:{end-start} seconds")

    # Evaluate the learned means 
    results = evaluate_halfplane(halfplane,data,labels)

