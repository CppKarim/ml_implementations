import torch 
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression

from enum import Enum
from typing import Optional

# Function that returns data to be clustered
class ClassificationDataset(Enum):
    RANDOM = 0
    IRIS = 1
    WINE = 2
    BCANCER = 3
    TITANIC = 4
    
def get_data(dataset: ClassificationDataset = ClassificationDataset.IRIS, N:Optional[int]=100, dim:Optional[int]=5,with_labels:bool=False,binary:bool=False):
    if dataset == ClassificationDataset.IRIS:
        iris = load_iris()
        data = torch.tensor(iris.data, dtype=torch.float32)
        labels = torch.tensor(iris.target, dtype=torch.long)
    elif dataset == ClassificationDataset.WINE:
        wine = load_wine()
        data = torch.tensor(wine.data, dtype=torch.float32)
        labels = torch.tensor(wine.target, dtype=torch.long)
    elif dataset == ClassificationDataset.BCANCER:
        bcancer = load_breast_cancer()
        data = torch.tensor(bcancer.data, dtype=torch.float32)
        labels = torch.tensor(bcancer.target, dtype=torch.long)
    elif dataset == ClassificationDataset.TITANIC:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        # Load Titanic dataset from seaborn or a CSV file
        try:
            import seaborn as sns
            df = sns.load_dataset('titanic')
        except:
            df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        # Select numeric and encode categorical columns
        df = df.drop(['embarked', 'deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, errors='ignore')
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df = df.dropna()
        labels = torch.tensor(df['survived'].astype(np.int64).values, dtype=torch.long)
        data = torch.tensor(df.drop('survived', axis=1).astype(np.float32).values, dtype=torch.float32)
    elif dataset == ClassificationDataset.RANDOM:
        data = torch.rand((N, dim))
        labels = torch.randint(0, 2, (N,))
    else:
        raise(NotImplementedError("ClassificationDataset not available"))
    if with_labels:
        if binary:
            return make_binary(data,labels)
        else:
            return data, labels
    else:
        return data
    
def make_binary(data:torch.tensor,labels:torch.tensor):
    unique_elements = torch.unique(labels)
    x = unique_elements[torch.randint(0,unique_elements.size(0),(1,)).item()]
    labels_binary = (labels==x).int()
    return data, labels_binary
    #indices = (labels==x) | (labels==y)
    #return data[indices],labels_binary[indices]

class RegressionDataset(Enum):
    DIABETES = 0
    CALIFORNIA = 1
    SYNTHETIC = 2
    BOSTON = 3

def get_data_regression(dataset: RegressionDataset = RegressionDataset.DIABETES, N: Optional[int] = 100, dim: Optional[int] = 1, noise: float = 10.0):
    if dataset == RegressionDataset.DIABETES:
        ds = load_diabetes()
        data = torch.tensor(ds.data, dtype=torch.float32)
        targets = torch.tensor(ds.target, dtype=torch.float32)
    elif dataset == RegressionDataset.CALIFORNIA:
        ds = fetch_california_housing()
        data = torch.tensor(ds.data, dtype=torch.float32)
        targets = torch.tensor(ds.target, dtype=torch.float32)
    elif dataset == RegressionDataset.BOSTON:
        # Fallback to OpenML if load_boston is unavailable
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name='boston', version=1, as_frame=False)
        data = torch.tensor(ds.data, dtype=torch.float32)
        targets = torch.tensor(ds.target.astype(np.float32), dtype=torch.float32)
    elif dataset == RegressionDataset.SYNTHETIC:
        X, y = make_regression(n_samples=N, n_features=dim, noise=noise)
        data = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
    else:
        raise NotImplementedError("Regression dataset not available")
    return data, targets

