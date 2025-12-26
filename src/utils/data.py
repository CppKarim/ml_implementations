import torch 
import numpy as np
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from enum import Enum
from typing import Optional

# Function that returns data to be clustered
class Dataset(Enum):
    RANDOM = 0
    IRIS = 1
    WINE = 2
    BCANCER = 3
    TITANIC = 4
    
def get_data(dataset: Dataset = Dataset.IRIS, N:Optional[int]=100, dim:Optional[int]=5,with_labels:bool=False):
    if dataset == Dataset.IRIS:
        iris = load_iris()
        data = torch.tensor(iris.data, dtype=torch.float32)
        labels = torch.tensor(iris.target, dtype=torch.long)
    elif dataset == Dataset.WINE:
        wine = load_wine()
        data = torch.tensor(wine.data, dtype=torch.float32)
        labels = torch.tensor(wine.target, dtype=torch.long)
    elif dataset == Dataset.BCANCER:
        bcancer = load_breast_cancer()
        data = torch.tensor(bcancer.data, dtype=torch.float32)
        labels = torch.tensor(bcancer.target, dtype=torch.long)
    elif dataset == Dataset.TITANIC:
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
    elif dataset == Dataset.RANDOM:
        data = torch.rand((N, dim))
        labels = torch.randint(0, 2, (N,))
    else:
        raise(NotImplementedError("Dataset not available"))
    if with_labels:
        return data, labels
    else:
        return data

