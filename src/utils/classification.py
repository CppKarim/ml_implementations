import numpy as np

class BinaryClassificationResults:
    def __init__(self,tp:int,tn:int,fp:int,fn:int):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.total = tp+tn+fp+fn
    
    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2*(precision*recall)/(precision+recall)

    def precision(self):
        return self.tp/(self.tp+self.fp)
    
    def recall(self):
        return self.tp/(self.tp+self.fn)
    
    def accuracy(self):
        return (self.tp+self.tn)/self.total
    
class ClassificationResults:
    def __init__(self, labels, predictions):
        self.labels = np.array(labels.cpu())
        self.predictions = np.array(predictions.cpu())
        self.classes = np.unique(np.concatenate([self.labels, self.predictions]))
        self.confusion = self._compute_confusion_matrix()
        self.total = self.labels.size

    def _compute_confusion_matrix(self):
        n = len(self.classes)
        matrix = np.zeros((n, n), dtype=int)
        for t, p in zip(self.labels, self.predictions):
            i = np.where(self.classes == t)[0][0]
            j = np.where(self.classes == p)[0][0]
            matrix[i, j] += 1
        return matrix

    def accuracy(self):
        return np.trace(self.confusion) / self.total

    def precision(self, class_idx):
        # class_idx: index in self.classes
        tp = self.confusion[class_idx, class_idx]
        fp = self.confusion[:, class_idx].sum() - tp
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, class_idx):
        tp = self.confusion[class_idx, class_idx]
        fn = self.confusion[class_idx, :].sum() - tp
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1(self, class_idx):
        p = self.precision(class_idx)
        r = self.recall(class_idx)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def confusion_matrix(self):
        return self.confusion

    def print_confusion_matrix(self):
        print("Confusion Matrix (Columns:labels, rows:predictions):")
        print(" " * 12 + " ".join(f"{c:>6}" for c in self.classes))
        for i, c in enumerate(self.classes):
            row = " ".join(f"{self.confusion[i, j]:>6}" for j in range(len(self.classes)))
            print(f"{str(c):>10} {row}")