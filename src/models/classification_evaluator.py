import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
class ClassificationEvaluator:
    
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        
        self.y_prob = model.predict_proba(X_val)[:, 1]
        self.y_pred = model.predict(X_val)
    
    def basic_metrics(self):
        print("Accuracy :", accuracy_score(self.y_val, self.y_pred))
        print("Precision:", precision_score(self.y_val, self.y_pred))
        print("Recall   :", recall_score(self.y_val, self.y_pred))
        print("F1 Score :", f1_score(self.y_val, self.y_pred))
        print("ROC-AUC  :", roc_auc_score(self.y_val, self.y_prob))
    
    def classification_report(self):
        print("\nClassification Report:\n")
        print(classification_report(self.y_val, self.y_pred))
    
    def confusion_matrix_plot(self):
        cm = confusion_matrix(self.y_val, self.y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    def roc_curve_plot(self):
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(self.y_val, self.y_prob)
        auc = roc_auc_score(self.y_val, self.y_prob)
        
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
        
        return auc
    
    def threshold_analysis(self):
        print("\nThreshold Analysis:")
        for t in [0.4, 0.5, 0.6]:
            y_pred_t = (self.y_prob >= t).astype(int)
            cm = confusion_matrix(self.y_val, y_pred_t)
            print(f"\nThreshold = {t}")
            print(cm)