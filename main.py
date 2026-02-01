import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc, 
                            precision_recall_curve, f1_score)
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from ucimlrepo import fetch_ucirepo
import os 
import time

# Create necessary directories 
folders = [
    "data",
    os.path.join("results", "plots"),
    os.path.join("results", "metrics")
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING

class WineQualityDataset:
    """
    Class to handle Wine Quality dataset loading and preprocessing
    """
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load Wine Quality dataset from UCI repository"""
        print("="*60)
        print("STEP 1: LOADING WINE QUALITY DATASET")
        print("="*60)
        
        try:
            # Fetch dataset from UCI repo
            wine_quality = fetch_ucirepo(id=186)
            X = wine_quality.data.features
            y = wine_quality.data.targets
            
            print(f"Dataset loaded successfully!")
            print(f"Number of samples: {X.shape[0]}") 
            print(f"Number of features: {X.shape[1]}")  
            print(f"\nFeature names: {list(X.columns)}")
            
            self.feature_names = list(X.columns)
            X = X.values
            y = y.values.ravel()
            
            print(f"\nQuality score distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for score, count in zip(unique, counts):
                print(f"  Quality {score}: {count} samples")
                
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Loading from alternative source...")
            return self._load_from_url()
    
    def _load_from_url(self):
        """Alternative method to load data from URL"""
        url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        
        red_wine = pd.read_csv(url_red, sep=';')
        white_wine = pd.read_csv(url_white, sep=';')
        
        red_wine['wine_type'] = 0
        white_wine['wine_type'] = 1
        wine_data = pd.concat([red_wine, white_wine], axis=0)
        
        X = wine_data.drop('quality', axis=1).values
        y = wine_data['quality'].values
        self.feature_names = list(wine_data.columns[:-1])
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.3, random_state=42):
        """Preprocess data: split and normalize"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        # Convert to binary classification
        y_binary = (y > 5).astype(int)
        
        print(f"\nConverting to binary classification:")
        print(f"  Low quality (≤5): {np.sum(y_binary == 0)} samples")
        print(f"  High quality (>5): {np.sum(y_binary == 1)} samples")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        print(f"\nData split:")
        print(f"  Training samples: {self.X_train.shape[0]}")  # FIXED
        print(f"  Testing samples: {self.X_test.shape[0]}")   # FIXED
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nFeature normalization completed!")
        return self.X_train, self.X_test, self.y_train, self.y_test

# =============================================================================
# SECTION 2: FUZZY INFERENCE SYSTEM
# =============================================================================

class FuzzyInferenceSystem:
    """Fuzzy Inference System for feature enhancement"""
    def __init__(self, n_rules=15):
        self.n_rules = n_rules
        self.centers = None
        self.widths = None
        self.kmeans = None
        
    def fit(self, X):
        """Generate fuzzy rules using K-means clustering"""
        print("\n" + "="*60)
        print("STEP 4: FUZZY INFERENCE SYSTEM TRAINING")
        print("="*60)
        print(f"Generating {self.n_rules} fuzzy rules using K-means clustering...")
        
        self.kmeans = KMeans(n_clusters=self.n_rules, random_state=42, n_init=10)
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        
        distances = cdist(self.centers, self.centers, metric='euclidean')
        distances[distances == 0] = np.inf
        self.widths = np.min(distances, axis=1) / 2.0
        
        print(f"Fuzzy rule centers shape: {self.centers.shape}")
        print(f"Fuzzy rules generated successfully!")
        
    def fuzzify(self, X):
        """Fuzzification: Convert input to fuzzy membership values"""
        n_samples = X.shape[0]  # FIXED: Correctly get number of samples
        fuzzy_features = np.zeros((n_samples, self.n_rules))
        
        for i in range(self.n_rules):
            distances = np.linalg.norm(X - self.centers[i], axis=1)
            fuzzy_features[:, i] = np.exp(-(distances ** 2) / (2 * self.widths[i] ** 2))
        
        return fuzzy_features
    
    def defuzzify(self, fuzzy_features, X):
        """Defuzzification: Convert fuzzy features to numerical features"""
        defuzz_features = np.dot(fuzzy_features, self.centers)
        membership_sum = np.sum(fuzzy_features, axis=1, keepdims=True) + 1e-10
        defuzz_features = defuzz_features / membership_sum
        return defuzz_features

# =============================================================================
# SECTION 3: RVFL BASE MODEL
# =============================================================================

class RVFL:
    """Single Random Vector Functional Link Neural Network"""
    def __init__(self, n_hidden=100, activation='sigmoid', C=1.0):
        self.n_hidden = n_hidden
        self.activation = activation
        self.C = C
        self.input_weights = None
        self.bias = None
        self.beta = None
        
    def _sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
    
    def _activate(self, X):
        if self.activation == 'sigmoid':
            return self._sigmoid(X)
        return X
    
    def fit(self, X, y):
        """Train RVFL network"""
        n_samples, n_features = X.shape
        
        np.random.seed(42)
        self.input_weights = np.random.randn(n_features, self.n_hidden) * 0.5
        self.bias = np.random.randn(self.n_hidden) * 0.5
        
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H)
        H_enhanced = np.concatenate([X, H], axis=1)
        
        HTH = np.dot(H_enhanced.T, H_enhanced)
        I = np.eye(H_enhanced.shape[1])
        HTy = np.dot(H_enhanced.T, y.reshape(-1, 1))
        
        self.beta = np.linalg.solve(HTH + self.C * I, HTy)
        
    def predict(self, X):
        """Predict using trained RVFL"""
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H)
        H_enhanced = np.concatenate([X, H], axis=1)
        output = np.dot(H_enhanced, self.beta)
        predictions = (output > 0.5).astype(int).ravel()
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities"""
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H)
        H_enhanced = np.concatenate([X, H], axis=1)
        output = np.dot(H_enhanced, self.beta)
        return output.ravel()

# =============================================================================
# SECTION 4: edRVFL-FIS MODEL
# =============================================================================

class edRVFL_FIS:
    """Ensemble Deep RVFL with Fuzzy Inference System"""
    def __init__(self, n_layers=5, n_hidden=100, n_fuzzy_rules=15, 
                 activation='sigmoid', C=1.0):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_fuzzy_rules = n_fuzzy_rules
        self.activation = activation
        self.C = C
        self.fuzzy_system = FuzzyInferenceSystem(n_rules=n_fuzzy_rules)
        self.base_models = []
        
    def fit(self, X_train, y_train):
        """Train the edRVFL-FIS model"""
        print("\n" + "="*60)
        print("STEP 3: TRAINING edRVFL-FIS MODEL")
        print("="*60)
        print(f"Model configuration:")
        print(f"  Number of layers: {self.n_layers}")
        print(f"  Hidden nodes per layer: {self.n_hidden}")
        print(f"  Fuzzy rules: {self.n_fuzzy_rules}")
        
        self.fuzzy_system.fit(X_train)
        
        print("\nGenerating fuzzy features...")
        fuzzy_features = self.fuzzy_system.fuzzify(X_train)
        defuzz_features = self.fuzzy_system.defuzzify(fuzzy_features, X_train)
        
        print(f"Fuzzy features shape: {fuzzy_features.shape}")
        print(f"Defuzzified features shape: {defuzz_features.shape}")
        
        print(f"\nTraining {self.n_layers} base RVFL models...")
        
        current_features = X_train.copy()
        
        for layer in range(self.n_layers):
            print(f"  Training layer {layer + 1}/{self.n_layers}...", end='')
            
            if layer == 0:
                enhanced_features = np.concatenate([current_features, defuzz_features], axis=1)
            else:
                enhanced_features = np.concatenate([X_train, defuzz_features, current_features], axis=1)
            
            base_model = RVFL(n_hidden=self.n_hidden, activation=self.activation, C=self.C)
            base_model.fit(enhanced_features, y_train)
            self.base_models.append(base_model)
            
            H = np.dot(enhanced_features, base_model.input_weights) + base_model.bias
            H = base_model._activate(H)
            current_features = H
            
            print(" Done!")
        
        print(f"\nModel training completed!")
        
    def predict(self, X_test):
        """Predict using majority voting"""
        fuzzy_features = self.fuzzy_system.fuzzify(X_test)
        defuzz_features = self.fuzzy_system.defuzzify(fuzzy_features, X_test)
        
        all_predictions = []
        current_features = X_test.copy()
        
        for layer, base_model in enumerate(self.base_models):
            if layer == 0:
                enhanced_features = np.concatenate([current_features, defuzz_features], axis=1)
            else:
                enhanced_features = np.concatenate([X_test, defuzz_features, current_features], axis=1)
            
            predictions = base_model.predict(enhanced_features)
            all_predictions.append(predictions)
            
            H = np.dot(enhanced_features, base_model.input_weights) + base_model.bias
            H = base_model._activate(H)
            current_features = H
        
        all_predictions = np.array(all_predictions)
        final_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X_test):
        """Predict probabilities"""
        fuzzy_features = self.fuzzy_system.fuzzify(X_test)
        defuzz_features = self.fuzzy_system.defuzzify(fuzzy_features, X_test)
        
        all_probas = []
        current_features = X_test.copy()
        
        for layer, base_model in enumerate(self.base_models):
            if layer == 0:
                enhanced_features = np.concatenate([current_features, defuzz_features], axis=1)
            else:
                enhanced_features = np.concatenate([X_test, defuzz_features, current_features], axis=1)
            
            probas = base_model.predict_proba(enhanced_features)
            all_probas.append(probas)
            
            H = np.dot(enhanced_features, base_model.input_weights) + base_model.bias
            H = base_model._activate(H)
            current_features = H
        
        avg_probas = np.mean(all_probas, axis=0)
        return avg_probas

# =============================================================================
# SECTION 5: EVALUATION AND VISUALIZATION
# =============================================================================

class ModelEvaluator:
    """Class for model evaluation and visualization"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="edRVFL-FIS"):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("STEP 5: MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1-Score: {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Low Quality', 'High Quality']))
        
        return y_pred, y_proba, accuracy, f1
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, save_path='results/plots/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Quality', 'High Quality'],
                   yticklabels=['Low Quality', 'High Quality'])
        plt.title('Confusion Matrix - Wine Quality Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_test, y_proba, save_path='results/plots/roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - edRVFL-FIS Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
        plt.close()
        
        return roc_auc
    
    @staticmethod
    def plot_precision_recall(y_test, y_proba, save_path='results/plots/precision_recall.png'):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - edRVFL-FIS Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
        plt.close()
    
    @staticmethod
    def save_results_csv(y_test, y_pred, y_proba, accuracy, f1, roc_auc, 
                        save_path='results/metrics/evaluation_results.csv'):
        """Save evaluation results to CSV"""
        results_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred,
            'Prediction_Probability': y_proba
        })
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nPrediction results saved to: {save_path}")
        
        # Save summary metrics
        summary_path = 'results/metrics/performance_summary.csv'
        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
            'Value': [accuracy, f1, roc_auc]
        })
        summary_df.to_csv(summary_path, index=False)
        print(f"Performance summary saved to: {summary_path}")

# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print(" WINE QUALITY CLASSIFICATION USING edRVFL-FIS")
    print(" Based on IEEE Paper Implementation")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Load and preprocess data
    dataset = WineQualityDataset()
    X, y = dataset.load_data()
    X_train, X_test, y_train, y_test = dataset.preprocess_data(X, y)
    
    # Step 2: Initialize and train model
    model = edRVFL_FIS(
        n_layers=5,
        n_hidden=100,
        n_fuzzy_rules=15,
        activation='sigmoid',
        C=1.0
    )
    
    model.fit(X_train, y_train)
    
    # Step 3: Evaluate model
    evaluator = ModelEvaluator()
    y_pred, y_proba, accuracy, f1 = evaluator.evaluate_model(model, X_test, y_test)
    
    # Step 4: Generate visualizations
    print("\n" + "="*60)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*60)
    
    evaluator.plot_confusion_matrix(y_test, y_pred)
    roc_auc = evaluator.plot_roc_curve(y_test, y_proba)
    evaluator.plot_precision_recall(y_test, y_proba)
    
    # Step 5: Save results
    evaluator.save_results_csv(y_test, y_pred, y_proba, accuracy, f1, roc_auc)
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print(" EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"\nAll results saved in 'results/' folder")
    print("  - Plots: results/plots/")
    print("  - Metrics: results/metrics/")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
