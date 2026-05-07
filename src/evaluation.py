"""
Model Evaluation Module untuk Prediksi Keberhasilan Pengobatan MDR-TB

Modul ini menangani:
- Perhitungan metrik: Accuracy, Precision, Recall, F1-Score, Specificity, AUC-ROC
- Confusion Matrix
- Classification Report
- Tabel perbandingan Training vs Testing
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List, Tuple


class ModelEvaluator:
    """Kelas untuk evaluasi model machine learning"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.comparison_table: Dict[str, Any] = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Menghitung semua metrik evaluasi
        
        Args:
            y_true: Label sebenarnya
            y_pred: Label prediksi
            y_proba: Probabilitas prediksi kelas positif (untuk AUC-ROC)
            
        Returns:
            Dictionary berisi metrik evaluasi
        """
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
            'specificity': specificity,
        }

        # AUC-ROC
        if y_proba is not None:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
        
        self.results = metrics
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Menghitung confusion matrix
        
        Returns:
            Dictionary dengan TP, TN, FP, FN
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm.shape[0] >= 1:
                if len(np.unique(y_true)) == 1:
                    if y_true[0] == 0:
                        tn = cm[0, 0]
                    else:
                        tp = cm[0, 0]
        
        return {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }
    
    def get_confusion_matrix_array(self, y_true: np.ndarray, y_pred: np.ndarray) -> list:
        """
        Mendapatkan confusion matrix sebagai 2D array
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   target_names: list = None) -> str:
        """
        Mendapatkan classification report lengkap
        """
        if target_names is None:
            target_names = ['Tidak Berhasil', 'Berhasil']
        
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def evaluate_all_models(self, models: Dict, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluasi semua model pada test set
        """
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
            
            metrics = self.evaluate(y_test, y_pred, y_proba)
            cm = self.get_confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'confusion_matrix_array': self.get_confusion_matrix_array(y_test, y_pred)
            }
            
            print(f"\n=== {name} ===")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            print(f"Confusion Matrix: TP={cm['true_positive']}, TN={cm['true_negative']}, "
                  f"FP={cm['false_positive']}, FN={cm['false_negative']}")
        
        return results

    def evaluate_train_test(self, models: Dict,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluasi semua model pada data training DAN testing.
        Menghitung: Akurasi, Sensitivitas (Recall), Spesifisitas, AUC-ROC
        untuk kedua set data.
        
        Returns:
            Dictionary: { model_name: { accuracy: {train, test}, sensitivity: {train, test}, ... } }
        """
        comparison = {}

        for name, model in models.items():
            # Training set predictions
            y_train_pred = model.predict(X_train)
            try:
                y_train_proba = model.predict_proba(X_train)[:, 1]
            except Exception:
                y_train_proba = None

            # Test set predictions
            y_test_pred = model.predict(X_test)
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_test_proba = None

            train_metrics = self.evaluate(y_train, y_train_pred, y_train_proba)
            test_metrics = self.evaluate(y_test, y_test_pred, y_test_proba)

            comparison[name] = {
                'accuracy': {
                    'train': round(train_metrics['accuracy'] * 100, 2),
                    'test': round(test_metrics['accuracy'] * 100, 2),
                },
                'sensitivity': {
                    'train': round(train_metrics['recall'] * 100, 2),
                    'test': round(test_metrics['recall'] * 100, 2),
                },
                'specificity': {
                    'train': round(train_metrics['specificity'] * 100, 2),
                    'test': round(test_metrics['specificity'] * 100, 2),
                },
                'auc_roc': {
                    'train': round(train_metrics.get('auc_roc', 0) * 100, 2),
                    'test': round(test_metrics.get('auc_roc', 0) * 100, 2),
                },
            }

            print(f"\n=== {name} (Train/Test Comparison) ===")
            print(f"  Accuracy:    Tr={comparison[name]['accuracy']['train']:.2f}% | Ts={comparison[name]['accuracy']['test']:.2f}%")
            print(f"  Sensitivity: Tr={comparison[name]['sensitivity']['train']:.2f}% | Ts={comparison[name]['sensitivity']['test']:.2f}%")
            print(f"  Specificity: Tr={comparison[name]['specificity']['train']:.2f}% | Ts={comparison[name]['specificity']['test']:.2f}%")
            print(f"  AUC-ROC:     Tr={comparison[name]['auc_roc']['train']:.2f}% | Ts={comparison[name]['auc_roc']['test']:.2f}%")

        self.comparison_table = comparison
        return comparison

    def get_roc_curve_data(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Menghitung data ROC curve untuk satu model.
        
        Returns:
            Dictionary berisi fpr, tpr, thresholds, auc
        """
        try:
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, y_proba)
            auc = float(roc_auc_score(y, y_proba))
            return {
                'fpr': [float(x) for x in fpr],
                'tpr': [float(x) for x in tpr],
                'thresholds': [float(x) for x in thresholds],
                'auc': auc
            }
        except Exception as e:
            print(f"Error computing ROC curve: {e}")
            return {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': 0.0}

    def get_pr_curve_data(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Menghitung data Precision-Recall curve untuk satu model.
        
        Returns:
            Dictionary berisi precision, recall, thresholds, average_precision
        """
        try:
            y_proba = model.predict_proba(X)[:, 1]
            prec, rec, thresholds = precision_recall_curve(y, y_proba)
            ap = float(average_precision_score(y, y_proba))
            return {
                'precision': [float(x) for x in prec],
                'recall': [float(x) for x in rec],
                'thresholds': [float(x) for x in thresholds],
                'average_precision': ap
            }
        except Exception as e:
            print(f"Error computing PR curve: {e}")
            return {'precision': [], 'recall': [], 'thresholds': [], 'average_precision': 0.0}

    def get_calibration_data(self, model, X: pd.DataFrame, y: pd.Series,
                             n_bins: int = 10) -> Dict[str, Any]:
        """
        Menghitung data Calibration curve + Brier Score untuk satu model.
        
        Returns:
            Dictionary berisi prob_true, prob_pred, brier_score
        """
        try:
            y_proba = model.predict_proba(X)[:, 1]
            prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy='uniform')
            brier = float(brier_score_loss(y, y_proba))
            return {
                'prob_true': [float(x) for x in prob_true],
                'prob_pred': [float(x) for x in prob_pred],
                'brier_score': brier
            }
        except Exception as e:
            print(f"Error computing calibration curve: {e}")
            return {'prob_true': [], 'prob_pred': [], 'brier_score': 1.0}

    def bootstrap_ci(self, y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray = None,
                     n_iter: int = 1000, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """
        Menghitung Bootstrap 95% Confidence Interval untuk metrik evaluasi.
        
        Returns:
            Dictionary: { metric_name: {mean, ci_lower, ci_upper} }
        """
        rng = np.random.RandomState(42)
        n = len(y_true)
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
        
        boot_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'specificity': [],
            'auc_roc': []
        }
        
        for _ in range(n_iter):
            idx = rng.choice(n, size=n, replace=True)
            bt_true = y_true[idx]
            bt_pred = y_pred[idx]
            
            # Skip jika hanya satu kelas di sample
            if len(np.unique(bt_true)) < 2:
                continue
            
            boot_metrics['accuracy'].append(float(accuracy_score(bt_true, bt_pred)))
            boot_metrics['precision'].append(float(precision_score(bt_true, bt_pred, zero_division=0)))
            boot_metrics['recall'].append(float(recall_score(bt_true, bt_pred, zero_division=0)))
            boot_metrics['f1_score'].append(float(f1_score(bt_true, bt_pred, zero_division=0)))
            
            cm = confusion_matrix(bt_true, bt_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            else:
                spec = 0.0
            boot_metrics['specificity'].append(spec)
            
            if y_proba is not None:
                bt_proba = y_proba[idx]
                try:
                    boot_metrics['auc_roc'].append(float(roc_auc_score(bt_true, bt_proba)))
                except Exception:
                    pass
        
        results = {}
        for metric_name, values in boot_metrics.items():
            if len(values) > 0:
                arr = np.array(values)
                results[metric_name] = {
                    'mean': float(np.mean(arr)),
                    'ci_lower': float(np.percentile(arr, alpha / 2 * 100)),
                    'ci_upper': float(np.percentile(arr, (1 - alpha / 2) * 100))
                }
            else:
                results[metric_name] = {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        
        return results

    def get_all_curves(self, models: Dict, X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Dict]:
        """
        Menghitung ROC, PR, dan Calibration curves untuk semua model.
        """
        curves = {}
        for name, model in models.items():
            curves[name] = {
                'roc': self.get_roc_curve_data(model, X_test, y_test),
                'pr': self.get_pr_curve_data(model, X_test, y_test),
                'calibration': self.get_calibration_data(model, X_test, y_test)
            }
        return curves

    def get_all_bootstrap_ci(self, models: Dict, X_test: pd.DataFrame,
                              y_test: pd.Series, n_iter: int = 1000) -> Dict[str, Dict]:
        """
        Menghitung Bootstrap 95% CI untuk semua model.
        """
        ci_results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
            ci_results[name] = self.bootstrap_ci(y_test, y_pred, y_proba, n_iter=n_iter)
        return ci_results


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from training import ModelTrainer
    
    # Load dan preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("../data/data_uji_ml.csv")
    df_processed = preprocessor.preprocess(df)
    X, y = preprocessor.get_features_and_target(df_processed)
    
    # Train models
    trainer = ModelTrainer(n_folds=5)
    training_results = trainer.train(X, y, test_size=0.2)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_all_models(
        trainer.models,
        training_results['X_test'],
        training_results['y_test']
    )
    
    print("\n=== Evaluation Complete ===")
