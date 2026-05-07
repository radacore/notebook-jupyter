"""
Model Training Module untuk Prediksi Keberhasilan Pengobatan MDR-TB

Modul ini menangani:
- Training model: Logistic Regression, Decision Tree, SVM
- 3-Way Data Split (Training/Validation/Test)
- Hyperparameter Tuning (GridSearchCV)
- K-Fold Cross Validation
- Model evaluation dan selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any
import joblib
import os


class ModelTrainer:
    """Kelas untuk training dan evaluasi model machine learning"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 4):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models: Dict[str, Pipeline] = {}
        self.best_model_name: str = None
        self.best_model: Pipeline = None
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.best_params: Dict[str, Dict] = {}
        self.scaler = StandardScaler()
        
        # Inisialisasi model-model
        self._init_models()
    
    def _init_models(self):
        """Inisialisasi semua model yang akan dilatih"""
        self.models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]),
            'Decision Tree': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(
                    random_state=self.random_state,
                ))
            ]),
            'Support Vector Machine': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    random_state=self.random_state,
                    probability=True,
                    class_weight='balanced'
                ))
            ])
        }
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Mendapatkan parameter grid untuk hyperparameter tuning"""
        return {
            'Logistic Regression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear'],
            },
            'Decision Tree': {
                'classifier__max_depth': [3, 5, 7, 10, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy'],
            },
            'Support Vector Machine': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto'],
            }
        }
    
    def split_data_3way(self, X: pd.DataFrame, y: pd.Series,
                        val_size: float = 0.15, test_size: float = 0.15) -> Tuple:
        """
        Membagi data menjadi training, validation, dan testing set (3-way split)
        Rasio default: 70% train / 15% val / 15% test
        """
        # Pertama: pisahkan test set
        # test_size relatif terhadap total = 0.15
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Kedua: pisahkan val set dari sisa (val_size relatif terhadap sisa)
        # val_size = 0.15 dari total, sisa = 0.85
        # jadi val relatif terhadap sisa = 0.15 / 0.85 ≈ 0.176
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Membagi data menjadi training dan testing set (backward compatibility)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict]:
        """
        Melakukan Hyperparameter Tuning menggunakan GridSearchCV
        untuk semua model pada data training.
        
        Returns:
            Dictionary berisi parameter terbaik untuk setiap model
        """
        param_grids = self._get_param_grids()
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        best_params = {}
        
        for name, model in self.models.items():
            print(f"\nTuning {name}...")
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=cv,
                scoring='f1',
                n_jobs=1,
                verbose=0,
                refit=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Simpan model terbaik (yang sudah di-refit)
            self.models[name] = grid_search.best_estimator_
            
            # Simpan parameter terbaik (bersihkan prefix 'classifier__')
            clean_params = {}
            for k, v in grid_search.best_params_.items():
                clean_key = k.replace('classifier__', '')
                clean_params[clean_key] = v
            
            best_params[name] = {
                'params': clean_params,
                'best_cv_score': float(grid_search.best_score_),
            }
            
            print(f"  Best params: {clean_params}")
            print(f"  Best CV F1: {grid_search.best_score_:.4f}")
        
        self.best_params = best_params
        return best_params
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Melakukan K-Fold Cross Validation untuk semua model
        (Setelah hyperparameter tuning, sehingga menggunakan model yang sudah optimal)
        
        Returns:
            Dictionary berisi hasil CV untuk setiap model
        """
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for name, model in self.models.items():
            print(f"\nCross-validating {name}...")
            model_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                model_results[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'scores': [float(s) for s in scores]
                }
            
            results[name] = model_results
            print(f"  Accuracy: {model_results['accuracy']['mean']:.4f} (+/- {model_results['accuracy']['std']:.4f})")
            print(f"  F1 Score: {model_results['f1']['mean']:.4f} (+/- {model_results['f1']['std']:.4f})")
        
        self.cv_results = results
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Melatih semua model pada data training (final fit setelah tuning)
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        
        # Pilih model terbaik berdasarkan F1 score dari CV
        if self.cv_results:
            best_f1 = -1
            for name, results in self.cv_results.items():
                f1_mean = results['f1']['mean']
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    self.best_model_name = name
                    self.best_model = self.models[name]
            
            print(f"\nBest model: {self.best_model_name} (F1: {best_f1:.4f})")
    
    def train(self, X: pd.DataFrame, y: pd.Series, use_smote: bool = False) -> Dict:
        """
        Pipeline training lengkap:
        1. Split data 3-way (70/15/15)
        2. (Opsional) SMOTE oversampling pada data training
        3. Hyperparameter Tuning (GridSearchCV pada data training)
        4. Cross Validation (pada data training dengan model optimal)
        5. Train final models pada data training
        6. Select best model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            use_smote: Jika True, terapkan SMOTE pada data training
        
        Returns:
            Dictionary berisi hasil training
        """
        # 3-way split: 70% train, 15% val, 15% test
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_3way(X, y)
        
        # Class distribution reporting
        class_dist_original = y.value_counts().to_dict()
        class_dist_train = y_train.value_counts().to_dict()
        print(f"\n=== Class Distribution ===")
        print(f"Original dataset: {class_dist_original} (total: {len(y)})")
        print(f"Training set: {class_dist_train} (total: {len(y_train)})")
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Opsional: SMOTE oversampling pada training data
        smote_applied = False
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                smote_applied = True
                class_dist_after_smote = pd.Series(y_train).value_counts().to_dict()
                print(f"\n=== SMOTE Applied ===")
                print(f"Training set after SMOTE: {class_dist_after_smote} (total: {len(y_train)})")
            except ImportError:
                print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
            except Exception as e:
                print(f"Warning: SMOTE failed: {e}. Continuing without SMOTE.")
        
        # Hyperparameter tuning (GridSearchCV on training data)
        print("\n=== Hyperparameter Tuning ===")
        best_params = self.hyperparameter_tuning(X_train, y_train)
        
        # Cross validation (with tuned models on training data)
        print("\n=== Cross Validation ===")
        cv_results = self.cross_validate(X_train, y_train)
        
        # Final training (refit on full training data with best params)
        print("\n=== Final Training ===")
        self.train_all_models(X_train, y_train)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'cv_results': cv_results,
            'best_model_name': self.best_model_name,
            'best_params': best_params,
            'class_distribution': {
                'original': {str(k): int(v) for k, v in class_dist_original.items()},
                'train': {str(k): int(v) for k, v in class_dist_train.items()},
            },
            'smote_applied': smote_applied,
        }
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Melakukan prediksi menggunakan model tertentu atau model terbaik
        """
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.best_model
        
        return model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Mendapatkan probabilitas prediksi
        """
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.best_model
        
        return model.predict_proba(X)
    
    def save_models(self, directory: str):
        """
        Menyimpan semua model ke direktori
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filename = name.replace(' ', '_').lower() + '.pkl'
            filepath = os.path.join(directory, filename)
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Simpan info model terbaik
        best_info = {
            'best_model_name': self.best_model_name,
            'cv_results': self.cv_results,
            'best_params': self.best_params,
        }
        joblib.dump(best_info, os.path.join(directory, 'best_model_info.pkl'))
    
    def load_models(self, directory: str):
        """
        Memuat semua model dari direktori
        """
        for name in list(self.models.keys()):
            filename = name.replace(' ', '_').lower() + '.pkl'
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} from {filepath}")
        
        # Load info model terbaik
        best_info_path = os.path.join(directory, 'best_model_info.pkl')
        if os.path.exists(best_info_path):
            best_info = joblib.load(best_info_path)
            self.best_model_name = best_info.get('best_model_name')
            self.cv_results = best_info.get('cv_results', {})
            self.best_params = best_info.get('best_params', {})
            if self.best_model_name:
                self.best_model = self.models.get(self.best_model_name)


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("../data/data_uji_ml.csv")
    df_processed = preprocessor.preprocess(df)
    X, y = preprocessor.get_features_and_target(df_processed)
    
    trainer = ModelTrainer(n_folds=5)
    results = trainer.train(X, y)
    
    trainer.save_models("../models")
    preprocessor.save("../models/preprocessor.pkl")
    
    print("\n=== Training Complete ===")
    print(f"Best Model: {results['best_model_name']}")
    print(f"Best Params: {results['best_params']}")
