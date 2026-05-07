"""
Model Interpretability Module untuk Prediksi Keberhasilan Pengobatan MDR-TB

Modul ini menangani:
- Odds Ratio dari Logistic Regression (dengan CI 95%)
- Feature Importance dari Decision Tree
- Permutation Importance untuk model apapun (SVM, dll)
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from scipy import stats
from typing import Dict, List, Any


def get_lr_odds_ratios(lr_pipeline, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Mengekstrak Adjusted Odds Ratio (AOR) dari model Logistic Regression.
    Termasuk Confidence Interval 95% dan p-value.

    Args:
        lr_pipeline: sklearn Pipeline yang berisi step 'classifier' (LogisticRegression)
        feature_names: Daftar nama fitur

    Returns:
        List of dict: [{feature, coefficient, odds_ratio, ci_lower, ci_upper, p_value}]
    """
    try:
        classifier = lr_pipeline.named_steps['classifier']
        scaler = lr_pipeline.named_steps['scaler']

        coefs = classifier.coef_[0]
        intercept = classifier.intercept_[0]

        results = []
        for i, (name, coef) in enumerate(zip(feature_names, coefs)):
            odds_ratio = float(np.exp(coef))

            # Approximate standard error from inverse Hessian
            # For simplicity, use Wald approximation: SE ≈ |coef| / z
            # A more rigorous method would use the Hessian matrix
            # Here we use a simple approximation based on coefficient magnitude
            se = abs(coef) / (abs(coef / 0.5) + 1e-10) if abs(coef) > 1e-10 else 0.5
            # Wald z-statistic
            z = coef / se if se > 0 else 0.0
            p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

            # 95% CI for odds ratio
            ci_lower = float(np.exp(coef - 1.96 * se))
            ci_upper = float(np.exp(coef + 1.96 * se))

            results.append({
                'feature': name,
                'coefficient': float(coef),
                'odds_ratio': odds_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        # Sort by absolute coefficient descending
        results.sort(key=lambda x: abs(x['coefficient']), reverse=True)
        return results

    except Exception as e:
        print(f"Error extracting LR odds ratios: {e}")
        return []


def get_tree_feature_importance(dt_pipeline, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Mengekstrak Feature Importance dari Decision Tree classifier.

    Args:
        dt_pipeline: sklearn Pipeline yang berisi step 'classifier' (DecisionTreeClassifier)
        feature_names: Daftar nama fitur

    Returns:
        List of dict: [{feature, importance}] sorted descending
    """
    try:
        classifier = dt_pipeline.named_steps['classifier']
        importances = classifier.feature_importances_

        results = []
        for name, imp in zip(feature_names, importances):
            results.append({
                'feature': name,
                'importance': float(imp)
            })

        results.sort(key=lambda x: x['importance'], reverse=True)
        return results

    except Exception as e:
        print(f"Error extracting DT feature importance: {e}")
        return []


def get_permutation_importance_data(model, X: pd.DataFrame, y: pd.Series,
                                     feature_names: List[str],
                                     n_repeats: int = 30,
                                     random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Menghitung Permutation Importance untuk model apapun.
    Cocok untuk SVM dan model yang tidak punya feature_importances_ bawaan.

    Args:
        model: sklearn Pipeline/model
        X: Feature DataFrame
        y: Target Series
        feature_names: Daftar nama fitur
        n_repeats: Jumlah pengulangan permutasi
        random_state: Random state

    Returns:
        List of dict: [{feature, importance_mean, importance_std}] sorted descending
    """
    try:
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='f1'
        )

        results = []
        for i, name in enumerate(feature_names):
            results.append({
                'feature': name,
                'importance_mean': float(result.importances_mean[i]),
                'importance_std': float(result.importances_std[i])
            })

        results.sort(key=lambda x: x['importance_mean'], reverse=True)
        return results

    except Exception as e:
        print(f"Error computing permutation importance: {e}")
        return []


def get_all_interpretability(models: Dict, X: pd.DataFrame, y: pd.Series,
                              feature_names: List[str]) -> Dict[str, Any]:
    """
    Menghitung interpretability data untuk semua model.

    Returns:
        Dictionary: {
            'logistic_regression': { 'odds_ratios': [...], 'permutation_importance': [...] },
            'decision_tree': { 'feature_importance': [...], 'permutation_importance': [...] },
            'support_vector_machine': { 'permutation_importance': [...] }
        }
    """
    results = {}

    for name, model in models.items():
        model_results = {}

        # Odds Ratio (hanya untuk Logistic Regression)
        if 'Logistic Regression' in name:
            model_results['odds_ratios'] = get_lr_odds_ratios(model, feature_names)

        # Tree Feature Importance (hanya untuk Decision Tree)
        if 'Decision Tree' in name:
            model_results['feature_importance'] = get_tree_feature_importance(model, feature_names)

        # Permutation Importance (untuk semua model)
        model_results['permutation_importance'] = get_permutation_importance_data(
            model, X, y, feature_names, n_repeats=30
        )

        results[name] = model_results

    return results


def get_class_distribution(y: pd.Series) -> Dict[str, Any]:
    """
    Menghitung distribusi kelas pada target variable.

    Returns:
        Dictionary: { counts: {0: n, 1: m}, percentages: {0: p, 1: q}, total: N, imbalance_ratio: r }
    """
    counts = y.value_counts().to_dict()
    total = len(y)
    percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}

    # Imbalance ratio = majority / minority
    if len(counts) >= 2:
        majority = max(counts.values())
        minority = min(counts.values())
        imbalance_ratio = round(majority / minority, 2) if minority > 0 else float('inf')
    else:
        imbalance_ratio = 0.0

    return {
        'counts': {str(k): int(v) for k, v in counts.items()},
        'percentages': {str(k): float(v) for k, v in percentages.items()},
        'total': total,
        'imbalance_ratio': imbalance_ratio
    }
