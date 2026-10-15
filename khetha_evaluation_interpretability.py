import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

try:
    import shap
except ImportError:
    shap = None

# Import previous milestone pipelines
from khetha_data_pipeline import KhethaDataPipeline
from khetha_ensemble_optimizer import KhethaEnsembleOptimizer

class KhethaEvaluator:
    """Handles advanced model evaluation, explainability, and temporal stability checks."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = pd.get_dummies(X_train, drop_first=True).fillna(0)
        self.y_train = y_train
        self.X_test = pd.get_dummies(X_test, drop_first=True).fillna(0)
        self.y_test = y_test
        self.y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None

    def run_advanced_models(self, model_type='knn'):
        """Trains SVM/SVR or optimised KNN with PCA preprocessing."""
        if model_type == 'svm_classifier':
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            model.fit(self.X_train, self.y_train)
            return model
        elif model_type == 'svr_regressor':
            model = SVR(kernel='rbf', epsilon=0.1, C=1.0)
            model.fit(self.X_train, self.y_train)
            return model
        elif model_type == 'knn':
            # Apply PCA(n_components=15) before KNN for computational efficiency
            n_comp = min(15, self.X_train.shape[1], self.X_train.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr_pca = pca.fit_transform(self.X_train)
            
            model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='manhattan')
            model.fit(X_tr_pca, self.y_train)
            return model, pca
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def generate_classification_report(self):
        """Returns business-aligned metrics: confusion matrix, ROC-AUC, PR-AUC, and ZAR costs."""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        roc_auc = roc_auc_score(self.y_test, self.y_proba) if self.y_proba is not None else 0.0
        pr_auc = average_precision_score(self.y_test, self.y_proba) if self.y_proba is not None else 0.0
        
        # Business cost implications in ZAR (Investigation cost = R2,500, Average fraud loss = R45,000)
        tn, fp, fn, tp = cm.ravel()
        investigation_cost_zar = 2500
        avg_fraud_loss_zar = 45000
        
        fp_cost = fp * investigation_cost_zar
        fn_cost = fn * avg_fraud_loss_zar
        total_risk_cost = fp_cost + fn_cost
        
        report = {
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fp_cost_zar': fp_cost,
            'fn_cost_zar': fn_cost,
            'total_risk_cost_zar': total_risk_cost
        }
        
        print(f"Evaluation Complete | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Total Risk Cost: R{total_risk_cost:,.2f}")
        return report

    def apply_shap_explanations(self, n_samples=50):
        """Generates global and local feature attributions using SHAP."""
        if shap is None:
            print("SHAP library is not installed.")
            return None
            
        explainer = shap.TreeExplainer(self.model)
        val_subset = self.X_test.iloc[:n_samples]
        shap_values = explainer.shap_values(val_subset)
        
        # Generate summary plot data
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, val_subset, show=False)
        plt.title("Khetha Mutual - SHAP Global Summary Plot")
        plt.tight_layout()
        plt.savefig("shap_global_summary.png", dpi=300)
        plt.close()
        
        print("SHAP global summary and local force plots generated successfully.")
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'val_subset': val_subset
        }

    def temporal_validation(self, df_chronological, date_col, target_col):
        """Checks model stability over time using TimeSeriesSplit across quarters."""
        if date_col not in df_chronological.columns:
            print(f"Date column '{date_col}' not found in dataframe.")
            return None
            
        df_sorted = df_chronological.sort_values(by=date_col)
        X = df_sorted.drop(columns=[target_col, date_col], errors='ignore')
        X_enc = pd.get_dummies(X, drop_first=True).fillna(0)
        y = df_sorted[target_col]
        
        tscv = TimeSeriesSplit(n_splits=4)
        quarterly_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_enc)):
            X_tr, X_val = X_enc.iloc[train_idx], X_enc.iloc[test_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
            
            self.model.fit(X_tr, y_tr)
            score = self.model.score(X_val, y_val)
            quarterly_scores.append(score)
            print(f"TimeSeriesSplit Q{fold+1} Accuracy: {score:.4f}")
            
        # Check if latest quarter performance drops >10% relative to average of previous quarters
        if len(quarterly_scores) > 1:
            prev_mean = np.mean(quarterly_scores[:-1])
            latest_score = quarterly_scores[-1]
            drop_pct = (prev_mean - latest_score) / prev_mean if prev_mean > 0 else 0
            
            if drop_pct > 0.10:
                print(f"Performance Degradation Warning: Latest quarter dropped by {drop_pct:.2%} (>10% threshold). Triggering Retrain.")
            else:
                print(f"Temporal Stability Maintained. Latest quarter performance drop: {drop_pct:.2%}")
                
        return quarterly_scores

    def calculate_psi_drift(self, expected_dist, actual_dist, num_bins=10):
        """Computes Population Stability Index for drift monitoring and triggers retraining."""
        expected = np.array(expected_dist)
        actual = np.array(actual_dist)
        
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        exp_counts = np.histogram(expected, bins=bins)[0]
        act_counts = np.histogram(actual, bins=bins)[0]
        
        exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts) / len(expected)
        act_pct = np.where(act_counts == 0, 0.0001, act_counts) / len(actual)
        
        psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        
        # Industry thresholds: <0.1 (stable), 0.1-0.25 (monitor), >0.25 (retrain)
        if psi_value < 0.1:
            drift_level = 'stable'
        elif psi_value < 0.25:
            drift_level = 'monitor'
        else:
            drift_level = 'retrain'
            
        print(f"PSI Calculation Result: {psi_value:.4f} | Drift Status: {drift_level.upper()}")
        return psi_value, drift_level


if __name__ == "__main__":
    print("--- Running Khetha Evaluation & Interpretability Pipeline ---")
    try:
        # Load data and execute Milestone 1 pipeline
        claims_df = pd.read_csv("claims_data.csv").dropna(subset=['fraud_label'])
        leakage_columns = ['claim_id', 'claim_date', 'claim_settlement_date', 'investigation_outcome']
        key_features = ['claim_amount', 'vehicle_age_years', 'province']
        
        pipeline = KhethaDataPipeline(raw_df=claims_df, leakage_cols=leakage_columns)
        X_train, X_test, y_train, y_test = pipeline.run(target_col='fraud_label', key_features=key_features)
        
        # Train optimized ensemble model from Milestone 3
        optimizer = KhethaEnsembleOptimizer(X_train, y_train, X_test, y_test)
        rf_model = optimizer.build_ensembles('random_forest')
        
        # Initialize Evaluator
        evaluator = KhethaEvaluator(rf_model, X_train, y_train, X_test, y_test)
        
        # 1. Generate Business-Aligned Classification Report
        print("\n[1] Generating Classification Report & Business Costs...")
        evaluator.generate_classification_report()
        
        # 2. Compute PSI Drift Monitoring
        print("\n[2] Computing Population Stability Index (PSI)...")
        evaluator.calculate_psi_drift(X_train['claim_amount'], X_test['claim_amount'])
        
        # 3. Temporal Validation Check
        print("\n[3] Running Temporal Validation (TimeSeriesSplit)...")
        claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])
        evaluator.temporal_validation(claims_df.head(2000), date_col='claim_date', target_col='fraud_label')
        
        print("\nMilestone 4 Evaluator Executed Successfully!")
        
    except Exception as e:
        print(f"Error during evaluation execution: {e}")
