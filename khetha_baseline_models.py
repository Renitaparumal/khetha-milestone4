import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import the Milestone 1 data pipeline
from khetha_data_pipeline import KhethaDataPipeline

class KhethaBaselineEngine:
    """Builds and validates baseline models using decision trees and logistic regression."""
    
    def __init__(self, X_train, y_train, X_test, y_test, task_type):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.task_type = task_type
        self.best_model = None

    def build_tree_baseline(self, max_depth=None, ccp_alpha=0.0):
        """Trains a single decision tree with complexity controls."""
        if self.task_type == 'classification':
            model = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=42)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=42)
        
        # One-hot encode categorical features for tree fitting
        X_tr_enc = pd.get_dummies(self.X_train, drop_first=True)
        model.fit(X_tr_enc, self.y_train)
        self.best_model = model
        return model

    def build_logistic_baseline(self, multi_class=False):
        """Trains logistic regression for classification tasks."""
        if self.task_type != 'classification':
            raise ValueError("Logistic regression baseline is intended for classification tasks.")
        
        unique_classes = np.unique(self.y_train)
        multi = 'multinomial' if (len(unique_classes) > 2 or multi_class) else 'auto'
        
        # Build pipeline with StandardScaler and LogisticRegression for robust convergence
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver='lbfgs',
                max_iter=2000,
                class_weight='balanced',
                multi_class=multi,
                random_state=42
            )
        )
        
        X_tr_enc = pd.get_dummies(self.X_train, drop_first=True).fillna(0)
        model.fit(X_tr_enc, self.y_train)
        return model

    def run_cross_validation(self, model, cv_folds=5):
        """Executes K-Fold or StratifiedKFold and returns aggregated metrics."""
        X_enc = pd.get_dummies(self.X_train, drop_first=True).fillna(0)
        
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'r2'
            
        scores = cross_val_score(model, X_enc, self.y_train, cv=cv, scoring=scoring)
        
        # Calculate scores for diagnostic logging
        train_score = model.score(X_enc, self.y_train)
        mean_val_score = scores.mean()
        std_val_score = scores.std()
        
        # Diagnostic logging to flag overfitting or underfitting
        if train_score - mean_val_score > 0.15:
            print(f"Diagnostic Warning [Overfitting]: Train score ({train_score:.3f}) exceeds validation mean ({mean_val_score:.3f}) by >15%.")
        elif train_score < 0.7 and mean_val_score < 0.7:
            print(f"Diagnostic Warning [Underfitting]: Both train score ({train_score:.3f}) and validation mean ({mean_val_score:.3f}) are below 0.7.")
        else:
            print(f"Diagnostic Status [Balanced]: Train score = {train_score:.3f}, Validation Mean = {mean_val_score:.3f} (+/- {std_val_score:.3f})")
            
        return {'mean': mean_val_score, 'std': std_val_score, 'scores': scores}

    def control_tree_complexity(self, max_candidates=10):
        """Finds optimal ccp_alpha to balance bias-variance."""
        X_enc = pd.get_dummies(self.X_train, drop_first=True).fillna(0)
        
        if self.task_type == 'classification':
            base_tree = DecisionTreeClassifier(random_state=42)
            scoring = 'accuracy'
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            base_tree = DecisionTreeRegressor(random_state=42)
            scoring = 'r2'
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            
        path = base_tree.cost_complexity_pruning_path(X_enc, self.y_train)
        ccp_alphas = path.ccp_alphas
        
        if len(ccp_alphas) > max_candidates:
            ccp_alphas = np.linspace(ccp_alphas[0], ccp_alphas[-1], max_candidates)
            
        best_alpha = 0.0
        best_score = -np.inf
        best_model = None
        
        for alpha in ccp_alphas:
            if self.task_type == 'classification':
                model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
            else:
                model = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
                
            scores = cross_val_score(model, X_enc, self.y_train, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                model.fit(X_enc, self.y_train)
                best_model = model
                
        self.best_model = best_model
        print(f"Optimal ccp_alpha selected: {best_alpha:.5f} with validation score: {best_score:.3f}")
        return best_alpha, best_model

    def visualize_top_splits(self, model, max_depth=3):
        """Plots first 3 levels of the tree for interpretability review."""
        plt.figure(figsize=(14, 8))
        X_enc = pd.get_dummies(self.X_train, drop_first=True).fillna(0)
        
        # If model is part of a pipeline or raw estimator
        tree_estimator = model.named_steps['decisiontreeclassifier'] if hasattr(model, 'named_steps') else model
        
        plot_tree(
            tree_estimator, 
            max_depth=max_depth, 
            feature_names=list(X_enc.columns), 
            filled=True, 
            rounded=True,
            proportion=True
        )
        plt.title(f"Khetha Mutual - Decision Tree Top {max_depth} Splits Interpretability")
        plt.tight_layout()
        plt.savefig("tree_top_splits.png", dpi=300)
        plt.close()
        print("Tree visualization saved successfully as 'tree_top_splits.png'.")
        return "tree_top_splits.png"


if __name__ == "__main__":
    print("--- Running Khetha Baseline Engine Pipeline ---")
    try:
        # Load and execute data preparation using Milestone 1 pipeline
        claims_df = pd.read_csv("claims_data.csv").dropna(subset=['fraud_label'])
        leakage_columns = ['claim_id', 'claim_date', 'claim_settlement_date', 'investigation_outcome']
        key_features = ['claim_amount', 'vehicle_age_years', 'province']
        
        pipeline = KhethaDataPipeline(raw_df=claims_df, leakage_cols=leakage_columns)
        X_train, X_test, y_train, y_test = pipeline.run(target_col='fraud_label', key_features=key_features)
        
        # Initialize Baseline Engine
        engine = KhethaBaselineEngine(X_train, y_train, X_test, y_test, task_type=pipeline.task_type)
        
        # 1. Build and Evaluate Tree Baseline
        print("\n[1] Training Decision Tree Baseline...")
        tree_model = engine.build_tree_baseline(max_depth=5)
        tree_cv = engine.run_cross_validation(tree_model)
        
        # 2. Build and Evaluate Logistic Regression Baseline
        print("\n[2] Training Logistic Regression Baseline...")
        log_model = engine.build_logistic_baseline()
        log_cv = engine.run_cross_validation(log_model)
        
        # 3. Control Tree Complexity (Pruning)
        print("\n[3] Optimizing Tree Complexity (Cost-Complexity Pruning)...")
        best_alpha, pruned_tree = engine.control_tree_complexity()
        
        # 4. Visualise Top Splits
        print("\n[4] Generating Tree Visualisation...")
        engine.visualize_top_splits(pruned_tree, max_depth=3)
        
        print("\nMilestone 2 Baseline Engine Executed Successfully!")
        
    except Exception as e:
        print(f"Error during baseline execution: {e}")
