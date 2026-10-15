import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from scipy.stats import randint, uniform
import optuna

# Import data preparation output from Milestone 1
from khetha_data_pipeline import KhethaDataPipeline

class KhethaEnsembleOptimizer:
    """Manages ensemble training, hyperparameter search, and nested validation."""
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = pd.get_dummies(X_train, drop_first=True).fillna(0)
        self.y_train = y_train
        self.X_test = pd.get_dummies(X_test, drop_first=True).fillna(0)
        self.y_test = y_test
        self.tuned_model = None

    def build_ensembles(self, strategy='random_forest'):
        """Initialises RF, GBM, or AdaBoost based on bias-variance diagnosis."""
        if strategy == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                oob_score=True,
                max_features='sqrt',
                random_state=42
            )
        elif strategy == 'gradient_boosting':
            model = GradientBoostingClassifier(
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                random_state=42
            )
        elif strategy == 'adaboost':
            base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
            model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
        
        model.fit(self.X_train, self.y_train)
        return model

    def run_grid_search(self, estimator, param_grid, cv=5):
        """Exhaustively searches hyperparameter space."""
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        print(f"GridSearchCV Best Score: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def run_randomized_search(self, estimator, param_distributions, n_iter=20, cv=5):
        """Samples distributions to reduce compute while finding near-optimal configs."""
        rand_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        rand_search.fit(self.X_train, self.y_train)
        print(f"RandomizedSearchCV Best Score: {rand_search.best_score_:.4f} with params: {rand_search.best_params_}")
        return {
            'best_params': rand_search.best_params_,
            'best_score': rand_search.best_score_,
            'cv_results': rand_search.cv_results_
        }

    def run_bayesian_optimization(self, n_trials=30):
        """Uses Optuna TPE sampler to intelligently guide search."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                oob_score=True,
                random_state=42
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        print(f"Optuna Bayesian Optimization Best Score: {study.best_value:.4f} with params: {best_params}")
        
        self.tuned_model = RandomForestClassifier(**best_params, oob_score=True, random_state=42)
        self.tuned_model.fit(self.X_train, self.y_train)
        return best_params, study.best_value

    def nested_cross_validation(self, outer_cv=5, inner_cv=3):
        """Provides unbiased performance estimate by separating tuning from evaluation."""
        outer_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        outer_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_splitter.split(self.X_train, self.y_train)):
            X_tr_outer = self.X_train.iloc[train_idx]
            y_tr_outer = self.y_train.iloc[train_idx]
            X_val_outer = self.X_train.iloc[val_idx]
            y_val_outer = self.y_train.iloc[val_idx]
            
            # Inner loop hyperparameter tuning
            base_rf = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [4, 8]
            }
            inner_search = GridSearchCV(
                estimator=base_rf,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
            inner_search.fit(X_tr_outer, y_tr_outer)
            best_inner_model = inner_search.best_estimator_
            
            # Evaluate on outer validation fold
            score = best_inner_model.score(X_val_outer, y_val_outer)
            outer_scores.append(score)
            
        outer_scores = np.array(outer_scores)
        mean_score = outer_scores.mean()
        std_err = outer_scores.std(ddof=1) / np.sqrt(len(outer_scores))
        ci_95 = (mean_score - 1.96 * std_err, mean_score + 1.96 * std_err)
        
        print(f"Nested CV Evaluation Complete | Mean Accuracy: {mean_score:.4f} | 95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        return {
            'mean_score': mean_score,
            'ci_95': ci_95,
            'outer_scores': outer_scores
        }


if __name__ == "__main__":
    print("--- Running Khetha Ensemble Optimizer Pipeline ---")
    try:
        # Load claims data and execute data preparation pipeline from Milestone 1
        claims_df = pd.read_csv("claims_data.csv").dropna(subset=['fraud_label'])
        leakage_columns = ['claim_id', 'claim_date', 'claim_settlement_date', 'investigation_outcome']
        key_features = ['claim_amount', 'vehicle_age_years', 'province']
        
        pipeline = KhethaDataPipeline(raw_df=claims_df, leakage_cols=leakage_columns)
        X_train, X_test, y_train, y_test = pipeline.run(target_col='fraud_label', key_features=key_features)
        
        # Initialize Ensemble Optimizer
        optimizer = KhethaEnsembleOptimizer(X_train, y_train, X_test, y_test)
        
        # 1. Build Ensembles
        print("\n[1] Building Random Forest with OOB Scoring...")
        rf_model = optimizer.build_ensembles('random_forest')
        print(f"Random Forest OOB Score: {rf_model.oob_score_:.4f}")
        
        print("\n[2] Building Gradient Boosting Ensemble...")
        gbm_model = optimizer.build_ensembles('gradient_boosting')
        
        # 2. Run Hyperparameter Searches
        print("\n[3] Running RandomizedSearchCV...")
        param_dist = {
            'n_estimators': randint(50, 150),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 10)
        }
        optimizer.run_randomized_search(RandomForestClassifier(random_state=42), param_dist, n_iter=5, cv=3)
        
        # 3. Run Bayesian Optimization via Optuna
        print("\n[4] Running Optuna Bayesian Optimization...")
        optimizer.run_bayesian_optimization(n_trials=10)
        
        # 4. Run Nested Cross-Validation
        print("\n[5] Executing Nested Cross-Validation...")
        optimizer.nested_cross_validation(outer_cv=3, inner_cv=2)
        
        print("\nMilestone 3 Ensemble Optimizer Executed Successfully!")
        
    except Exception as e:
        print(f"Error during ensemble optimization execution: {e}")
