import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class KhethaDataPipeline:
    """Handles data loading, leakage removal, and stratified splitting for Khetha Mutual."""
    
    def __init__(self, raw_df, task_type=None, leakage_cols=None):
        self.raw_df = raw_df.copy()
        self.task_type = task_type
        self.leakage_cols = leakage_cols or []
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def detect_task_type(self, target_col):
        """Determines if the problem is regression or classification."""
        target_series = self.raw_df[target_col]
        nunique = target_series.nunique()
        desc = target_series.describe()
        
        # Infer task type based on unique values and data type using nunique() and describe()
        if nunique <= 20 or target_series.dtype == 'object' or target_series.dtype == 'bool':
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
            
        return self.task_type

    def isolate_features(self, target_col):
        """Separates predictors (X) from outcome (y) and removes leakage."""
        # Validate that the target column has no null values
        assert self.raw_df[target_col].isnull().sum() == 0, f"Data Validation Error: Target column '{target_col}' contains null values."

        # Drop leakage columns and target from raw_df to create X
        cols_to_drop = self.leakage_cols + [target_col]
        valid_cols_to_drop = [col for col in cols_to_drop if col in self.raw_df.columns]
        
        self.X = self.raw_df.drop(columns=valid_cols_to_drop)
        self.y = self.raw_df[target_col]

        # Assert index alignment and shape consistency
        assert self.X.shape[0] == len(self.y), "Data Validation Error: Row count mismatch between features (X) and target (y)."
        assert (self.X.index == self.y.index).all(), "Data Validation Error: Index alignment mismatch between X and y."

    def split_with_stratification(self, test_size=0.2, random_state=42):
        """Splits data while preserving class/regression distributions."""
        stratify_arg = None
        
        if self.task_type == 'classification':
            # Class imbalance diagnostics: compute ratio and flag if minority class < 5%
            value_counts = self.y.value_counts(normalize=True)
            min_class_prop = value_counts.min()
            if min_class_prop < 0.05:
                print(f"Class Imbalance Warning: Minority class proportion is {min_class_prop:.2%} (< 5%).")
            
            stratify_arg = self.y

        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        # Verify no index overlap between train and test sets
        train_index_set = set(self.X_train.index)
        test_index_set = set(self.X_test.index)
        assert train_index_set.isdisjoint(test_index_set), "Data Leakage Error: Index overlap detected between train and test sets."

    def check_representativeness(self, key_features):
        """Validates that train/test distributions mirror the original data."""
        for col in key_features:
            if col not in self.X.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(self.X[col]):
                train_mean = self.X_train[col].mean()
                test_mean = self.X_test[col].mean()
                if train_mean != 0 and not np.isnan(train_mean):
                    diff_mean = abs(train_mean - test_mean) / abs(train_mean)
                    if diff_mean > 0.05:
                        print(f"Representativeness Warning: Mean deviation for numeric feature '{col}' is {diff_mean:.2%} (> 5%).")
            else:
                train_prop = self.X_train[col].value_counts(normalize=True).to_dict()
                test_prop = self.X_test[col].value_counts(normalize=True).to_dict()
                for cat, p_train in train_prop.items():
                    p_test = test_prop.get(cat, 0.0)
                    if p_train != 0:
                        diff_prop = abs(p_train - p_test) / p_train
                        if diff_prop > 0.05:
                            print(f"Representativeness Warning: Proportion deviation for category '{cat}' in '{col}' is {diff_prop:.2%} (> 5%).")

    def run(self, target_col, key_features):
        """Orchestrates the pipeline."""
        self.task_type = self.detect_task_type(target_col)
        self.isolate_features(target_col)
        self.split_with_stratification()
        self.check_representativeness(key_features)
        print(f"Pipeline Complete | Task: {self.task_type} | Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

if __name__ == "__main__":
    try:
        # Load claims data for local execution and pipeline testing
        raw_df = pd.read_csv("claims_data.csv").dropna(subset=['fraud_label'])
        leakage_columns = ['claim_id', 'claim_date', 'claim_settlement_date', 'investigation_outcome']
        key_features = ['claim_amount', 'vehicle_age_years', 'province']
        
        pipeline = KhethaDataPipeline(raw_df=raw_df, leakage_cols=leakage_columns)
        X_train, X_test, y_train, y_test = pipeline.run(target_col='fraud_label', key_features=key_features)
    except FileNotFoundError:
        print("Dataset files not found in the current directory.")
