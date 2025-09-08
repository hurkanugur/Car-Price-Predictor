import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
import config

class CarPriceDataset:
    def __init__(self):
        """Initialize dataset handler and internal state."""
        self.feature_transformer = None
        self.y_scaler = StandardScaler()

        self.max_train_year = None
        self.numeric_min = None
        self.numeric_max = None

        self.numeric_cols = ["Car_Age", "Engine_Size", "Cylinders","Mileage", "Horsepower", "Doors", "Weight"]
        self.categorical_cols = ["Brand", "Model", "Transmission", "Fuel_Type", "Color"]

    # ----------------- Public Methods -----------------

    def get_flattened_input_size(self, data_loader):
        """Return number of input features per sample after flattening (for MLPs)."""
        sample_X, _ = next(iter(data_loader))
        input_dim = sample_X[0].numel()
        print(f"• Input dimension: {input_dim}")
        return input_dim

    def prepare_data_for_training(self):
        """Prepare training, validation, and test DataLoaders from CSV."""
        df = pd.read_csv(config.DATASET_CSV_PATH)
        df = self._compute_car_age(df)
        df = self._clip_numeric_features(df)
        X = self._fit_feature_transformer(df)
        y = torch.tensor(df["Price"].values.astype(np.float32), dtype=torch.float32).reshape(-1, 1)

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        y_train_norm, y_val_norm, y_test_norm = self._normalize_targets(y_train, y_val, y_test)
        train_loader, val_loader, test_loader = self._create_data_loaders(X_train, X_val, X_test, y_train_norm, y_val_norm, y_test_norm)

        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, df):
        """Prepare feature tensor for inference from new data."""
        df = self._compute_car_age(df)
        df = self._clip_numeric_features(df)
        X = self._transform_features(df)
        return X

    def denormalize_targets(self, y_norm):
        """Convert normalized target values back to original scale."""
        y_original_np_array = self.y_scaler.inverse_transform(y_norm.detach().cpu().numpy())
        y_original_tensor = torch.tensor(y_original_np_array, dtype=torch.float32, device=y_norm.device)
        return y_original_tensor

    # ----------------- Save / Load -----------------
    
    def save_statistics(self):
        """Save statistics to file."""
        stats = {
            "y_scaler": self.y_scaler,
            "max_train_year": self.max_train_year,
            "numeric_min": self.numeric_min,
            "numeric_max": self.numeric_max
        }
        os.makedirs(os.path.dirname(config.STATISTICS_PATH), exist_ok=True)
        with open(config.STATISTICS_PATH, "wb") as f:
            pickle.dump(stats, f)
        print(f"• Statistics saved to {config.STATISTICS_PATH}")

    def load_statistics(self):
        """Load statistics from file."""
        with open(config.STATISTICS_PATH, "rb") as f:
            stats = pickle.load(f)
        self.y_scaler = stats["y_scaler"]
        self.max_train_year = stats["max_train_year"]
        self.numeric_min = stats["numeric_min"]
        self.numeric_max = stats["numeric_max"]
        print(f"• Statistics loaded from {config.STATISTICS_PATH}")

    def save_feature_transformer(self):
        """Save the fitted feature transformer to file."""
        os.makedirs(os.path.dirname(config.FEATURE_TRANSFORMER_PATH), exist_ok=True)
        with open(config.FEATURE_TRANSFORMER_PATH, "wb") as f:
            pickle.dump(self.feature_transformer, f)
        print(f"• Feature transformer saved to {config.FEATURE_TRANSFORMER_PATH}")

    def load_feature_transformer(self):
        """Load the fitted feature transformer from file."""
        with open(config.FEATURE_TRANSFORMER_PATH, "rb") as f:
            self.feature_transformer = pickle.load(f)
        print(f"• Feature transformer loaded from {config.FEATURE_TRANSFORMER_PATH}")

    # ----------------- Private Helpers -----------------

    def _load_csv(self):
        """Load CSV into a pandas DataFrame."""
        df = pd.read_csv(config.DATASET_CSV_PATH)
        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _compute_car_age(self, df):
        """Compute car age feature based on max year in training data."""
        if self.max_train_year is None:
            self.max_train_year = df["Year"].max()
        df["Car_Age"] = self.max_train_year - df["Year"]
        return df

    def _clip_numeric_features(self, df):
        """Clip numeric features to their training min/max ranges."""
        if self.numeric_min is None or self.numeric_max is None:
            self.numeric_min = df[self.numeric_cols].min().values
            self.numeric_max = df[self.numeric_cols].max().values
        for i, col in enumerate(self.numeric_cols):
            df[col] = df[col].clip(self.numeric_min[i], self.numeric_max[i])
        return df

    def _fit_feature_transformer(self, df):
        """Fit the feature transformer on training data and return transformed features."""
        self.feature_transformer = ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
        ])
        X = self.feature_transformer.fit_transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)

    def _transform_features(self, df):
        """Apply the fitted feature transformer to new data."""
        X = self.feature_transformer.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)

    def _split_dataset(self, X, y):
        """Split dataset into training, validation, and test subsets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val/test.")
            return X, X, X, y, y, y
    
        dataset = TensorDataset(X, y)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=generator
        )

        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _normalize_targets(self, y_train, y_val, y_test):
        """Fit scaler on training set, transform train/val/test targets."""
        y_train_norm = torch.tensor(self.y_scaler.fit_transform(y_train), dtype=torch.float32)
        y_val_norm = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)
        y_test_norm = torch.tensor(self.y_scaler.transform(y_test), dtype=torch.float32)
        return y_train_norm, y_val_norm, y_test_norm

    def _create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Return train, val, test DataLoaders."""
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
