"""ML module for IMS prediction models."""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
from typing import Tuple, Optional, Dict, List
from .schemas import MLFeatures, MLPrediction
from .utils import get_models_dir

class MLManager:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.k0_regressor = None
        self.family_classifier = None
    
    def make_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature matrix from dataframe."""
        cols = ["mass_amu", "z", "ccs_A2", "gas_mass_amu", "T_K", "P_Pa", "E_over_N_Td"]
        X = df[cols].astype(float).to_numpy()
        return X
    
    def train_k0_regressor(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train K0 regression model."""
        # Clean data
        df = df.dropna(subset=["mass_amu", "z", "ccs_A2", "gas_mass_amu", 
                              "T_K", "P_Pa", "E_over_N_Td", "K0_cm2_Vs"])
        
        if len(df) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        X = self.make_features(df)
        y = df["K0_cm2_Vs"].astype(float).to_numpy()
        
        # Split data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model
        self.k0_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
        self.k0_regressor.fit(Xtr, ytr)
        
        # Evaluate
        yhat = self.k0_regressor.predict(Xte)
        r2 = r2_score(yte, yhat)
        
        # Save model
        joblib.dump(self.k0_regressor, self.models_dir / "k0_reg.joblib")
        
        return {"r2": r2, "n_samples": len(df)}
    
    def train_family_classifier(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train family classification model."""
        if "family" not in df.columns:
            raise ValueError("Family column not found")
        
        # Clean data
        df = df.dropna(subset=["mass_amu", "z", "ccs_A2", "gas_mass_amu", 
                              "T_K", "P_Pa", "E_over_N_Td", "family"])
        df = df[df["family"].astype(str).str.strip() != ""]
        
        if len(df) < 20:
            raise ValueError("Need at least 20 samples for classification")
        
        X = self.make_features(df)
        y = df["family"].fillna("unknown").astype(str).to_numpy()
        
        # Check for sufficient classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Split data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Train model
        self.family_classifier = RandomForestClassifier(
            n_estimators=300, 
            class_weight="balanced_subsample", 
            random_state=0
        )
        self.family_classifier.fit(Xtr, ytr)
        
        # Evaluate
        yhat = self.family_classifier.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat, average="macro")
        
        # Save model
        joblib.dump(self.family_classifier, self.models_dir / "family_cls.joblib")
        
        return {"accuracy": acc, "f1_macro": f1, "n_samples": len(df), "n_classes": len(unique_classes)}
    
    def load_models(self) -> Tuple[bool, bool]:
        """Load saved models."""
        k0_path = self.models_dir / "k0_reg.joblib"
        family_path = self.models_dir / "family_cls.joblib"
        
        k0_loaded = False
        family_loaded = False
        
        if k0_path.exists():
            try:
                self.k0_regressor = joblib.load(k0_path)
                k0_loaded = True
            except Exception:
                pass
        
        if family_path.exists():
            try:
                self.family_classifier = joblib.load(family_path)
                family_loaded = True
            except Exception:
                pass
        
        return k0_loaded, family_loaded
    
    def predict_k0(self, features: MLFeatures) -> float:
        """Predict K0 from features."""
        if self.k0_regressor is None:
            raise ValueError("K0 regressor not loaded")
        
        X = features.to_array().reshape(1, -1)
        return float(self.k0_regressor.predict(X)[0])
    
    def predict_family(self, features: MLFeatures) -> str:
        """Predict family from features."""
        if self.family_classifier is None:
            raise ValueError("Family classifier not loaded")
        
        X = features.to_array().reshape(1, -1)
        return str(self.family_classifier.predict(X)[0])
    
    def predict_all(self, features: MLFeatures) -> MLPrediction:
        """Make all predictions."""
        prediction = MLPrediction(K0_pred=0.0)
        
        try:
            prediction.K0_pred = self.predict_k0(features)
            # Add uncertainty estimation (simplified)
            if self.k0_regressor:
                X = features.to_array().reshape(1, -1)
                tree_predictions = [tree.predict(X)[0] for tree in self.k0_regressor.estimators_]
                prediction.K0_uncertainty = float(np.std(tree_predictions))
        except Exception:
            pass
        
        try:
            prediction.family_pred = self.predict_family(features)
            # Add confidence (simplified)
            if self.family_classifier:
                X = features.to_array().reshape(1, -1)
                proba = self.family_classifier.predict_proba(X)[0]
                prediction.family_confidence = float(np.max(proba))
        except Exception:
            pass
        
        return prediction
    
    def invert_ccs_for_target_k0(self, mass_amu: float, z: int, target_k0: float,
                                gas_mass_amu: float, T_K: float, P_Pa: float, 
                                E_over_N_Td: float) -> float:
        """Find CCS that gives target K0."""
        if self.k0_regressor is None:
            raise ValueError("K0 regressor not loaded")
        
        # Grid search over CCS values
        ccs_grid = np.linspace(50, 400, 400)  # Å²
        X = np.column_stack([
            np.full_like(ccs_grid, mass_amu),
            np.full_like(ccs_grid, z),
            ccs_grid,
            np.full_like(ccs_grid, gas_mass_amu),
            np.full_like(ccs_grid, T_K),
            np.full_like(ccs_grid, P_Pa),
            np.full_like(ccs_grid, E_over_N_Td)
        ])
        
        predictions = self.k0_regressor.predict(X)
        best_idx = np.argmin(np.abs(predictions - target_k0))
        
        return float(ccs_grid[best_idx])
    
    def have_models(self) -> Tuple[bool, bool]:
        """Check if models are available."""
        return (self.k0_regressor is not None, self.family_classifier is not None)