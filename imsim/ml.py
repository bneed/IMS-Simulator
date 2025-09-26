"""ML module for IMS prediction models."""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, Optional, Dict, List
from .schemas import MLFeatures, MLPrediction
from .utils import get_models_dir
import warnings
warnings.filterwarnings('ignore')

class MLManager:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.k0_regressor = None
        self.family_classifier = None
        self.k0_scaler = None
        self.family_scaler = None
        self.family_encoder = None
        self.feature_names = None
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced feature matrix with engineered features."""
        # Base features
        base_cols = ["mass_amu", "z", "ccs_A2", "gas_mass_amu", "T_K", "P_Pa", "E_over_N_Td"]
        X = df[base_cols].copy()
        
        # Feature engineering
        X['mass_to_charge'] = X['mass_amu'] / X['z']
        X['ccs_to_mass'] = X['ccs_A2'] / X['mass_amu']
        X['ccs_to_charge'] = X['ccs_A2'] / X['z']
        X['reduced_mass'] = (X['mass_amu'] * X['gas_mass_amu']) / (X['mass_amu'] + X['gas_mass_amu'])
        X['collision_frequency'] = X['E_over_N_Td'] * X['P_Pa'] / X['T_K']
        X['mobility_parameter'] = X['mass_amu'] / (X['ccs_A2'] * X['z'])
        X['gas_density_effect'] = X['gas_mass_amu'] * X['P_Pa'] / X['T_K']
        X['temperature_effect'] = X['T_K'] / 298.15  # Normalized to room temp
        X['pressure_effect'] = X['P_Pa'] / 101325.0  # Normalized to 1 atm
        
        # Log transforms for skewed features
        X['log_mass'] = np.log1p(X['mass_amu'])
        X['log_ccs'] = np.log1p(X['ccs_A2'])
        X['log_collision_freq'] = np.log1p(X['collision_frequency'])
        
        # Polynomial features for key relationships
        X['mass_squared'] = X['mass_amu'] ** 2
        X['ccs_squared'] = X['ccs_A2'] ** 2
        X['mass_ccs_interaction'] = X['mass_amu'] * X['ccs_A2']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X
    
    def make_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature matrix from dataframe (backward compatibility)."""
        X_enhanced = self.create_enhanced_features(df)
        return X_enhanced.astype(float).to_numpy()
    
    def train_k0_regressor(self, df: pd.DataFrame, use_cv: bool = True, optimize_hyperparams: bool = True) -> Dict[str, float]:
        """Train K0 regression model with enhanced features and cross-validation."""
        # Clean data
        df = df.dropna(subset=["mass_amu", "z", "ccs_A2", "gas_mass_amu", 
                              "T_K", "P_Pa", "E_over_N_Td", "K0_cm2_Vs"])
        
        if len(df) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Create enhanced features
        X_df = self.create_enhanced_features(df)
        X = X_df.astype(float).to_numpy()
        y = df["K0_cm2_Vs"].astype(float).to_numpy()
        
        # Feature scaling
        self.k0_scaler = StandardScaler()
        X_scaled = self.k0_scaler.fit_transform(X)
        
        # Split data
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        
        # Hyperparameter optimization
        if optimize_hyperparams and len(df) >= 50:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(Xtr, ytr)
            self.k0_regressor = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Default parameters for smaller datasets
            self.k0_regressor = RandomForestRegressor(
                n_estimators=300, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.k0_regressor.fit(Xtr, ytr)
            best_params = {}
        
        # Cross-validation evaluation
        if use_cv:
            cv_scores = cross_val_score(
                self.k0_regressor, X_scaled, y, cv=5, scoring='r2'
            )
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()
        else:
            cv_r2_mean = cv_r2_std = 0.0
        
        # Final evaluation
        yhat = self.k0_regressor.predict(Xte)
        test_r2 = r2_score(yte, yhat)
        test_mae = mean_absolute_error(yte, yhat)
        test_rmse = np.sqrt(mean_squared_error(yte, yhat))
        
        # Save models and scaler
        joblib.dump(self.k0_regressor, self.models_dir / "k0_reg.joblib")
        joblib.dump(self.k0_scaler, self.models_dir / "k0_scaler.joblib")
        joblib.dump(self.feature_names, self.models_dir / "k0_features.joblib")
        
        return {
            "r2": test_r2,
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "mae": test_mae,
            "rmse": test_rmse,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "best_params": best_params
        }
    
    def train_family_classifier(self, df: pd.DataFrame, use_cv: bool = True, optimize_hyperparams: bool = True) -> Dict[str, float]:
        """Train family classification model with enhanced features and cross-validation."""
        if "family" not in df.columns:
            raise ValueError("Family column not found")
        
        # Clean data
        df = df.dropna(subset=["mass_amu", "z", "ccs_A2", "gas_mass_amu", 
                              "T_K", "P_Pa", "E_over_N_Td", "family"])
        df = df[df["family"].astype(str).str.strip() != ""]
        
        if len(df) < 20:
            raise ValueError("Need at least 20 samples for classification")
        
        # Create enhanced features
        X_df = self.create_enhanced_features(df)
        X = X_df.astype(float).to_numpy()
        y = df["family"].fillna("unknown").astype(str).to_numpy()
        
        # Check for sufficient classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Encode labels
        self.family_encoder = LabelEncoder()
        y_encoded = self.family_encoder.fit_transform(y)
        
        # Feature scaling
        self.family_scaler = StandardScaler()
        X_scaled = self.family_scaler.fit_transform(X)
        
        # Split data with stratification
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        # Hyperparameter optimization
        if optimize_hyperparams and len(df) >= 50:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(class_weight="balanced", random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=0
            )
            grid_search.fit(Xtr, ytr)
            self.family_classifier = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Default parameters for smaller datasets
            self.family_classifier = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
            self.family_classifier.fit(Xtr, ytr)
            best_params = {}
        
        # Cross-validation evaluation
        if use_cv:
            cv_scores = cross_val_score(
                self.family_classifier, X_scaled, y_encoded, cv=5, scoring='f1_macro'
            )
            cv_f1_mean = cv_scores.mean()
            cv_f1_std = cv_scores.std()
        else:
            cv_f1_mean = cv_f1_std = 0.0
        
        # Final evaluation
        yhat = self.family_classifier.predict(Xte)
        test_acc = accuracy_score(yte, yhat)
        test_f1 = f1_score(yte, yhat, average="macro")
        
        # Save models and scalers
        joblib.dump(self.family_classifier, self.models_dir / "family_cls.joblib")
        joblib.dump(self.family_scaler, self.models_dir / "family_scaler.joblib")
        joblib.dump(self.family_encoder, self.models_dir / "family_encoder.joblib")
        joblib.dump(self.feature_names, self.models_dir / "family_features.joblib")
        
        return {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "n_samples": len(df),
            "n_classes": len(unique_classes),
            "n_features": X.shape[1],
            "best_params": best_params
        }
    
    def load_models(self) -> Tuple[bool, bool]:
        """Load saved models and scalers."""
        k0_path = self.models_dir / "k0_reg.joblib"
        family_path = self.models_dir / "family_cls.joblib"
        k0_scaler_path = self.models_dir / "k0_scaler.joblib"
        family_scaler_path = self.models_dir / "family_scaler.joblib"
        family_encoder_path = self.models_dir / "family_encoder.joblib"
        k0_features_path = self.models_dir / "k0_features.joblib"
        family_features_path = self.models_dir / "family_features.joblib"
        
        k0_loaded = False
        family_loaded = False
        
        # Load K0 regressor
        if k0_path.exists():
            try:
                self.k0_regressor = joblib.load(k0_path)
                if k0_scaler_path.exists():
                    self.k0_scaler = joblib.load(k0_scaler_path)
                if k0_features_path.exists():
                    self.feature_names = joblib.load(k0_features_path)
                k0_loaded = True
            except Exception:
                pass
        
        # Load family classifier
        if family_path.exists():
            try:
                self.family_classifier = joblib.load(family_path)
                if family_scaler_path.exists():
                    self.family_scaler = joblib.load(family_scaler_path)
                if family_encoder_path.exists():
                    self.family_encoder = joblib.load(family_encoder_path)
                if family_features_path.exists():
                    self.feature_names = joblib.load(family_features_path)
                family_loaded = True
            except Exception:
                pass
        
        return k0_loaded, family_loaded
    
    def predict_k0(self, features: MLFeatures) -> float:
        """Predict K0 from features with enhanced uncertainty."""
        if self.k0_regressor is None:
            raise ValueError("K0 regressor not loaded")
        
        # Create feature array with enhanced features
        X_df = self.create_enhanced_features(pd.DataFrame([{
            'mass_amu': features.mass_amu,
            'z': features.z,
            'ccs_A2': features.ccs_A2,
            'gas_mass_amu': features.gas_mass_amu,
            'T_K': features.T_K,
            'P_Pa': features.P_Pa,
            'E_over_N_Td': features.E_over_N_Td
        }]))
        
        X = X_df.astype(float).to_numpy()
        
        # Scale features if scaler is available
        if self.k0_scaler is not None:
            X = self.k0_scaler.transform(X)
        
        return float(self.k0_regressor.predict(X)[0])
    
    def predict_family(self, features: MLFeatures) -> str:
        """Predict family from features with enhanced uncertainty."""
        if self.family_classifier is None:
            raise ValueError("Family classifier not loaded")
        
        # Create feature array with enhanced features
        X_df = self.create_enhanced_features(pd.DataFrame([{
            'mass_amu': features.mass_amu,
            'z': features.z,
            'ccs_A2': features.ccs_A2,
            'gas_mass_amu': features.gas_mass_amu,
            'T_K': features.T_K,
            'P_Pa': features.P_Pa,
            'E_over_N_Td': features.E_over_N_Td
        }]))
        
        X = X_df.astype(float).to_numpy()
        
        # Scale features if scaler is available
        if self.family_scaler is not None:
            X = self.family_scaler.transform(X)
        
        # Predict and decode
        y_pred = self.family_classifier.predict(X)[0]
        if self.family_encoder is not None:
            return str(self.family_encoder.inverse_transform([y_pred])[0])
        else:
            return str(y_pred)
    
    def predict_all(self, features: MLFeatures) -> MLPrediction:
        """Make all predictions with enhanced uncertainty quantification."""
        prediction = MLPrediction(K0_pred=0.0)
        
        try:
            prediction.K0_pred = self.predict_k0(features)
            
            # Enhanced uncertainty estimation using ensemble variance
            if self.k0_regressor:
                # Create feature array with enhanced features
                X_df = self.create_enhanced_features(pd.DataFrame([{
                    'mass_amu': features.mass_amu,
                    'z': features.z,
                    'ccs_A2': features.ccs_A2,
                    'gas_mass_amu': features.gas_mass_amu,
                    'T_K': features.T_K,
                    'P_Pa': features.P_Pa,
                    'E_over_N_Td': features.E_over_N_Td
                }]))
                
                X = X_df.astype(float).to_numpy()
                if self.k0_scaler is not None:
                    X = self.k0_scaler.transform(X)
                
                # Get predictions from all trees
                tree_predictions = [tree.predict(X)[0] for tree in self.k0_regressor.estimators_]
                prediction.K0_uncertainty = float(np.std(tree_predictions))
                
                # Add prediction interval (95% confidence)
                prediction.K0_lower = float(np.percentile(tree_predictions, 2.5))
                prediction.K0_upper = float(np.percentile(tree_predictions, 97.5))
                
        except Exception:
            pass
        
        try:
            prediction.family_pred = self.predict_family(features)
            
            # Enhanced confidence estimation
            if self.family_classifier:
                # Create feature array with enhanced features
                X_df = self.create_enhanced_features(pd.DataFrame([{
                    'mass_amu': features.mass_amu,
                    'z': features.z,
                    'ccs_A2': features.ccs_A2,
                    'gas_mass_amu': features.gas_mass_amu,
                    'T_K': features.T_K,
                    'P_Pa': features.P_Pa,
                    'E_over_N_Td': features.E_over_N_Td
                }]))
                
                X = X_df.astype(float).to_numpy()
                if self.family_scaler is not None:
                    X = self.family_scaler.transform(X)
                
                # Get class probabilities
                proba = self.family_classifier.predict_proba(X)[0]
                prediction.family_confidence = float(np.max(proba))
                
                # Get top 3 predictions
                if self.family_encoder is not None:
                    classes = self.family_encoder.classes_
                    top_indices = np.argsort(proba)[-3:][::-1]
                    prediction.top_predictions = [
                        {"class": classes[i], "confidence": float(proba[i])} 
                        for i in top_indices
                    ]
                
        except Exception:
            pass
        
        return prediction
    
    def train_models_easy(self, df: pd.DataFrame, model_types: List[str] = None) -> Dict[str, Dict]:
        """Easy one-step model training with automatic optimization."""
        if model_types is None:
            model_types = ["k0", "family"]
        
        results = {}
        
        # Train K0 regressor
        if "k0" in model_types:
            try:
                results["k0"] = self.train_k0_regressor(df, use_cv=True, optimize_hyperparams=True)
            except Exception as e:
                results["k0"] = {"error": str(e)}
        
        # Train family classifier
        if "family" in model_types:
            try:
                results["family"] = self.train_family_classifier(df, use_cv=True, optimize_hyperparams=True)
            except Exception as e:
                results["family"] = {"error": str(e)}
        
        return results
    
    def get_model_performance_summary(self) -> Dict[str, any]:
        """Get comprehensive model performance summary."""
        summary = {
            "models_loaded": self.have_models(),
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names if self.feature_names else []
        }
        
        # Add model-specific info if available
        if self.k0_regressor:
            summary["k0_model"] = {
                "n_estimators": self.k0_regressor.n_estimators,
                "max_depth": self.k0_regressor.max_depth,
                "feature_importance": dict(zip(
                    self.feature_names or [], 
                    self.k0_regressor.feature_importances_
                )) if self.feature_names else {}
            }
        
        if self.family_classifier:
            summary["family_model"] = {
                "n_estimators": self.family_classifier.n_estimators,
                "max_depth": self.family_classifier.max_depth,
                "n_classes": len(self.family_classifier.classes_),
                "classes": self.family_classifier.classes_.tolist(),
                "feature_importance": dict(zip(
                    self.feature_names or [], 
                    self.family_classifier.feature_importances_
                )) if self.feature_names else {}
            }
        
        return summary
    
    def invert_ccs_for_target_k0(self, mass_amu: float, z: int, target_k0: float,
                                gas_mass_amu: float, T_K: float, P_Pa: float, 
                                E_over_N_Td: float) -> float:
        """Find CCS that gives target K0 with enhanced features."""
        if self.k0_regressor is None:
            raise ValueError("K0 regressor not loaded")
        
        # Grid search over CCS values
        ccs_grid = np.linspace(50, 400, 400)  # Å²
        
        # Create feature matrix with enhanced features
        feature_data = []
        for ccs in ccs_grid:
            feature_data.append({
                'mass_amu': mass_amu,
                'z': z,
                'ccs_A2': ccs,
                'gas_mass_amu': gas_mass_amu,
                'T_K': T_K,
                'P_Pa': P_Pa,
                'E_over_N_Td': E_over_N_Td
            })
        
        X_df = self.create_enhanced_features(pd.DataFrame(feature_data))
        X = X_df.astype(float).to_numpy()
        
        if self.k0_scaler is not None:
            X = self.k0_scaler.transform(X)
        
        predictions = self.k0_regressor.predict(X)
        best_idx = np.argmin(np.abs(predictions - target_k0))
        
        return float(ccs_grid[best_idx])
    
    def have_models(self) -> Tuple[bool, bool]:
        """Check if models are available."""
        return (self.k0_regressor is not None, self.family_classifier is not None)