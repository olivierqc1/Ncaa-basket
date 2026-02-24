#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NCAA Model
Adapté pour le basket universitaire américain (NCAAB)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from ncaa_data_collector import NCAADataCollector


class XGBoostNCAAModel:
    """
    Modèle XGBoost pour prédictions NCAAB
    Optimisé pour les petits datasets (20-40 matchs par saison NCAA)
    """

    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.training_stats = {}

        self.stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB'
        }
        self.target_column = self.stat_map.get(stat_type, 'PTS')
        self.collector = NCAADataCollector()

    def train(self, player_name, season_year=None, save_model=True):
        """
        Entraîne le modèle XGBoost pour un joueur NCAA

        Note: Les saisons NCAA sont plus courtes que NBA (~30 matchs vs ~82)
        → Hyperparamètres ajustés pour petits datasets
        """

        if season_year is None:
            season_year = self.collector.season_year

        print(f"\n{'='*60}")
        print(f"🎓 TRAINING NCAA: {player_name} - {self.stat_type.upper()}")
        print(f"   Saison: {season_year-1}-{str(season_year)[-2:]}")
        print(f"{'='*60}")

        try:
            # 1. Collecte données
            df = self.collector.get_complete_player_data(player_name, season_year)

            if df is None or len(df) < 10:
                return {
                    'status': 'ERROR',
                    'message': f'Données insuffisantes: {len(df) if df is not None else 0} matchs',
                    'player': player_name,
                    'stat': self.stat_type
                }

            print(f"   ✅ {len(df)} matchs collectés")

            # 2. Prépare features
            X, y = self._prepare_training_data(df)

            if X is None or len(X) < 8:
                return {
                    'status': 'ERROR',
                    'message': 'Données alignées insuffisantes',
                    'player': player_name,
                    'stat': self.stat_type
                }

            # 3. Split train/test
            # NCAA: saisons courtes → test_size=0.2 (moins de données perdues)
            test_size = 0.2 if len(X) < 25 else 0.25

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                shuffle=True  # IMPORTANT: shuffle=True pour éviter distribution shift
            )

            print(f"\n🔄 Split: {len(X_train)} train / {len(X_test)} test")

            # 4. Hyperparamètres OPTIMISÉS pour petits datasets NCAA
            # Plus conservateurs que NBA (moins de matchs → plus de régularisation)
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 2,           # Peu profond (évite overfitting)
                'learning_rate': 0.05,    # Lent = plus stable
                'n_estimators': 25,       # Peu d'arbres (petits datasets)
                'subsample': 0.7,         # Bagging
                'colsample_bytree': 0.8,  # Feature sampling
                'reg_alpha': 0.2,         # L1 plus fort (NCAA = moins de données)
                'reg_lambda': 2.0,        # L2 plus fort
                'min_child_weight': 2,    # Évite feuilles avec peu d'exemples
                'random_state': 42,
                'verbosity': 0
            }

            print(f"\n🤖 Entraînement XGBoost NCAA (régularisé)...")

            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)

            # 5. Métriques
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            train_r2 = float(r2_score(y_train, y_pred_train))
            test_r2 = float(r2_score(y_test, y_pred_test))

            # Cap R² négatif (dataset trop petit → modèle peu fiable)
            if test_r2 < 0:
                print(f"   ⚠️  R² négatif ({test_r2:.3f}) → 0.01 (données insuffisantes)")
                test_r2 = 0.01

            train_metrics = {
                'r2': train_r2,
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train))
            }

            test_metrics = {
                'r2': test_r2,
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test))
            }

            print(f"\n📈 RÉSULTATS:")
            print(f"   Train R²:  {train_metrics['r2']:.3f}")
            print(f"   Test R²:   {test_metrics['r2']:.3f}")
            print(f"   Test RMSE: {test_metrics['rmse']:.2f} pts")

            if train_r2 - test_r2 > 0.3:
                print(f"   ⚠️  Overfitting (gap: {train_r2 - test_r2:.3f}) - normal pour petits datasets")

            # 6. Predictability score
            pred_score = max(0, min(100, test_r2 * 100))
            pred_category = 'HIGH' if pred_score >= 40 else 'MEDIUM' if pred_score >= 20 else 'LOW'

            predictability = {
                'score': float(pred_score),
                'category': pred_category
            }

            # 7. Sauvegarde stats
            self.training_stats = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictability': predictability,
                'data': {
                    'total_games': len(df),
                    'clean_games': len(X),
                    'season_year': season_year
                },
                'cv_results': {
                    'r2_mean': test_r2
                }
            }

            print(f"   Predictability: {pred_score:.1f}/100 ({pred_category})")
            print(f"✅ Entraînement NCAA terminé!")

            return {
                'status': 'SUCCESS',
                'player': player_name,
                'stat': self.stat_type,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'predictability': predictability,
                'games_analyzed': len(X),
                'season_year': season_year
            }

        except Exception as e:
            print(f"\n❌ Erreur entraînement: {e}")
            import traceback
            traceback.print_exc()

            return {
                'status': 'ERROR',
                'message': str(e),
                'player': player_name,
                'stat': self.stat_type
            }

    def _prepare_training_data(self, df):
        """Prépare features d'entraînement"""

        if self.target_column not in df.columns:
            print(f"   ❌ Colonne {self.target_column} manquante")
            return None, None

        y = df[self.target_column].copy()

        # Sélectionne features disponibles
        feature_candidates = [
            f'avg_{self.target_column.lower()}_last_5',
            f'avg_{self.target_column.lower()}_last_10',
            'avg_pts_last_5',
            'avg_pts_last_10',
            'home',
            'rest_days',
            'minutes_avg'
        ]

        feature_cols = [f for f in feature_candidates if f in df.columns]

        # Évite features redondantes si target = PTS
        if self.target_column == 'PTS':
            feature_cols = [f for f in feature_cols if f not in ['avg_pts_last_5', 'avg_pts_last_10']
                            or f'avg_pts_last' in f]

        if len(feature_cols) < 2:
            print(f"   ❌ Pas assez de features ({len(feature_cols)})")
            return None, None

        self.feature_columns = feature_cols
        X = df[feature_cols].copy()

        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        # Drop premiers matchs (features instables)
        if len(X) > 10:
            X = X.iloc[5:]
            y = y.iloc[5:]
            print(f"   🔧 Suppression 5 premiers matchs (features instables)")

        print(f"\n✅ Données propres: {len(X)} matchs, {len(feature_cols)} features")
        print(f"   Features: {feature_cols}")

        return X, y

    def predict(self, features_dict):
        """Prédit une valeur statistique"""
        if self.model is None:
            raise ValueError("Modèle non entraîné!")

        X = pd.DataFrame([features_dict])
        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]
        return float(prediction)


# ============================================================================
# MODEL MANAGER - Cache les modèles NCAA
# ============================================================================

class ModelManager:
    """Gère entraînement et cache des modèles XGBoost NCAA"""

    def __init__(self):
        self.models = {}
        self.collector = NCAADataCollector()

    def predict(self, player, stat_type, opponent, is_home):
        """
        Prédit en entraînant le modèle si besoin

        Returns:
            dict avec prediction, confidence_interval
        """
        model_key = f"{player}_{stat_type}"

        if model_key not in self.models:
            print(f"🔄 Entraînement modèle NCAA: {player} ({stat_type})...")

            model = XGBoostNCAAModel(stat_type=stat_type)
            result = model.train(player, save_model=False)

            if result['status'] != 'SUCCESS':
                raise ValueError(f"Entraînement échoué: {result.get('message')}")

            self.models[model_key] = model
            print(f"✅ Modèle en cache: {model_key}")

        model = self.models[model_key]

        # Prépare features récentes
        df = self.collector.get_complete_player_data(player)
        if df is None or len(df) == 0:
            raise ValueError("Pas de données disponibles")

        latest = df.iloc[0]

        features = {}
        for col in model.feature_columns:
            features[col] = float(latest[col]) if col in df.columns else 0.0

        features['home'] = 1.0 if is_home else 0.0

        # Prédiction
        prediction = model.predict(features)

        # Intervalle de confiance ± 2 RMSE
        rmse = model.training_stats['test_metrics']['rmse']
        ci = {
            'lower': round(max(0, prediction - 2 * rmse), 1),
            'upper': round(prediction + 2 * rmse, 1)
        }

        return {
            'prediction': round(prediction, 1),
            'confidence_interval': ci
        }


if __name__ == "__main__":
    print("XGBoost NCAA Basketball Model")
    print("Optimisé pour petits datasets (~30 matchs/saison)")

    # Test
    model = XGBoostNCAAModel(stat_type='points')
    result = model.train("Cooper Flagg")

    if result['status'] == 'SUCCESS':
        print(f"\n✅ Test R²: {result['test_metrics']['r2']:.3f}")
        print(f"   RMSE: {result['test_metrics']['rmse']:.2f} pts")
    else:
        print(f"\n❌ Erreur: {result['message']}")
