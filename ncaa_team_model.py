#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCAA TEAM MODEL
Modèle XGBoost pour prédire:
  1. Total de points du match (Over/Under)
  2. Vainqueur (Moneyline)
  3. Points par équipe (Team Totals)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import xgboost as xgb
from scipy import stats as scipy_stats

from ncaa_team_data_collector import NCAATeamDataCollector, NCAA_NATIONAL_AVG_OE, NCAA_NATIONAL_AVG_DE, NCAA_NATIONAL_TEMPO


class NCAATeamModel:
    """
    Modèle de prédiction pour les marchés d'équipe NCAA:
    - Totaux (Over/Under)
    - Moneyline (qui gagne)
    - Team Totals (score individuel par équipe)
    """

    def __init__(self):
        self.collector = NCAATeamDataCollector()
        self.total_model = None
        self.moneyline_model = None
        self.home_score_model = None
        self.away_score_model = None
        self.feature_columns = None
        self.training_stats = {}
        self.is_trained = False

    def train(self):
        """Entraîne les 4 modèles XGBoost sur données historiques"""
        print("\n" + "="*60)
        print("🏀 ENTRAÎNEMENT MODÈLE ÉQUIPE NCAA")
        print("="*60)

        df = self.collector.get_historical_games_for_training(n_teams=20)

        if df is None or len(df) < 100:
            return {'status': 'ERROR', 'message': 'Données insuffisantes'}

        print(f"\n✅ {len(df)} matchs pour entraînement")

        df_features = self._add_features(df)

        self.feature_columns = [
            'home_adj_oe', 'home_adj_de', 'home_adj_tempo',
            'away_adj_oe', 'away_adj_de', 'away_adj_tempo',
            'home_win_pct', 'away_win_pct',
            'oe_diff', 'de_diff', 'tempo_avg', 'pace_factor',
            'predicted_total_formula',
            'home_efficiency_ratio', 'away_efficiency_ratio'
        ]
        available_features = [f for f in self.feature_columns if f in df_features.columns]
        self.feature_columns = available_features

        X = df_features[self.feature_columns].fillna(df_features[self.feature_columns].mean())
        y_total = df_features['total_points']
        y_home_win = df_features['home_won']
        y_home_score = df_features['home_score']
        y_away_score = df_features['away_score']

        X_train, X_test, y_tot_train, y_tot_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
        _, _, y_ml_train, y_ml_test = train_test_split(X, y_home_win, test_size=0.2, random_state=42)
        _, _, y_hs_train, y_hs_test = train_test_split(X, y_home_score, test_size=0.2, random_state=42)
        _, _, y_as_train, y_as_test = train_test_split(X, y_away_score, test_size=0.2, random_state=42)

        print(f"\n🔄 Split: {len(X_train)} train / {len(X_test)} test")

        reg_params = {
            'objective': 'reg:squarederror', 'max_depth': 4,
            'learning_rate': 0.05, 'n_estimators': 150,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'min_child_weight': 5, 'random_state': 42, 'verbosity': 0
        }
        clf_params = {
            'objective': 'binary:logistic', 'max_depth': 4,
            'learning_rate': 0.05, 'n_estimators': 150,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'min_child_weight': 5, 'random_state': 42, 'verbosity': 0,
            'eval_metric': 'logloss'
        }

        print("\n🤖 Entraînement des 4 modèles...")

        self.total_model = xgb.XGBRegressor(**reg_params)
        self.total_model.fit(X_train, y_tot_train)

        self.moneyline_model = xgb.XGBClassifier(**clf_params)
        self.moneyline_model.fit(X_train, y_ml_train)

        self.home_score_model = xgb.XGBRegressor(**reg_params)
        self.home_score_model.fit(X_train, y_hs_train)

        self.away_score_model = xgb.XGBRegressor(**reg_params)
        self.away_score_model.fit(X_train, y_as_train)

        # Métriques
        total_pred = self.total_model.predict(X_test)
        ml_pred = self.moneyline_model.predict(X_test)
        hs_pred = self.home_score_model.predict(X_test)
        as_pred = self.away_score_model.predict(X_test)

        total_r2 = float(r2_score(y_tot_test, total_pred))
        total_rmse = float(np.sqrt(mean_squared_error(y_tot_test, total_pred)))
        total_mae = float(mean_absolute_error(y_tot_test, total_pred))
        ml_accuracy = float(accuracy_score(y_ml_test, ml_pred))
        home_score_rmse = float(np.sqrt(mean_squared_error(y_hs_test, hs_pred)))
        away_score_rmse = float(np.sqrt(mean_squared_error(y_as_test, as_pred)))

        self.training_stats = {
            'total_model': {'r2': total_r2, 'rmse': total_rmse, 'mae': total_mae},
            'moneyline_model': {'accuracy': ml_accuracy},
            'home_score_model': {'rmse': home_score_rmse},
            'away_score_model': {'rmse': away_score_rmse},
            'n_games': len(df)
        }
        self.is_trained = True

        print(f"\n📈 RÉSULTATS:")
        print(f"   Total R²:    {total_r2:.3f}")
        print(f"   Total RMSE:  {total_rmse:.2f} pts")
        print(f"   ML Accuracy: {ml_accuracy:.1%}")
        print("✅ Entraînement terminé!")

        return {
            'status': 'SUCCESS',
            'total_r2': total_r2,
            'total_rmse': total_rmse,
            'ml_accuracy': ml_accuracy,
            'n_games': len(df)
        }

    def _add_features(self, df):
        df = df.copy()
        for col in ['home_adj_oe', 'home_adj_de', 'home_adj_tempo',
                    'away_adj_oe', 'away_adj_de', 'away_adj_tempo',
                    'home_win_pct', 'away_win_pct']:
            if col not in df.columns:
                if 'oe' in col or 'de' in col:
                    df[col] = NCAA_NATIONAL_AVG_OE
                elif 'tempo' in col:
                    df[col] = NCAA_NATIONAL_TEMPO
                else:
                    df[col] = 0.5

        df['oe_diff'] = df['home_adj_oe'] - df['away_adj_oe']
        df['de_diff'] = df['home_adj_de'] - df['away_adj_de']
        df['tempo_avg'] = (df['home_adj_tempo'] + df['away_adj_tempo']) / 2
        df['pace_factor'] = df['tempo_avg'] / NCAA_NATIONAL_TEMPO
        df['predicted_total_formula'] = (
            df['tempo_avg'] *
            (df['home_adj_oe'] / NCAA_NATIONAL_AVG_OE + df['away_adj_oe'] / NCAA_NATIONAL_AVG_OE) *
            (NCAA_NATIONAL_AVG_DE / df['home_adj_de'] + NCAA_NATIONAL_AVG_DE / df['away_adj_de']) / 2
        )
        df['home_efficiency_ratio'] = df['home_adj_oe'] / df['away_adj_de'].clip(lower=1)
        df['away_efficiency_ratio'] = df['away_adj_oe'] / df['home_adj_de'].clip(lower=1)

        if 'home_won' not in df.columns and 'home_score' in df.columns:
            df['home_won'] = (df['home_score'] > df['away_score']).astype(int)
        elif 'home_won' not in df.columns:
            df['home_won'] = (df['total_points'] > df['total_points'].median()).astype(int)

        return df

    def predict_game(self, home_team, away_team, total_line=None, home_ml=None, away_ml=None):
        """Prédit toutes les métriques pour un match"""
        if not self.is_trained:
            print("⚠️  Modèle non entraîné - entraînement automatique...")
            self.train()

        all_stats = self.collector.get_barttorvik_stats()
        home_stats = self.collector.find_team_stats(home_team, all_stats)
        away_stats = self.collector.find_team_stats(away_team, all_stats)

        features = self._build_prediction_features(home_stats, away_stats)
        X = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_columns].fillna(0)

        pred_total = float(self.total_model.predict(X)[0])
        home_win_prob = float(self.moneyline_model.predict_proba(X)[0][1])
        pred_home_score = float(self.home_score_model.predict(X)[0])
        pred_away_score = float(self.away_score_model.predict(X)[0])

        total_rmse = self.training_stats['total_model']['rmse']
        home_rmse = self.training_stats['home_score_model']['rmse']

        formula_total = self._pomeroy_formula(home_stats, away_stats)

        total_analysis = self._analyze_total(pred_total, total_rmse, total_line)
        moneyline_analysis = self._analyze_moneyline(home_win_prob, home_ml, away_ml)

        return {
            'status': 'SUCCESS',
            'home_team': home_team,
            'away_team': away_team,
            'home_stats': {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in home_stats.items() if k != 'source'},
            'away_stats': {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in away_stats.items() if k != 'source'},
            'predicted_total': round(pred_total, 1),
            'predicted_home_score': round(pred_home_score, 1),
            'predicted_away_score': round(pred_away_score, 1),
            'home_win_probability': round(home_win_prob * 100, 1),
            'away_win_probability': round((1 - home_win_prob) * 100, 1),
            'formula_total': round(formula_total, 1),
            'total_ci': {
                'lower': round(pred_total - 1.96 * total_rmse, 1),
                'upper': round(pred_total + 1.96 * total_rmse, 1)
            },
            'home_score_ci': {
                'lower': round(pred_home_score - 1.96 * home_rmse, 1),
                'upper': round(pred_home_score + 1.96 * home_rmse, 1)
            },
            'total_analysis': total_analysis,
            'moneyline_analysis': moneyline_analysis,
            'model_stats': {
                'total_r2': round(self.training_stats['total_model']['r2'], 3),
                'total_rmse': round(total_rmse, 2),
                'ml_accuracy': round(self.training_stats['moneyline_model']['accuracy'], 3),
                'n_training_games': self.training_stats['n_games']
            }
        }

    def _build_prediction_features(self, home_stats, away_stats):
        f = {
            'home_adj_oe': home_stats.get('adj_oe', NCAA_NATIONAL_AVG_OE),
            'home_adj_de': home_stats.get('adj_de', NCAA_NATIONAL_AVG_DE),
            'home_adj_tempo': home_stats.get('adj_tempo', NCAA_NATIONAL_TEMPO),
            'home_win_pct': home_stats.get('win_pct', 0.5),
            'away_adj_oe': away_stats.get('adj_oe', NCAA_NATIONAL_AVG_OE),
            'away_adj_de': away_stats.get('adj_de', NCAA_NATIONAL_AVG_DE),
            'away_adj_tempo': away_stats.get('adj_tempo', NCAA_NATIONAL_TEMPO),
            'away_win_pct': away_stats.get('win_pct', 0.5),
        }
        f['oe_diff'] = f['home_adj_oe'] - f['away_adj_oe']
        f['de_diff'] = f['home_adj_de'] - f['away_adj_de']
        f['tempo_avg'] = (f['home_adj_tempo'] + f['away_adj_tempo']) / 2
        f['pace_factor'] = f['tempo_avg'] / NCAA_NATIONAL_TEMPO
        f['predicted_total_formula'] = (
            f['tempo_avg'] *
            (f['home_adj_oe'] / NCAA_NATIONAL_AVG_OE + f['away_adj_oe'] / NCAA_NATIONAL_AVG_OE) *
            (NCAA_NATIONAL_AVG_DE / max(f['away_adj_de'], 1) + NCAA_NATIONAL_AVG_DE / max(f['home_adj_de'], 1)) / 2
        )
        f['home_efficiency_ratio'] = f['home_adj_oe'] / max(f['away_adj_de'], 1)
        f['away_efficiency_ratio'] = f['away_adj_oe'] / max(f['home_adj_de'], 1)
        return f

    def _pomeroy_formula(self, home_stats, away_stats):
        avg_tempo = (home_stats.get('adj_tempo', NCAA_NATIONAL_TEMPO) +
                    away_stats.get('adj_tempo', NCAA_NATIONAL_TEMPO)) / 2
        home_score = avg_tempo * home_stats.get('adj_oe', NCAA_NATIONAL_AVG_OE) * \
                    (away_stats.get('adj_de', NCAA_NATIONAL_AVG_DE) / NCAA_NATIONAL_AVG_DE) / 100
        away_score = avg_tempo * away_stats.get('adj_oe', NCAA_NATIONAL_AVG_OE) * \
                    (home_stats.get('adj_de', NCAA_NATIONAL_AVG_DE) / NCAA_NATIONAL_AVG_DE) / 100
        home_score += 1.4
        away_score -= 1.4
        return home_score + away_score

    def _analyze_total(self, prediction, rmse, bookmaker_line):
        if bookmaker_line is None:
            return {'recommendation': 'NO_LINE', 'bookmaker_line': None}
        z = (bookmaker_line - prediction) / rmse
        over_prob = (1 - scipy_stats.norm.cdf(z)) * 100
        under_prob = 100 - over_prob
        implied_prob = 52.4
        if over_prob > implied_prob + 3:
            edge, recommendation, bet_prob = over_prob - implied_prob, 'OVER', over_prob
        elif under_prob > implied_prob + 3:
            edge, recommendation, bet_prob = under_prob - implied_prob, 'UNDER', under_prob
        else:
            edge, recommendation, bet_prob = 0, 'SKIP', max(over_prob, under_prob)

        full_kelly = (bet_prob / 100 - (1 - bet_prob / 100)) * 100 if edge > 3 else 0
        quarter_kelly = max(min(full_kelly / 4, 5), 0)

        return {
            'recommendation': recommendation,
            'bookmaker_line': bookmaker_line,
            'predicted_total': round(prediction, 1),
            'diff_from_line': round(prediction - bookmaker_line, 1),
            'over_probability': round(over_prob, 1),
            'under_probability': round(under_prob, 1),
            'edge': round(edge, 1),
            'kelly_criterion': round(quarter_kelly, 2),
            'confidence': 'HIGH' if edge >= 8 else 'MEDIUM' if edge >= 4 else 'LOW'
        }

    def _analyze_moneyline(self, model_home_win_prob, home_ml, away_ml):
        if home_ml is None or away_ml is None:
            return {'recommendation': 'NO_LINE', 'home_ml': None, 'away_ml': None}

        home_implied = self._american_to_prob(home_ml)
        away_implied = self._american_to_prob(away_ml)
        total_implied = home_implied + away_implied
        home_fair = home_implied / total_implied * 100
        away_fair = away_implied / total_implied * 100

        model_home_pct = model_home_win_prob * 100
        model_away_pct = (1 - model_home_win_prob) * 100
        home_edge = model_home_pct - home_fair
        away_edge = model_away_pct - away_fair

        if home_edge > away_edge and home_edge > 3:
            recommendation, edge, bet_odds, bet_prob = 'HOME', home_edge, home_ml, model_home_pct
        elif away_edge > 3:
            recommendation, edge, bet_odds, bet_prob = 'AWAY', away_edge, away_ml, model_away_pct
        else:
            recommendation, edge, bet_odds, bet_prob = 'SKIP', max(home_edge, away_edge), None, max(model_home_pct, model_away_pct)

        quarter_kelly = 0
        if bet_odds and recommendation != 'SKIP':
            decimal_odds = self._american_to_decimal(bet_odds)
            kelly = (bet_prob / 100 * decimal_odds - 1) / (decimal_odds - 1)
            quarter_kelly = max(min(kelly * 100 / 4, 5), 0)

        return {
            'recommendation': recommendation,
            'home_ml': home_ml, 'away_ml': away_ml,
            'model_home_win_pct': round(model_home_pct, 1),
            'model_away_win_pct': round(model_away_pct, 1),
            'home_implied_prob': round(home_fair, 1),
            'away_implied_prob': round(away_fair, 1),
            'home_edge': round(home_edge, 1),
            'away_edge': round(away_edge, 1),
            'best_edge': round(edge, 1),
            'kelly_criterion': round(quarter_kelly, 2),
            'confidence': 'HIGH' if edge >= 8 else 'MEDIUM' if edge >= 4 else 'LOW'
        }

    def _american_to_prob(self, odds):
        if odds > 0:
            return 100 / (odds + 100) * 100
        return abs(odds) / (abs(odds) + 100) * 100

    def _american_to_decimal(self, odds):
        return odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1


class TeamModelManager:
    """Singleton - entraîne une seule fois, réutilise le modèle"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = NCAATeamModel()
            cls._instance._trained = False
        return cls._instance

    def ensure_trained(self):
        if not self._trained:
            result = self.model.train()
            if result.get('status') == 'SUCCESS':
                self._trained = True
        return self._trained

    def predict_game(self, home_team, away_team, **kwargs):
        self.ensure_trained()
        return self.model.predict_game(home_team, away_team, **kwargs)


if __name__ == "__main__":
    model = NCAATeamModel()
    result = model.train()
    if result['status'] == 'SUCCESS':
        pred = model.predict_game(
            "Duke Blue Devils", "North Carolina Tar Heels",
            total_line=158.5, home_ml=-180, away_ml=155
        )
        print(f"\nDuke vs UNC: Total prédit={pred['predicted_total']}, ML Duke={pred['home_win_probability']}%")
        print(f"Total rec: {pred['total_analysis']['recommendation']} (edge: {pred['total_analysis']['edge']}%)")