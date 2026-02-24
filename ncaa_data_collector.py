#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCAA BASKETBALL DATA COLLECTOR
Source: sportsipy (sports-reference.com)
Remplace nba_api par les données NCAA Men's Basketball
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from sportsipy.ncaab.roster import Player, Roster
    from sportsipy.ncaab.teams import Teams
    SPORTSIPY_AVAILABLE = True
except ImportError:
    SPORTSIPY_AVAILABLE = False
    print("⚠️  sportsipy non disponible - pip install sportsipy")


# Cache global pour éviter de re-télécharger
_PLAYER_CACHE = {}
_TEAM_ROSTER_CACHE = {}

# Top programmes NCAA pour recherche rapide
TOP_NCAAB_TEAMS = [
    'DUKE', 'KANSAS', 'KENTUCKY', 'NORTH-CAROLINA', 'GONZAGA',
    'VILLANOVA', 'MICHIGAN-STATE', 'LOUISVILLE', 'ARIZONA', 'UCLA',
    'INDIANA', 'CONNECTICUT', 'HOUSTON', 'TENNESSEE', 'PURDUE',
    'BAYLOR', 'VIRGINIA', 'CREIGHTON', 'MARQUETTE', 'AUBURN',
    'ALABAMA', 'ARKANSAS', 'FLORIDA', 'IOWA-STATE', 'MICHIGAN',
    'OHIO-STATE', 'OREGON', 'TEXAS', 'MARYLAND', 'ILLINOIS',
    'WISCONSIN', 'FLORIDA-STATE', 'XAVIER', 'BUTLER', 'SAN-DIEGO-STATE'
]


class NCAADataCollector:
    """
    Collecteur de données NCAA Basketball
    Utilise sportsipy (sports-reference.com)
    """

    def __init__(self):
        self.cache = {}
        self.season_year = self._get_current_ncaa_year()

    def _get_current_ncaa_year(self):
        """
        NCAA utilise l'année de FIN de saison
        Ex: saison 2024-25 → year=2025
        """
        now = datetime.now()
        # Saison NBA/NCAA: Oct-Avril
        # Si on est en Oct-Dec → saison commence → year = now.year + 1
        # Si on est en Jan-Juin → saison en cours → year = now.year
        if now.month >= 10:
            return now.year + 1
        return now.year

    def get_complete_player_data(self, player_name, season_year=None):
        """
        Récupère les game logs d'un joueur NCAA

        Returns DataFrame avec:
        - GAME_DATE
        - PTS / AST / REB / MIN
        - avg_pts_last_5, avg_pts_last_10
        - home (0/1)
        - rest_days
        - minutes_avg
        """
        if season_year is None:
            season_year = self.season_year

        cache_key = f"{player_name}_{season_year}"
        if cache_key in self.cache:
            print(f"   📦 Cache hit: {player_name}")
            return self.cache[cache_key]

        try:
            print(f"\n📥 Collecting NCAA data: {player_name} ({season_year})...")

            if not SPORTSIPY_AVAILABLE:
                return self._generate_mock_data(player_name)

            # Cherche le joueur
            player_obj = self._find_player(player_name, season_year)

            if player_obj is None:
                print(f"❌ Joueur non trouvé: {player_name}")
                return None

            # Récupère game logs
            df = self._extract_game_logs(player_obj, season_year)

            if df is None or len(df) < 5:
                print(f"❌ Pas assez de matchs pour {player_name}")
                return None

            # Ajoute features
            df = self._add_features(df)

            print(f"   ✅ Final: {len(df)} games, {len(df.columns)} features")

            if len(df) < 10:
                print(f"   ⚠️  Seulement {len(df)} matchs")
                return None

            self.cache[cache_key] = df
            return df

        except Exception as e:
            print(f"❌ ERROR {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_player(self, player_name, season_year):
        """
        Cherche un joueur NCAA par nom
        Parcourt les rosters des principales équipes
        """
        player_name_lower = player_name.lower()

        print(f"   🔍 Searching for: {player_name}")

        for team_abbr in TOP_NCAAB_TEAMS:
            cache_key = f"{team_abbr}_{season_year}"

            try:
                if cache_key not in _TEAM_ROSTER_CACHE:
                    roster = Roster(team_abbr, year=season_year, slim=True)
                    _TEAM_ROSTER_CACHE[cache_key] = roster
                    time.sleep(0.3)
                else:
                    roster = _TEAM_ROSTER_CACHE[cache_key]

                for player_id, player_name_ref in roster.players.items():
                    if player_name_lower in player_name_ref.lower():
                        print(f"   ✅ Found: {player_name_ref} (ID: {player_id}) @ {team_abbr}")
                        player_obj = Player(player_id)
                        time.sleep(0.3)
                        return player_obj

            except Exception:
                continue

        # Essai direct par slug (format: prenom-nom-1)
        try:
            slug = self._name_to_slug(player_name)
            player_obj = Player(slug)
            if player_obj is not None:
                print(f"   ✅ Found via slug: {slug}")
                return player_obj
        except Exception:
            pass

        return None

    def _name_to_slug(self, player_name):
        """Convertit 'Cooper Flagg' → 'cooper-flagg-1'"""
        parts = player_name.lower().split()
        return '-'.join(parts) + '-1'

    def _extract_game_logs(self, player_obj, season_year):
        """Extrait game logs depuis objet Player sportsipy"""
        try:
            # sportsipy retourne un DataFrame avec toutes les saisons
            # On filtre par saison
            df = player_obj.dataframe

            if df is None or len(df) == 0:
                return None

            # Filtre la saison courante si multi-saisons
            if 'season' in df.columns:
                season_str = f"{season_year-1}-{str(season_year)[-2:]}"
                df = df[df['season'] == season_str].copy()

            if len(df) == 0:
                # Prend tout si pas de filtre possible
                df = player_obj.dataframe.copy()

            # Renomme colonnes vers notre format standard
            column_mapping = {
                'points': 'PTS',
                'assists': 'AST',
                'total_rebounds': 'REB',
                'offensive_rebounds': 'OREB',
                'defensive_rebounds': 'DREB',
                'minutes_played': 'MIN',
                'date': 'GAME_DATE',
                'opponent_id': 'MATCHUP',
                'location': 'HOME_AWAY',
                'result': 'WL',
                'field_goals': 'FGM',
                'field_goal_attempts': 'FGA',
                'three_point_field_goals': 'FG3M',
                'free_throws': 'FTM',
                'free_throw_attempts': 'FTA',
                'steals': 'STL',
                'blocks': 'BLK',
                'turnovers': 'TOV'
            }

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # S'assure qu'on a les colonnes essentielles
            for col in ['PTS', 'AST', 'REB']:
                if col not in df.columns:
                    df[col] = 0

            if 'MIN' not in df.columns:
                df['MIN'] = 30.0

            if 'GAME_DATE' not in df.columns and 'date' in df.index.names:
                df['GAME_DATE'] = df.index

            return df

        except Exception as e:
            print(f"   ❌ Extract error: {e}")
            return None

    def _add_features(self, df):
        """Ajoute les features pour le modèle XGBoost"""

        # Parse date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            df = df.dropna(subset=['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        else:
            df['GAME_DATE'] = pd.date_range(end=datetime.now(), periods=len(df), freq='3D')

        # Parse minutes
        if 'MIN' in df.columns:
            df['MIN'] = df['MIN'].fillna('0:00')
            df['MIN'] = df['MIN'].apply(self._parse_minutes)
        else:
            df['MIN'] = 28.0  # NCAA moyenne typique

        # Stats numériques
        for col in ['PTS', 'AST', 'REB']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mean_val = df[col].dropna().mean()
                df[col] = df[col].fillna(mean_val if not pd.isna(mean_val) else 0)
            else:
                df[col] = 0

        # Home/Away
        if 'HOME_AWAY' in df.columns:
            df['home'] = df['HOME_AWAY'].apply(lambda x: 1 if str(x).upper() in ['', 'HOME'] else 0)
        elif 'MATCHUP' in df.columns:
            df['home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
        else:
            df['home'] = 0

        # Rest days
        df['rest_days'] = df['GAME_DATE'].diff(-1).dt.days.fillna(2)
        df['rest_days'] = df['rest_days'].clip(0, 7)

        # Moyennes glissantes (shift pour éviter data leakage)
        df['avg_pts_last_5'] = df['PTS'].shift(1).rolling(5, min_periods=1).mean()
        df['avg_ast_last_5'] = df['AST'].shift(1).rolling(5, min_periods=1).mean()
        df['avg_reb_last_5'] = df['REB'].shift(1).rolling(5, min_periods=1).mean()

        df['avg_pts_last_10'] = df['PTS'].shift(1).rolling(10, min_periods=1).mean()
        df['avg_ast_last_10'] = df['AST'].shift(1).rolling(10, min_periods=1).mean()
        df['avg_reb_last_10'] = df['REB'].shift(1).rolling(10, min_periods=1).mean()

        df['minutes_avg'] = df['MIN'].shift(1).rolling(10, min_periods=1).mean()

        # Fill NaN final
        for col in df.select_dtypes(include=[np.float64, np.int64]).columns:
            mean_val = df[col].dropna().mean()
            df[col] = df[col].fillna(mean_val if not pd.isna(mean_val) else 0)

        # Supprime premiers matchs (features instables)
        if len(df) > 15:
            df = df[10:].reset_index(drop=True)

        return df

    def _parse_minutes(self, min_val):
        """Parse minutes: '32:45' → 32.75"""
        try:
            if isinstance(min_val, (int, float)):
                return float(min_val)
            if ':' in str(min_val):
                parts = str(min_val).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_val)
        except Exception:
            return 28.0  # Défaut NCAA

    def _generate_mock_data(self, player_name):
        """Génère données simulées si sportsipy indisponible (test)"""
        print(f"   ⚠️  Mode MOCK pour {player_name}")

        np.random.seed(hash(player_name) % (2**31))
        n_games = 30

        pts_mean = np.random.uniform(8, 25)
        ast_mean = np.random.uniform(1, 7)
        reb_mean = np.random.uniform(2, 10)

        dates = pd.date_range(end=datetime.now(), periods=n_games, freq='3D')[::-1]

        df = pd.DataFrame({
            'GAME_DATE': dates,
            'MATCHUP': [f'TEAM-A vs. TEAM-B' if i % 2 == 0 else 'TEAM-A @ TEAM-B' for i in range(n_games)],
            'PTS': np.random.normal(pts_mean, 4, n_games).clip(0, 45),
            'AST': np.random.normal(ast_mean, 2, n_games).clip(0, 15),
            'REB': np.random.normal(reb_mean, 2, n_games).clip(0, 20),
            'MIN': np.random.normal(30, 5, n_games).clip(10, 40),
        })

        df['PTS'] = df['PTS'].round(0).astype(int)
        df['AST'] = df['AST'].round(0).astype(int)
        df['REB'] = df['REB'].round(0).astype(int)

        return self._add_features(df)

    def prepare_features_for_prediction(self, player_name, opponent='', is_home=True, current_features=None):
        """Prépare features pour une prédiction"""
        try:
            if current_features is not None and isinstance(current_features, pd.DataFrame):
                return current_features

            df = self.get_complete_player_data(player_name)

            if df is None or len(df) == 0:
                return None

            features = df.iloc[0:1].copy()
            features = features.select_dtypes(include=[np.number])
            features = features.drop(columns=['PTS', 'AST', 'REB'], errors='ignore')

            return features

        except Exception as e:
            print(f"❌ Prepare features error: {e}")
            return None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    collector = NCAADataCollector()

    print(f"📅 Saison NCAA courante: {collector.season_year}")

    # Test avec un joueur connu
    test_players = ["Cooper Flagg", "Ace Bailey", "Tre Johnson"]

    for player in test_players:
        df = collector.get_complete_player_data(player)

        if df is not None:
            print(f"\n{'='*60}")
            print(f"✅ {player}: {len(df)} matchs")
            print(f"   Colonnes: {list(df.columns)}")
            print(f"   Moyennes: PTS={df['PTS'].mean():.1f}, AST={df['AST'].mean():.1f}, REB={df['REB'].mean():.1f}")
        else:
            print(f"\n❌ {player}: Pas de données")
