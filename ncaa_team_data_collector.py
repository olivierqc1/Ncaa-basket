#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCAA TEAM DATA COLLECTOR
Sources:
  - BartTorvik (AdjOE, AdjDE, AdjTempo) via scraping
  - sportsipy (game logs par équipe)
  - The Odds API (lignes totaux, moneyline)
"""

import os
import re
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from difflib import SequenceMatcher

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️  beautifulsoup4 non disponible - pip install beautifulsoup4")

try:
    from sportsipy.ncaab.teams import Teams
    from sportsipy.ncaab.schedule import Schedule
    SPORTSIPY_AVAILABLE = True
except ImportError:
    SPORTSIPY_AVAILABLE = False
    print("⚠️  sportsipy non disponible")


# =============================================================================
# CONSTANTES
# =============================================================================

# Mapping noms Odds API → abbr BartTorvik/sportsipy
TEAM_NAME_MAP = {
    # Format: "Odds API name": "sportsipy abbreviation"
    "Duke Blue Devils": "DUKE",
    "Kansas Jayhawks": "KANSAS",
    "Kentucky Wildcats": "KENTUCKY",
    "North Carolina Tar Heels": "NORTH-CAROLINA",
    "Gonzaga Bulldogs": "GONZAGA",
    "Houston Cougars": "HOUSTON",
    "Tennessee Volunteers": "TENNESSEE",
    "Auburn Tigers": "AUBURN",
    "Alabama Crimson Tide": "ALABAMA",
    "Purdue Boilermakers": "PURDUE",
    "Connecticut Huskies": "CONNECTICUT",
    "Baylor Bears": "BAYLOR",
    "Iowa State Cyclones": "IOWA-STATE",
    "Wisconsin Badgers": "WISCONSIN",
    "Michigan State Spartans": "MICHIGAN-STATE",
    "Arizona Wildcats": "ARIZONA",
    "Illinois Fighting Illini": "ILLINOIS",
    "Florida Gators": "FLORIDA",
    "Texas Tech Red Raiders": "TEXAS-TECH",
    "Creighton Bluejays": "CREIGHTON",
    "Marquette Golden Eagles": "MARQUETTE",
    "San Diego State Aztecs": "SAN-DIEGO-STATE",
    "St. John's Red Storm": "ST-JOHNS",
    "Pittsburgh Panthers": "PITTSBURGH",
    "Clemson Tigers": "CLEMSON",
    "Ole Miss Rebels": "MISSISSIPPI",
    "Mississippi State Bulldogs": "MISSISSIPPI-STATE",
    "Louisville Cardinals": "LOUISVILLE",
    "Texas A&M Aggies": "TEXAS-AM",
    "Florida State Seminoles": "FLORIDA-STATE",
    "Villanova Wildcats": "VILLANOVA",
    "Oregon Ducks": "OREGON",
    "UCLA Bruins": "UCLA",
    "Indiana Hoosiers": "INDIANA",
    "Ohio State Buckeyes": "OHIO-STATE",
    "Michigan Wolverines": "MICHIGAN",
    "Maryland Terrapins": "MARYLAND",
    "Xavier Musketeers": "XAVIER",
    "Butler Bulldogs": "BUTLER",
    "Virginia Cavaliers": "VIRGINIA",
    "Arkansas Razorbacks": "ARKANSAS",
    "Missouri Tigers": "MISSOURI",
    "Oklahoma Sooners": "OKLAHOMA",
    "West Virginia Mountaineers": "WEST-VIRGINIA",
    "TCU Horned Frogs": "TCU",
    "Kansas State Wildcats": "KANSAS-STATE",
    "Utah Utes": "UTAH",
    "Colorado Buffaloes": "COLORADO",
    "Washington Huskies": "WASHINGTON",
    "Stanford Cardinal": "STANFORD",
    "Notre Dame Fighting Irish": "NOTRE-DAME",
    "Georgetown Hoyas": "GEORGETOWN",
    "Seton Hall Pirates": "SETON-HALL",
    "Providence Friars": "PROVIDENCE",
    "Rutgers Scarlet Knights": "RUTGERS",
    "Minnesota Golden Gophers": "MINNESOTA",
    "Penn State Nittany Lions": "PENN-STATE",
    "Nebraska Cornhuskers": "NEBRASKA",
    "Iowa Hawkeyes": "IOWA",
    "Northwestern Wildcats": "NORTHWESTERN",
    "Wake Forest Demon Deacons": "WAKE-FOREST",
    "Syracuse Orange": "SYRACUSE",
    "Georgia Tech Yellow Jackets": "GEORGIA-TECH",
    "Boston College Eagles": "BOSTON-COLLEGE",
    "Virginia Tech Hokies": "VIRGINIA-TECH",
    "Miami Hurricanes": "MIAMI-FL",
    "Georgia Bulldogs": "GEORGIA",
    "South Carolina Gamecocks": "SOUTH-CAROLINA",
    "Tennessee State Tigers": "TENNESSEE-STATE",
    "LSU Tigers": "LSU",
    "Texas Longhorns": "TEXAS",
    "BYU Cougars": "BRIGHAM-YOUNG",
    "Cincinnati Bearcats": "CINCINNATI",
    "Memphis Tigers": "MEMPHIS",
    "Saint Louis Billikens": "SAINT-LOUIS",
    "Dayton Flyers": "DAYTON",
    "Rhode Island Rams": "RHODE-ISLAND",
    "VCU Rams": "VCU",
    "Davidson Wildcats": "DAVIDSON",
    "Wichita State Shockers": "WICHITA-STATE",
}

# Moyenne nationale NCAA 2024-25 (approximatif)
NCAA_NATIONAL_AVG_OE = 102.5   # pts per 100 poss
NCAA_NATIONAL_AVG_DE = 102.5
NCAA_NATIONAL_TEMPO = 68.5     # poss per 40 min


class NCAATeamDataCollector:
    """
    Collecteur de données d'équipe NCAA Basketball
    Combine BartTorvik + sportsipy + The Odds API
    """

    def __init__(self):
        self.season_year = self._get_current_ncaa_year()
        self._barttorvik_cache = {}
        self._team_stats_cache = {}
        self.odds_api_key = os.environ.get('ODDS_API_KEY')
        self.odds_base_url = 'https://api.the-odds-api.com/v4'

    def _get_current_ncaa_year(self):
        now = datetime.now()
        return now.year + 1 if now.month >= 10 else now.year

    # =========================================================================
    # BARTTORVIK - stats d'efficacité ajustée
    # =========================================================================

    def get_barttorvik_stats(self, force_refresh=False):
        """
        Scrape BartTorvik pour obtenir AdjOE, AdjDE, AdjTempo de toutes les équipes
        URL: https://barttorvik.com/trank.php (tableau principal)
        """
        if self._barttorvik_cache and not force_refresh:
            return self._barttorvik_cache

        print("📥 Récupération BartTorvik (AdjOE, AdjDE, Tempo)...")

        try:
            year = self.season_year
            url = f"https://barttorvik.com/trank.php?year={year}&sort=&top=0&conlimit=All&state=All&begin=20241101&end=20260401&toplimit=All"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

            resp = requests.get(url, headers=headers, timeout=20)

            if not BS4_AVAILABLE:
                print("   ⚠️  BeautifulSoup requis pour scraper BartTorvik")
                return self._get_mock_barttorvik_stats()

            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'id': 'dataTable'}) or soup.find('table')

            if not table:
                print("   ⚠️  Table BartTorvik non trouvée - mode mock")
                return self._get_mock_barttorvik_stats()

            rows = table.find_all('tr')[1:]  # Skip header
            stats = {}

            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) < 8:
                    continue

                try:
                    team_name = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    conf = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                    # Colonnes BartTorvik: Rank, Team, Conf, G, Rec, AdjOE, AdjDE, AdjTempo, ...
                    adj_oe = float(cols[5].get_text(strip=True)) if len(cols) > 5 else NCAA_NATIONAL_AVG_OE
                    adj_de = float(cols[6].get_text(strip=True)) if len(cols) > 6 else NCAA_NATIONAL_AVG_DE
                    adj_tempo = float(cols[7].get_text(strip=True)) if len(cols) > 7 else NCAA_NATIONAL_TEMPO

                    # Record W-L
                    record = cols[4].get_text(strip=True) if len(cols) > 4 else "0-0"
                    wins, losses = self._parse_record(record)

                    stats[team_name] = {
                        'team': team_name,
                        'conference': conf,
                        'adj_oe': adj_oe,
                        'adj_de': adj_de,
                        'adj_tempo': adj_tempo,
                        'wins': wins,
                        'losses': losses,
                        'win_pct': wins / max(wins + losses, 1),
                        'source': 'BartTorvik'
                    }
                except (ValueError, IndexError):
                    continue

            if stats:
                print(f"   ✅ {len(stats)} équipes récupérées depuis BartTorvik")
                self._barttorvik_cache = stats
                return stats
            else:
                print("   ⚠️  Données BartTorvik vides - mode mock")
                return self._get_mock_barttorvik_stats()

        except Exception as e:
            print(f"   ❌ Erreur BartTorvik: {e} - mode mock")
            return self._get_mock_barttorvik_stats()

    def _parse_record(self, record_str):
        """Parse '18-5' → (18, 5)"""
        try:
            parts = record_str.split('-')
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    def _get_mock_barttorvik_stats(self):
        """Données simulées si BartTorvik indisponible"""
        print("   ℹ️  Mode MOCK BartTorvik")
        mock_teams = [
            ("Duke", 120.5, 95.2, 72.1),
            ("Auburn", 118.3, 96.1, 69.8),
            ("Alabama", 117.8, 97.5, 73.2),
            ("Houston", 115.2, 94.8, 65.3),
            ("Tennessee", 114.9, 95.3, 66.1),
            ("Iowa State", 116.1, 96.7, 67.4),
            ("Florida", 115.7, 97.1, 68.9),
            ("Kansas", 114.3, 97.8, 70.5),
            ("Michigan State", 113.8, 96.9, 67.2),
            ("Purdue", 113.5, 97.4, 65.8),
            ("Texas A&M", 112.9, 98.1, 68.3),
            ("Wisconsin", 112.4, 97.9, 64.7),
            ("Arizona", 116.2, 99.1, 71.4),
            ("Gonzaga", 117.5, 100.3, 70.2),
            ("Illinois", 113.1, 98.5, 68.1),
            ("UConn", 114.7, 96.3, 66.9),
            ("St. John's", 115.3, 98.7, 69.5),
            ("Pittsburgh", 113.6, 99.2, 68.8),
            ("Marquette", 114.1, 98.3, 70.1),
            ("Clemson", 112.8, 99.5, 67.3),
            ("Ole Miss", 113.4, 100.1, 70.8),
            ("Louisville", 112.1, 100.5, 69.4),
        ]

        stats = {}
        for i, (team, adj_oe, adj_de, tempo) in enumerate(mock_teams):
            wins = np.random.randint(10, 20)
            losses = np.random.randint(3, 10)
            stats[team] = {
                'team': team,
                'conference': 'ACC' if i % 3 == 0 else 'SEC' if i % 3 == 1 else 'B12',
                'adj_oe': adj_oe,
                'adj_de': adj_de,
                'adj_tempo': tempo,
                'wins': wins,
                'losses': losses,
                'win_pct': wins / (wins + losses),
                'source': 'MOCK'
            }
        return stats

    def find_team_stats(self, team_name, all_stats=None):
        """
        Trouve les stats d'une équipe par fuzzy matching
        Retourne dict avec adj_oe, adj_de, adj_tempo
        """
        if all_stats is None:
            all_stats = self.get_barttorvik_stats()

        if not all_stats:
            return self._default_team_stats(team_name)

        # Match exact
        if team_name in all_stats:
            return all_stats[team_name]

        # Match fuzzy
        best_match = None
        best_score = 0

        for key in all_stats:
            score = SequenceMatcher(None, team_name.lower(), key.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = key

        if best_score >= 0.6:
            print(f"   🔍 Match fuzzy: '{team_name}' → '{best_match}' ({best_score:.2f})")
            return all_stats[best_match]

        print(f"   ⚠️  Équipe non trouvée: {team_name} - stats par défaut")
        return self._default_team_stats(team_name)

    def _default_team_stats(self, team_name):
        """Stats par défaut si équipe introuvable"""
        return {
            'team': team_name,
            'adj_oe': NCAA_NATIONAL_AVG_OE,
            'adj_de': NCAA_NATIONAL_AVG_DE,
            'adj_tempo': NCAA_NATIONAL_TEMPO,
            'wins': 10,
            'losses': 10,
            'win_pct': 0.5,
            'source': 'DEFAULT'
        }

    # =========================================================================
    # THE ODDS API - marchés d'équipe
    # =========================================================================

    def get_upcoming_games_with_odds(self):
        """
        Récupère les matchs NCAA avec lignes:
        - totals (Over/Under total de points)
        - h2h (moneyline)

        Returns: list de dicts avec infos match + cotes
        """
        if not self.odds_api_key:
            print("   ⚠️  ODDS_API_KEY non configurée")
            return self._get_mock_games()

        print("\n📥 Récupération matchs NCAA + lignes (Odds API)...")

        try:
            url = f"{self.odds_base_url}/sports/basketball_ncaab/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'totals,h2h,spreads',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }

            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            games_raw = resp.json()

            print(f"   ✅ {len(games_raw)} matchs trouvés")

            games = []
            for game in games_raw:
                parsed = self._parse_game(game)
                if parsed:
                    games.append(parsed)

            return games

        except Exception as e:
            print(f"   ❌ Erreur Odds API: {e}")
            return self._get_mock_games()

    def _parse_game(self, game_raw):
        """Parse un match depuis l'API Odds"""
        try:
            home = game_raw['home_team']
            away = game_raw['away_team']
            game_time = game_raw.get('commence_time', '')

            result = {
                'game_id': game_raw['id'],
                'home_team': home,
                'away_team': away,
                'game_time': game_time,
                'total_line': None,
                'total_over_odds': None,
                'total_under_odds': None,
                'home_moneyline': None,
                'away_moneyline': None,
                'spread': None,
                'home_spread_odds': None,
                'bookmakers': []
            }

            for bk in game_raw.get('bookmakers', []):
                result['bookmakers'].append(bk['title'])

                for market in bk.get('markets', []):
                    if market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over' and result['total_line'] is None:
                                result['total_line'] = outcome.get('point')
                                result['total_over_odds'] = outcome.get('price')
                            elif outcome['name'] == 'Under' and result['total_under_odds'] is None:
                                result['total_under_odds'] = outcome.get('price')

                    elif market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home and result['home_moneyline'] is None:
                                result['home_moneyline'] = outcome.get('price')
                            elif outcome['name'] == away and result['away_moneyline'] is None:
                                result['away_moneyline'] = outcome.get('price')

                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home and result['spread'] is None:
                                result['spread'] = outcome.get('point')
                                result['home_spread_odds'] = outcome.get('price')

                if result['total_line']:  # On a les infos du premier bookmaker
                    break

            return result

        except Exception as e:
            print(f"   ❌ Parse error: {e}")
            return None

    def _get_mock_games(self):
        """Matchs simulés pour tests"""
        print("   ℹ️  Mode MOCK - 5 matchs simulés")
        return [
            {
                'game_id': 'mock_1',
                'home_team': 'Duke Blue Devils',
                'away_team': 'North Carolina Tar Heels',
                'game_time': datetime.now().isoformat(),
                'total_line': 158.5,
                'total_over_odds': -110,
                'total_under_odds': -110,
                'home_moneyline': -180,
                'away_moneyline': +155,
                'spread': -4.5,
                'home_spread_odds': -110,
                'bookmakers': ['BetOnline']
            },
            {
                'game_id': 'mock_2',
                'home_team': 'Auburn Tigers',
                'away_team': 'Alabama Crimson Tide',
                'game_time': datetime.now().isoformat(),
                'total_line': 162.5,
                'total_over_odds': -115,
                'total_under_odds': -105,
                'home_moneyline': -140,
                'away_moneyline': +120,
                'spread': -3.0,
                'home_spread_odds': -110,
                'bookmakers': ['BetOnline']
            },
            {
                'game_id': 'mock_3',
                'home_team': 'Houston Cougars',
                'away_team': 'Iowa State Cyclones',
                'game_time': datetime.now().isoformat(),
                'total_line': 138.5,
                'total_over_odds': -108,
                'total_under_odds': -112,
                'home_moneyline': +105,
                'away_moneyline': -125,
                'spread': 1.5,
                'home_spread_odds': -110,
                'bookmakers': ['BetOnline']
            },
            {
                'game_id': 'mock_4',
                'home_team': 'Gonzaga Bulldogs',
                'away_team': 'Saint Mary\'s Gaels',
                'game_time': datetime.now().isoformat(),
                'total_line': 155.0,
                'total_over_odds': -110,
                'total_under_odds': -110,
                'home_moneyline': -250,
                'away_moneyline': +205,
                'spread': -7.5,
                'home_spread_odds': -110,
                'bookmakers': ['BetOnline']
            },
            {
                'game_id': 'mock_5',
                'home_team': 'Tennessee Volunteers',
                'away_team': 'Florida Gators',
                'game_time': datetime.now().isoformat(),
                'total_line': 145.5,
                'total_over_odds': -105,
                'total_under_odds': -115,
                'home_moneyline': -160,
                'away_moneyline': +135,
                'spread': -3.5,
                'home_spread_odds': -110,
                'bookmakers': ['BetOnline']
            }
        ]

    # =========================================================================
    # HISTORICAL DATA pour entraînement
    # =========================================================================

    def get_historical_games_for_training(self, n_teams=20):
        """
        Collecte données historiques pour entraîner le modèle
        Utilise sportsipy pour récupérer les résultats de matchs récents
        """
        print(f"\n📥 Collecte données d'entraînement ({n_teams} équipes)...")

        teams_to_sample = [
            'DUKE', 'AUBURN', 'ALABAMA', 'HOUSTON', 'TENNESSEE',
            'IOWA-STATE', 'FLORIDA', 'KANSAS', 'MICHIGAN-STATE', 'PURDUE',
            'CONNECTICUT', 'BAYLOR', 'ARIZONA', 'GONZAGA', 'ILLINOIS',
            'WISCONSIN', 'MARQUETTE', 'CREIGHTON', 'KENTUCKY', 'NORTH-CAROLINA'
        ][:n_teams]

        all_games = []

        if not SPORTSIPY_AVAILABLE:
            print("   ⚠️  sportsipy non disponible - génération données simulées")
            return self._generate_mock_training_data()

        for team_abbr in teams_to_sample:
            try:
                print(f"   📊 {team_abbr}...")
                schedule = Schedule(team_abbr, year=self.season_year)
                df = schedule.dataframe

                if 