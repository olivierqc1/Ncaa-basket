#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Odds API Client - NCAA BASKETBALL (NCAAB)
Sport key: basketball_ncaab
"""

import os
import requests
from datetime import datetime, timedelta


class OddsAPIClient:
    def __init__(self):
        self.api_key = os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not found in environment variables")

        self.base_url = 'https://api.the-odds-api.com/v4'
        self.sport = 'basketball_ncaab'  # ← NCAA au lieu de basketball_nba

    def get_player_props(self, days=2):
        """
        Récupère les player props NCAA pour X jours

        Returns:
            list: Liste de toutes les props trouvées
        """
        all_props = []

        today_props = self._fetch_props_for_date(datetime.now())
        all_props.extend(today_props)

        if days >= 2:
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_props = self._fetch_props_for_date(tomorrow)
            all_props.extend(tomorrow_props)

        print(f"Total NCAA props retrieved: {len(all_props)} ({days} jour(s))")
        return all_props

    def _fetch_props_for_date(self, date):
        """Récupère les props pour une date"""

        games_url = f'{self.base_url}/sports/{self.sport}/odds'

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            response = requests.get(games_url, params=params, timeout=15)
            response.raise_for_status()
            games = response.json()

            print(f"Matchs NCAA trouvés ({date.strftime('%Y-%m-%d')}): {len(games)}")

            if not games:
                return []

        except requests.exceptions.RequestException as e:
            print(f"Erreur récupération matchs NCAA: {e}")
            return []

        # Récupère props pour chaque match
        params_props = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'player_points,player_rebounds,player_assists',
            'oddsFormat': 'american'
        }

        all_props = []

        for game in games:
            event_id = game['id']

            try:
                props_response = requests.get(
                    f'{self.base_url}/sports/{self.sport}/events/{event_id}/odds',
                    params=params_props,
                    timeout=15
                )
                props_response.raise_for_status()
                event_data = props_response.json()

                props = self._parse_props(event_data, game, date)
                all_props.extend(props)

            except requests.exceptions.RequestException as e:
                print(f"Erreur props match {event_id}: {e}")
                continue

        return all_props

    def _parse_props(self, event_data, game, date):
        """Parse les props d'un événement NCAA"""
        props = []

        home_team = game['home_team']
        away_team = game['away_team']
        game_time = game.get('commence_time', '')

        if 'bookmakers' not in event_data:
            return props

        stat_type_map = {
            'player_points': 'points',
            'player_rebounds': 'rebounds',
            'player_assists': 'assists'
        }

        seen = set()  # Déduplique par joueur+stat+bookmaker

        for bookmaker in event_data['bookmakers']:
            bookmaker_name = bookmaker['key']

            for market in bookmaker.get('markets', []):
                market_type = market['key']
                stat_type = stat_type_map.get(market_type)
                if not stat_type:
                    continue

                for outcome in market.get('outcomes', []):
                    player = outcome.get('description', '')
                    line = outcome.get('point')
                    bet_type = outcome.get('name', '').upper()  # OVER/UNDER

                    if not player or line is None:
                        continue

                    # Garde seulement OVER pour éviter doublons
                    if bet_type != 'OVER':
                        continue

                    dedup_key = f"{player}_{stat_type}_{bookmaker_name}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    props.append({
                        'player': player,
                        'stat_type': stat_type,
                        'line': float(line),
                        'over_odds': outcome.get('price', -110),
                        'under_odds': -110,
                        'bookmaker': bookmaker_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game_time,
                        'date': date.strftime('%Y-%m-%d'),
                        'sport': 'NCAA Basketball'
                    })

        return props

    def get_usage_stats(self):
        """Vérifie l'utilisation de l'API"""
        url = f'{self.base_url}/sports/{self.sport}/odds'
        params = {'apiKey': self.api_key, 'regions': 'us'}

        try:
            response = requests.get(url, params=params, timeout=10)
            return {
                'used': response.headers.get('X-Requests-Used', 'unknown'),
                'remaining': response.headers.get('X-Requests-Remaining', 'unknown'),
                'total': 500,
                'sport': 'NCAA Basketball'
            }
        except Exception as e:
            return {'error': str(e)}


if __name__ == '__main__':
    client = OddsAPIClient()

    print("\n=== TEST: NCAA Basketball Props ===\n")
    props = client.get_player_props(days=1)

    print(f"\nTotal props: {len(props)}")
    if props:
        print("\nExemple:")
        p = props[0]
        print(f"  Joueur: {p['player']}")
        print(f"  Stat:   {p['stat_type']}")
        print(f"  Ligne:  {p['line']}")
        print(f"  Match:  {p['away_team']} @ {p['home_team']}")