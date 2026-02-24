#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCAA Basketball Betting Analyzer - API Backend
Adapté de NBA → NCAAB
"""

import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np

try:
    from ncaa_data_collector import NCAADataCollector
    from ncaa_xgboost_model import XGBoostNCAAModel, ModelManager
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost NCAA activé")
except ImportError as e:
    print(f"⚠️  XGBoost indisponible: {e}")
    XGBOOST_AVAILABLE = False

try:
    from ncaa_odds_api_client import OddsAPIClient
    ODDS_API_AVAILABLE = True
    odds_client = OddsAPIClient()
except Exception as e:
    print(f"⚠️  Odds API indisponible: {e}")
    ODDS_API_AVAILABLE = False
    odds_client = None

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

collector = NCAADataCollector() if XGBOOST_AVAILABLE else None
model_manager = ModelManager() if XGBOOST_AVAILABLE else None


# ============================================================================
# ANALYSE XGBOOST
# ============================================================================

def analyze_with_xgboost(player, opponent, is_home, stat_type, line):
    """Analyse XGBoost pour un joueur NCAA"""

    print(f"🎓 Analyse NCAA: {player} vs {opponent} ({stat_type})")

    features = collector.prepare_features_for_prediction(player, opponent, is_home)
    if features is None:
        return {'status': 'ERROR', 'error': 'Impossible de collecter les données'}

    try:
        prediction_result = model_manager.predict(player, stat_type, opponent, is_home)
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Erreur modèle: {str(e)}'}

    prediction = prediction_result['prediction']
    confidence_interval = prediction_result['confidence_interval']

    line_analysis = analyze_betting_line(prediction, confidence_interval, line)

    df = collector.get_complete_player_data(player)
    stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[stat_type]

    season_stats = {
        'games_played': len(df),
        'weighted_avg': round(df[stat_col].mean(), 1),
        'std_dev': round(df[stat_col].std(), 1),
        'min': int(df[stat_col].min()),
        'max': int(df[stat_col].max())
    }

    model_key = f"{player}_{stat_type}"
    if model_key in model_manager.models:
        model_stats = model_manager.models[model_key].training_stats
        r_squared = model_stats.get('test_metrics', {}).get('r2', 0.3)
        rmse = model_stats.get('test_metrics', {}).get('rmse', 5.0)
    else:
        r_squared = 0.30
        rmse = 5.0

    return {
        'status': 'SUCCESS',
        'player': player,
        'opponent': opponent,
        'is_home': is_home,
        'stat_type': stat_type,
        'prediction': prediction,
        'confidence_interval': confidence_interval,
        'line_analysis': line_analysis,
        'season_stats': season_stats,
        'regression_stats': {
            'r_squared': round(r_squared, 3),
            'rmse': round(rmse, 2),
            'model_type': 'XGBoost NCAA'
        },
        'data_source': 'Sports-Reference + XGBoost',
        'sport': 'NCAA Basketball'
    }


def analyze_betting_line(prediction, confidence_interval, line):
    """Analyse la ligne bookmaker"""

    if line is None:
        return {'recommendation': 'NO_LINE', 'bookmaker_line': None}

    std = (confidence_interval['upper'] - confidence_interval['lower']) / 3.92

    from scipy import stats
    z_score = (line - prediction) / std
    over_prob = (1 - stats.norm.cdf(z_score)) * 100
    under_prob = 100 - over_prob

    implied_prob = 52.4

    if over_prob > implied_prob:
        edge = over_prob - implied_prob
        recommendation = 'OVER'
        bet_prob = over_prob
    elif under_prob > implied_prob:
        edge = under_prob - implied_prob
        recommendation = 'UNDER'
        bet_prob = under_prob
    else:
        edge = 0
        recommendation = 'SKIP'
        bet_prob = max(over_prob, under_prob)

    kelly = (bet_prob / 100 - (1 - bet_prob / 100)) * 100 if edge > 5 else 0
    kelly = max(min(kelly, 10), 0)

    confidence = 'HIGH' if edge >= 10 else 'MEDIUM' if edge >= 5 else 'LOW'

    return {
        'recommendation': recommendation,
        'bookmaker_line': line,
        'over_probability': round(over_prob, 1),
        'under_probability': round(under_prob, 1),
        'edge': round(edge, 1),
        'kelly_criterion': round(kelly, 1),
        'bet_confidence': confidence
    }


# ============================================================================
# ENDPOINTS NCAA
# ============================================================================

@app.route('/api/daily-opportunities-points', methods=['GET'])
def daily_opportunities_points():
    return scan_opportunities_by_type('points', limit=10)


@app.route('/api/daily-opportunities-assists', methods=['GET'])
def daily_opportunities_assists():
    return scan_opportunities_by_type('assists', limit=15)


@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    return scan_opportunities_by_type('rebounds', limit=15)


def scan_opportunities_by_type(stat_type, limit=15):
    """Scan opportunités NCAA pour un type de stat"""

    min_edge = request.args.get('min_edge', 5.0, type=float)
    min_r2 = request.args.get('min_r2', 0.30, type=float)  # Seuil plus bas pour NCAA

    if not ODDS_API_AVAILABLE or not odds_client:
        return jsonify({
            'status': 'ERROR',
            'message': 'Odds API non disponible - configurer ODDS_API_KEY'
        }), 503

    if not XGBOOST_AVAILABLE:
        return jsonify({
            'status': 'ERROR',
            'message': 'XGBoost non disponible'
        }), 503

    print(f"\n{'='*70}")
    print(f"🎓 SCAN NCAA {stat_type.upper()} - {limit} PROPS")
    print(f"{'='*70}\n")

    try:
        all_props = odds_client.get_player_props(days=1)

        filtered_props = [
            p for p in all_props
            if p.get('stat_type') == stat_type
        ]

        print(f"📊 Props NCAA {stat_type} disponibles: {len(filtered_props)}")

        if len(filtered_props) == 0:
            return jsonify({
                'status': 'SUCCESS',
                'stat_type': stat_type,
                'sport': 'NCAA Basketball',
                'total_available': 0,
                'total_analyzed': 0,
                'opportunities_found': 0,
                'scan_time': datetime.now().isoformat(),
                'message': f'Aucune prop NCAA {stat_type} disponible aujourd\'hui',
                'opportunities': []
            })

        random.shuffle(filtered_props)
        selected_props = filtered_props[:limit]

        opportunities = []
        analyzed_count = 0

        for prop in selected_props:
            player = prop.get('player', 'Unknown')
            line = prop.get('line', 0)
            opponent = prop.get('away_team', 'Unknown')
            is_home = bool(prop.get('home_team'))

            try:
                result = analyze_with_xgboost(player, opponent, is_home, stat_type, line)
                analyzed_count += 1

                if result.get('status') != 'SUCCESS':
                    continue

                r2 = result['regression_stats']['r_squared']
                edge = result['line_analysis']['edge']
                rec = result['line_analysis']['recommendation']

                if rec == 'SKIP' or edge < min_edge or r2 < min_r2:
                    continue

                result['game_info'] = {
                    'date': prop.get('date', ''),
                    'time': prop.get('game_time', ''),
                    'home_team': prop.get('home_team', ''),
                    'away_team': prop.get('away_team', ''),
                    'sport': 'NCAA Basketball'
                }

                result['bookmaker_info'] = {
                    'bookmaker': prop.get('bookmaker', 'Unknown'),
                    'line': line,
                    'over_odds': prop.get('over_odds', -110),
                    'under_odds': prop.get('under_odds', -110)
                }

                opportunities.append(result)

            except Exception as e:
                print(f"❌ Erreur {player}: {e}")
                continue

        opportunities.sort(
            key=lambda x: x['regression_stats']['r_squared'],
            reverse=True
        )

        print(f"✅ {analyzed_count} props analysées")
        print(f"✅ {len(opportunities)} opportunités trouvées")

        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'sport': 'NCAA Basketball',
            'total_available': len(filtered_props),
            'total_analyzed': analyzed_count,
            'opportunities_found': len(opportunities),
            'scan_time': datetime.now().isoformat(),
            'model_type': 'XGBoost NCAA',
            'filters': {'min_edge': min_edge, 'min_r2': min_r2},
            'opportunities': opportunities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ============================================================================
# PLAYER HISTORY (NCAA)
# ============================================================================

@app.route('/api/player-history/<player_name>', methods=['GET'])
def get_player_history(player_name):
    """Récupère les 10 derniers matchs d'un joueur NCAA"""

    if not XGBOOST_AVAILABLE or not collector:
        return jsonify({'status': 'ERROR', 'message': 'Collecteur non disponible'}), 503

    try:
        df = collector.get_complete_player_data(player_name)

        if df is None or len(df) == 0:
            return jsonify({
                'status': 'ERROR',
                'message': f'Aucune donnée pour {player_name}'
            }), 404

        df = df.sort_values('GAME_DATE', ascending=False)
        recent_games = df.head(10)

        games = []
        for _, row in recent_games.iterrows():
            matchup = str(row.get('MATCHUP', ''))
            opponent = matchup.split()[-1] if matchup else 'N/A'
            is_home = 'vs.' in matchup or row.get('home', 0) == 1

            games.append({
                'date': str(row['GAME_DATE'])[:10],
                'opponent': opponent,
                'is_home': bool(is_home),
                'points': int(row.get('PTS', 0)),
                'assists': int(row.get('AST', 0)),
                'rebounds': int(row.get('REB', 0)),
                'minutes': int(row.get('MIN', 0)),
                'result': str(row.get('WL', '-'))
            })

        pts_last_5 = df.head(5)['PTS'].mean()
        pts_prev_5 = df.iloc[5:10]['PTS'].mean() if len(df) >= 10 else df['PTS'].mean()
        pts_trend = round(pts_last_5 - pts_prev_5, 1)

        wl_counts = df.head(5).get('WL', pd.Series()).value_counts().to_dict() if 'WL' in df.columns else {}
        wins = wl_counts.get('W', 0)
        losses = wl_counts.get('L', 0)

        return jsonify({
            'status': 'SUCCESS',
            'player': player_name,
            'sport': 'NCAA Basketball',
            'games': games,
            'stats': {
                'games_played': len(df),
                'avg_points': round(df['PTS'].mean(), 1),
                'avg_assists': round(df['AST'].mean(), 1),
                'avg_rebounds': round(df['REB'].mean(), 1),
                'avg_minutes': round(df['MIN'].mean(), 1)
            },
            'trends': {
                'points_trend': pts_trend,
                'form': f"{wins}W-{losses}L",
                'minutes_avg': round(df.head(5)['MIN'].mean(), 1),
                'minutes_stable': bool(df.head(10)['MIN'].std() < 4)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ============================================================================
# AUTRES ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'sport': 'NCAA Basketball',
        'xgboost_enabled': XGBOOST_AVAILABLE,
        'odds_api_enabled': ODDS_API_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/debug-odds', methods=['GET'])
def debug_odds():
    """Debug de l'API Odds NCAA"""
    if not ODDS_API_AVAILABLE or not odds_client:
        return jsonify({'status': 'ERROR', 'message': 'Odds API non initialisée'})

    try:
        raw_props = odds_client.get_player_props(days=1)
        stat_types = {}
        for p in raw_props:
            st = p.get('stat_type', 'unknown')
            stat_types[st] = stat_types.get(st, 0) + 1

        return jsonify({
            'status': 'SUCCESS',
            'sport': 'basketball_ncaab',
            'total_props': len(raw_props),
            'stat_types': stat_types,
            'sample': raw_props[:3] if raw_props else []
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)})


@app.route('/api/test-model-quality', methods=['GET'])
def test_model_quality():
    """Teste R² sur joueurs NCAA stars"""
    if not XGBOOST_AVAILABLE:
        return jsonify({'error': 'XGBoost non disponible'}), 503

    test_players = ['Cooper Flagg', 'Ace Bailey', 'Tre Johnson', 'Dylan Harper']
    results = []

    for player in test_players:
        for stat_type in ['points', 'assists', 'rebounds']:
            try:
                model = XGBoostNCAAModel(stat_type=stat_type)
                result = model.train(player, save_model=False)

                if result['status'] == 'SUCCESS':
                    results.append({
                        'player': player,
                        'stat': stat_type,
                        'test_r2': round(result['test_metrics']['r2'], 3),
                        'train_r2': round(result['train_metrics']['r2'], 3),
                        'games': result['games_analyzed']
                    })
                else:
                    results.append({
                        'player': player,
                        'stat': stat_type,
                        'status': result['status'],
                        'message': result.get('message', 'Erreur')
                    })
            except Exception as e:
                results.append({'player': player, 'stat': stat_type, 'error': str(e)})

    return jsonify({'status': 'SUCCESS', 'sport': 'NCAA Basketball', 'results': results})


@app.route('/api/odds/usage', methods=['GET'])
def get_odds_usage():
    if ODDS_API_AVAILABLE and odds_client:
        return jsonify(odds_client.get_usage_stats())
    return jsonify({'error': 'Odds API non disponible'}), 503


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"\n🎓 NCAA Basketball Analyzer démarré (port {port})")
    app.run(host='0.0.0.0', port=port, debug=debug)