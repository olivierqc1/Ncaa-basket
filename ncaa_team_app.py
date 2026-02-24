#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCAA TEAM BETTING ANALYZER - Flask Backend
Marchés: Totaux (O/U), Moneyline, Team Totals
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

try:
    from ncaa_team_data_collector import NCAATeamDataCollector
    from ncaa_team_model import TeamModelManager
    TEAM_MODEL_AVAILABLE = True
    print("✅ Modèle équipe NCAA activé")
except ImportError as e:
    TEAM_MODEL_AVAILABLE = False
    print(f"⚠️  Modèle équipe indisponible: {e}")

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

collector = NCAATeamDataCollector() if TEAM_MODEL_AVAILABLE else None
model_manager = TeamModelManager() if TEAM_MODEL_AVAILABLE else None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK',
        'sport': 'NCAA Basketball - Team Markets',
        'model_available': TEAM_MODEL_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/scan-games', methods=['GET'])
def scan_games():
    """
    Scanne tous les matchs NCAA du jour:
    - Récupère les lignes Totaux + Moneyline via Odds API
    - Prédit chaque match avec le modèle XGBoost
    - Identifie les edges positifs

    Query params:
        min_edge: float (défaut 4.0%) - edge minimum pour afficher
        market: 'totals' | 'moneyline' | 'all' (défaut 'all')
    """
    if not TEAM_MODEL_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Modèle non disponible'}), 503

    min_edge = request.args.get('min_edge', 4.0, type=float)
    market_filter = request.args.get('market', 'all')

    print(f"\n{'='*60}")
    print(f"🏀 SCAN MATCHS NCAA - Edge min: {min_edge}%")
    print(f"{'='*60}\n")

    try:
        # 1. Assure que le modèle est entraîné
        model_manager.ensure_trained()

        # 2. Récupère matchs avec lignes
        games = collector.get_upcoming_games_with_odds()
        print(f"📊 {len(games)} matchs trouvés")

        if not games:
            return jsonify({
                'status': 'SUCCESS',
                'total_games': 0,
                'opportunities': [],
                'scan_time': datetime.now().isoformat(),
                'message': 'Aucun match NCAA disponible'
            })

        # 3. Prédit chaque match
        opportunities = []

        for game in games:
            try:
                home = game['home_team']
                away = game['away_team']
                total_line = game.get('total_line')
                home_ml = game.get('home_moneyline')
                away_ml = game.get('away_moneyline')

                result = model_manager.predict_game(
                    home_team=home,
                    away_team=away,
                    total_line=total_line,
                    home_ml=home_ml,
                    away_ml=away_ml
                )

                if result.get('status') != 'SUCCESS':
                    continue

                # Ajoute infos du match
                result['game_info'] = {
                    'game_id': game.get('game_id', ''),
                    'game_time': game.get('game_time', ''),
                    'bookmakers': game.get('bookmakers', []),
                    'spread': game.get('spread'),
                    'total_over_odds': game.get('total_over_odds'),
                    'total_under_odds': game.get('total_under_odds'),
                }

                # Filtre par market et edge minimum
                total_edge = result.get('total_analysis', {}).get('edge', 0)
                ml_edge = result.get('moneyline_analysis', {}).get('best_edge', 0)

                has_total_opportunity = (
                    result['total_analysis'].get('recommendation', 'SKIP') != 'SKIP' and
                    total_edge >= min_edge and
                    market_filter in ('totals', 'all')
                )
                has_ml_opportunity = (
                    result['moneyline_analysis'].get('recommendation', 'SKIP') != 'SKIP' and
                    ml_edge >= min_edge and
                    market_filter in ('moneyline', 'all')
                )

                if has_total_opportunity or has_ml_opportunity:
                    result['has_total_opportunity'] = has_total_opportunity
                    result['has_ml_opportunity'] = has_ml_opportunity
                    result['best_opportunity_edge'] = max(total_edge, ml_edge)
                    opportunities.append(result)

            except Exception as e:
                print(f"   ❌ Erreur {game.get('home_team', '?')} vs {game.get('away_team', '?')}: {e}")
                continue

        # Trie par meilleur edge
        opportunities.sort(key=lambda x: x.get('best_opportunity_edge', 0), reverse=True)

        print(f"\n✅ {len(opportunities)}/{len(games)} matchs avec opportunités (edge ≥ {min_edge}%)")

        return jsonify({
            'status': 'SUCCESS',
            'total_games': len(games),
            'opportunities_found': len(opportunities),
            'min_edge_filter': min_edge,
            'scan_time': datetime.now().isoformat(),
            'model_stats': model_manager.model.training_stats if model_manager.model.is_trained else {},
            'opportunities': opportunities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/predict-game', methods=['POST'])
def predict_game():
    """
    Prédit un match spécifique.
    Body JSON:
    {
        "home_team": "Duke Blue Devils",
        "away_team": "North Carolina Tar Heels",
        "total_line": 158.5,        // optionnel
        "home_ml": -180,             // optionnel
        "away_ml": 155               // optionnel
    }
    """
    if not TEAM_MODEL_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Modèle non disponible'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'status': 'ERROR', 'message': 'Corps JSON requis'}), 400

    home = data.get('home_team', '').strip()
    away = data.get('away_team', '').strip()

    if not home or not away:
        return jsonify({'status': 'ERROR', 'message': 'home_team et away_team requis'}), 400

    try:
        result = model_manager.predict_game(
            home_team=home,
            away_team=away,
            total_line=data.get('total_line'),
            home_ml=data.get('home_ml'),
            away_ml=data.get('away_ml')
        )
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/team-stats/<team_name>', methods=['GET'])
def get_team_stats(team_name):
    """Retourne les stats BartTorvik d'une équipe"""
    if not TEAM_MODEL_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Collecteur non disponible'}), 503
    try:
        all_stats = collector.get_barttorvik_stats()
        stats = collector.find_team_stats(team_name, all_stats)
        return jsonify({'status': 'SUCCESS', 'team': team_name, 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Force un ré-entraînement du modèle"""
    if not TEAM_MODEL_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Modèle non disponible'}), 503
    try:
        model_manager._trained = False
        model_manager.model.is_trained = False
        result = model_manager.model.train()
        if result.get('status') == 'SUCCESS':
            model_manager._trained = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/debug-odds', methods=['GET'])
def debug_odds():
    """Debug: affiche les matchs et lignes disponibles"""
    if not TEAM_MODEL_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Collecteur non disponible'}), 503
    try:
        games = collector.get_upcoming_games_with_odds()
        return jsonify({
            'status': 'SUCCESS',
            'total_games': len(games),
            'games': games[:10]
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"\n🏀 NCAA Team Analyzer démarré (port {port})")
    app.run(host='0.0.0.0', port=port, debug=debug)