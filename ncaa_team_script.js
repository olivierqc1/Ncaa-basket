const API_URL = 'https://ncaa-team-betting.onrender.com'; // ← Mettre à jour avec ton URL Render

// ============================================================================
// SCAN MATCHS DU JOUR
// ============================================================================

async function scanGames() {
    const minEdge = document.getElementById('minEdge').value;
    const market = document.getElementById('marketFilter').value;

    setLoading(true);
    clearResults();
    hideError();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 240000); // 4 min timeout

    try {
        const resp = await fetch(
            `${API_URL}/api/scan-games?min_edge=${minEdge}&market=${market}`,
            { signal: controller.signal }
        );
        clearTimeout(timeoutId);

        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        const data = await resp.json();

        if (data.status === 'SUCCESS') {
            displayScanResults(data);
        } else {
            showError(data.message || 'Erreur inconnue');
        }
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
            showError('⏱ Timeout (>4 min). Le modèle s\'entraîne lors du premier appel — réessaie dans quelques secondes.');
        } else {
            showError(`❌ ${err.message}`);
        }
    } finally {
        setLoading(false);
    }
}

// ============================================================================
// PRÉDICTION MANUELLE
// ============================================================================

async function predictManual() {
    const homeTeam = document.getElementById('homeTeam').value.trim();
    const awayTeam = document.getElementById('awayTeam').value.trim();

    if (!homeTeam || !awayTeam) {
        showError('Remplis les deux noms d\'équipe');
        return;
    }

    const totalLine = parseFloat(document.getElementById('totalLine').value) || null;
    const homeML = parseInt(document.getElementById('homeML').value) || null;
    const awayML = parseInt(document.getElementById('awayML').value) || null;

    setLoading(true, 'Prédiction en cours...');
    clearResults();
    hideError();

    try {
        const resp = await fetch(`${API_URL}/api/predict-game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, total_line: totalLine, home_ml: homeML, away_ml: awayML })
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (data.status === 'SUCCESS') {
            displaySingleGame(data);
        } else {
            showError(data.message || 'Erreur');
        }
    } catch (err) {
        showError(`❌ ${err.message}`);
    } finally {
        setLoading(false);
    }
}

// ============================================================================
// DEBUG
// ============================================================================

async function debugOdds() {
    try {
        const resp = await fetch(`${API_URL}/api/debug-odds`);
        const data = await resp.json();
        const resultsDiv = document.getElementById('resultsDiv');
        resultsDiv.innerHTML = `
            <div style="background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; font-family: var(--mono); font-size: 11px; color: var(--muted);">
                <div style="color: var(--accent); margin-bottom: 12px; font-size: 13px;">DEBUG — ${data.total_games} matchs disponibles</div>
                <pre style="white-space: pre-wrap; overflow-x: auto;">${JSON.stringify(data.games?.slice(0,3), null, 2)}</pre>
            </div>
        `;
    } catch (err) {
        showError(`Debug error: ${err.message}`);
    }
}

// ============================================================================
// AFFICHAGE DES RÉSULTATS (SCAN)
// ============================================================================

function displayScanResults(data) {
    const statsBar = document.getElementById('statsBar');
    statsBar.classList.remove('hidden');

    document.getElementById('statGames').textContent = data.total_games;
    document.getElementById('statOpps').textContent = data.opportunities_found;

    const modelStats = data.model_stats?.total_model;
    if (modelStats) {
        document.getElementById('statR2').textContent = (modelStats.r2 * 100).toFixed(0) + '%';
    }
    const mlAcc = data.model_stats?.moneyline_model?.accuracy;
    if (mlAcc) {
        document.getElementById('statAcc').textContent = (mlAcc * 100).toFixed(0) + '%';
    }

    const resultsDiv = document.getElementById('resultsDiv');

    if (data.opportunities_found === 0) {
        resultsDiv.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div>Aucune opportunité avec edge ≥ ${data.min_edge_filter}%</div>
                <div style="margin-top: 8px; font-size: 11px;">Réduis le seuil ou consulte tous les matchs</div>
            </div>
        `;
        return;
    }

    resultsDiv.innerHTML = data.opportunities.map(opp => createGameCard(opp)).join('');
}

// ============================================================================
// AFFICHAGE PRÉDICTION UNIQUE
// ============================================================================

function displaySingleGame(data) {
    document.getElementById('resultsDiv').innerHTML = createGameCard(data);
}

// ============================================================================
// GAME CARD
// ============================================================================

function createGameCard(opp) {
    const ta = opp.total_analysis || {};
    const ml = opp.moneyline_analysis || {};
    const gi = opp.game_info || {};

    const homeWinPct = opp.home_win_probability ?? 50;
    const awayWinPct = opp.away_win_probability ?? 50;

    const gameTimeStr = gi.game_time
        ? new Date(gi.game_time).toLocaleString('fr-FR', { weekday: 'short', hour: '2-digit', minute: '2-digit' })
        : '';

    // Total recommendation
    const totalRec = ta.recommendation || 'NO_LINE';
    const totalRecClass = totalRec === 'OVER' ? 'rec-over' : totalRec === 'UNDER' ? 'rec-under' : 'rec-skip';

    // ML recommendation
    const mlRec = ml.recommendation || 'NO_LINE';
    let mlRecClass = 'rec-skip';
    let mlLabel = mlRec;
    if (mlRec === 'HOME') {
        mlRecClass = 'rec-home';
        mlLabel = `⌂ ${opp.home_team?.split(' ').slice(-1)[0]?.toUpperCase() || 'HOME'}`;
    } else if (mlRec === 'AWAY') {
        mlRecClass = 'rec-away';
        mlLabel = `✈ ${opp.away_team?.split(' ').slice(-1)[0]?.toUpperCase() || 'AWAY'}`;
    }

    // Edge color
    const totalEdgeClass = ta.edge >= 8 ? 'edge-high' : ta.edge >= 4 ? 'edge-medium' : 'edge-low';
    const mlEdgeClass = ml.best_edge >= 8 ? 'edge-high' : ml.best_edge >= 4 ? 'edge-medium' : 'edge-low';

    // Home stats
    const hs = opp.home_stats || {};
    const as_ = opp.away_stats || {};

    return `
    <div class="game-card">
      <!-- HEADER -->
      <div class="game-header">
        <div class="teams-display">
          <div class="team-name">${opp.home_team || '?'}</div>
          <div class="vs-badge">VS</div>
          <div class="team-name">${opp.away_team || '?'}</div>
        </div>
        <div class="game-time">${gameTimeStr}</div>
      </div>

      <!-- PREDICTIONS -->
      <div class="predictions-grid">
        <!-- Total -->
        <div class="pred-section">
          <div class="pred-section-label">Total de points</div>
          <div class="pred-main-value">${opp.predicted_total ?? '—'}</div>
          <div class="pred-sub">
            ${ta.bookmaker_line ? `Ligne: <strong>${ta.bookmaker_line}</strong> · Δ ${ta.diff_from_line > 0 ? '+' : ''}${ta.diff_from_line ?? ''}` : 'Pas de ligne disponible'}
          </div>
          <div class="pred-sub" style="margin-top:6px; font-size:10px; color: var(--muted);">
            Formule: ${opp.formula_total ?? '—'} · CI: [${opp.total_ci?.lower ?? '—'}, ${opp.total_ci?.upper ?? '—'}]
          </div>
        </div>

        <!-- Score prédit -->
        <div class="pred-section">
          <div class="pred-section-label">Score prédit</div>
          <div class="pred-main-value" style="font-size:26px;">
            ${opp.predicted_home_score ?? '—'} – ${opp.predicted_away_score ?? '—'}
          </div>
          <div class="pred-sub">${opp.home_team?.split(' ').slice(-1)[0]} – ${opp.away_team?.split(' ').slice(-1)[0]}</div>
        </div>

        <!-- Moneyline -->
        <div class="pred-section">
          <div class="pred-section-label">Moneyline</div>
          <div class="pred-main-value" style="font-size:22px;">
            ${homeWinPct}% – ${awayWinPct}%
          </div>
          <div class="pred-sub">
            ${ml.home_implied_prob ? `Implied: ${ml.home_implied_prob}% – ${ml.away_implied_prob}%` : ''}
          </div>
          ${ml.home_ml ? `<div class="pred-sub" style="margin-top:4px;">Cotes: <strong>${ml.home_ml > 0 ? '+' : ''}${ml.home_ml}</strong> / <strong>${ml.away_ml > 0 ? '+' : ''}${ml.away_ml}</strong></div>` : ''}
        </div>
      </div>

      <!-- OPPORTUNITIES -->
      <div class="opps-row">
        <!-- Total opportunity -->
        <div class="opp-cell">
          <div class="rec-badge ${totalRecClass}">
            ${totalRec === 'NO_LINE' ? 'PAS DE LIGNE' : totalRec}
            ${ta.bookmaker_line ? ` ${ta.bookmaker_line}` : ''}
          </div>
          ${ta.recommendation && ta.recommendation !== 'NO_LINE' ? `
          <div class="opp-details">
            <div class="opp-detail-label">TOTAL O/U</div>
            <div class="opp-detail-value ${totalEdgeClass}">
              Edge: ${ta.edge > 0 ? '+' : ''}${ta.edge ?? 0}% · Kelly: ${ta.kelly_criterion ?? 0}%
            </div>
            <div class="opp-detail-label">${ta.over_probability}% OVER · ${ta.under_probability}% UNDER · ${ta.confidence}</div>
          </div>
          ` : `<div style="font-family: var(--mono); font-size: 11px; color: var(--muted);">TOTAUX — ${ta.bookmaker_line ? 'Pas d\'edge' : 'Ligne indisponible'}</div>`}
        </div>

        <!-- Moneyline opportunity -->
        <div class="opp-cell">
          <div class="rec-badge ${mlRecClass}">${mlLabel}</div>
          ${ml.recommendation && ml.recommendation !== 'NO_LINE' && ml.recommendation !== 'SKIP' ? `
          <div class="opp-details">
            <div class="opp-detail-label">MONEYLINE</div>
            <div class="opp-detail-value ${mlEdgeClass}">
              Edge: ${ml.best_edge > 0 ? '+' : ''}${ml.best_edge ?? 0}% · Kelly: ${ml.kelly_criterion ?? 0}%
            </div>
            <div class="opp-detail-label">Modèle vs Implied · ${ml.confidence}</div>
          </div>
          ` : `<div style="font-family: var(--mono); font-size: 11px; color: var(--muted);">MONEYLINE — ${ml.home_ml ? 'Pas d\'edge' : 'Cotes indisponibles'}</div>`}
        </div>
      </div>

      <!-- TEAM STATS -->
      <div class="team-stats-row">
        <div class="team-stats-cell">
          <div class="team-stats-name">⌂ ${opp.home_team?.toUpperCase() || 'HOME'}</div>
          <div class="team-stats-metrics">
            <div class="team-metric">
              <div class="team-metric-val">${hs.adj_oe?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">AdjOE</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${hs.adj_de?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">AdjDE</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${hs.adj_tempo?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">Tempo</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${hs.wins ?? '?'}-${hs.losses ?? '?'}</div>
              <div class="team-metric-lbl">W-L</div>
            </div>
          </div>
        </div>
        <div class="team-stats-cell">
          <div class="team-stats-name">✈ ${opp.away_team?.toUpperCase() || 'AWAY'}</div>
          <div class="team-stats-metrics">
            <div class="team-metric">
              <div class="team-metric-val">${as_.adj_oe?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">AdjOE</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${as_.adj_de?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">AdjDE</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${as_.adj_tempo?.toFixed(1) ?? '—'}</div>
              <div class="team-metric-lbl">Tempo</div>
            </div>
            <div class="team-metric">
              <div class="team-metric-val">${as_.wins ?? '?'}-${as_.losses ?? '?'}</div>
              <div class="team-metric-lbl">W-L</div>
            </div>
          </div>
        </div>
      </div>

    </div>`;
}

// ============================================================================
// HELPERS
// ============================================================================

function setLoading(active, text = 'Analyse en cours — XGBoost + BartTorvik') {
    const loadingDiv = document.getElementById('loadingDiv');
    const loadingText = loadingDiv.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = text;

    document.getElementById('loadingDiv').classList.toggle('hidden', !active);
    document.getElementById('btnScan').disabled = active;
    document.getElementById('btnPredict').disabled = active;
}

function clearResults() {
    document.getElementById('resultsDiv').innerHTML = '';
    document.getElementById('statsBar').classList.add('hidden');
}

function showError(msg) {
    const div = document.getElementById('errorDiv');
    div.textContent = msg;
    div.classList.remove('hidden');
}

function hideError() {
    document.getElementById('errorDiv').classList.add('hidden');
}