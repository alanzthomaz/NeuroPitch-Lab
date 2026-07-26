import React, { useState, useEffect, useRef } from 'react';

const FEATURE_CONFIGS = {
  elo_home: { min: 1000, max: 2500, step: 10, label: "Elo Home", default: 1500 },
  elo_away: { min: 1000, max: 2500, step: 10, label: "Elo Away", default: 1500 },
  rank_home: { min: 1, max: 200, step: 1, label: "FIFA Rank Home", default: 50 },
  rank_away: { min: 1, max: 200, step: 1, label: "FIFA Rank Away", default: 50 },
  rank_diff: { min: -150, max: 150, step: 1, label: "Rank Difference", default: 0 },
  form_home: { min: 0, max: 3.0, step: 0.05, label: "Recent Form Home", default: 1.5 },
  form_away: { min: 0, max: 3.0, step: 0.05, label: "Recent Form Away", default: 1.5 },
  gs_home: { min: 0, max: 5.0, step: 0.1, label: "Avg Goals Scored Home", default: 1.5 },
  gs_away: { min: 0, max: 5.0, step: 0.1, label: "Avg Goals Scored Away", default: 1.5 },
  gc_home: { min: 0, max: 5.0, step: 0.1, label: "Avg Goals Conceded Home", default: 1.5 },
  gc_away: { min: 0, max: 5.0, step: 0.1, label: "Avg Goals Conceded Away", default: 1.5 },
  win_rate_home: { min: 0, max: 1.0, step: 0.01, label: "Win Rate Home", default: 0.5 },
  win_rate_away: { min: 0, max: 1.0, step: 0.01, label: "Win Rate Away", default: 0.5 },
  h2h_win_rate_home: { min: 0, max: 1.0, step: 0.01, label: "H2H Win Rate Home", default: 0.33 },
  h2h_win_rate_away: { min: 0, max: 1.0, step: 0.01, label: "H2H Win Rate Away", default: 0.33 },
  home_adv: { min: 0, max: 1, step: 1, label: "Home Advantage", default: 0 },
  rest_home: { min: 0, max: 15, step: 1, label: "Rest Days Home", default: 4 },
  rest_away: { min: 0, max: 15, step: 1, label: "Rest Days Away", default: 4 },
  poss_home: { min: 0.2, max: 0.8, step: 0.01, label: "Possession Home", default: 0.5 },
  poss_away: { min: 0.2, max: 0.8, step: 0.01, label: "Possession Away", default: 0.5 },
  shots_home: { min: 0, max: 40, step: 1, label: "Shots Home", default: 10 },
  shots_away: { min: 0, max: 40, step: 1, label: "Shots Away", default: 10 },
  player_avail_home: { min: 0, max: 1.0, step: 0.05, label: "Player Avail Home", default: 1.0 },
  player_avail_away: { min: 0, max: 1.0, step: 0.05, label: "Player Avail Away", default: 1.0 },
  stat_prob_home: { min: 0, max: 1.0, step: 0.01, label: "Stat Win Prob Home", default: 0.33 },
  stat_prob_draw: { min: 0, max: 1.0, step: 0.01, label: "Stat Win Prob Draw", default: 0.33 },
  stat_prob_away: { min: 0, max: 1.0, step: 0.01, label: "Stat Win Prob Away", default: 0.33 },
  stat_xg_home: { min: 0, max: 5.0, step: 0.1, label: "Stat xG Home", default: 1.5 },
  stat_xg_away: { min: 0, max: 5.0, step: 0.1, label: "Stat xG Away", default: 1.5 },
  age_home: { min: 18, max: 40, step: 0.5, label: "Avg Age Home", default: 26.0 },
  age_away: { min: 18, max: 40, step: 0.5, label: "Avg Age Away", default: 26.0 },
  market_value_home: { min: 0, max: 1000, step: 10, label: "Market Value Home ($M)", default: 150 },
  market_value_away: { min: 0, max: 1000, step: 10, label: "Market Value Away ($M)", default: 150 },
  def_rating_home: { min: 0, max: 100, step: 1, label: "Def Rating Home", default: 75 },
  def_rating_away: { min: 0, max: 100, step: 1, label: "Def Rating Away", default: 75 },
  off_rating_home: { min: 0, max: 100, step: 1, label: "Off Rating Home", default: 75 },
  off_rating_away: { min: 0, max: 100, step: 1, label: "Off Rating Away", default: 75 },
  exp_home: { min: 0, max: 12.0, step: 0.1, label: "Exp Rating Home", default: 6.0 },
  exp_away: { min: 0, max: 12.0, step: 0.1, label: "Exp Rating Away", default: 6.0 }
};

export default function SandboxView({
  refMatch = { team1: 'france', team2: 'spain', home_team: 'france' },
  activeFeatures = [],
  refFeatures = {},
  onActivationsUpdate,
  isTraining = false,
  customWeightOverrides = {},
  onClearOverrides,
  explanation = [],
  onExplanationUpdate
}) {
  const [featuresOverride, setFeaturesOverride] = useState({});
  const [targetClass, setTargetClass] = useState(0); // 0: Home Win, 1: Draw, 2: Away Win
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef(null);

  // Helper to format team name
  const formatTeamName = (slug) => {
    if (!slug) return '';
    return slug
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Synchronize overrides when refFeatures or activeFeatures change
  useEffect(() => {
    const initialOverrides = {};
    activeFeatures.forEach(feat => {
      if (refFeatures[feat] !== undefined) {
        initialOverrides[feat] = refFeatures[feat];
      } else {
        initialOverrides[feat] = FEATURE_CONFIGS[feat]?.default ?? 0.5;
      }
    });
    setFeaturesOverride(initialOverrides);
  }, [refFeatures, activeFeatures]);

  // Run prediction request
  const fetchSandboxPrediction = async (overrides, currentClass) => {
    if (activeFeatures.length === 0 || isTraining) return;
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/sandbox/predict?target_class=${currentClass}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features_override: overrides,
          weight_overrides: customWeightOverrides
        })
      });
      if (response.ok) {
        const data = await response.json();
        setPrediction(data.prediction);
        if (onExplanationUpdate) {
          onExplanationUpdate(data.explanation || []);
        }
        if (onActivationsUpdate && data.activations) {
          onActivationsUpdate(data.activations);
        }
      }
    } catch (err) {
      console.error("Sandbox prediction failed:", err);
    } finally {
      setLoading(false);
    }
  };

  // Debounced/Triggered prediction on override updates
  useEffect(() => {
    if (Object.keys(featuresOverride).length === 0) return;
    
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      fetchSandboxPrediction(featuresOverride, targetClass);
    }, 150);

    return () => clearTimeout(timerRef.current);
  }, [featuresOverride, targetClass, customWeightOverrides]);

  const handleSliderChange = (feature, value) => {
    setFeaturesOverride(prev => ({
      ...prev,
      [feature]: parseFloat(value)
    }));
  };

  const handleReset = () => {
    const resetOverrides = {};
    activeFeatures.forEach(feat => {
      resetOverrides[feat] = refFeatures[feat] !== undefined ? refFeatures[feat] : (FEATURE_CONFIGS[feat]?.default ?? 0.5);
    });
    setFeaturesOverride(resetOverrides);
    if (onClearOverrides) onClearOverrides();
    if (onExplanationUpdate) onExplanationUpdate([]);
  };

  if (activeFeatures.length === 0) {
    return (
      <div className="glass-card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        ⚙️ Neural network configuration has no selected features. Compile or load a model to activate the sandbox.
      </div>
    );
  }

  const homePct = prediction ? (prediction.home * 100).toFixed(1) : "33.3";
  const drawPct = prediction ? (prediction.draw * 100).toFixed(1) : "33.3";
  const awayPct = prediction ? (prediction.away * 100).toFixed(1) : "33.3";

  return (
    <div className="slide-up" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', marginTop: '1.5rem' }}>
      
      {/* LEFT COLUMN: Feature Override Sliders */}
      <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 className="card-title" style={{ margin: 0 }}>🎛️ SANDBOX INFERENCE SLIDERS</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '0.2rem' }}>
              Override match variables in real-time to watch signals flow through the network.
            </p>
          </div>
          <button onClick={handleReset} className="nav-item" style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}>
            RESET
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', maxHeight: '420px', overflowY: 'auto', paddingRight: '0.5rem' }}>
          {activeFeatures.map(feat => {
            const config = FEATURE_CONFIGS[feat] || { min: 0, max: 1, step: 0.01, label: feat };
            const currentVal = featuresOverride[feat] !== undefined ? featuresOverride[feat] : (config.default ?? 0.5);
            return (
              <div key={feat} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-display)' }}>{config.label}</span>
                  <span style={{ fontWeight: 800 }}>{currentVal.toFixed(config.step < 1 ? 2 : 0)}</span>
                </div>
                <input
                  type="range"
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  value={currentVal}
                  onChange={(e) => handleSliderChange(feat, e.target.value)}
                  disabled={isTraining}
                  className="time-machine-slider"
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* RIGHT COLUMN: Realtime Predictions & Attributions */}
      <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div>
          <h3 className="card-title" style={{ margin: 0 }}>📊 PREDICTIVE OUTPUTS & ATTRIBUTIONS</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '0.2rem' }}>
            Simulated outcome probabilities and neural attributions for {formatTeamName(refMatch.team1)} vs {formatTeamName(refMatch.team2)}.
          </p>
        </div>

        {/* Prediction Results Meters */}
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div style={{ flex: 1, textAlign: 'center', padding: '0.75rem', background: 'rgba(0,114,255,0.06)', border: '1px solid rgba(0,114,255,0.15)', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>HOME WIN</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--cyan)', fontFamily: 'var(--font-display)', marginTop: '0.2rem' }}>
              {homePct}%
            </div>
          </div>
          <div style={{ flex: 1, textAlign: 'center', padding: '0.75rem', background: 'rgba(255,193,7,0.06)', border: '1px solid rgba(255,193,7,0.15)', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>DRAW</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--gold)', fontFamily: 'var(--font-display)', marginTop: '0.2rem' }}>
              {drawPct}%
            </div>
          </div>
          <div style={{ flex: 1, textAlign: 'center', padding: '0.75rem', background: 'rgba(255,0,127,0.06)', border: '1px solid rgba(255,0,127,0.15)', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>AWAY WIN</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--pink)', fontFamily: 'var(--font-display)', marginTop: '0.2rem' }}>
              {awayPct}%
            </div>
          </div>
        </div>

        {/* Attribution Selector & Explanation Chart */}
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)' }}>EXPLAIN ATTRIBUTIONS FOR</span>
            <select
              value={targetClass}
              onChange={(e) => setTargetClass(parseInt(e.target.value))}
              disabled={isTraining}
              style={{
                background: '#070a0f',
                color: '#fff',
                border: '1px solid var(--border-glow)',
                borderRadius: '6px',
                padding: '0.3rem 0.6rem',
                fontSize: '0.8rem',
                fontFamily: 'var(--font-display)'
              }}
            >
              <option value={0}>Home Win</option>
              <option value={1}>Draw</option>
              <option value={2}>Away Win</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '220px', overflowY: 'auto', paddingRight: '0.5rem' }}>
            {explanation.length === 0 ? (
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem', padding: '1rem' }}>
                No attribution data computed. Adjust overrides to calculate.
              </div>
            ) : (
              explanation.map(exp => {
                const config = FEATURE_CONFIGS[exp.feature] || { label: exp.feature };
                // Using integrated gradients attribution score
                const val = exp.attribution;
                const absVal = Math.abs(val);
                const isPositive = val >= 0;
                
                // Scale bar width relative to maximum attribution found
                const maxAttr = Math.max(...explanation.map(e => Math.abs(e.attribution)), 0.001);
                const width = (absVal / maxAttr) * 100;

                return (
                  <div key={exp.feature} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 50px', gap: '1rem', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={config.label}>
                      {config.label}
                    </span>
                    <div style={{ height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          background: isPositive ? 'var(--cyan)' : 'var(--pink)',
                          width: `${width}%`,
                          float: isPositive ? 'left' : 'right',
                          borderRadius: '4px',
                          transition: 'width 0.2s ease'
                        }}
                      />
                    </div>
                    <span style={{ fontSize: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-display)', color: isPositive ? 'var(--cyan)' : 'var(--pink)' }}>
                      {isPositive ? '+' : ''}{val.toFixed(4)}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
