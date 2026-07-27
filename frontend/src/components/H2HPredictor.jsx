// frontend/src/components/H2HPredictor.jsx
import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';

const TEAMS_LIST = [
  "argentina", "france", "spain", "brazil", "england", "portugal", "netherlands", 
  "germany", "belgium", "italy", "colombia", "uruguay", "croatia", "morocco", 
  "switzerland", "usa", "mexico", "japan", "senegal", "denmark", "ecuador", 
  "australia", "south-korea", "iran", "poland", "canada", "serbia", "wales", 
  "ghana", "tunisia", "ivory-coast", "nigeria", "saudi-arabia", "qatar", "egypt", 
  "algeria", "scotland", "cameroon", "paraguay", "venezuela", "chile", "peru", 
  "czech-republic", "bosnia-and-herzegovina", "south-africa", "new-zealand", 
  "panama", "jamaica", "honduras", "jordan", "haiti", "el-salvador", "guatemala"
];

export default function H2HPredictor() {
  const [t1, setT1] = useState("france");
  const [t2, setT2] = useState("spain");
  const [venue, setVenue] = useState("neutral"); // team1, team2, or neutral
  const [statWeight, setStatWeight] = useState(60); // percentages
  
  const [prediction, setPrediction] = useState(null);
  const [t1Profile, setT1Profile] = useState(null);
  const [t2Profile, setT2Profile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchH2H = async () => {
    if (t1 === t2) {
      setError("Please select two different teams.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const homeTeam = venue === 'team1' ? t1 : (venue === 'team2' ? t2 : '');
      const sW = statWeight / 100.0;
      const aW = (100 - statWeight) / 100.0;
      
      const predUrl = `${API_BASE_URL}/predict?team1=${t1}&team2=${t2}&home_team=${homeTeam}&stat_weight=${sW}&ann_weight=${aW}`;
      const [predRes, t1Res, t2Res] = await Promise.all([
        fetch(predUrl).then(r => r.json()),
        fetch(`${API_BASE_URL}/team/${t1}`).then(r => r.json()),
        fetch(`${API_BASE_URL}/team/${t2}`).then(r => r.json())
      ]);
      
      setPrediction(predRes);
      setT1Profile(t1Res);
      setT2Profile(t2Res);
    } catch (err) {
      setError("Failed to fetch predictions or profiles.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchH2H();
  }, [t1, t2, venue, statWeight]);

  const getConfidenceText = (p) => {
    const margin = Math.abs(p.ensemble_home - p.ensemble_away);
    const draw = p.ensemble_draw;
    
    if (margin > 0.4) return { label: "EXTREME CONFIDENCE", color: "var(--cyan)", val: 92 };
    if (margin > 0.25) return { label: "HIGH CONFIDENCE", color: "var(--cyan)", val: 78 };
    if (margin > 0.1) return { label: "MODERATE", color: "var(--gold)", val: 56 };
    return { label: "LOW / VOLATILE", color: "var(--pink)", val: 32 };
  };

  const capitalize = (s) => s.replace("-", " ").replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Selection Control Panel */}
      <div className="glass-card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', alignItems: 'center' }}>
        
        {/* Team 1 Select */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', marginBottom: '0.5rem' }}>TEAM A</label>
          <select 
            value={t1} 
            onChange={(e) => setT1(e.target.value)}
            style={{ width: '100%', padding: '0.6rem', background: '#0d1423', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '8px', fontSize: '0.9rem' }}
          >
            {TEAMS_LIST.map(t => <option key={t} value={t}>{capitalize(t)}</option>)}
          </select>
        </div>

        {/* Team 2 Select */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', marginBottom: '0.5rem' }}>TEAM B</label>
          <select 
            value={t2} 
            onChange={(e) => setT2(e.target.value)}
            style={{ width: '100%', padding: '0.6rem', background: '#0d1423', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '8px', fontSize: '0.9rem' }}
          >
            {TEAMS_LIST.map(t => <option key={t} value={t}>{capitalize(t)}</option>)}
          </select>
        </div>

        {/* Venue Select */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', marginBottom: '0.5rem' }}>VENUE ADVANTAGE</label>
          <select 
            value={venue} 
            onChange={(e) => setVenue(e.target.value)}
            style={{ width: '100%', padding: '0.6rem', background: '#0d1423', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '8px', fontSize: '0.9rem' }}
          >
            <option value="neutral">Neutral Venue</option>
            <option value="team1">Team A at Home</option>
            <option value="team2">Team B at Home</option>
          </select>
        </div>

        {/* Weights Slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--cyan)', fontFamily: 'var(--font-display)', marginBottom: '0.5rem' }}>
            <span>MODEL BLEND</span>
            <span>{statWeight}% STAT / {100 - statWeight}% ANN</span>
          </div>
          <input 
            type="range" 
            min="0" 
            max="100" 
            value={statWeight} 
            onChange={(e) => setStatWeight(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--cyan)', cursor: 'pointer' }}
          />
        </div>
      </div>

      {error && (
        <div style={{ color: 'var(--pink)', padding: '1rem', background: 'rgba(255,0,127,0.1)', borderRadius: '8px', border: '1px solid rgba(255,0,127,0.2)' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="glass-card" style={{ display: 'flex', justifyContent: 'center', padding: '6rem' }}>
          <div style={{ fontFamily: 'var(--font-display)', animation: 'pulse-glow 1.5s infinite alternate', fontSize: '1.2rem', color: 'var(--cyan)' }}>
            ANALYZING MATCH MATRIX...
          </div>
        </div>
      ) : prediction && t1Profile && t2Profile ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem' }}>
          
          {/* Main Ensemble Probabilities Card */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <h3 className="card-title">⚔️ MATCH PROBABILITY BREAKDOWN</h3>
            
            {/* Blended Predictions */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              
              {/* Home Team Probability */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <span style={{ fontWeight: 600 }}>{prediction.team1_name} Win</span>
                  <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>{(prediction.ensemble_home*100).toFixed(1)}%</span>
                </div>
                <div className="progress-container">
                  <div className="progress-bar-fill" style={{ width: `${prediction.ensemble_home*100}%` }} />
                </div>
              </div>
              
              {/* Draw Probability */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <span style={{ fontWeight: 600 }}>Draw</span>
                  <span style={{ color: 'var(--gold)', fontWeight: 700 }}>{(prediction.ensemble_draw*100).toFixed(1)}%</span>
                </div>
                <div className="progress-container">
                  <div className="progress-bar-fill" style={{ width: `${prediction.ensemble_draw*100}%`, background: 'var(--gold)', boxShadow: '0 0 8px var(--gold)' }} />
                </div>
              </div>

              {/* Away Team Probability */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <span style={{ fontWeight: 600 }}>{prediction.team2_name} Win</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 700 }}>{(prediction.ensemble_away*100).toFixed(1)}%</span>
                </div>
                <div className="progress-container">
                  <div className="progress-bar-fill" style={{ width: `${prediction.ensemble_away*100}%`, background: 'var(--pink)', boxShadow: '0 0 8px var(--pink)' }} />
                </div>
              </div>
            </div>

            {/* Expected Goals section */}
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>EXPECTED GOALS A</div>
                <div style={{ fontSize: '1.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)', fontWeight: 800 }}>{prediction.expectedGoalsA.toFixed(2)}</div>
              </div>
              <div style={{ borderLeft: '1px solid rgba(255,255,255,0.08)' }} />
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>EXPECTED GOALS B</div>
                <div style={{ fontSize: '1.75rem', fontFamily: 'var(--font-display)', color: 'var(--pink)', fontWeight: 800 }}>{prediction.expectedGoalsB.toFixed(2)}</div>
              </div>
            </div>
            
            {/* Confidence Gauge */}
            {(() => {
              const conf = getConfidenceText(prediction);
              return (
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem 1rem', background: 'rgba(0,242,254,0.04)', border: '1px solid rgba(0,242,254,0.1)', borderRadius: '8px' }}>
                  <div style={{ width: '40px', height: '40px' }}>
                    <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%' }}>
                      <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
                      <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke={conf.color} strokeDasharray={`${conf.val}, 100`} strokeWidth="3" strokeLinecap="round" />
                    </svg>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-display)' }}>CONFIDENCE SCORE</div>
                    <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: conf.color, fontSize: '0.9rem' }}>{conf.label} ({conf.val}%)</div>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Model Comparisons Card */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <h3 className="card-title">🤖 MODEL COMPARISON</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 700, paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <span>OUTCOME</span>
                <span style={{ textAlign: 'center' }}>STAT ENGINE</span>
                <span style={{ textAlign: 'center' }}>ANN LAYER</span>
              </div>
              
              {/* Home Win comparing */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', fontSize: '0.9rem' }}>
                <span style={{ fontWeight: 600 }}>Home Win</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)' }}>{(prediction.winA*100).toFixed(0)}%</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>{(prediction.ann_prob_home*100).toFixed(0)}%</span>
              </div>

              {/* Draw comparing */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', fontSize: '0.9rem' }}>
                <span style={{ fontWeight: 600 }}>Draw</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)' }}>{(prediction.draw*100).toFixed(0)}%</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)', color: 'var(--gold)' }}>{(prediction.ann_prob_draw*100).toFixed(0)}%</span>
              </div>

              {/* Away Win comparing */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', fontSize: '0.9rem' }}>
                <span style={{ fontWeight: 600 }}>Away Win</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)' }}>{(prediction.winB*100).toFixed(0)}%</span>
                <span style={{ textAlign: 'center', fontFamily: 'var(--font-display)', color: 'var(--pink)' }}>{(prediction.ann_prob_away*100).toFixed(0)}%</span>
              </div>
            </div>

            {/* Team details comparison sheet */}
            <div style={{ marginTop: '1rem' }}>
              <h4 style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--text-secondary)', marginBottom: '0.75rem', letterSpacing: '1px' }}>
                STATS SHEET COMPARISON
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {/* FIFA Rank Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>#{t1Profile.fifa_rank}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>FIFA Ranking</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 600 }}>#{t2Profile.fifa_rank}</span>
                </div>
                {/* Elo Rating Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>{t1Profile.elo}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>Elo Strength</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 600 }}>{t2Profile.elo}</span>
                </div>
                {/* Recent Form Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>{t1Profile.form.toFixed(2)}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>Form (Last 5 pts/m)</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 600 }}>{t2Profile.form.toFixed(2)}</span>
                </div>
                {/* Goals Scored Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>{t1Profile.avg_goals_scored.toFixed(1)}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>Goals Scored (L5 avg)</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 600 }}>{t2Profile.avg_goals_scored.toFixed(1)}</span>
                </div>
                {/* Win Rate Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>{t1Profile.win_rate}%</span>
                  <span style={{ color: 'var(--text-secondary)' }}>Win Rate (L10 avg)</span>
                  <span style={{ color: 'var(--pink)', fontWeight: 600 }}>{t2Profile.win_rate}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
