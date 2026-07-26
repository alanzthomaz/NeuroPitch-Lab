// frontend/src/components/BracketSimulator.jsx
import React, { useState, useEffect } from 'react';

export default function BracketSimulator() {
  const [bracket, setBracket] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedRound, setSelectedRound] = useState("Quarter-final");

  const fetchBracket = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/bracket");
      if (!response.ok) {
        throw new Error("Failed to load bracket data");
      }
      const data = await response.json();
      setBracket(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBracket();
  }, []);

  const rounds = [
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final"
  ];

  // Helper to filter matches of the active round
  const matches = bracket.filter(m => {
    if (selectedRound === "Semi-final") {
      return m.round === "Semi-final" || m.round === "Final";
    }
    return m.round === selectedRound;
  });

  return (
    <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Round Selection Tabs */}
      <div className="glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', marginBottom: '0.25rem' }}>
            🏆 TOURNAMENT BRACKET EXPLORER
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Explore actual matches, finished scores, and ensemble predictions of upcoming rounds.
          </p>
        </div>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', background: 'rgba(0,0,0,0.2)', padding: '0.25rem', borderRadius: '8px', border: '1px solid var(--border-glow)' }}>
          {rounds.map(r => (
            <button
              key={r}
              className={`nav-item ${selectedRound === r ? 'active' : ''}`}
              onClick={() => setSelectedRound(r)}
              style={{ padding: '0.4rem 1rem', fontSize: '0.75rem', borderRadius: '6px' }}
            >
              {r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div style={{ color: 'var(--pink)', padding: '1rem', background: 'rgba(255,0,127,0.1)', borderRadius: '8px', border: '1px solid rgba(255,0,127,0.2)' }}>
          Error: {error}
        </div>
      )}

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '6rem' }}>
          <div style={{ fontFamily: 'var(--font-display)', animation: 'pulse-glow 1.5s infinite alternate', fontSize: '1.2rem', color: 'var(--cyan)' }}>
            LOADING BRACKET DATA...
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          {matches.map((m, idx) => {
            const isFinished = m.status && m.status !== 'NS';
            return (
              <div key={idx} className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyBetween: 'space-between', gap: '1rem', borderLeftWidth: '4px', borderLeftColor: isFinished ? 'var(--cyan)' : 'var(--pink)' }}>
                
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-display)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
                  <span>{m.round.toUpperCase()}</span>
                  <span>{m.date}</span>
                </div>

                {/* Team 1 / 2 block */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {/* Team 1 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: m.winner === m.team1 ? 700 : 400, color: m.winner === m.team1 ? '#fff' : (m.winner ? 'var(--text-secondary)' : '#fff') }}>
                      ⚽ {m.team1_name}
                    </span>
                    {isFinished ? (
                      <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: m.winner === m.team1 ? 'var(--cyan)' : 'var(--text-secondary)' }}>
                        {m.score ? m.score.split(" - ")[0] : '-'}
                      </span>
                    ) : (
                      m.prediction && (
                        <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
                          {(m.prediction.home_win*100).toFixed(0)}%
                        </span>
                      )
                    )}
                  </div>
                  
                  {/* Team 2 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: m.winner === m.team2 ? 700 : 400, color: m.winner === m.team2 ? '#fff' : (m.winner ? 'var(--text-secondary)' : '#fff') }}>
                      ⚽ {m.team2_name}
                    </span>
                    {isFinished ? (
                      <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: m.winner === m.team2 ? 'var(--cyan)' : 'var(--text-secondary)' }}>
                        {m.score ? m.score.split(" - ")[1] : '-'}
                      </span>
                    ) : (
                      m.prediction && (
                        <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--pink)' }}>
                          {(m.prediction.away_win*100).toFixed(0)}%
                        </span>
                      )
                    )}
                  </div>
                </div>

                {/* Draw or Expected goals footer for pending games */}
                {!isFinished && m.prediction ? (
                  <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '6px', fontSize: '0.75rem', display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.04)' }}>
                    <span>Draw Probability: <strong style={{ color: 'var(--gold)' }}>{(m.prediction.draw*100).toFixed(0)}%</strong></span>
                    <span>Expected: <strong style={{ color: '#fff' }}>{m.prediction.stat_xg_home.toFixed(1)}–{m.prediction.stat_xg_away.toFixed(1)}</strong></span>
                  </div>
                ) : (
                  isFinished && (
                    <div style={{ background: 'rgba(0,242,254,0.03)', padding: '0.5rem', borderRadius: '6px', fontSize: '0.75rem', textAlign: 'center', color: 'var(--cyan)', fontFamily: 'var(--font-display)' }}>
                      MATCH RESOLVED - WINNER: {m.winner_name.toUpperCase()}
                    </div>
                  )
                )}
              </div>
            );
          })}
          {matches.length === 0 && (
            <div style={{ gridColumn: '1/-1', textAlign: 'center', color: 'var(--text-secondary)', padding: '4rem' }}>
              No matches found for the selected round.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
