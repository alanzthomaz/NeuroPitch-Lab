// frontend/src/components/PredictionHistory.jsx
import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';

export default function PredictionHistory() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/history`);
      if (!response.ok) {
        throw new Error("Failed to load prediction history");
      }
      const res = await response.json();
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Accuracy Header Panel */}
      {data && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem' }}>
          
          {/* Accuracy Card */}
          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <div style={{ width: '60px', height: '60px' }}>
              <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%' }}>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831" fill="none" stroke="var(--cyan)" strokeDasharray={`${data.accuracy}, 100`} strokeWidth="3" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-display)' }}>MODEL ACCURACY</div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--cyan)', fontSize: '1.75rem' }}>
                {data.accuracy}%
              </div>
            </div>
          </div>

          {/* Record Count Card */}
          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <div style={{ fontSize: '2rem' }}>📊</div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-display)' }}>TOTAL MATCHES EVALUATED</div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: '#fff', fontSize: '1.75rem' }}>
                {data.correct_count} / {data.total_count}
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--pink)', padding: '1rem', background: 'rgba(255,0,127,0.1)', borderRadius: '8px', border: '1px solid rgba(255,0,127,0.2)' }}>
          Error: {error}
        </div>
      )}

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '6rem' }}>
          <div style={{ fontFamily: 'var(--font-display)', animation: 'pulse-glow 1.5s infinite alternate', fontSize: '1.2rem', color: 'var(--cyan)' }}>
            RETRIEVING COMPLETED PREDICTIONS RECORD...
          </div>
        </div>
      ) : data ? (
        <div className="glass-card">
          <h3 className="card-title">📜 HISTORICAL PREDICTION LOG</h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="custom-table">
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>ROUND</th>
                  <th>MATCHUP</th>
                  <th style={{ textAlign: 'center' }}>ACTUAL SCORE</th>
                  <th style={{ textAlign: 'center' }}>MODEL'S PICK (PROBS)</th>
                  <th style={{ width: '80px', textAlign: 'center' }}>RESULT</th>
                </tr>
              </thead>
              <tbody>
                {data.predictions.map((p, idx) => (
                  <tr key={idx}>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{p.date}</td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-display)' }}>
                      {p.round.toUpperCase()}
                    </td>
                    <td style={{ fontWeight: 600 }}>
                      {p.team1_name} vs {p.team2_name}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'var(--font-display)', fontWeight: 700 }}>
                      {p.score}
                    </td>
                    <td style={{ textAlign: 'center', fontSize: '0.85rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', fontFamily: 'var(--font-display)' }}>
                        <span style={{ color: 'var(--cyan)' }}>A: {p.prediction.win1}%</span>
                        <span style={{ color: 'var(--gold)' }}>D: {p.prediction.draw}%</span>
                        <span style={{ color: 'var(--pink)' }}>B: {p.prediction.win2}%</span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {p.correct ? (
                        <span style={{ color: 'var(--cyan)', fontWeight: 800, textShadow: '0 0 8px rgba(0,242,254,0.4)' }}>✅</span>
                      ) : (
                        <span style={{ color: 'var(--pink)', fontWeight: 800, textShadow: '0 0 8px rgba(255,0,127,0.4)' }}>❌</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}
