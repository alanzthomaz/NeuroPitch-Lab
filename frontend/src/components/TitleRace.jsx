// frontend/src/components/TitleRace.jsx
import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';

export default function TitleRace() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [numSims, setNumSims] = useState(2500);
  const [condition, setCondition] = useState(true);
  const [search, setSearch] = useState('');

  const fetchSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/simulate?num_sims=${numSims}&condition=${condition}`);
      if (!response.ok) {
        throw new Error("Failed to run Monte Carlo simulations");
      }
      const res = await response.json();
      setData(res.title_odds || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulation();
  }, [numSims, condition]);

  const filteredTeams = data.filter(t => 
    t.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Simulation Controls Panel */}
      <div className="glass-card" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', marginBottom: '0.5rem' }}>
            🔮 MONTE CARLO TOURNAMENT SIMULATOR
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Simulate all remaining matches in the bracket to compute championship advancement odds.
          </p>
        </div>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'center' }}>
          
          {/* Conditioning Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Condition on live results:</span>
            <button 
              className={`nav-item ${condition ? 'active' : ''}`}
              onClick={() => setCondition(!condition)}
              style={{ border: '1px solid rgba(255,255,255,0.1)', padding: '0.4rem 1rem' }}
            >
              {condition ? 'ON (Lock Played)' : 'OFF (Start Scratch)'}
            </button>
          </div>

          {/* Num Sims Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Simulations:</span>
            <select 
              value={numSims} 
              onChange={(e) => setNumSims(Number(e.target.value))}
              style={{
                background: '#0d1423',
                color: '#fff',
                border: '1px solid var(--border-glow)',
                padding: '0.4rem 1rem',
                borderRadius: '6px',
                fontFamily: 'var(--font-display)',
                fontSize: '0.8rem',
                cursor: 'pointer'
              }}
            >
              <option value={1000}>1,000 runs</option>
              <option value={2500}>2,500 runs</option>
              <option value={5000}>5,000 runs</option>
              <option value={10000}>10,000 runs</option>
            </select>
          </div>

          <button className="btn-cyber" onClick={fetchSimulation} disabled={loading}>
            {loading ? 'RUNNING...' : 'RE-RUN SIM'}
          </button>
        </div>
      </div>

      {/* Main Results Board */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
          <h3 className="card-title" style={{ margin: 0 }}>🏆 TITLE RACE LEADERBOARD</h3>
          
          <input 
            type="text" 
            placeholder="Search team..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-glow)',
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              color: '#fff',
              outline: 'none',
              width: '250px',
              fontSize: '0.9rem',
              transition: 'border 0.3s'
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--cyan)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--border-glow)'}
          />
        </div>

        {error && (
          <div style={{ color: 'var(--pink)', padding: '1rem', background: 'rgba(255,0,127,0.1)', borderRadius: '8px', border: '1px solid rgba(255,0,127,0.2)' }}>
            Error: {error}
          </div>
        )}

        {loading && data.length === 0 ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
            <div style={{ fontFamily: 'var(--font-display)', animation: 'pulse-glow 1.5s infinite alternate', fontSize: '1.2rem', color: 'var(--cyan)' }}>
              SIMULATING WORLD CUP BRACKETS...
            </div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="custom-table">
              <thead>
                <tr>
                  <th style={{ width: '60px' }}>RANK</th>
                  <th>TEAM</th>
                  <th style={{ width: '80px', textAlign: 'center' }}>ELO</th>
                  <th style={{ width: '90px', textAlign: 'center' }}>R32 %</th>
                  <th style={{ width: '90px', textAlign: 'center' }}>R16 %</th>
                  <th style={{ width: '90px', textAlign: 'center' }}>QF %</th>
                  <th style={{ width: '90px', textAlign: 'center' }}>SF %</th>
                  <th style={{ width: '90px', textAlign: 'center' }}>FINAL %</th>
                  <th style={{ width: '220px' }}>CHAMPION ODDS</th>
                </tr>
              </thead>
              <tbody>
                {filteredTeams.map((team, idx) => (
                  <tr key={team.slug}>
                    <td style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: idx < 3 ? 'var(--cyan)' : 'var(--text-secondary)' }}>
                      #{idx + 1}
                    </td>
                    <td style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '1.1rem' }}>⚽</span>
                      {team.name}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'var(--font-display)', fontSize: '0.85rem' }}>{team.rating}</td>
                    <td style={{ textAlign: 'center', color: team.r32 === 100 ? 'var(--cyan)' : '#fff' }}>{team.r32}%</td>
                    <td style={{ textAlign: 'center', color: team.r16 === 100 ? 'var(--cyan)' : '#fff' }}>{team.r16}%</td>
                    <td style={{ textAlign: 'center', color: team.qf === 100 ? 'var(--cyan)' : '#fff' }}>{team.qf}%</td>
                    <td style={{ textAlign: 'center', color: team.sf === 100 ? 'var(--cyan)' : '#fff' }}>{team.sf}%</td>
                    <td style={{ textAlign: 'center', color: team.final === 100 ? 'var(--cyan)' : '#fff' }}>{team.final}%</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div className="progress-container" style={{ flex: 1, height: '6px' }}>
                          <div className="progress-bar-fill" style={{ width: `${team.win}%`, background: idx === 0 ? 'linear-gradient(90deg, #ff007f, #f5a623)' : 'var(--cyan)' }} />
                        </div>
                        <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, minWidth: '45px', textAlign: 'right', color: team.win > 0 ? 'var(--cyan)' : 'var(--text-secondary)' }}>
                          {team.win}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
                {filteredTeams.length === 0 && (
                  <tr>
                    <td colSpan="9" style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '2rem' }}>
                      No teams match your search filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
