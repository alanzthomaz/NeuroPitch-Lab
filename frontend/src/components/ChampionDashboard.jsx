import React from 'react';

export default function ChampionDashboard({ data, isTraining }) {
  if (!data) {
    return <div className="glass-card" style={{padding:'1rem'}}>No tournament data available. Train a model or adjust settings to see predictions.</div>;
  }

  const { title_odds, confidence } = data;

  return (
    <div className="glass-card" style={{padding:'1rem'}}>
      <h3 className="card-title">🏆 Champion Dashboard</h3>
      <div style={{display:'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap:'1rem', marginTop:'1rem'}}>
        {Object.entries(title_odds).map(([team, odds]) => (
          <div key={team} style={{background:'rgba(255,255,255,0.05)', padding:'0.75rem', borderRadius:'4px', textAlign:'center'}}>
            <div style={{fontSize:'0.9rem', opacity:0.8}}>{team}</div>
            <div style={{fontSize:'1.5rem', fontWeight:'bold', margin:'0.5rem 0'}}>
              {odds.toFixed(1)}%
            </div>
            <div style={{fontSize:'0.8rem', opacity:0.7}}>Win Probability</div>
          </div>
        ))}
      </div>
      {confidence !== undefined && (
        <div style={{marginTop:'1.5rem', paddingTop:'1rem', borderTop:'1px solid var(--border-glow)'}}>
          <div style={{fontSize:'0.9rem', opacity:0.8}}>Model Confidence</div>
          <div style={{fontSize:'1.8rem', fontWeight:'bold', margin:'0.5rem 0'}}>
            {confidence.toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
