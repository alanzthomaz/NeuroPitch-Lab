import React from 'react';

export default function ExplainabilityPanel({ explanation, isTraining }) {
  if (!explanation || explanation.length === 0) {
    return <div className="glass-card" style={{padding:'1rem'}}>No explanation available. Select a match and train a model to see feature contributions.</div>;
  }

  // Find max absolute contribution for scaling bars
  const maxContribution = Math.max(...explanation.map(e => Math.abs(e.contribution)), 0.01);

  return (
    <div className="glass-card" style={{padding:'1rem'}}>
      <h3 className="card-title">🔬 Explainability Panel</h3>
      <div style={{marginTop:'1rem'}}>
        {explanation.map((exp, idx) => {
          const { feature, contribution } = exp;
          const isPositive = contribution >= 0;
          const absContribution = Math.abs(contribution);
          const width = (absContribution / maxContribution) * 100;
          
          return (
            <div key={feature} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span>{feature}</span>
                <span style={{ color: isPositive ? 'var(--cyan)' : 'var(--pink)', fontWeight: 600 }}>
                  {isPositive ? '+' : ''}{contribution.toFixed(3)}
                </span>
              </div>
              <div style={{ width: '100%', height: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '999px', position: 'relative', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    background: isPositive ? 'var(--cyan)' : 'var(--pink)',
                    width: `${width}%`,
                    transition: 'width 0.3s ease'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
