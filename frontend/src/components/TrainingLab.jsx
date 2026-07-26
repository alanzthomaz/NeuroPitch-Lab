// frontend/src/components/TrainingLab.jsx
import React, { useMemo } from 'react';

// Custom SVG Line Chart Component
function SVGLineChart({ title, trainValues, valValues, maxEpochs, valColor="var(--cyan)", trainColor="rgba(255,255,255,0.25)" }) {
  const width = 300;
  const height = 150;
  const padding = 25;

  const getPoints = (values) => {
    if (values.length === 0) return "";
    const minVal = Math.min(...values, 0);
    const maxVal = Math.max(...values, 1);
    const valRange = maxVal - minVal || 1.0;

    return values.map((val, idx) => {
      const x = padding + (idx / (maxEpochs - 1 || 1)) * (width - 2 * padding);
      const y = height - padding - ((val - minVal) / valRange) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');
  };

  const trainPoints = getPoints(trainValues);
  const valPoints = getPoints(valValues);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: '#070a0f', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
        <span>{title}</span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span style={{ color: trainColor }}>■ Train</span>
          <span style={{ color: valColor }}>■ Val</span>
        </div>
      </div>
      
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }}>
        {/* Grid lines */}
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.08)" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.08)" />
        
        {/* Train Line */}
        {trainPoints && (
          <polyline fill="none" stroke={trainColor} strokeWidth="1.2" points={trainPoints} />
        )}
        
        {/* Val Line */}
        {valPoints && (
          <polyline fill="none" stroke={valColor} strokeWidth="2.0" points={valPoints} />
        )}
      </svg>
    </div>
  );
}

export default function TrainingLab({
  epochHistory = [],
  confusionMatrix = [],
  rocCurve = {},
  epochs = 30
}) {
  const trainLosses = epochHistory.map(h => h.train_loss);
  const valLosses = epochHistory.map(h => h.val_loss);
  const trainAccs = epochHistory.map(h => h.train_acc);
  const valAccs = epochHistory.map(h => h.val_acc);

  // Compute macro metrics from Confusion Matrix
  const metrics = useMemo(() => {
    if (!confusionMatrix || confusionMatrix.length < 3) {
      return { precision: 0, recall: 0, f1: 0 };
    }
    
    let precisionSum = 0;
    let recallSum = 0;
    let validClassesP = 0;
    let validClassesR = 0;

    for (let i = 0; i < 3; i++) {
      const tp = confusionMatrix[i][i];
      
      // Sum of column i (predicted class i)
      const colSum = confusionMatrix.reduce((sum, row) => sum + row[i], 0);
      // Sum of row i (actual class i)
      const rowSum = confusionMatrix[i].reduce((sum, val) => sum + val, 0);

      const p = colSum > 0 ? tp / colSum : 0;
      const r = rowSum > 0 ? tp / rowSum : 0;

      if (colSum > 0) {
        precisionSum += p;
        validClassesP++;
      }
      if (rowSum > 0) {
        recallSum += r;
        validClassesR++;
      }
    }

    const precision = validClassesP > 0 ? precisionSum / validClassesP : 0;
    const recall = validClassesR > 0 ? recallSum / validClassesR : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      precision: precision * 100,
      recall: recall * 100,
      f1: f1 * 100
    };
  }, [confusionMatrix]);

  // Render Confusion Matrix
  const renderConfusionMatrix = () => {
    if (!confusionMatrix || confusionMatrix.length === 0) {
      return <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Awaiting training...</div>;
    }
    const classes = ['HOME', 'DRAW', 'AWAY'];
    const total = confusionMatrix.reduce((sum, row) => sum + row.reduce((s, val) => s + val, 0), 0) || 1.0;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: '#070a0f', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)', flex: 1.2 }}>
        <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>CONFUSION MATRIX</span>
        <div style={{ display: 'grid', gridTemplateColumns: '60px 1fr 1fr 1fr', gap: '0.3rem', textAlign: 'center', fontSize: '0.7rem' }}>
          <span />
          {classes.map(c => <span key={c} style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{c}</span>)}
          
          {classes.map((c, rIdx) => (
            <React.Fragment key={c}>
              <span style={{ textAlign: 'left', fontWeight: 700, color: 'var(--text-secondary)', alignSelf: 'center' }}>{c}</span>
              {confusionMatrix[rIdx]?.map((val, cIdx) => {
                const pct = (val / total) * 100;
                const isDiag = rIdx === cIdx;
                const cellBg = isDiag 
                  ? `rgba(0, 114, 255, ${Math.min(0.8, 0.15 + pct / 70)})` // Blue
                  : `rgba(255, 0, 127, ${Math.min(0.8, 0.05 + pct / 70)})`; // Red
                
                return (
                  <div
                    key={cIdx}
                    style={{
                      background: cellBg,
                      padding: '0.5rem',
                      borderRadius: '4px',
                      fontWeight: 800,
                      color: isDiag ? '#fff' : 'rgba(255,255,255,0.8)',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      minWidth: '45px'
                    }}
                  >
                    <span>{val}</span>
                    <span style={{ fontSize: '0.55rem', opacity: 0.8 }}>{pct.toFixed(0)}%</span>
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  };

  // Render ROC Curves
  const renderROCCurves = () => {
    if (!rocCurve || Object.keys(rocCurve).length === 0) {
      return null;
    }
    const width = 200;
    const height = 150;
    const padding = 20;

    const colors = ['var(--cyan)', 'var(--gold)', 'var(--pink)'];
    const labels = ['Home', 'Draw', 'Away'];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: '#070a0f', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.03)', flex: 1 }}>
        <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>ROC CURVES</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          
          <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100px', height: 'auto' }}>
            <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.08)" />
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.08)" />
            <line x1={padding} y1={height - padding} x2={width - padding} y2={padding} stroke="rgba(255,255,255,0.05)" strokeDasharray="3" />
            
            {Object.entries(rocCurve).map(([classIdx, curve]) => {
              const points = curve.fpr.map((fpr, i) => {
                const x = padding + fpr * (width - 2 * padding);
                const y = height - padding - curve.tpr[i] * (height - 2 * padding);
                return `${x},${y}`;
              }).join(' ');
              
              return (
                <polyline
                  key={classIdx}
                  fill="none"
                  stroke={colors[classIdx]}
                  strokeWidth="1.5"
                  points={points}
                />
              );
            })}
          </svg>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.6rem' }}>
            {labels.map((lbl, idx) => (
              <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                <div style={{ width: '6px', height: '6px', background: colors[idx] }} />
                <span>{lbl} ({rocCurve[idx]?.auc.toFixed(2) ?? '0.0'})</span>
              </div>
            ))}
          </div>

        </div>
      </div>
    );
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      
      {/* Header and Macro Metrics */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <h3 className="card-title" style={{ margin: 0 }}>📊 REAL-TIME TRAINING STATS</h3>
        {confusionMatrix.length > 0 && (
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', fontFamily: 'var(--font-display)' }}>
            <span>P: <strong style={{ color: 'var(--cyan)' }}>{metrics.precision.toFixed(1)}%</strong></span>
            <span>R: <strong style={{ color: 'var(--gold)' }}>{metrics.recall.toFixed(1)}%</strong></span>
            <span>F1: <strong style={{ color: 'var(--pink)' }}>{metrics.f1.toFixed(1)}%</strong></span>
          </div>
        )}
      </div>
      
      {/* Loss & Accuracy Curves */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <SVGLineChart
          title="LOSS CURVE"
          trainValues={trainLosses}
          valValues={valLosses}
          maxEpochs={epochs}
        />
        <SVGLineChart
          title="ACCURACY CURVE"
          trainValues={trainAccs}
          valValues={valAccs}
          maxEpochs={epochs}
        />
      </div>

      {/* Confusion Matrix and ROC Curves */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        {renderConfusionMatrix()}
        {renderROCCurves()}
      </div>
    </div>
  );
}
