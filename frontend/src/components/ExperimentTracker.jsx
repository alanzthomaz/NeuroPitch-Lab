// frontend/src/components/ExperimentTracker.jsx
import React, { useState, useEffect } from 'react';

export default function ExperimentTracker({
  currentConfig,
  currentMetrics,
  onLoadConfig
}) {
  const [experiments, setExperiments] = useState([]);
  const [modelA, setModelA] = useState(null);
  const [modelB, setModelB] = useState(null);
  const [error, setError] = useState(null);
  const [saveName, setSaveName] = useState('');

  const fetchExperiments = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/experiments");
      if (response.ok) {
        const data = await response.json();
        setExperiments(data);
      }
    } catch (err) {
      console.error("Failed to load experiments history");
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const saveCurrentExperiment = async () => {
    if (!currentMetrics || !currentConfig) return;
    
    const name = saveName.trim() || `Experiment #${experiments.length + 1}`;
    const newExp = {
      id: `exp-${Date.now()}`,
      name,
      config: currentConfig,
      metrics: currentMetrics,
      timestamp: new Date().toLocaleTimeString()
    };

    try {
      const response = await fetch("http://localhost:8000/api/experiments", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newExp)
      });
      if (response.ok) {
        setSaveName('');
        fetchExperiments();
      }
    } catch (err) {
      setError("Failed to save experiment to persistent log");
    }
  };

  const clearLog = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/experiments", {
        method: 'DELETE'
      });
      if (response.ok) {
        setExperiments([]);
        setModelA(null);
        setModelB(null);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const renderComparisonMatrix = (matrix) => {
    if (!matrix || matrix.length === 0) return null;
    const classes = ['Home', 'Draw', 'Away'];
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.2rem', fontSize: '0.7rem', textAlign: 'center', marginTop: '0.5rem' }}>
        <span />
        {classes.map(c => <span key={c} style={{ fontWeight: 700 }}>{c}</span>)}
        {classes.map((c, rIdx) => (
          <React.Fragment key={c}>
            <span style={{ fontWeight: 700, textAlign: 'left' }}>{c}</span>
            {matrix[rIdx]?.map((val, cIdx) => (
              <div key={cIdx} style={{ background: rIdx === cIdx ? 'rgba(0,242,254,0.1)' : 'rgba(255,0,127,0.05)', padding: '0.25rem', borderRadius: '2px' }}>
                {val}
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    );
  };

  return (
    <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Save Experiment Control */}
      {currentMetrics && (
        <div className="glass-card" style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>💾 SAVE COMPLETED RUN</h4>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Record the current architecture and metrics in the persistent log.</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <input
              type="text"
              placeholder="Experiment Name (optional)..."
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              style={{
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid var(--border-glow)',
                padding: '0.5rem 1rem',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '0.8rem',
                width: '230px'
              }}
            />
            <button className="btn-cyber" onClick={saveCurrentExperiment} style={{ padding: '0.5rem 1.25rem' }}>
              SAVE TO LOG
            </button>
          </div>
        </div>
      )}

      {/* Model A vs Model B Side-by-Side Comparison Workspace */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem' }}>
        
        {/* Model A Selector & Summary */}
        <div className="glass-card">
          <h3 className="card-title">🔬 MODEL WORKSPACE A</h3>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--cyan)', marginBottom: '0.35rem' }}>SELECT MODEL A</label>
            <select
              value={modelA?.id || modelA?.experiment_id || ''}
              onChange={(e) => setModelA(experiments.find(ex => (ex.id || ex.experiment_id) === e.target.value))}
              style={{ width: '100%', padding: '0.5rem', background: '#0d1423', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '6px' }}
            >
              <option value="">-- Choose saved model --</option>
              {experiments.map(ex => (
                <option key={ex.id || ex.experiment_id} value={ex.id || ex.experiment_id}>
                  {ex.name || ex.experiment_id}
                </option>
              ))}
            </select>
          </div>

          {modelA && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Layers:</span>
                <span style={{ fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
                  [{modelA.config?.hiddenLayers?.join(',') || 'N/A'}]
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Activation:</span>
                <span>{modelA.config?.activation?.toUpperCase() || 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Optimizer:</span>
                <span>{modelA.config?.optimizer?.toUpperCase() || 'N/A'} {modelA.config?.learningRate ? `(LR: ${modelA.config.learningRate})` : ''}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Val Loss:</span>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800 }}>
                  {modelA.metrics?.val_loss !== undefined ? modelA.metrics.val_loss.toFixed(4) : (modelA.upgraded_log_loss !== undefined ? modelA.upgraded_log_loss.toFixed(4) : 'N/A')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Val Accuracy:</span>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--cyan)' }}>
                  {modelA.metrics?.val_acc !== undefined ? `${modelA.metrics.val_acc}%` : (modelA.upgraded_accuracy !== undefined ? `${(modelA.upgraded_accuracy * 100).toFixed(1)}%` : 'N/A')}
                </span>
              </div>
              
              {modelA.metrics?.confusion_matrix && (
                <div>
                  <span style={{ color: 'var(--text-secondary)' }}>Confusion Matrix:</span>
                  {renderComparisonMatrix(modelA.metrics.confusion_matrix)}
                </div>
              )}

              {modelA.config && (
                <button
                  className="nav-item"
                  onClick={() => onLoadConfig(modelA.config)}
                  style={{ alignSelf: 'flex-start', marginTop: '0.75rem', fontSize: '0.75rem', border: '1px solid rgba(255,255,255,0.15)' }}
                >
                  LOAD ARCHITECTURE TO LAB
                </button>
              )}
            </div>
          )}
        </div>

        {/* Model B Selector & Summary */}
        <div className="glass-card">
          <h3 className="card-title">🔬 MODEL WORKSPACE B</h3>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--cyan)', marginBottom: '0.35rem' }}>SELECT MODEL B</label>
            <select
              value={modelB?.id || modelB?.experiment_id || ''}
              onChange={(e) => setModelB(experiments.find(ex => (ex.id || ex.experiment_id) === e.target.value))}
              style={{ width: '100%', padding: '0.5rem', background: '#0d1423', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '6px' }}
            >
              <option value="">-- Choose saved model --</option>
              {experiments.map(ex => (
                <option key={ex.id || ex.experiment_id} value={ex.id || ex.experiment_id}>
                  {ex.name || ex.experiment_id}
                </option>
              ))}
            </select>
          </div>

          {modelB && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', fontSize: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Layers:</span>
                <span style={{ fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
                  [{modelB.config?.hiddenLayers?.join(',') || 'N/A'}]
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Activation:</span>
                <span>{modelB.config?.activation?.toUpperCase() || 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Optimizer:</span>
                <span>{modelB.config?.optimizer?.toUpperCase() || 'N/A'} {modelB.config?.learningRate ? `(LR: ${modelB.config.learningRate})` : ''}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Val Loss:</span>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800 }}>
                  {modelB.metrics?.val_loss !== undefined ? modelB.metrics.val_loss.toFixed(4) : (modelB.upgraded_log_loss !== undefined ? modelB.upgraded_log_loss.toFixed(4) : 'N/A')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Val Accuracy:</span>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--cyan)' }}>
                  {modelB.metrics?.val_acc !== undefined ? `${modelB.metrics.val_acc}%` : (modelB.upgraded_accuracy !== undefined ? `${(modelB.upgraded_accuracy * 100).toFixed(1)}%` : 'N/A')}
                </span>
              </div>
              
              {modelB.metrics?.confusion_matrix && (
                <div>
                  <span style={{ color: 'var(--text-secondary)' }}>Confusion Matrix:</span>
                  {renderComparisonMatrix(modelB.metrics.confusion_matrix)}
                </div>
              )}

              {modelB.config && (
                <button
                  className="nav-item"
                  onClick={() => onLoadConfig(modelB.config)}
                  style={{ alignSelf: 'flex-start', marginTop: '0.75rem', fontSize: '0.75rem', border: '1px solid rgba(255,255,255,0.15)' }}
                >
                  LOAD ARCHITECTURE TO LAB
                </button>
              )}
            </div>
          )}
        </div>

      </div>

      {/* Persistent Experiments History Grid */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 className="card-title" style={{ margin: 0 }}>📜 EXPERIMENT RECORD LOG</h3>
          <button className="nav-item" onClick={clearLog} style={{ padding: '0.35rem 1rem', fontSize: '0.75rem', border: '1px solid rgba(255,255,255,0.1)' }}>
            CLEAR ALL LOGS
          </button>
        </div>

        {experiments.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '2rem', fontSize: '0.9rem' }}>
            No saved experiments found. Train a network in the laboratory and save it above.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="custom-table">
              <thead>
                <tr>
                  <th>NAME</th>
                  <th>LAYERS</th>
                  <th>ACTIVATION</th>
                  <th>OPTIMIZER</th>
                  <th style={{ textAlign: 'center' }}>VAL LOSS</th>
                  <th style={{ textAlign: 'center' }}>VAL ACC %</th>
                  <th style={{ width: '120px', textAlign: 'center' }}>ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map(ex => (
                  <tr key={ex.id || ex.experiment_id}>
                    <td style={{ fontWeight: 700 }}>{ex.name || ex.experiment_id}</td>
                    <td style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem' }}>
                      {ex.config?.hiddenLayers ? `[${ex.config.hiddenLayers.join(',')}]` : 'N/A'}
                    </td>
                    <td style={{ fontSize: '0.8rem' }}>{ex.config?.activation?.toUpperCase() || 'N/A'}</td>
                    <td style={{ fontSize: '0.8rem' }}>
                      {ex.config?.optimizer?.toUpperCase() || 'N/A'} {ex.config?.learningRate ? `(LR: ${ex.config.learningRate})` : ''}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'var(--font-display)' }}>
                      {ex.metrics?.val_loss !== undefined ? ex.metrics.val_loss.toFixed(4) : (ex.upgraded_log_loss !== undefined ? ex.upgraded_log_loss.toFixed(4) : 'N/A')}
                    </td>
                    <td style={{ textAlign: 'center', fontFamily: 'var(--font-display)', color: 'var(--cyan)', fontWeight: 800 }}>
                      {ex.metrics?.val_acc !== undefined ? `${ex.metrics.val_acc}%` : (ex.upgraded_accuracy !== undefined ? `${(ex.upgraded_accuracy * 100).toFixed(1)}%` : 'N/A')}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', gap: '0.4rem', justifyContent: 'center' }}>
                        <button className="nav-item" onClick={() => setModelA(ex)} style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem', border: '1px solid rgba(255,255,255,0.1)' }}>SET A</button>
                        <button className="nav-item" onClick={() => setModelB(ex)} style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem', border: '1px solid rgba(255,255,255,0.1)' }}>SET B</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
