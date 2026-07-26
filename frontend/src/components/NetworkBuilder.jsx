// frontend/src/components/NetworkBuilder.jsx
import React, { useState, useRef } from 'react';

const FEATURE_CATEGORIES = {
  "Ratings & Rankings": [
    { id: "elo_home", label: "Elo Strength (Home)" },
    { id: "elo_away", label: "Elo Strength (Away)" },
    { id: "rank_home", label: "FIFA Ranking (Home)" },
    { id: "rank_away", label: "FIFA Ranking (Away)" },
    { id: "rank_diff", label: "Ranking Difference" }
  ],
  "Recent Form & Activity": [
    { id: "form_home", label: "Recent Form (Home)" },
    { id: "form_away", label: "Recent Form (Away)" },
    { id: "win_rate_home", label: "Win Rate L10 (Home)" },
    { id: "win_rate_away", label: "Win Rate L10 (Away)" },
    { id: "rest_home", label: "Rest Days (Home)" },
    { id: "rest_away", label: "Rest Days (Away)" }
  ],
  "Match Statistics": [
    { id: "gs_home", label: "Avg Goals Scored (Home)" },
    { id: "gs_away", label: "Avg Goals Scored (Away)" },
    { id: "gc_home", label: "Avg Goals Conceded (Home)" },
    { id: "gc_away", label: "Avg Goals Conceded (Away)" },
    { id: "poss_home", label: "Possession Ratio (Home)" },
    { id: "poss_away", label: "Possession Ratio (Away)" },
    { id: "shots_home", label: "Avg Shots Taken (Home)" },
    { id: "shots_away", label: "Avg Shots Taken (Away)" }
  ],
  "Statistical Engine Inputs": [
    { id: "stat_prob_home", label: "Stat Win Prob (Home)" },
    { id: "stat_prob_draw", label: "Stat Draw Prob" },
    { id: "stat_prob_away", label: "Stat Win Prob (Away)" },
    { id: "stat_xg_home", label: "Stat Expected Goals (Home)" },
    { id: "stat_xg_away", label: "Stat Expected Goals (Away)" }
  ],
  "Design Studio Synthetic Features": [
    { id: "market_value_home", label: "Team Market Value (Home)" },
    { id: "market_value_away", label: "Team Market Value (Away)" },
    { id: "age_home", label: "Average Team Age (Home)" },
    { id: "age_away", label: "Average Team Age (Away)" },
    { id: "def_rating_home", label: "Defensive Rating (Home)" },
    { id: "def_rating_away", label: "Defensive Rating (Away)" },
    { id: "off_rating_home", label: "Offensive Rating (Home)" },
    { id: "off_rating_away", label: "Offensive Rating (Away)" },
    { id: "exp_home", label: "World Cup Experience (Home)" },
    { id: "exp_away", label: "World Cup Experience (Away)" }
  ],
  "Venue & Metadata": [
    { id: "home_adv", label: "Home Turf Advantage" },
    { id: "player_avail_home", label: "Squad Fitness (Home)" },
    { id: "player_avail_away", label: "Squad Fitness (Away)" }
  ]
};

export default function NetworkBuilder({
  config,
  onChange,
  onTrain,
  isTraining = false,
  networkState = [],
  onImportModel,
  scalerParams
}) {
  const fileInputRef = useRef(null);
  const [openSections, setOpenSections] = useState({
    topology: true,
    hyperparams: true,
    features: false,
    advanced: false,
    utilities: false
  });

  const toggleSection = (sect) => {
    setOpenSections(prev => ({ ...prev, [sect]: !prev[sect] }));
  };

  const toggleFeature = (fId) => {
    const list = [...config.features];
    const idx = list.indexOf(fId);
    if (idx >= 0) {
      list.splice(idx, 1);
    } else {
      list.push(fId);
    }
    onChange({ ...config, features: list });
  };

  const handleLayerChange = (idx, val) => {
    const layers = [...config.hiddenLayers];
    layers[idx] = Math.max(1, Math.min(256, Number(val)));
    onChange({ ...config, hiddenLayers: layers });
  };

  const addLayer = () => {
    if (config.hiddenLayers.length < 5) {
      onChange({ ...config, hiddenLayers: [...config.hiddenLayers, 8] });
    }
  };

  const removeLayer = () => {
    if (config.hiddenLayers.length > 1) {
      onChange({ ...config, hiddenLayers: config.hiddenLayers.slice(0, -1) });
    }
  };

  // Export current config & state dict to file
  const handleExportModel = () => {
    const hasValidScaler = scalerParams && 
      scalerParams.mean && scalerParams.mean.length === config.features.length &&
      scalerParams.var && scalerParams.var.length === config.features.length &&
      scalerParams.scale && scalerParams.scale.length === config.features.length;

    const exportData = {
      config,
      // Retrieve scaling parameters from locally saved scaler or backend
      scaler_mean: hasValidScaler ? scalerParams.mean : Array(config.features.length).fill(0.0), // Baseline mean
      scaler_var: hasValidScaler ? scalerParams.var : Array(config.features.length).fill(1.0),
      scaler_scale: hasValidScaler ? scalerParams.scale : Array(config.features.length).fill(1.0),
      state_dict: {}
    };

    // Serialize weights/biases from networkState
    networkState.forEach(layer => {
      // Rebuild PyTorch state dict keys from the actual Layer name (e.g. "Layer 0" -> "net.0")
      const key = layer.name.replace('Layer ', 'net.');
      exportData.state_dict[`${key}.weight`] = layer.weights;
      exportData.state_dict[`${key}.bias`] = layer.biases;
    });

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `worldcup_ann_${config.activation}_${config.optimizer}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import JSON model configuration
  const handleImportModel = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const data = JSON.parse(event.target.result);
        if (onImportModel) {
          await onImportModel(data);
        }
      } catch (err) {
        alert("Failed to parse JSON file: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, padding: '1.25rem' }}>
      
      {/* Title */}
      <div>
        <h3 className="card-title" style={{ margin: 0 }}>⚙️ DYNAMIC MODEL STUDIO</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '0.2rem' }}>Configure topology, feature vectors, loss, optimizer, and export variables.</p>
      </div>

      {/* SECTION 1: Topology */}
      <div style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
        <div onClick={() => toggleSection('topology')} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
            {openSections.topology ? '▼' : '▶'} 1. NEURAL TOPOLOGY
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{config.hiddenLayers.length} Layers</span>
        </div>

        {openSections.topology && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem', paddingLeft: '0.5rem' }}>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              {config.hiddenLayers.map((hDim, idx) => (
                <div key={idx} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.2rem' }}>
                  <input
                    type="number"
                    value={hDim}
                    min="1"
                    max="128"
                    onChange={(e) => handleLayerChange(idx, e.target.value)}
                    disabled={isTraining}
                    style={{
                      width: '45px',
                      padding: '0.3rem',
                      background: '#070a0f',
                      color: '#fff',
                      border: '1px solid var(--border-glow)',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontFamily: 'var(--font-display)',
                      fontSize: '0.75rem'
                    }}
                  />
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>L{idx+1}</span>
                </div>
              ))}
              <div style={{ display: 'flex', gap: '0.2rem' }}>
                <button className="nav-item" onClick={addLayer} disabled={isTraining || config.hiddenLayers.length >= 5} style={{ padding: '0.1rem 0.4rem', fontSize: '0.7rem', border: '1px solid rgba(255,255,255,0.1)' }}>+</button>
                <button className="nav-item" onClick={removeLayer} disabled={isTraining || config.hiddenLayers.length <= 1} style={{ padding: '0.1rem 0.4rem', fontSize: '0.7rem', border: '1px solid rgba(255,255,255,0.1)' }}>-</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SECTION 2: Core Hyperparameters */}
      <div style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
        <div onClick={() => toggleSection('hyperparams')} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
            {openSections.hyperparams ? '▼' : '▶'} 2. HYPERPARAMETERS
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{config.optimizer.toUpperCase()} ({config.learningRate})</span>
        </div>

        {openSections.hyperparams && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginTop: '0.75rem', paddingLeft: '0.5rem' }}>
            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>ACTIVATION</label>
              <select
                value={config.activation}
                onChange={(e) => onChange({ ...config, activation: e.target.value })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="relu">ReLU</option>
                <option value="leaky_relu">Leaky ReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="elu">ELU</option>
                <option value="selu">SELU</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>OPTIMIZER</label>
              <select
                value={config.optimizer}
                onChange={(e) => onChange({ ...config, optimizer: e.target.value })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
                <option value="adamw">AdamW</option>
                <option value="rmsprop">RMSprop</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>LEARNING RATE</label>
              <input
                type="number" step="0.001" min="0.001" max="0.1"
                value={config.learningRate}
                onChange={(e) => onChange({ ...config, learningRate: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>WEIGHT INIT</label>
              <select
                value={config.weightInit}
                onChange={(e) => onChange({ ...config, weightInit: e.target.value })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="xavier_uniform">Xavier Uniform</option>
                <option value="xavier_normal">Xavier Normal</option>
                <option value="kaiming_uniform">He Uniform</option>
                <option value="kaiming_normal">He Normal</option>
                <option value="orthogonal">Orthogonal</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>BATCH SIZE</label>
              <input
                type="number" min="8" max="256" step="8"
                value={config.batchSize}
                onChange={(e) => onChange({ ...config, batchSize: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>EPOCHS</label>
              <input
                type="number" min="5" max="200" step="5"
                value={config.epochs}
                onChange={(e) => onChange({ ...config, epochs: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* SECTION 3: Advanced Optimization Parameters */}
      <div style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
        <div onClick={() => toggleSection('advanced')} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
            {openSections.advanced ? '▼' : '▶'} 3. REGULARIZATION & LOSS
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Seed ({config.randomSeed || 42})</span>
        </div>

        {openSections.advanced && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginTop: '0.75rem', paddingLeft: '0.5rem' }}>
            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>LOSS FUNCTION</label>
              <select
                value={config.lossFunc || 'crossentropy'}
                onChange={(e) => onChange({ ...config, lossFunc: e.target.value })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="crossentropy">Cross Entropy</option>
                <option value="mse">MSE (Regression)</option>
                <option value="l1">MAE (L1 Loss)</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>LR SCHEDULER</label>
              <select
                value={config.scheduler || 'none'}
                onChange={(e) => onChange({ ...config, scheduler: e.target.value })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="none">None</option>
                <option value="step">StepLR (decay 10 epoch)</option>
                <option value="exponential">ExponentialLR</option>
                <option value="cosine">CosineAnnealingLR</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>WEIGHT DECAY (L2)</label>
              <input
                type="number" step="0.0001" min="0" max="0.01"
                value={config.weightDecay || 0.0001}
                onChange={(e) => onChange({ ...config, weightDecay: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>MOMENTUM (SGD)</label>
              <input
                type="number" step="0.05" min="0" max="0.99"
                value={config.momentum || 0.9}
                onChange={(e) => onChange({ ...config, momentum: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>EARLY STOPPATIENCE</label>
              <input
                type="number" min="0" max="30" step="1"
                value={config.earlyStopping || 0}
                onChange={(e) => onChange({ ...config, earlyStopping: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>RANDOM SEED</label>
              <input
                type="number" min="1" max="9999"
                value={config.randomSeed || 42}
                onChange={(e) => onChange({ ...config, randomSeed: Number(e.target.value) })}
                disabled={isTraining}
                style={{ width: '100%', padding: '0.35rem', background: '#070a0f', color: '#fff', border: '1px solid var(--border-glow)', borderRadius: '4px', fontSize: '0.75rem' }}
              />
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', cursor: 'pointer', gridColumn: 'span 2' }}>
              <input
                type="checkbox"
                checked={config.gradClipping}
                onChange={(e) => onChange({ ...config, gradClipping: e.target.checked })}
                disabled={isTraining}
                style={{ accentColor: 'var(--cyan)' }}
              />
              Gradient Clipping (max norm = 1.0)
            </label>
          </div>
        )}
      </div>

      {/* SECTION 4: Input Feature Selector */}
      <div style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
        <div onClick={() => toggleSection('features')} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
            {openSections.features ? '▼' : '▶'} 4. SELECT FEATURES
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{config.features.length} Active</span>
        </div>

        {openSections.features && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem', paddingLeft: '0.5rem', maxHeight: '180px', overflowY: 'auto' }}>
            {Object.entries(FEATURE_CATEGORIES).map(([cat, feats]) => (
              <div key={cat}>
                <div style={{ fontSize: '0.65rem', fontWeight: 800, color: 'var(--text-secondary)', marginBottom: '0.2rem', textTransform: 'uppercase' }}>
                  {cat}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  {feats.map(f => {
                    const isChecked = config.features.includes(f.id);
                    return (
                      <label key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.7rem', color: isChecked ? '#fff' : 'var(--text-secondary)', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleFeature(f.id)}
                          disabled={isTraining}
                          style={{ accentColor: 'var(--cyan)' }}
                        />
                        {f.label}
                      </label>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* SECTION 5: Export / Import Utilities */}
      <div style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
        <div onClick={() => toggleSection('utilities')} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}>
          <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>
            {openSections.utilities ? '▼' : '▶'} 5. STUDIO WEIGHTS UTILITIES
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Import/Export</span>
        </div>

        {openSections.utilities && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.75rem', paddingLeft: '0.5rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              <button 
                className="nav-item" 
                onClick={handleExportModel}
                style={{ padding: '0.35rem', fontSize: '0.75rem', border: '1px solid var(--border-glow)', background: 'rgba(0, 242, 254, 0.05)' }}
              >
                📥 EXPORT JSON
              </button>
              
              <button 
                className="nav-item" 
                onClick={() => fileInputRef.current.click()}
                style={{ padding: '0.35rem', fontSize: '0.75rem', border: '1px solid var(--border-glow)', background: 'rgba(0, 242, 254, 0.05)' }}
              >
                📤 IMPORT JSON
              </button>
            </div>
            
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleImportModel} 
              style={{ display: 'none' }} 
              accept=".json" 
            />
          </div>
        )}
      </div>

      {/* Compile button */}
      <button 
        className="btn-cyber" 
        onClick={onTrain} 
        disabled={isTraining || config.features.length === 0} 
        style={{ width: '100%', padding: '0.75rem', fontSize: '0.85rem' }}
      >
        {isTraining ? '⚡ COMPILED & TRAINING...' : '🏁 COMPILE & RUN TRAINING'}
      </button>

    </div>
  );
}
