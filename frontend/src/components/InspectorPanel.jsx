// frontend/src/components/InspectorPanel.jsx
import React, { useState } from 'react';

export default function InspectorPanel({
  selectedNode,
  selectedConnection,
  activationName = 'relu',
  networkState = [],
  customWeightOverrides = {},
  onWeightOverrideChange
}) {
  const [learningMode, setLearningMode] = useState('beginner'); // beginner | intermediate | advanced

  const renderTabs = () => (
    <div style={{ display: 'flex', gap: '0.2rem', padding: '0.2rem', background: '#070a0f', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
      {['beginner', 'intermediate', 'advanced'].map(mode => (
        <button
          key={mode}
          onClick={() => setLearningMode(mode)}
          style={{
            flex: 1,
            background: learningMode === mode ? 'var(--cyan)' : 'transparent',
            border: 'none',
            color: learningMode === mode ? '#000' : 'var(--text-secondary)',
            fontFamily: 'var(--font-display)',
            fontWeight: 800,
            fontSize: '0.65rem',
            padding: '0.25rem 0.4rem',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            textTransform: 'uppercase'
          }}
        >
          {mode}
        </button>
      ))}
    </div>
  );

  const renderNodeInspector = () => {
    const isDead = selectedNode.value === 0 && activationName.toLowerCase() === 'relu';
    
    // Retrieve incoming weights
    const incomingWeights = [];
    if (selectedNode.layerIdx > 0 && networkState[selectedNode.layerIdx - 1]) {
      const wMatrix = networkState[selectedNode.layerIdx - 1].weights || [];
      const nodeRow = wMatrix[selectedNode.nodeIdx] || [];
      nodeRow.forEach((val, idx) => {
        incomingWeights.push({ idx, weight: val });
      });
    }

    // Dynamic Weight Histogram calculations
    const weightsArr = incomingWeights.map(w => w.weight);
    const binsCount = 8;
    const bins = new Array(binsCount).fill(0);
    let minW = weightsArr.length > 0 ? Math.min(...weightsArr) : -1;
    let maxW = weightsArr.length > 0 ? Math.max(...weightsArr) : 1;
    if (minW === maxW) {
      minW -= 0.1;
      maxW += 0.1;
    }
    const step = (maxW - minW) / binsCount;
    weightsArr.forEach(w => {
      let bIdx = Math.floor((w - minW) / step);
      if (bIdx >= binsCount) bIdx = binsCount - 1;
      if (bIdx < 0) bIdx = 0;
      bins[bIdx]++;
    });
    const maxBin = Math.max(...bins, 1);

    const renderWeightHistogram = () => {
      if (incomingWeights.length === 0) return null;
      return (
        <div style={{ marginTop: '0.8rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.8rem' }}>
          <span style={{ color: 'var(--cyan)', fontSize: '0.65rem', fontWeight: 800, textTransform: 'uppercase' }}>📊 Synapse Weight Distribution</span>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.62rem', margin: '0.1rem 0 0.4rem 0' }}>
            Density of incoming weights to this node (range: {minW.toFixed(2)} to {maxW.toFixed(2)}).
          </p>
          <svg width="100%" height="60" style={{ background: '#070a0f', borderRadius: '4px', padding: '4px' }}>
            {bins.map((val, idx) => {
              const h = (val / maxBin) * 40;
              const w = 100 / binsCount;
              const x = idx * w;
              const y = 44 - h;
              return (
                <g key={idx}>
                  <rect
                    x={`${x}%`}
                    y={y}
                    width={`${w - 2}%`}
                    height={h}
                    fill="var(--cyan)"
                    opacity={0.3 + (h / 40) * 0.7}
                    rx="1.5"
                  />
                  <text
                    x={`${x + w/2}%`}
                    y="54"
                    textAnchor="middle"
                    fill="var(--text-secondary)"
                    fontSize="5.5px"
                    fontFamily="monospace"
                  >
                    {((minW + idx * step) + step/2).toFixed(1)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      );
    };

    return (
      <div className="glass-card slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', height: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="card-title" style={{ margin: 0 }}>🔍 NEURON INSPECTOR</h3>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{selectedNode.id.toUpperCase()}</span>
        </div>

        {renderTabs()}

        {/* BEGINNER MODE */}
        {learningMode === 'beginner' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem', lineHeight: '1.4', color: 'var(--text-secondary)' }}>
            <p>
              💡 <strong style={{ color: '#fff' }}>Activation value ({selectedNode.value.toFixed(4)}):</strong> This is the numerical strength of the signal leaving this neuron. A higher value means this node is actively firing.
            </p>
            <p>
              💡 <strong style={{ color: '#fff' }}>Bias value ({selectedNode.bias.toFixed(4)}):</strong> Think of bias as a threshold helper. It shifts the activation function trigger point, deciding how easy it is for the neuron to activate.
            </p>
            {isDead && (
              <div style={{ background: 'rgba(255,0,127,0.1)', border: '1px solid rgba(255,0,127,0.2)', padding: '0.4rem', borderRadius: '4px', color: 'var(--pink)', fontSize: '0.7rem' }}>
                ⚠️ <strong>Dead Neuron:</strong> Because you are using the <strong>ReLU</strong> activation function, this neuron is currently outputs exactly 0. This stops any backpropagation adjustments through this node.
              </div>
            )}
          </div>
        )}

        {/* INTERMEDIATE MODE */}
        {learningMode === 'intermediate' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <div>
              <span style={{ color: 'var(--cyan)', fontWeight: 800 }}>MATHEMATICAL EQUATION</span>
              <div style={{ background: '#070a0f', padding: '0.4rem', borderRadius: '4px', marginTop: '0.2rem', fontFamily: 'monospace', color: '#fff', fontSize: '0.7rem' }}>
                z = Σ (w_i * x_i) + b <br/>
                a = max(0, z)  # For ReLU
              </div>
            </div>
            <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.4rem' }}>
              <strong>Layer Calculation:</strong>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.2rem' }}>
                <span>Bias term (b):</span>
                <span style={{ color: '#fff' }}>{selectedNode.bias.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Output (a):</span>
                <span style={{ color: 'var(--cyan)' }}>{selectedNode.value.toFixed(5)}</span>
              </div>
            </div>
          </div>
        )}

        {/* ADVANCED MODE */}
        {learningMode === 'advanced' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Layer Activation:</span>
              <span style={{ fontFamily: 'var(--font-display)', color: 'var(--cyan)', fontWeight: 800 }}>{selectedNode.value.toFixed(6)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Bias Tensor:</span>
              <span style={{ fontFamily: 'var(--font-display)', color: 'var(--gold)', fontWeight: 800 }}>{selectedNode.bias.toFixed(6)}</span>
            </div>

            {incomingWeights.length > 0 && (
              <div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: 800 }}>RAW PYTORCH WEIGHT VECTOR</span>
                <div style={{ maxHeight: '100px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.25rem', padding: '0.3rem', background: '#070a0f', borderRadius: '4px', fontSize: '0.65rem', fontFamily: 'monospace' }}>
                  {incomingWeights.map(w => (
                    <div key={w.idx} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>w[{w.idx}]</span>
                      <span style={{ color: w.weight > 0 ? 'var(--cyan)' : 'var(--pink)' }}>{w.weight.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Dynamic Weight Histogram */}
        {renderWeightHistogram()}

      </div>
    );
  };

  const renderConnectionInspector = () => {
    const overrideKey = selectedConnection
      ? `${selectedConnection.layerIdx}_${selectedConnection.srcIdx}_${selectedConnection.dstIdx}`
      : "";
    const wVal = (selectedConnection && customWeightOverrides[overrideKey] !== undefined)
      ? customWeightOverrides[overrideKey]
      : selectedConnection?.weight ?? 0;
      
    const gVal = selectedConnection.gradient ?? 0;
    const absW = Math.abs(wVal);
    
    return (
      <div className="glass-card slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', height: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="card-title" style={{ margin: 0 }}>🔗 LINK INSPECTOR</h3>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>WEIGHT CONNECTOR</span>
        </div>
 
        {renderTabs()}
 
        {/* BEGINNER MODE */}
        {learningMode === 'beginner' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem', lineHeight: '1.4', color: 'var(--text-secondary)' }}>
            <p>
              💡 <strong style={{ color: '#fff' }}>Weight ({wVal.toFixed(4)}):</strong> The connection strength between the source and target neurons. Think of it as a gatekeeper that scales the signal.
            </p>
            <p>
              💡 <strong style={{ color: '#fff' }}>Gradient ({gVal.toFixed(4)}):</strong> Calculated via **Backpropagation**. It tells us how the overall prediction error shifts when we adjust this weight.
            </p>
            <p>
              💡 <strong style={{ color: '#fff' }}>Intuitive Update:</strong> If gradient is positive (<span style={{ color: 'var(--gold)' }}>{gVal > 0 ? 'Yes' : 'No'}</span>), it means increasing the weight increases the error. The optimizer will **decrease** the weight to reduce the error.
            </p>
          </div>
        )}
 
        {/* INTERMEDIATE MODE */}
        {learningMode === 'intermediate' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <div>
              <span style={{ color: 'var(--cyan)', fontWeight: 800 }}>WEIGHT OPTIMIZATION STEP</span>
              <div style={{ background: '#070a0f', padding: '0.45rem', borderRadius: '4px', marginTop: '0.2rem', fontFamily: 'monospace', color: '#fff', fontSize: '0.68rem', lineHeight: 1.5 }}>
                W_new = W - η * (dL/dW)<br/>
                W_new = {wVal.toFixed(5)} - η * ({gVal.toFixed(5)})
              </div>
            </div>
            <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Current Weight (W):</span>
                <span style={{ color: '#fff' }}>{wVal.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Gradient (dL/dW):</span>
                <span style={{ color: 'var(--gold)' }}>{gVal.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Signal Impact:</span>
                <span style={{ color: 'var(--cyan)' }}>{absW.toFixed(5)}</span>
              </div>
            </div>
            <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.4rem', fontSize: '0.7rem' }}>
              ℹ️ The **learning rate (η)** controls the step size downhill on the error surface.
            </div>
          </div>
        )}
 
        {/* ADVANCED MODE */}
        {learningMode === 'advanced' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span>Source Node (a_i):</span>
              <span style={{ color: '#fff' }}>{selectedConnection.src.toUpperCase()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span>Target Node (z_j):</span>
              <span style={{ color: '#fff' }}>{selectedConnection.dst.toUpperCase()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span>Weight Tensor (W_ij):</span>
              <span style={{ fontFamily: 'var(--font-display)', color: wVal > 0 ? 'var(--cyan)' : 'var(--pink)', fontWeight: 800 }}>
                {wVal.toFixed(6)}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.2rem' }}>
              <span>Autograd dL/dW:</span>
              <span style={{ fontFamily: 'var(--font-display)', color: 'var(--gold)', fontWeight: 800 }}>
                {gVal.toFixed(6)}
              </span>
            </div>
 
            <div style={{ marginTop: '0.4rem' }}>
              <span style={{ color: 'var(--cyan)', fontSize: '0.65rem', fontWeight: 800 }}>BACKPROPAGATION CHAIN RULE</span>
              <div style={{ background: '#070a0f', padding: '0.45rem', borderRadius: '4px', marginTop: '0.2rem', fontFamily: 'monospace', color: '#fff', fontSize: '0.68rem', lineHeight: 1.5 }}>
                ∂L/∂W_ij = (∂L/∂z_j) * a_i<br/>
                ∂L/∂W_ij = δ_j * a_i
              </div>
              <p style={{ fontSize: '0.65rem', marginTop: '0.3rem', lineHeight: '1.45' }}>
                The error gradient is the product of the target neuron's error sensitivity (δ_j) and the source neuron's active firing output (a_i).
              </p>
            </div>
          </div>
        )}

        {/* ⚡ Neuron Surgery (Weight Overrides) */}
        <div style={{ marginTop: '0.8rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.8rem' }}>
          <span style={{ color: 'var(--cyan)', fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase' }}>⚡ Neuron Surgery (Weight Edit)</span>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.62rem', margin: '0.1rem 0 0.4rem 0' }}>
            Manually override this connection weight to test custom network responses.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <input
              type="range"
              min="-3.0"
              max="3.0"
              step="0.01"
              value={wVal}
              onChange={(e) => onWeightOverrideChange?.(overrideKey, parseFloat(e.target.value))}
              style={{ flex: 1 }}
              className="time-machine-slider"
            />
            <span style={{ fontSize: '0.85rem', fontFamily: 'monospace', fontWeight: 800, width: '45px', textAlign: 'right', color: wVal >= 0 ? 'var(--cyan)' : 'var(--pink)' }}>
              {wVal.toFixed(2)}
            </span>
          </div>
          {customWeightOverrides[overrideKey] !== undefined && (
            <button
              onClick={() => onWeightOverrideChange?.(overrideKey, undefined)}
              style={{
                background: 'rgba(255, 0, 127, 0.1)',
                border: '1px solid var(--pink)',
                color: 'var(--pink)',
                fontSize: '0.65rem',
                padding: '0.25rem 0.5rem',
                borderRadius: '6px',
                cursor: 'pointer',
                marginTop: '0.5rem',
                fontFamily: 'var(--font-display)',
                width: '100%',
                fontWeight: 800
              }}
            >
              RESET OVERRIDE
            </button>
          )}
        </div>
 
      </div>
    );
  };

  const renderDefault = () => (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', alignItems: 'center', textAlign: 'center', height: '100%', minHeight: '220px' }}>
      <div style={{ fontSize: '2.5rem', animation: 'pulse-glow 2s infinite alternate' }}>🔬</div>
      <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem', margin: 0 }}>STUDIO INSPECTOR</h4>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', maxWidth: '230px', margin: 0 }}>
        Select or hover any neuron circle or connection path link in the SVG topological map. Choose Beginner, Intermediate, or Advanced explainers.
      </p>
      {renderTabs()}
    </div>
  );

  if (selectedNode) return renderNodeInspector();
  if (selectedConnection) return renderConnectionInspector();
  return renderDefault();
}
