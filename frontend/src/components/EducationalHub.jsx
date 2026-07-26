// frontend/src/components/EducationalHub.jsx
import React, { useState, useMemo } from 'react';

export default function EducationalHub() {
  const [activeTab, setActiveTab] = useState('feedforward'); // feedforward | loss | backprop
  const [isOpen, setIsOpen] = useState(true);

  // 1. Feedforward Calculator State
  const [ffInput, setFfInput] = useState(1.5);
  const [ffWeight, setFfWeight] = useState(0.8);
  const [ffBias, setFfBias] = useState(-0.5);
  const [ffActivation, setFfActivation] = useState('ReLU');

  // 2. Loss Calculator State
  const [targetClass, setTargetClass] = useState(0); // 0: Home, 1: Draw, 2: Away
  const [rawProbs, setRawProbs] = useState([0.6, 0.25, 0.15]); // Home, Draw, Away

  // 3. Backpropagation Calculator State
  const [bpWeight, setBpWeight] = useState(0.75);
  const [bpGrad, setBpGrad] = useState(1.2);
  const [bpLR, setBpLR] = useState(0.01);

  // --- Feedforward Calculations ---
  const ffResult = useMemo(() => {
    const z = ffInput * ffWeight + ffBias;
    let a = z;
    if (ffActivation === 'ReLU') a = Math.max(0, z);
    else if (ffActivation === 'Sigmoid') a = 1 / (1 + Math.exp(-z));
    else if (ffActivation === 'Tanh') a = Math.tanh(z);
    else if (ffActivation === 'Leaky ReLU') a = z >= 0 ? z : 0.1 * z;

    return { z, a };
  }, [ffInput, ffWeight, ffBias, ffActivation]);

  // --- Loss Calculations ---
  const normalizedProbs = useMemo(() => {
    const sum = rawProbs.reduce((a, b) => a + b, 0) || 1;
    return rawProbs.map(p => p / sum);
  }, [rawProbs]);

  const lossResult = useMemo(() => {
    const targetProb = normalizedProbs[targetClass];
    // Cross Entropy Loss
    const crossEntropy = -Math.log(Math.max(1e-15, targetProb));

    // Mean Squared Error (MSE)
    const targetVector = [0, 0, 0];
    targetVector[targetClass] = 1.0;
    const mse = targetVector.reduce((sum, t, idx) => sum + Math.pow(normalizedProbs[idx] - t, 2), 0) / 3;

    return { crossEntropy, mse, targetProb };
  }, [normalizedProbs, targetClass]);

  const handleProbChange = (idx, value) => {
    const newVal = Math.max(0, Math.min(1, parseFloat(value) || 0));
    setRawProbs(prev => {
      const updated = [...prev];
      updated[idx] = newVal;
      return updated;
    });
  };

  // --- Backpropagation Calculations ---
  const bpResult = useMemo(() => {
    const step = bpLR * bpGrad;
    const newWeight = bpWeight - step;
    const percentChange = bpWeight !== 0 ? (step / bpWeight) * 100 : 0;
    return { step, newWeight, percentChange };
  }, [bpWeight, bpGrad, bpLR]);

  const C = {
    cyan: '#00f2fe',
    gold: '#ffc107',
    pink: '#ff007f',
    green: '#00ff80',
    text2: 'rgba(148,163,184,0.8)',
    bg: '#070a0f',
    bgDark: '#04060b'
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      {/* Header Banner */}
      <div 
        onClick={() => setIsOpen(!isOpen)} 
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', userSelect: 'none' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '1.25rem' }}>📚</span>
          <h3 className="card-title" style={{ margin: 0 }}>ARTIFICIAL NEURAL NETWORK (ANN) THEORY & MATH HUB</h3>
        </div>
        <button style={{ background: 'none', border: 'none', color: C.cyan, cursor: 'pointer', fontWeight: 800 }}>
          {isOpen ? '❌ COLLAPSE LESSONS' : '📖 EXPAND LESSONS'}
        </button>
      </div>

      {isOpen && (
        <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1rem' }}>
          
          {/* Tabs */}
          <div style={{ display: 'flex', gap: '0.4rem', padding: '0.25rem', background: C.bgDark, borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            {[
              { id: 'feedforward', label: '1. FEEDFORWARD COMPUTATION' },
              { id: 'loss', label: '2. ERROR & LOSS CALCULATION' },
              { id: 'backprop', label: '3. BACKPROPAGATION & OPTIMIZATION' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  flex: 1,
                  background: activeTab === tab.id ? 'rgba(0, 242, 254, 0.15)' : 'transparent',
                  border: activeTab === tab.id ? `1px solid ${C.cyan}` : '1px solid transparent',
                  color: activeTab === tab.id ? C.cyan : C.text2,
                  fontFamily: 'var(--font-display)',
                  fontWeight: 800,
                  fontSize: '0.72rem',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* TAB 1: FEEDFORWARD */}
          {activeTab === 'feedforward' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
              
              {/* Theory */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', lineHeight: '1.5', color: C.text2 }}>
                <h4 style={{ margin: 0, color: '#fff', fontSize: '0.9rem', fontFamily: 'var(--font-display)' }}>THE FORWARD PASS</h4>
                <p>
                  A Neural Network makes predictions by moving signals forward layer-by-layer. At each neuron, two main operations happen:
                </p>
                <ol style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <li>
                    <strong style={{ color: '#fff' }}>Linear Summation (z):</strong> Inputs ($x_i$) are scaled by weights ($w_i$), summed together, and adjusted by a bias ($b$).
                    <div style={{ background: C.bgDark, padding: '0.4rem 0.6rem', borderRadius: '4px', fontFamily: 'monospace', color: C.cyan, marginTop: '0.2rem', fontSize: '0.75rem' }}>
                      z = w₁x₁ + w₂x₂ + ... + b = Σ(w_i * x_i) + b
                    </div>
                  </li>
                  <li>
                    <strong style={{ color: '#fff' }}>Activation Function (a):</strong> The linear logit $z$ is passed through a non-linear function $f(z)$. Non-linearity is what allows networks to learn complex, non-linear boundaries.
                    <div style={{ background: C.bgDark, padding: '0.4rem 0.6rem', borderRadius: '4px', fontFamily: 'monospace', color: C.cyan, marginTop: '0.2rem', fontSize: '0.75rem' }}>
                      a = f(z)
                    </div>
                  </li>
                </ol>
                <p>
                  Our model uses **ReLU** ($a = \max(0, z)$) for hidden layers, and **Softmax** for the output layer to convert output logits into probability distributions.
                </p>
              </div>

              {/* Calculator */}
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', background: C.bgDark }}>
                <span style={{ color: C.cyan, fontSize: '0.75rem', fontWeight: 800, fontFamily: 'var(--font-display)' }}>🧪 LIVE FORWARD PASS PLAYGROUND</span>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Input Value ($x$):</span>
                    <input 
                      type="number" 
                      step="0.1" 
                      value={ffInput} 
                      onChange={e => setFfInput(parseFloat(e.target.value) || 0)} 
                      style={{ width: '80px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.75rem' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Weight ($w$):</span>
                    <input 
                      type="number" 
                      step="0.05" 
                      value={ffWeight} 
                      onChange={e => setFfWeight(parseFloat(e.target.value) || 0)} 
                      style={{ width: '80px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.75rem' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Bias ($b$):</span>
                    <input 
                      type="number" 
                      step="0.1" 
                      value={ffBias} 
                      onChange={e => setFfBias(parseFloat(e.target.value) || 0)} 
                      style={{ width: '80px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.75rem' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Activation Function:</span>
                    <select 
                      value={ffActivation} 
                      onChange={e => setFfActivation(e.target.value)} 
                      style={{ width: '100px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.72rem' }}
                    >
                      {['ReLU', 'Sigmoid', 'Tanh', 'Leaky ReLU'].map(act => <option key={act} value={act}>{act}</option>)}
                    </select>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.72rem', fontFamily: 'monospace' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>z = ({ffInput} * {ffWeight}) + ({ffBias}) =</span>
                    <strong style={{ color: '#fff' }}>{ffResult.z.toFixed(4)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>a = f({ffResult.z.toFixed(4)}) =</span>
                    <strong style={{ color: C.green }}>{ffResult.a.toFixed(4)}</strong>
                  </div>
                </div>

                {ffActivation === 'ReLU' && ffResult.z < 0 && (
                  <div style={{ background: 'rgba(255,0,127,0.1)', border: '1px solid rgba(255,0,127,0.15)', color: C.pink, borderRadius: '4px', padding: '0.4rem', fontSize: '0.68rem', lineHeight: '1.4' }}>
                    ⚠️ **Dead Node (ReLU Clamp):** Since $z \le 0$, the activation $a$ is clamped to exactly **0**. The gradient through this node will be zero during backpropagation, meaning weights feeding into it cannot update.
                  </div>
                )}
              </div>

            </div>
          )}

          {/* TAB 2: LOSS CALCULATION */}
          {activeTab === 'loss' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
              
              {/* Theory */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', lineHeight: '1.5', color: C.text2 }}>
                <h4 style={{ margin: 0, color: '#fff', fontSize: '0.9rem', fontFamily: 'var(--font-display)' }}>LOSS & ERROR CALCULATION</h4>
                <p>
                  To train the model, we must measure how "wrong" its predictions are. This numerical rating is called the **Loss** (or cost/error).
                </p>
                <p>
                  For multi-class classifiers like ours (predicting **Home Win**, **Draw**, or **Away Win**), the standard error metric is **Categorical Cross-Entropy**:
                </p>
                <div style={{ background: C.bgDark, padding: '0.5rem 0.75rem', borderRadius: '4px', fontFamily: 'monospace', color: C.cyan, fontSize: '0.75rem', lineHeight: 1.4 }}>
                  Loss = -Σ(y_i * ln(p_i)) = -ln(p_target)
                </div>
                <p>
                  {"where $p_{target}$ is the predicted probability for the actual correct class."}
                </p>
                <p>
                  **Why Cross-Entropy?** It penalizes the network logarithmicly. If the correct team wins, but the model only gave it a $5\%$ probability, the penalty is very high ($-\ln(0.05) \approx 3.0$). If it correctly gave it a $95\%$ probability, the loss is very low ($-\ln(0.95) \approx 0.05$).
                </p>
              </div>

              {/* Calculator */}
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', background: C.bgDark }}>
                <span style={{ color: C.cyan, fontSize: '0.75rem', fontWeight: 800, fontFamily: 'var(--font-display)' }}>🧪 LIVE ERROR / LOSS PLAYGROUND</span>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.75rem' }}>
                  <div>
                    <span style={{ display: 'block', marginBottom: '0.2rem', color: C.text2 }}>Actual Match Outcome (Target Class):</span>
                    <div style={{ display: 'flex', gap: '0.25rem' }}>
                      {['Home Win', 'Draw', 'Away Win'].map((lbl, idx) => (
                        <button
                          key={lbl}
                          onClick={() => setTargetClass(idx)}
                          style={{
                            flex: 1,
                            background: targetClass === idx ? 'var(--cyan)' : C.bg,
                            border: '1px solid rgba(255,255,255,0.08)',
                            color: targetClass === idx ? '#000' : '#fff',
                            fontSize: '0.68rem',
                            fontWeight: 800,
                            padding: '0.3rem',
                            borderRadius: '4px',
                            cursor: 'pointer'
                          }}
                        >
                          {lbl}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', marginTop: '0.2rem' }}>
                    <span style={{ color: C.text2 }}>Set Predicted Class Probabilities (Sliders):</span>
                    {['Home', 'Draw', 'Away'].map((lbl, idx) => (
                      <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '45px', fontSize: '0.7rem' }}>{lbl}:</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={rawProbs[idx]}
                          onChange={e => handleProbChange(idx, e.target.value)}
                          style={{ flex: 1 }}
                        />
                        <span style={{ width: '40px', textAlign: 'right', fontWeight: 'bold', fontFamily: 'monospace' }}>
                          {(normalizedProbs[idx] * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.72rem', fontFamily: 'monospace' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Correct Class Prob (p_target):</span>
                    <strong style={{ color: C.cyan }}>{(lossResult.targetProb * 100).toFixed(1)}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Cross-Entropy Loss = -ln({lossResult.targetProb.toFixed(4)}):</span>
                    <strong style={{ color: lossResult.crossEntropy > 1.2 ? C.pink : C.green }}>{lossResult.crossEntropy.toFixed(5)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Mean Squared Error (MSE):</span>
                    <strong style={{ color: '#fff' }}>{lossResult.mse.toFixed(5)}</strong>
                  </div>
                </div>

                {lossResult.crossEntropy > 1.5 && (
                  <div style={{ background: 'rgba(255,0,127,0.1)', border: '1px solid rgba(255,0,127,0.15)', color: C.pink, borderRadius: '4px', padding: '0.4rem', fontSize: '0.68rem', lineHeight: '1.4' }}>
                    🚨 **High Loss Penalty:** Because the model gave low probability to the actual winning outcome, the Cross Entropy loss penalty spiked dramatically!
                  </div>
                )}
              </div>

            </div>
          )}

          {/* TAB 3: BACKPROPAGATION */}
          {activeTab === 'backprop' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
              
              {/* Theory */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', lineHeight: '1.5', color: C.text2 }}>
                <h4 style={{ margin: 0, color: '#fff', fontSize: '0.9rem', fontFamily: 'var(--font-display)' }}>BACKPROP & OPTIMIZATION</h4>
                <p>
                  Once the Loss is calculated, we must adjust weights and biases to reduce this Loss. We do this by calculating gradients.
                </p>
                <p>
                  {"**The Chain Rule**: Since the weight resides deep in the network, we propagate the error backwards from the output. We evaluate the derivative of the Loss ($L$) with respect to each weight ($W_{ij}$):"}
                </p>
                <div style={{ background: C.bgDark, padding: '0.5rem 0.75rem', borderRadius: '4px', fontFamily: 'monospace', color: C.cyan, fontSize: '0.75rem', lineHeight: 1.4 }}>
                  dL/dW_ij = (dL/da_j) * (da_j/dz_j) * (dz_j/dW_ij) = δ_j * a_i
                </div>
                <p>
                  **Gradient Descent**: Once we know the gradient (the slope of the error curve), we update the weight by moving **downhill** (opposite direction of the gradient):
                </p>
                <div style={{ background: C.bgDark, padding: '0.5rem 0.75rem', borderRadius: '4px', fontFamily: 'monospace', color: C.cyan, fontSize: '0.75rem', lineHeight: 1.4 }}>
                  W_new = W - η * (dL/dW)
                </div>
                <p>
                  where $\eta$ is the **Learning Rate**. If the gradient is negative, subtracting it will increase the weight. If positive, it will decrease the weight.
                </p>
              </div>

              {/* Calculator */}
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', background: C.bgDark }}>
                <span style={{ color: C.cyan, fontSize: '0.75rem', fontWeight: 800, fontFamily: 'var(--font-display)' }}>🧪 LIVE GRADIENT UPDATE PLAYGROUND</span>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Current Weight (W):</span>
                    <input 
                      type="number" 
                      step="0.05" 
                      value={bpWeight} 
                      onChange={e => setBpWeight(parseFloat(e.target.value) || 0)} 
                      style={{ width: '80px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.75rem' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Error Gradient (dL/dW):</span>
                    <input 
                      type="number" 
                      step="0.1" 
                      value={bpGrad} 
                      onChange={e => setBpGrad(parseFloat(e.target.value) || 0)} 
                      style={{ width: '80px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.75rem' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem' }}>Learning Rate (η):</span>
                    <select 
                      value={bpLR} 
                      onChange={e => setBpLR(parseFloat(e.target.value))} 
                      style={{ width: '90px', background: C.bg, border: '1px solid rgba(255,255,255,0.1)', color: '#fff', borderRadius: '4px', padding: '0.2rem .4rem', fontSize: '0.72rem' }}
                    >
                      {[0.001, 0.005, 0.01, 0.05, 0.1, 0.5].map(lr => <option key={lr} value={lr}>{lr}</option>)}
                    </select>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.72rem', fontFamily: 'monospace' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Adjustment Step (-η * dL/dW):</span>
                    <strong style={{ color: bpResult.step > 0 ? C.pink : C.cyan }}>
                      {bpResult.step > 0 ? '-' : '+'}{Math.abs(bpResult.step).toFixed(5)}
                    </strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>New Calibrated Weight:</span>
                    <strong style={{ color: C.green }}>{bpResult.newWeight.toFixed(5)}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Weight Shift Percentage:</span>
                    <strong style={{ color: '#fff' }}>{bpResult.percentChange.toFixed(1)}%</strong>
                  </div>
                </div>

                <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', color: C.text2, borderRadius: '4px', padding: '0.4rem', fontSize: '0.68rem', lineHeight: '1.4' }}>
                  💡 **Optimization Path:** Gradient is **{bpGrad > 0 ? 'positive' : 'negative'}**. To minimize loss, the weight is shifted **{bpGrad > 0 ? 'lower' : 'higher'}** from **{bpWeight.toFixed(3)}** to **{bpResult.newWeight.toFixed(3)}**.
                </div>
              </div>

            </div>
          )}

        </div>
      )}
    </div>
  );
}
