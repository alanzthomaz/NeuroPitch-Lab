// frontend/src/components/ModelExplainer.jsx
import React, { useState, useMemo } from 'react';

export default function ModelExplainer({ config, epochHistory = [], isTraining = false }) {
  const [showMathSection, setShowMathSection] = useState(false);

  // Get final metrics from history
  const metrics = useMemo(() => {
    if (!epochHistory || epochHistory.length === 0) return null;
    const lastEpoch = epochHistory[epochHistory.length - 1];
    if (!lastEpoch || lastEpoch.train_loss === undefined || lastEpoch.val_loss === undefined) return null;
    return {
      trainLoss: lastEpoch.train_loss,
      valLoss: lastEpoch.val_loss,
      trainAcc: lastEpoch.train_acc !== undefined ? (lastEpoch.train_acc * 100).toFixed(1) : null,
      valAcc: lastEpoch.val_acc !== undefined ? (lastEpoch.val_acc * 100).toFixed(1) : null,
      epoch: lastEpoch.epoch
    };
  }, [epochHistory]);

  const C = {
    cyan: '#00f2fe',
    gold: '#ffc107',
    pink: '#ff007f',
    green: '#00ff80',
    text2: 'rgba(148,163,184,0.8)',
    bg: '#070a0f',
    bgDark: '#04060b'
  };

  // Get explanations based on current config settings
  const featureExplanation = useMemo(() => {
    const active = config.features || [];
    if (active.length === 0) return 'No features selected. Please select at least one feature in the Studio.';
    
    let parts = [];
    if (active.some(f => f.startsWith('elo'))) {
      parts.push('Elo relative ratings (to capture base historical strength differences)');
    }
    if (active.some(f => f.startsWith('rank'))) {
      parts.push('FIFA ranks (to represent official global tier categories)');
    }
    if (active.some(f => f.startsWith('form') || f.startsWith('win_rate'))) {
      parts.push('Recent team form and win rates (capturing short-term momentum)');
    }
    if (active.some(f => f.startsWith('poss') || f.startsWith('shots') || f.startsWith('gs') || f.startsWith('gc'))) {
      parts.push('Match goal and efficiency stats (shots taken, average goals, possession control)');
    }
    if (active.some(f => f.startsWith('stat_prob') || f.startsWith('stat_xg'))) {
      parts.push('Predictions from the statistical Poisson engine (acts as a hybrid baseline guide)');
    }
    if (active.some(f => f.startsWith('market_value') || f.startsWith('def_rating') || f.startsWith('off_rating') || f.startsWith('exp'))) {
      parts.push('Synthetic properties (squad values, experience tiers, tactical ratings)');
    }
    if (active.includes('home_adv')) {
      parts.push('Venue advantage (home turf benefit weighting)');
    }

    return `The model makes decisions using ${active.length} inputs, representing: ${parts.join(', and ')}.`;
  }, [config.features]);

  const activationExplanation = useMemo(() => {
    const act = (config.activation || 'relu').toLowerCase();
    switch (act) {
      case 'relu':
        return 'ReLU (Rectified Linear Unit) outputs max(0, x). It creates sparse node patterns by silencing negative values, which is fast and efficient but can cause dead nodes if weight updates push inputs into negative territory.';
      case 'leaky_relu':
        return 'Leaky ReLU outputs x for positive values and 0.01x for negative values. It resolves the "dying ReLU" problem by ensuring nodes always pass a tiny signal, maintaining constant optimization flow.';
      case 'sigmoid':
        return 'Sigmoid outputs values between 0 and 1, mapping signals to a probability-like shape. It is popular for final layers but can cause "vanishing gradients" in deep networks because its gradient gets extremely small at high or low activation values.';
      case 'tanh':
        return 'Tanh outputs values between -1 and 1, centering the node signals around zero. It often converges faster than sigmoid, but still suffers from vanishing gradients when outputs saturate.';
      case 'elu':
        return 'ELU (Exponential Linear Unit) matches ReLU for positive values but scales negative values exponentially. It smooths the transition around zero, pushing mean activations closer to zero and improving noise robustness.';
      case 'selu':
        return 'SELU (Scaled Exponential Linear Unit) automatically self-normalizes activations during forward passes. Each layer keeps mean output near 0 and variance near 1, preventing exploding/vanishing gradients without needing Batch Normalization.';
      default:
        return 'Linear activation. Signals pass through unmodified.';
    }
  }, [config.activation]);

  const lossExplanation = useMemo(() => {
    const loss = (config.lossFunc || 'crossentropy').toLowerCase();
    switch (loss) {
      case 'crossentropy':
        return 'Cross-Entropy Loss measures the difference between prediction probabilities and target categories. It applies an exponential penalty when the model is confident and incorrect, driving output nodes toward clear, categorical decisions.';
      case 'mse':
        return 'Mean Squared Error (MSE) measures the average squared difference between predictions and target categories (treated as 1s and 0s). It focuses heavily on minimizing large errors, acting like a regression boundary.';
      case 'l1':
        return 'MAE (L1 Loss) measures absolute difference. It is more robust to outlier matches than MSE but has a constant gradient, which can lead to oscillations near convergence.';
      default:
        return 'Cross-Entropy Loss.';
    }
  }, [config.lossFunc]);

  const optimizerExplanation = useMemo(() => {
    const opt = (config.optimizer || 'adam').toLowerCase();
    const lr = config.learningRate || 0.005;
    switch (opt) {
      case 'adam':
        return `Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate per weight based on gradient momentum (1st moment) and gradient variance (2nd moment). It navigates saddle points quickly at a learning rate of ${lr}.`;
      case 'adamw':
        return `AdamW decouples weight decay (L2 regularization) directly from gradient updates. This ensures true L2 regularization for adaptive learning rates, yielding better generalization. Current learning rate: ${lr}.`;
      case 'sgd':
        return `Stochastic Gradient Descent (SGD) adjusts weights in the direct opposite path of the batch gradient. We use momentum (0.9) to build speed down valleys and damp oscillations. Current learning rate: ${lr}.`;
      case 'rmsprop':
        return `RMSprop scales the learning rate by dividing by a running average of the squared gradients. This prevents learning rates from shrinking too fast, resolving steep gradients. Current learning rate: ${lr}.`;
      default:
        return `Optimizer updates weights at a learning rate of ${lr}.`;
    }
  }, [config.optimizer, config.learningRate]);

  // Topology description
  const topologyExplanation = useMemo(() => {
    const layers = config.hiddenLayers || [];
    if (layers.length === 0) return 'The network is linear (mapping inputs directly to outputs).';
    
    const layerRoles = [
      "Layer 1 (Inputs & Basic Patterns): Takes scaled input features (Elo, ranking difference, home advantage) and projects them into the initial hidden space. It models simple, individual feature correlations.",
      "Layer 2 (Feature Integration): Combines simple signals from the first layer into more complex interactions (e.g. coupling rest days with recent form to evaluate squad form decay).",
      "Layer 3 (Abstract Tactical Balance): Synthesizes composite properties into high-level features (e.g. weighing a team's home scoring form against the opponent's defensive ranking).",
      "Layer 4 (Decision Boundary Shaping): Condenses complex abstract features into distinct margins that help differentiate between Home Win, Draw, and Away Win logits.",
      "Layer 5 (Pre-logits Refinement): Performs the final smoothing of the activation space to prepare clean inputs for the final classification layer, minimizing predictions entropy."
    ];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.2rem' }}>
        <span>
          Propagates inputs through a feedforward path of <strong>{layers.length} hidden layer(s)</strong>: <code>{layers.join(' → ')}</code>.
        </span>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
          background: 'rgba(255,255,255,0.02)',
          padding: '0.5rem 0.75rem',
          borderRadius: '6px',
          border: '1px solid rgba(255,255,255,0.06)',
          fontSize: '0.72rem',
          lineHeight: '1.4',
          color: 'rgba(255, 255, 255, 0.85)'
        }}>
          {layers.map((neurons, idx) => {
            const role = layerRoles[idx] || `Layer ${idx + 1}: General feature abstraction.`;
            return (
              <div key={idx} style={{ borderBottom: idx < layers.length - 1 ? '1px solid rgba(255,255,255,0.03)' : 'none', paddingBottom: idx < layers.length - 1 ? '0.3rem' : 0 }}>
                <strong style={{ color: C.cyan }}>Layer {idx + 1} ({neurons} Neurons):</strong> {role}
              </div>
            );
          })}
        </div>
      </div>
    );
  }, [config.hiddenLayers]);

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', padding: '1.25rem' }}>
      
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="card-title" style={{ margin: 0 }}>🔬 DECISION & ARCHITECTURE EXPLAINER</h3>
        {isTraining && (
          <span className="pulse-ring" style={{ fontSize: '0.7rem', color: C.cyan, fontWeight: 800 }}>
            🧠 CALCULATING...
          </span>
        )}
      </div>

      <p style={{ color: C.text2, fontSize: '0.75rem', margin: 0, lineHeight: 1.4 }}>
        This dynamic panel interprets how your customized hyperparameters and topology settings translate to decisions inside the PyTorch network.
      </p>

      {/* Dynamic Summary Blocks */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.3rem', fontSize: '0.75rem', lineHeight: '1.45' }}>
        
        {/* Features */}
        <div>
          <strong style={{ color: '#fff', display: 'block', marginBottom: '0.15rem' }}>📋 Input Features Selection:</strong>
          <span style={{ color: C.text2 }}>{featureExplanation}</span>
        </div>

        {/* Topology */}
        <div>
          <strong style={{ color: '#fff', display: 'block', marginBottom: '0.15rem' }}>🕸️ Network Structure:</strong>
          <span style={{ color: C.text2 }}>{topologyExplanation}</span>
        </div>

        {/* Activation */}
        <div>
          <strong style={{ color: '#fff', display: 'block', marginBottom: '0.15rem' }}>⚡ Node Firing Behavior:</strong>
          <span style={{ color: C.text2 }}>{activationExplanation}</span>
        </div>

        {/* Loss */}
        <div>
          <strong style={{ color: '#fff', display: 'block', marginBottom: '0.15rem' }}>⚖️ Optimization Goal (Loss):</strong>
          <span style={{ color: C.text2 }}>{lossExplanation}</span>
        </div>

        {/* Optimizer */}
        <div>
          <strong style={{ color: '#fff', display: 'block', marginBottom: '0.15rem' }}>🚀 Learning Rate & Step Updates:</strong>
          <span style={{ color: C.text2 }}>{optimizerExplanation}</span>
        </div>

      </div>

      {/* Performance verdict */}
      {metrics && (
        <div style={{
          marginTop: '0.5rem',
          padding: '0.7rem 0.9rem',
          background: 'rgba(0,242,254,0.04)',
          border: `1px solid ${C.cyan}25`,
          borderRadius: '6px',
          fontSize: '0.75rem',
          lineHeight: 1.45
        }}>
          <div style={{ fontWeight: 800, color: C.cyan, fontFamily: 'var(--font-display)', fontSize: '0.72rem', letterSpacing: '0.04em', marginBottom: '0.2rem' }}>
            📊 TRAINING OUTCOME (EPOCH {metrics.epoch})
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.2rem', color: C.text2 }}>
            <div>Validation Acc: <strong style={{ color: C.green }}>{metrics.valAcc ?? "0.0"}%</strong></div>
            <div>Train Loss: <strong style={{ color: '#fff' }}>{(metrics.trainLoss ?? 0).toFixed(4)}</strong></div>
            <div>Training Acc: <strong style={{ color: C.cyan }}>{metrics.trainAcc ?? "0.0"}%</strong></div>
            <div>Val Loss: <strong style={{ color: '#fff' }}>{(metrics.valLoss ?? 0).toFixed(4)}</strong></div>
          </div>
          <p style={{ margin: '0.4rem 0 0 0', fontSize: '0.7rem', color: C.text2 }}>
            💡 This customized configuration achieves <strong style={{ color: '#fff' }}>{metrics.valAcc}% validation accuracy</strong>. 
            {parseFloat(metrics.valAcc) > 58 ? ' This represents excellent generalisation performance for football prediction, outperforming basic Elo models.' : 
             parseFloat(metrics.valAcc) > 55 ? ' This is a solid, stable baseline classifier.' : 
             ' Accuracy is slightly lower than typical baselines. Try adding more features or adjusting layer sizes to increase representation capacity.'}
          </p>
        </div>
      )}

      {/* Collapsible step-by-step prediction math */}
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.6rem', marginTop: '0.2rem' }}>
        <button
          onClick={() => setShowMathSection(!showMathSection)}
          style={{
            background: 'none',
            border: 'none',
            color: C.cyan,
            cursor: 'pointer',
            fontSize: '0.72rem',
            fontFamily: 'var(--font-display)',
            fontWeight: 800,
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem'
          }}
        >
          {showMathSection ? '▼ HIDE PREDICTION MATH STEPS' : '▶ SHOW HOW PREDICTION IS MADE (MATH)'}
        </button>

        {showMathSection && (
          <div className="slide-up" style={{
            fontSize: '0.7rem',
            color: C.text2,
            lineHeight: '1.5',
            marginTop: '0.6rem',
            background: C.bgDark,
            padding: '0.65rem 0.8rem',
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.04)',
            fontFamily: 'monospace',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem'
          }}>
            <div>
              <strong style={{ color: '#fff' }}>[STEP 1: Normalization]</strong><br/>
              Subtract mean and divide by scale for all selected inputs:<br/>
              <span style={{ color: C.cyan }}>x_scaled = (x - mean) / scale</span>
            </div>
            
            <div>
              <strong style={{ color: '#fff' }}>[STEP 2: Hidden Feedforward]</strong><br/>
              For hidden layer $L_k$ with weights $W_k$ and biases $b_k$:<br/>
              <span style={{ color: C.cyan }}>z_k = W_k * a_(k-1) + b_k</span><br/>
              Apply activation:<br/>
              <span style={{ color: C.cyan }}>a_k = {config.activation.toUpperCase()}(z_k)</span>
            </div>

            <div>
              <strong style={{ color: '#fff' }}>[STEP 3: Output Projection]</strong><br/>
              Propagate final hidden activations to output layer:<br/>
              <span style={{ color: C.cyan }}>z_out = W_out * a_last + b_out</span>
            </div>

            <div>
              <strong style={{ color: '#fff' }}>[STEP 4: Softmax Probabilities]</strong><br/>
              Calculate class probabilities using Softmax function:<br/>
              <span style={{ color: C.cyan }}>P(class_i) = exp(z_out[i]) / Σ(exp(z_out[j]))</span><br/>
              The highest probability determines the predicted match outcome.
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
