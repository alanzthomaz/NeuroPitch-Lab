// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import NetworkBuilder from './components/NetworkBuilder';
import ANNVisualizer from './components/ANNVisualizer';
import TrainingLab from './components/TrainingLab';
import InspectorPanel from './components/InspectorPanel';
import SandboxView from './components/SandboxView';
import SVGBracket from './components/SVGBracket';
import ExperimentTracker from './components/ExperimentTracker';
import EducationalHub from './components/EducationalHub';
import ModelExplainer from './components/ModelExplainer';

export default function App() {
  // Collapsible panel toggles
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);

  // Studio Configuration State
  const [labConfig, setLabConfig] = useState({
    hiddenLayers: [16, 8],
    activation: 'relu',
    optimizer: 'adam',
    learningRate: 0.005,
    epochs: 30,
    batchSize: 64,
    features: ['elo_home', 'elo_away', 'rank_diff', 'home_adv', 'stat_prob_home', 'stat_prob_away'],
    batchNorm: false,
    gradClipping: false,
    weightInit: 'xavier_uniform',
    lossFunc: 'crossentropy',
    scheduler: 'none',
    momentum: 0.9,
    weightDecay: 0.0001,
    earlyStopping: 0,
    randomSeed: 42
  });

  // Training Stream States
  const [isTraining, setIsTraining] = useState(false);
  const [epochHistoryStates, setEpochHistoryStates] = useState([]); 
  const [timeMachineIndex, setTimeMachineIndex] = useState(-1); 

  // Visualizer states
  const [networkState, setNetworkState] = useState([]);
  const [activations, setActivations] = useState([]);
  const [confusionMatrix, setConfusionMatrix] = useState([]);
  const [rocCurve, setRocCurve] = useState({});
  const [epochHistory, setEpochHistory] = useState([]);
  const [customWeightOverrides, setCustomWeightOverrides] = useState({});
  const [sandboxExplanation, setSandboxExplanation] = useState([]);

  // Inspection states
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedConnection, setSelectedConnection] = useState(null);

  // Tournament Bracket & Win probabilities data
  const [bracketData, setBracketData] = useState([]);
  const [winnerFreq, setWinnerFreq] = useState([]);

  // Sandbox reference matchup
  const [refMatch, setRefMatch] = useState({ team1: 'france', team2: 'spain', home_team: 'france' });
  const [refFeatures, setRefFeatures] = useState({});
  const [scalerParams, setScalerParams] = useState(null);

  // Fetch active model config, weights, and activations
  const fetchActiveModel = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/model/active?team1=${refMatch.team1}&team2=${refMatch.team2}&home_team=${refMatch.home_team}`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === "active") {
          setLabConfig(prev => ({
            ...prev,
            features: data.config.features,
            hiddenLayers: data.config.hidden_layers,
            activation: data.config.activation
          }));
          setNetworkState(data.network_state || []);
          setActivations(data.ref_activations || []);
          setRefFeatures(data.ref_features || {});
          setScalerParams({
            mean: data.scaler_mean || [],
            var: data.scaler_var || [],
            scale: data.scaler_scale || []
          });
        }
      }
    } catch (err) {
      console.error("Failed to fetch active model details:", err);
    }
  };

  useEffect(() => {
    fetchActiveModel();
  }, [refMatch]);

  // Fetch bracket state
  const fetchBracket = async () => {
    try {
      const response = await fetch("http://localhost:8000/bracket");
      if (response.ok) {
        const data = await response.json();
        setBracketData(data);
      }
    } catch (e) {
      console.error("Failed to fetch bracket state", e);
    }
  };

  // Fetch simulation winner odds
  const fetchSimulation = async () => {
    try {
      const response = await fetch("http://localhost:8000/simulate?num_sims=500&condition=true");
      if (response.ok) {
        const data = await response.json();
        setWinnerFreq(data.winner_freq || []);
      }
    } catch (e) {
      console.error("Failed to run tournament simulations", e);
    }
  };

  useEffect(() => {
    fetchBracket();
    fetchSimulation();
  }, []);

  // Handle SSE training stream
  const handleTrainANN = () => {
    setIsTraining(false);
    setIsTraining(true);
    setEpochHistoryStates([]);
    setTimeMachineIndex(-1);
    
    setEpochHistory([]);
    setNetworkState([]);
    setActivations([]);
    setConfusionMatrix([]);
    setRocCurve({});
    setCustomWeightOverrides({});
    setSandboxExplanation([]);

    const params = new URLSearchParams({
      hidden_layers: labConfig.hiddenLayers.join(','),
      activation: labConfig.activation,
      optimizer: labConfig.optimizer,
      learning_rate: labConfig.learningRate.toString(),
      epochs: labConfig.epochs.toString(),
      batch_size: labConfig.batchSize.toString(),
      features: labConfig.features.join(','),
      l2_regularization: labConfig.weightDecay.toString(),
      weight_init: labConfig.weightInit,
      grad_clipping: labConfig.gradClipping ? 'true' : 'false',
      loss_func: labConfig.lossFunc,
      scheduler: labConfig.scheduler,
      momentum: labConfig.momentum.toString(),
      weight_decay: labConfig.weightDecay.toString(),
      early_stopping: labConfig.earlyStopping.toString(),
      random_seed: labConfig.randomSeed.toString(),
      team1: refMatch.team1,
      team2: refMatch.team2,
      home_team: refMatch.home_team
    });

    const eventSource = new EventSource(`http://localhost:8000/api/train/stream?${params.toString()}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === 'starting' || data.status === 'compiling') {
        console.log("Autograd training stream status update:", data.status, data.message || "");
      } else if (data.status === 'complete' || data.status === 'early_stopped') {
        eventSource.close();
        setIsTraining(false);
        fetchActiveModel();
        fetchBracket(); 
        fetchSimulation(); // Refresh simulation probabilities based on new weights!
        if (data.status === 'early_stopped') {
          alert(`Early stopping triggered at Epoch ${data.epoch} due to validation loss stagnation!`);
        }
      } else if (data.error) {
        alert("Training Error: " + data.error);
        eventSource.close();
        setIsTraining(false);
      } else {
        setEpochHistoryStates(prev => {
          const lastStored = prev[prev.length - 1] || {};
          const mergedData = {
            ...data,
            network_state: (data.network_state && data.network_state.length > 0) ? data.network_state : (lastStored.network_state || []),
            ref_activations: (data.ref_activations && data.ref_activations.length > 0) ? data.ref_activations : (lastStored.ref_activations || []),
            confusion_matrix: (data.confusion_matrix && data.confusion_matrix.length > 0) ? data.confusion_matrix : (lastStored.confusion_matrix || []),
            roc_curve: (data.roc_curve && Object.keys(data.roc_curve).length > 0) ? data.roc_curve : (lastStored.roc_curve || {})
          };
          const updated = [...prev, mergedData];
          setTimeMachineIndex(updated.length - 1);
          return updated;
        });

        setEpochHistory(prev => [...prev, {
          epoch: data.epoch,
          train_loss: data.train_loss,
          val_loss: data.val_loss,
          train_acc: data.train_acc,
          val_acc: data.val_acc
        }]);

        if (data.network_state && data.network_state.length > 0) {
          setNetworkState(data.network_state);
        }
        if (data.ref_activations && data.ref_activations.length > 0) {
          setActivations(data.ref_activations);
        }
        if (data.confusion_matrix && data.confusion_matrix.length > 0) {
          setConfusionMatrix(data.confusion_matrix);
        }
        if (data.roc_curve && Object.keys(data.roc_curve).length > 0) {
          setRocCurve(data.roc_curve);
        }
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE stream error:", err);
      eventSource.close();
      setIsTraining(false);
    };
  };

  // Time Machine Epoch Scrubbing
  const handleTimeMachineChange = (val) => {
    const idx = Number(val);
    setTimeMachineIndex(idx);
    
    if (epochHistoryStates[idx]) {
      const state = epochHistoryStates[idx];
      setNetworkState(state.network_state || []);
      setActivations(state.ref_activations || []);
      setConfusionMatrix(state.confusion_matrix || []);
      setRocCurve(state.roc_curve || {});
      
      setEpochHistory(epochHistoryStates.slice(0, idx + 1).map(h => ({
        epoch: h.epoch,
        train_loss: h.train_loss,
        val_loss: h.val_loss,
        train_acc: h.train_acc,
        val_acc: h.val_acc
      })));
    }
  };

  // Import model parameter JSON
  const handleImportModel = async (modelData) => {
    try {
      const response = await fetch("http://localhost:8000/api/model/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelData)
      });
      if (response.ok) {
        const data = await response.json();
        alert("Model imported and cached successfully on prediction engine!");
        setLabConfig(modelData.config);
        if (data.network_state) setNetworkState(data.network_state);
        if (data.ref_activations) setActivations(data.ref_activations);
        if (data.ref_features) setRefFeatures(data.ref_features);
        setScalerParams({
          mean: modelData.scaler_mean || [],
          var: modelData.scaler_var || [],
          scale: modelData.scaler_scale || []
        });
        fetchBracket();
        fetchSimulation();
      } else {
        const err = await response.json();
        alert("Model import failed: " + err.detail);
      }
    } catch (e) {
      alert("Network error importing model variables: " + e.message);
    }
  };

  const handleSelectNode = (node) => {
    setSelectedConnection(null);
    setSelectedNode(node);
  };

  const handleSelectConnection = (conn) => {
    setSelectedNode(null);
    setSelectedConnection(conn);
  };

  const handleSelectBracketMatch = (teamA, teamB) => {
    setRefMatch({ team1: teamA, team2: teamB, home_team: teamA });
    document.getElementById("sandbox-section")?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="app-container">
      {/* Header Banner */}
      <header className="app-header">
        <div className="logo-section">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" fill="none" style={{ filter: 'drop-shadow(0 0 4px #00f2fe)', marginRight: '10px' }}>
            <style>{`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
              @keyframes pulse {
                0%, 100% { opacity: 0.4; }
                50% { opacity: 0.9; }
              }
              .spinning-ball {
                transform-origin: 16px 16px;
                animation: spin 10s linear infinite;
              }
              .pulsing-line {
                animation: pulse 2s ease-in-out infinite;
              }
            `}</style>
            <circle cx="16" cy="16" r="14" stroke="#ffffff" strokeWidth="1.2" opacity="0.15" />
            <g className="spinning-ball">
              <path d="M16 4 L11 10 L16 16 L21 10 Z" stroke="#8a2be2" strokeWidth="1" className="pulsing-line" style={{ animationDelay: '0s' }} />
              <path d="M16 28 L11 22 L16 16 L21 22 Z" stroke="#00f2fe" strokeWidth="1" className="pulsing-line" style={{ animationDelay: '0.5s' }} />
              <path d="M4 16 L11 10 L11 22 Z" stroke="#8a2be2" strokeWidth="1" opacity="0.5" />
              <path d="M28 16 L21 10 L21 22 Z" stroke="#00f2fe" strokeWidth="1" opacity="0.5" />

              <line x1="4" y1="16" x2="11" y2="10" stroke="#8a2be2" strokeWidth="1" opacity="0.5" />
              <line x1="28" y1="16" x2="21" y2="10" stroke="#00f2fe" strokeWidth="1" opacity="0.5" />
              <line x1="4" y1="16" x2="11" y2="22" stroke="#8a2be2" strokeWidth="1" opacity="0.5" />
              <line x1="28" y1="16" x2="21" y2="22" stroke="#00f2fe" strokeWidth="1" opacity="0.5" />
              <line x1="16" y1="4" x2="11" y2="10" stroke="#8a2be2" strokeWidth="1" opacity="0.6" />
              <line x1="16" y1="4" x2="21" y2="10" stroke="#00f2fe" strokeWidth="1" opacity="0.6" />
              <line x1="16" y1="28" x2="11" y2="22" stroke="#8a2be2" strokeWidth="1" opacity="0.6" />
              <line x1="16" y1="28" x2="21" y2="22" stroke="#00f2fe" strokeWidth="1" opacity="0.6" />

              <circle cx="16" cy="4" r="2" fill="#8a2be2" />
              <circle cx="16" cy="28" r="2" fill="#00f2fe" />
              <circle cx="4" cy="16" r="2" fill="#8a2be2" />
              <circle cx="28" cy="16" r="2" fill="#00f2fe" />
              
              <circle cx="11" cy="10" r="2" fill="#8a2be2" />
              <circle cx="21" cy="10" r="2" fill="#00f2fe" />
              <circle cx="11" cy="22" r="2" fill="#00f2fe" />
              <circle cx="21" cy="22" r="2" fill="#8a2be2" />
              <circle cx="16" cy="16" r="2.5" fill="#ffffff" />
            </g>
          </svg>
          <h1 className="logo-text">NEUROPITCH LAB</h1>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button 
            className="nav-item" 
            onClick={() => setLeftPanelOpen(!leftPanelOpen)}
            style={{ border: leftPanelOpen ? '1px solid var(--cyan)' : '1px solid rgba(255,255,255,0.1)' }}
          >
            {leftPanelOpen ? '❌ HIDE STUDIO CONTROLS' : '⚙️ SHOW CONTROLS'}
          </button>
          <button 
            className="nav-item" 
            onClick={() => setRightPanelOpen(!rightPanelOpen)}
            style={{ border: rightPanelOpen ? '1px solid var(--cyan)' : '1px solid rgba(255,255,255,0.1)' }}
          >
            {rightPanelOpen ? '❌ HIDE INSPECTOR' : '🔬 SHOW INSPECTOR'}
          </button>
        </div>
      </header>

      {/* Main Studio Body */}
      <main className="dashboard-body" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: '100%', padding: '1.5rem' }}>
        
        {/* Three Column Studio Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: `${leftPanelOpen ? '320px' : '0px'} 1fr ${rightPanelOpen ? '320px' : '0px'}`,
          gap: (leftPanelOpen || rightPanelOpen) ? '1.5rem' : '0px',
          transition: 'grid-template-columns 0.3s ease, gap 0.3s ease',
          alignItems: 'start',
          overflow: 'hidden'
        }}>
          
          {/* LEFT PANEL: Builder & Config */}
          <div style={{ display: leftPanelOpen ? 'flex' : 'none', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Short Educational Dataset Header */}
            <div className="glass-card">
              <h3 className="card-title" style={{ margin: 0 }}>📊 STUDY DATASET</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '0.4rem', lineHeight: '1.4' }}>
                Modeling <strong>49,000+ historical international football fixtures</strong> to map weight patterns and output nodes in real-time.
              </p>
            </div>

            <NetworkBuilder
              config={labConfig}
              onChange={setLabConfig}
              onTrain={handleTrainANN}
              isTraining={isTraining}
              networkState={networkState}
              onImportModel={handleImportModel}
              scalerParams={scalerParams}
            />
          </div>

          {/* CENTER PANEL: Interactive SVG Map */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: '0' }}>
            
            <ANNVisualizer
              features={labConfig.features}
              hiddenLayers={labConfig.hiddenLayers}
              networkState={networkState}
              activations={activations}
              isTraining={isTraining}
              onSelectNode={handleSelectNode}
              onSelectConnection={handleSelectConnection}
              selectedNodeId={selectedNode?.id}
              selectedConnectionId={selectedConnection?.id}
              refMatch={refMatch}
              activationName={labConfig.activation}
              customWeightOverrides={customWeightOverrides}
              onWeightOverrideChange={(key, val) => setCustomWeightOverrides(prev => {
                const next = { ...prev };
                if (val === undefined) {
                  delete next[key];
                } else {
                  next[key] = val;
                }
                return next;
              })}
              explanation={sandboxExplanation}
            />

             {/* Time Machine Epoch Scrubber */}
            {epochHistoryStates.length > 0 && (
              <div className="glass-card slide-up" style={{ padding: '0.85rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                  <span style={{ fontFamily: 'var(--font-display)', color: 'var(--cyan)' }}>⏳ TIME MACHINE SCROLLER</span>
                  <span style={{ fontWeight: 800 }}>
                    Epoch {timeMachineIndex + 1} / {epochHistoryStates.length}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={epochHistoryStates.length - 1}
                  value={timeMachineIndex}
                  onChange={(e) => handleTimeMachineChange(e.target.value)}
                  className="time-machine-slider"
                />
                <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                  Drag the slider to restore connection weights, bias glow, error curves, and win probabilities of that specific epoch.
                </span>
              </div>
            )}

            <ModelExplainer
              config={labConfig}
              epochHistory={epochHistory}
              isTraining={isTraining}
            />
          </div>

          {/* RIGHT PANEL: Inspector & Realtime Metrics */}
          <div style={{ display: rightPanelOpen ? 'flex' : 'none', flexDirection: 'column', gap: '1.5rem' }}>
            
            <InspectorPanel
              selectedNode={selectedNode}
              selectedConnection={selectedConnection}
              activationName={labConfig.activation}
              networkState={networkState}
              customWeightOverrides={customWeightOverrides}
              onWeightOverrideChange={(key, val) => setCustomWeightOverrides(prev => {
                const next = { ...prev };
                if (val === undefined) {
                  delete next[key];
                } else {
                  next[key] = val;
                }
                return next;
              })}
            />

            <TrainingLab
              epochHistory={epochHistory}
              confusionMatrix={confusionMatrix}
              rocCurve={rocCurve}
              epochs={labConfig.epochs}
            />
          </div>

        </div>

        {/* BOTTOM AREA: Sandbox & SVG Bracket Symmetrical */}
        
        {/* Prediction Sandbox */}
        <div id="sandbox-section">
          <SandboxView
            refMatch={refMatch}
            activeFeatures={labConfig.features}
            refFeatures={refFeatures}
            onActivationsUpdate={setActivations}
            isTraining={isTraining}
            customWeightOverrides={customWeightOverrides}
            onClearOverrides={() => setCustomWeightOverrides({})}
            explanation={sandboxExplanation}
            onExplanationUpdate={setSandboxExplanation}
          />
        </div>

        {/* Symmetrical connected Bracket and Winner Probabilities Split */}
        <SVGBracket
          bracketData={bracketData}
          winnerFreq={winnerFreq}
          onSelectMatch={handleSelectBracketMatch}
        />

        {/* Persistent Hyperparameter Experiment comparisons */}
        <ExperimentTracker
          currentConfig={labConfig}
          currentMetrics={
            epochHistoryStates.length > 0 ? {
              val_loss: epochHistoryStates[epochHistoryStates.length - 1].val_loss,
              val_acc: (epochHistoryStates[epochHistoryStates.length - 1].val_acc * 100).toFixed(1),
              confusion_matrix: confusionMatrix
            } : null
          }
          onLoadConfig={(config) => {
            setLabConfig(config);
          }}
        />

        {/* Educational Theory & Math Hub */}
        <EducationalHub />

      </main>
    </div>
  );
}
