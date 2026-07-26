// frontend/src/components/ANNVisualizer.jsx
// TRUE TENSOR-DRIVEN FORWARD-PASS VISUALIZER
// Every particle = real activation × weight from model tensors.
// No decorative or random animation.
import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';

// ── Activation functions (mirroring the Python model exactly) ──────────────
const ACTIVATIONS = {
  relu:    x => Math.max(0, x),
  leaky_relu: x => x >= 0 ? x : 0.01 * x,
  sigmoid: x => 1 / (1 + Math.exp(-x)),
  tanh:    x => Math.tanh(x),
  elu:     x => x >= 0 ? x : Math.exp(x) - 1,
  selu:    x => {
    const alpha = 1.6732632423543772, scale = 1.0507009873554805;
    return scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1));
  },
  none:    x => x,
};

function applyActivation(x, name) {
  const fn = ACTIVATIONS[(name || 'relu').toLowerCase()] || ACTIVATIONS.relu;
  return fn(x);
}

// ── Contribution colour ────────────────────────────────────────────────────
function contributionColor(val) {
  const abs = Math.abs(val);
  if (abs < 0.01) return { r: 100, g: 110, b: 130 };   // near-zero → slate gray
  if (val > 0)   return { r: 0, g: 242, b: 254 };       // positive  → cyan
  return           { r: 255, g: 0, b: 127 };             // negative  → magenta
}
function colorString({ r, g, b }, alpha = 1) {
  return `rgba(${r},${g},${b},${alpha})`;
}

const W = 780, H = 460, PAD = 68, MAX_NODES = 12;

export default function ANNVisualizer({
  features = [],
  hiddenLayers = [],
  networkState = [],
  activations = [],
  isTraining = false,
  onSelectNode,
  onSelectConnection,
  selectedNodeId,
  selectedConnectionId,
  refMatch = { team1: 'France', team2: 'Spain' },
  activationName = 'relu',
  customWeightOverrides = {},
  onWeightOverrideChange,
  explanation = []
}) {
  // ── Refs ─────────────────────────────────────────────────────────────────
  const svgRef    = useRef(null);
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);

  // pan / zoom
  const [scale, setScale] = useState(1.0);
  const [pan,   setPan  ] = useState({ x: 0, y: 0 });
  const [drag,  setDrag ] = useState(null);
  const [hovNode, setHovNode] = useState(null);

  // speed & explain mode (kept in refs so RAF loop reads them without re-render lag)
  const speedRef   = useRef(1);
  const explainRef = useRef(false);
  const [speed,   setSpeed  ] = useState(1);
  const [explain, setExplain] = useState(false);

  // animation FSM
  //   state: 'idle' | 'animating' | 'paused' | 'done'
  const fsmRef = useRef({ state: 'idle', layerIdx: 0 });
  const [fsmState,   setFsmState  ] = useState('idle');
  const [fsmLayer,   setFsmLayer  ] = useState(0);

  // live particles – mutated directly inside RAF, never trigger re-render
  const particlesRef = useRef([]);
  // per-connection glow value (0→1, decays each frame)
  const connGlowRef  = useRef({});   // key: `${srcId}->${dstId}`
  // per-node receive-glow
  const nodeGlowRef  = useRef({});   // key: nodeId → 0..1

  // tooltip state (particle hover)
  const [tooltip, setTooltip] = useState(null);

  // selected-neuron panel data (updated on selectedNodeId change)
  const [neuronPanel, setNeuronPanel] = useState(null);

  // ── Layer/Node structure (derived from props) ─────────────────────────────
  const layers = useMemo(() => {
    const out = [];
    // Input
    const inputAct = activations.find(a => a.type === 'input');
    out.push({
      name: 'Input', type: 'input',
      nodes: features.map((f, i) => ({
        id: `in-${i}`, label: f.replace(/_/g, ' ').toUpperCase(),
        index: i, layerIdx: 0, bias: 0,
        value: inputAct?.values?.[i] ?? 0,
      })),
    });
    // Hidden
    hiddenLayers.forEach((dim, li) => {
      const disp = Math.min(dim, MAX_NODES);
      const hidAct = activations.find(a => a.name === `Hidden_${li}`);
      const lState = networkState[li] || {};
      out.push({
        name: `Hidden ${li + 1}`, type: 'hidden',
        nodes: Array.from({ length: disp }, (_, i) => ({
          id: `h-${li}-${i}`, label: `H${li+1}_${i}`,
          index: i, layerIdx: li + 1,
          bias: lState.biases?.[i] ?? 0,
          value: hidAct?.values?.[i] ?? 0,
        })),
        totalNodes: dim,
      });
    });
    // Output
    const outAct  = activations.find(a => a.name === 'Output');
    const lsOut   = networkState[networkState.length - 1] || {};
    out.push({
      name: 'Output', type: 'output',
      nodes: ['HOME WIN', 'DRAW', 'AWAY WIN'].map((label, i) => ({
        id: `out-${i}`, label,
        index: i, layerIdx: hiddenLayers.length + 1,
        bias: lsOut.biases?.[i] ?? 0,
        value: outAct?.values?.[i] ?? 0,
      })),
    });
    return out;
  }, [features, hiddenLayers, networkState, activations]);

  // ── Node positions ────────────────────────────────────────────────────────
  const nodePos = useMemo(() => {
    const map = {};
    const xStep = (W - 2 * PAD) / Math.max(1, layers.length - 1);
    layers.forEach((layer, li) => {
      const x = PAD + li * xStep;
      const n = layer.nodes.length;
      const yStep = n > 1 ? (H - 2 * PAD) / (n - 1) : 0;
      layer.nodes.forEach((node, ni) => {
        map[node.id] = {
          x, y: n === 1 ? H / 2 : PAD + ni * yStep,
          ...node,
        };
      });
    });
    return map;
  }, [layers]);

  // ── Connection list (static structure – weights from networkState) ────────
  const connections = useMemo(() => {
    const list = [];
    for (let li = 0; li < layers.length - 1; li++) {
      const src = layers[li], dst = layers[li + 1];
      const ws  = networkState[li]?.weights || [];
      const grads = networkState[li]?.gradients || [];
      src.nodes.forEach(s => {
        dst.nodes.forEach(d => {
          const overrideKey = `${li}_${s.index}_${d.index}`;
          const w_base = ws[d.index]?.[s.index] ?? 0;
          const w = customWeightOverrides[overrideKey] !== undefined ? customWeightOverrides[overrideKey] : w_base;
          const g = grads[d.index]?.[s.index] ?? 0;
          list.push({
            id: `${s.id}->${d.id}`,
            srcId: s.id, dstId: d.id,
            layerIdx: li, srcIdx: s.index, dstIdx: d.index, weight: w, gradient: g,
            x1: nodePos[s.id]?.x ?? 0, y1: nodePos[s.id]?.y ?? 0,
            x2: nodePos[d.id]?.x ?? 0, y2: nodePos[d.id]?.y ?? 0,
          });
        });
      });
    }
    return list;
  }, [layers, nodePos, networkState, customWeightOverrides]);

  // ── Per-node weighted sums (computed JS-side from tensors) ───────────────
  const weightedSums = useMemo(() => {
    const sums = {};
    layers[0]?.nodes.forEach(n => { sums[n.id] = n.value; });
    for (let li = 0; li < layers.length - 1; li++) {
      const src = layers[li], dst = layers[li + 1];
      const ws  = networkState[li]?.weights || [];
      dst.nodes.forEach(d => {
        let z = d.bias;
        src.nodes.forEach(s => {
          const overrideKey = `${li}_${s.index}_${d.index}`;
          const w_base = ws[d.index]?.[s.index] ?? 0;
          const w = customWeightOverrides[overrideKey] !== undefined ? customWeightOverrides[overrideKey] : w_base;
          z += s.value * w;
        });
        sums[d.id] = z;
      });
    }
    return sums;
  }, [layers, networkState, customWeightOverrides]);

  // ── Softmax on output logits ──────────────────────────────────────────────
  const outputProbs = useMemo(() => {
    const outLayer = layers[layers.length - 1];
    if (!outLayer) return [];
    const logits = outLayer.nodes.map(n => n.value);
    const m = Math.max(...logits);
    const ex = logits.map(v => Math.exp(v - m));
    const s  = ex.reduce((a, b) => a + b, 0);
    return ex.map(e => e / (s || 1));
  }, [layers]);

  const winnerIdx = useMemo(() => {
    let mi = 0;
    outputProbs.forEach((p, i) => { if (p > outputProbs[mi]) mi = i; });
    return mi;
  }, [outputProbs]);

  // ── Emit particles for one layer transition ───────────────────────────────
  const emitLayer = useCallback((li) => {
    if (li >= layers.length - 1) return;
    const srcLayer = layers[li];
    const dstLayer = layers[li + 1];
    const ws = networkState[li]?.weights || [];
    const newParticles = [];

    srcLayer.nodes.forEach(s => {
      dstLayer.nodes.forEach(d => {
        const w    = ws[d.index]?.[s.index] ?? 0;
        const contrib = s.value * w;
        const col  = contributionColor(contrib);
        const abs  = Math.abs(contrib);
        // speed proportional to |contribution| so strong signals travel fast
        const baseSpeed = 0.012 * speedRef.current;
        const spd  = baseSpeed * (0.5 + Math.min(1.5, abs * 3));
        const sz   = 2.5 + Math.min(5, abs * 8);

        newParticles.push({
          srcId: s.id, dstId: d.id,
          x1: nodePos[s.id]?.x ?? 0, y1: nodePos[s.id]?.y ?? 0,
          x2: nodePos[d.id]?.x ?? 0, y2: nodePos[d.id]?.y ?? 0,
          x: nodePos[s.id]?.x ?? 0,  y: nodePos[s.id]?.y ?? 0,
          progress: 0, speed: spd,
          weight: w, activation: s.value,
          contrib, col, sz,
          srcLabel: s.label, dstLabel: d.label,
          dstBias: d.bias,
          done: false,
        });
      });
    });

    particlesRef.current = newParticles;
  }, [layers, nodePos, networkState]);

  // ── Emit backward gradient particles for backpropagation ─────────────────
  const emitBackwardLayer = useCallback((li) => {
    if (li < 0 || li >= layers.length - 1) return;
    const srcLayer = layers[li];     // Left layer (receives gradient)
    const dstLayer = layers[li + 1]; // Right layer (sends gradient)
    const grads = networkState[li]?.gradients || [];
    const newParticles = [];

    dstLayer.nodes.forEach(d => {
      srcLayer.nodes.forEach(s => {
        const g = grads[d.index]?.[s.index] ?? 0;
        const col = contributionColor(-g * 5.0); // scale up for visual representation
        const abs = Math.abs(g);
        const baseSpeed = 0.012 * speedRef.current;
        const spd = baseSpeed * (0.5 + Math.min(1.5, abs * 12.0));
        const sz = 2.5 + Math.min(5, abs * 20.0);

        newParticles.push({
          srcId: d.id, dstId: s.id, // travelling from d (right) to s (left)
          x1: nodePos[d.id]?.x ?? 0, y1: nodePos[d.id]?.y ?? 0,
          x2: nodePos[s.id]?.x ?? 0, y2: nodePos[s.id]?.y ?? 0,
          x: nodePos[d.id]?.x ?? 0,  y: nodePos[d.id]?.y ?? 0,
          progress: 0, speed: spd,
          gradient: g, col, sz,
          srcLabel: d.label, dstLabel: s.label,
          done: false,
          isBackward: true
        });
      });
    });

    particlesRef.current = newParticles;
  }, [layers, nodePos, networkState]);

  // ── FSM: restart ──────────────────────────────────────────────────────────
  const restart = useCallback(() => {
    particlesRef.current = [];
    connGlowRef.current  = {};
    nodeGlowRef.current  = {};
    fsmRef.current = { state: 'forward_animating', layerIdx: 0 };
    setFsmState('forward_animating');
    setFsmLayer(0);
    emitLayer(0);
  }, [emitLayer]);

  // ── FSM: resume after pause (Explain mode) ───────────────────────────────
  const resume = useCallback(() => {
    const isBack = fsmRef.current.state === 'backward_paused' && fsmRef.current.layerIdx > 0;
    if (isBack) {
      const nextLi = fsmRef.current.layerIdx - 1;
      fsmRef.current = { state: 'backward_animating', layerIdx: nextLi };
      setFsmState('backward_animating');
      setFsmLayer(nextLi);
      emitBackwardLayer(nextLi);
    } else {
      const nextLi = fsmRef.current.layerIdx + 1;
      if (nextLi >= layers.length - 1) {
        // start backward pass
        const startBackLi = layers.length - 2;
        fsmRef.current = { state: 'backward_animating', layerIdx: startBackLi };
        setFsmState('backward_animating');
        setFsmLayer(startBackLi);
        emitBackwardLayer(startBackLi);
        return;
      }
      fsmRef.current = { state: 'forward_animating', layerIdx: nextLi };
      setFsmState('forward_animating');
      setFsmLayer(nextLi);
      emitLayer(nextLi);
    }
  }, [layers, emitLayer, emitBackwardLayer]);

  // ── RAF render loop ───────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      ctx.clearRect(0, 0, W, H);
      ctx.save();
      ctx.translate(pan.x, pan.y);
      ctx.scale(scale, scale);

      const fsm = fsmRef.current;
      const particles = particlesRef.current;

      // ── Advance particles ─────────────────────────────────────────────
      if (fsm.state === 'forward_animating') {
        let allDone = particles.length > 0;
        particles.forEach(p => {
          if (p.done) return;
          p.progress += p.speed * speedRef.current;
          if (p.progress >= 1) {
            p.progress = 1;
            p.done = true;
            // light up destination node
            nodeGlowRef.current[p.dstId] = 1.0;
            // light up connection
            connGlowRef.current[p.srcId + '->' + p.dstId] = 1.0;
          } else {
            allDone = false;
          }
          p.x = p.x1 + (p.x2 - p.x1) * p.progress;
          p.y = p.y1 + (p.y2 - p.y1) * p.progress;
        });

        // when all particles land → check next layer or pause
        if (allDone && particles.length > 0) {
          particlesRef.current = [];
          const nextLi = fsm.layerIdx + 1;
          if (nextLi >= layers.length - 1) {
            fsmRef.current = { state: 'forward_paused', layerIdx: nextLi };
            setFsmState('forward_paused');
            setFsmLayer(nextLi);
            setTimeout(() => {
              if (fsmRef.current.state === 'forward_paused') {
                const startBackLi = layers.length - 2;
                fsmRef.current = { state: 'backward_animating', layerIdx: startBackLi };
                setFsmState('backward_animating');
                setFsmLayer(startBackLi);
                emitBackwardLayer(startBackLi);
              }
            }, 800 / speedRef.current);
          } else if (explainRef.current) {
            fsmRef.current = { state: 'forward_paused', layerIdx: fsm.layerIdx };
            setFsmState('forward_paused');
            setFsmLayer(fsm.layerIdx);
          } else {
            // brief inter-layer delay then emit next
            setTimeout(() => {
              const nextNextLi = fsmRef.current.layerIdx + 1;
              fsmRef.current = { state: 'forward_animating', layerIdx: nextNextLi };
              setFsmState('forward_animating');
              setFsmLayer(nextNextLi);
              emitLayer(nextNextLi);
            }, 260 / speedRef.current);
            fsmRef.current = { state: 'forward_animating', layerIdx: nextLi };
          }
        }
      } else if (fsm.state === 'backward_animating') {
        let allDone = particles.length > 0;
        particles.forEach(p => {
          if (p.done) return;
          p.progress += p.speed * speedRef.current;
          if (p.progress >= 1) {
            p.progress = 1;
            p.done = true;
            nodeGlowRef.current[p.dstId] = 1.0;
            // light up backward connection (left node -> right node)
            connGlowRef.current[p.dstId + '->' + p.srcId] = 1.0;
          } else {
            allDone = false;
          }
          p.x = p.x1 + (p.x2 - p.x1) * p.progress;
          p.y = p.y1 + (p.y2 - p.y1) * p.progress;
        });

        if (allDone && particles.length > 0) {
          particlesRef.current = [];
          const prevLi = fsm.layerIdx - 1;
          if (prevLi < 0) {
            fsmRef.current = { state: 'backward_paused', layerIdx: 0 };
            setFsmState('backward_paused');
            setFsmLayer(0);
            setTimeout(() => {
              if (fsmRef.current.state === 'backward_paused') {
                fsmRef.current = { state: 'forward_animating', layerIdx: 0 };
                setFsmState('forward_animating');
                setFsmLayer(0);
                emitLayer(0);
              }
            }, 800 / speedRef.current);
          } else if (explainRef.current) {
            fsmRef.current = { state: 'backward_paused', layerIdx: fsm.layerIdx };
            setFsmState('backward_paused');
            setFsmLayer(fsm.layerIdx);
          } else {
            setTimeout(() => {
              const prevPrevLi = fsmRef.current.layerIdx - 1;
              fsmRef.current = { state: 'backward_animating', layerIdx: prevPrevLi };
              setFsmState('backward_animating');
              setFsmLayer(prevPrevLi);
              emitBackwardLayer(prevPrevLi);
            }, 260 / speedRef.current);
            fsmRef.current = { state: 'backward_animating', layerIdx: prevLi };
          }
        }
      }

      // ── Decay glows ───────────────────────────────────────────────────
      const decay = 0.025;
      Object.keys(connGlowRef.current).forEach(k => {
        connGlowRef.current[k] = Math.max(0, connGlowRef.current[k] - decay);
      });
      Object.keys(nodeGlowRef.current).forEach(k => {
        nodeGlowRef.current[k] = Math.max(0, nodeGlowRef.current[k] - decay * 0.6);
      });

      // ── Draw connection highlights (glow while particle travels) ──────
      particles.forEach(p => {
        if (p.done) return;
        const val = p.isBackward ? p.gradient : p.contrib;
        const abs  = Math.abs(val);
        const glow = 0.15 + Math.min(0.75, abs * 4);
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(p.x1, p.y1);
        ctx.lineTo(p.x, p.y);
        ctx.strokeStyle = colorString(p.col, glow);
        ctx.lineWidth   = 1 + Math.min(4, abs * 6);
        ctx.shadowColor = colorString(p.col, 0.7);
        ctx.shadowBlur  = 8;
        ctx.stroke();
        ctx.restore();
      });

      // ── Draw lingering connection glows (after particle arrives) ──────
      Object.entries(connGlowRef.current).forEach(([key, g]) => {
        if (g <= 0.01) return;
        const conn = connections.find(c => c.id === key);
        if (!conn) return;
        const abs = Math.abs(conn.weight);
        const col = contributionColor(conn.weight);
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(conn.x1, conn.y1);
        ctx.lineTo(conn.x2, conn.y2);
        ctx.strokeStyle = colorString(col, g * 0.4);
        ctx.lineWidth   = 1 + Math.min(3, abs * 4);
        ctx.shadowColor = colorString(col, g * 0.5);
        ctx.shadowBlur  = 6;
        ctx.stroke();
        ctx.restore();
      });

      // ── Draw particles ────────────────────────────────────────────────
      particles.forEach(p => {
        if (p.done) return;
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.sz, 0, Math.PI * 2);
        ctx.fillStyle   = colorString(p.col, 0.95);
        ctx.shadowColor = colorString(p.col, 0.85);
        ctx.shadowBlur  = p.sz * 2.5;
        ctx.fill();
        ctx.restore();
      });

      // ── Draw node receive-pulse rings ─────────────────────────────────
      Object.entries(nodeGlowRef.current).forEach(([id, g]) => {
        if (g <= 0.01) return;
        const np = nodePos[id];
        if (!np) return;
        const val = np.value;
        const col = val >= 0
          ? { r: 0, g: 242, b: 254 }
          : { r: 255, g: 0,   b: 127 };
        ctx.save();
        ctx.beginPath();
        ctx.arc(np.x, np.y, 14 + (1 - g) * 14, 0, Math.PI * 2);
        ctx.strokeStyle = colorString(col, g * 0.6);
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = colorString(col, g);
        ctx.shadowBlur  = 14;
        ctx.stroke();
        ctx.restore();
      });

      ctx.restore(); // end transform
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pan, scale, layers, connections, nodePos]);

  // ── Restart whenever activations/networkState change ─────────────────────
  useEffect(() => {
    if (activations.length > 0 && Object.keys(nodePos).length > 0) {
      const running = fsmRef.current.state === 'forward_animating' || fsmRef.current.state === 'backward_animating';
      if (!running) {
        setTimeout(restart, 100);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activations, networkState]);

  // ── Selected neuron panel ─────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedNodeId) { setNeuronPanel(null); return; }
    let found = null, layerType = '';
    for (const l of layers) {
      const n = l.nodes.find(n => n.id === selectedNodeId);
      if (n) { found = n; layerType = l.type; break; }
    }
    if (!found) { setNeuronPanel(null); return; }

    const incoming = [];
    if (found.layerIdx > 0 && layers[found.layerIdx - 1]) {
      const prev = layers[found.layerIdx - 1];
      const ws   = networkState[found.layerIdx - 1]?.weights || [];
      prev.nodes.forEach(s => {
        const w = ws[found.index]?.[s.index] ?? 0;
        incoming.push({ srcLabel: s.label, weight: w, activation: s.value, contrib: s.value * w });
      });
      incoming.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
    }

    const outgoing = [];
    if (found.layerIdx < layers.length - 1 && layers[found.layerIdx + 1]) {
      const next = layers[found.layerIdx + 1];
      const ws   = networkState[found.layerIdx]?.weights || [];
      next.nodes.forEach(d => {
        const w = ws[d.index]?.[found.index] ?? 0;
        outgoing.push({ dstLabel: d.label, weight: w });
      });
      outgoing.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    }

    const z = weightedSums[found.id] ?? found.value;
    const a = layerType === 'input' ? found.value : applyActivation(z, activationName);

    setNeuronPanel({
      id: found.id, label: found.label, type: layerType,
      bias: found.bias, z, a,
      incoming, outgoing,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedNodeId, layers, networkState, weightedSums, activationName]);

  // ── Tooltip: particle hover ───────────────────────────────────────────────
  const handleSVGMouseMove = useCallback((e) => {
    if (drag) {
      setPan({ x: e.clientX - drag.ox, y: e.clientY - drag.oy });
      return;
    }
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mx = (e.clientX - rect.left - pan.x) / scale;
    const my = (e.clientY - rect.top  - pan.y) / scale;
    let found = null;
    for (const p of particlesRef.current) {
      if (!p.done && Math.hypot(p.x - mx, p.y - my) < 10) {
        found = { ...p, screenX: e.clientX - rect.left, screenY: e.clientY - rect.top };
        break;
      }
    }
    setTooltip(found);
  }, [drag, pan, scale]);

  // ── Zoom / pan ────────────────────────────────────────────────────────────
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.06 : 1 / 1.06;
    setScale(s => Math.min(3, Math.max(0.35, s * f)));
  }, []);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  const handleMouseDown = (e) => {
    if (['svg', 'rect', 'g'].includes(e.target.tagName.toLowerCase()) && !e.target.closest('[data-node]')) {
      setDrag({ ox: e.clientX - pan.x, oy: e.clientY - pan.y });
    }
  };

  // ── CSS vars used inline ──────────────────────────────────────────────────
  const C = {
    cyan: '#00f2fe', gold: '#ffc107', pink: '#ff007f',
    text2: 'rgba(148,163,184,0.8)', bg: '#070a0f',
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="glass-card" style={{ display:'flex', flexDirection:'column', gap:'1rem', flex:1 }}>
      <style>{`
        @keyframes neon-pulse {
          0%,100% { opacity:.7; } 50% { opacity:1; }
        }
        .pulse-ring {
          animation: neon-pulse 1.4s ease-in-out infinite;
          transform-origin: center;
        }
        .fsm-badge {
          display:inline-flex; align-items:center; gap:.35rem;
          padding:.15rem .5rem; border-radius:999px; font-size:.65rem;
          font-family: var(--font-display); font-weight:800; letter-spacing:.04em;
        }
      `}</style>

      {/* ── Top Controls ───────────────────────────────────────────────────── */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:'.75rem' }}>
        <div>
          <h3 className="card-title" style={{ margin:0 }}>🧠 NEURAL LEARNING CYCLE VISUALIZER</h3>
          <p style={{ color:C.text2, fontSize:'.75rem', margin:'.1rem 0 0' }}>
            Visualize forward pass activations (left-to-right) and backward pass gradients (right-to-left) in real time.
            Hover a particle · Click a neuron.
          </p>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'.4rem', flexWrap:'wrap' }}>
          {/* FSM badge */}
          {fsmState === 'forward_animating' && (
            <span className="fsm-badge" style={{ background:'rgba(0,242,254,.12)', border:'1px solid rgba(0,242,254,.3)', color:C.cyan }}>
              ⚡ FORWARD PROPAGATING L{fsmLayer}→L{fsmLayer+1}
            </span>
          )}
          {fsmState === 'forward_paused' && (
            <span className="fsm-badge" style={{ background:'rgba(0,255,128,.1)', border:'1px solid rgba(0,255,128,.3)', color:'#00ff80' }}>
              ✓ FORWARD PASS COMPLETE
            </span>
          )}
          {fsmState === 'backward_animating' && (
            <span className="fsm-badge" style={{ background:'rgba(255,0,127,.12)', border:'1px solid rgba(255,0,127,.3)', color:C.pink }}>
              🔄 BACKPROPAGATING GRADIENTS L{fsmLayer+1}←L{fsmLayer}
            </span>
          )}
          {fsmState === 'backward_paused' && (
            <span className="fsm-badge" style={{ background:'rgba(138,43,226,.1)', border:'1px solid rgba(138,43,226,.3)', color:'#8a2be2' }}>
              ✓ BACKWARD PASS COMPLETE
            </span>
          )}

          {/* Speed */}
          <span style={{ fontSize:'.65rem', color:C.text2 }}>SPEED</span>
          {[0.25, 0.5, 1, 2, 4].map(s => (
            <button key={s} onClick={() => { speedRef.current = s; setSpeed(s); }} className="nav-item"
              style={{
                padding:'.15rem .4rem', fontSize:'.65rem',
                border: speed===s ? `1px solid ${C.cyan}` : '1px solid rgba(255,255,255,.08)',
                background: speed===s ? 'rgba(0,242,254,.15)' : 'transparent',
                color: speed===s ? C.cyan : C.text2,
              }}
            >{s}×</button>
          ))}

          {/* Explain mode */}
          <button onClick={() => { const v=!explain; explainRef.current=v; setExplain(v); restart(); }}
            className="nav-item"
            style={{
              padding:'.2rem .6rem', fontSize:'.7rem',
              border: explain ? `1px solid ${C.gold}` : '1px solid rgba(255,255,255,.08)',
              background: explain ? 'rgba(255,193,7,.12)' : 'transparent',
              color: explain ? C.gold : C.text2,
            }}
          >{explain ? '🔬 EXPLAIN ON' : '🔬 EXPLAIN'}</button>

          {/* Replay */}
          <button onClick={restart} className="nav-item"
            style={{ padding:'.2rem .6rem', fontSize:'.7rem', border:'1px solid rgba(255,255,255,.08)' }}
          >↺ REPLAY</button>

          {/* Zoom */}
          <button className="nav-item" onClick={() => setScale(s=>Math.min(3,s+.1))} style={{ padding:'.2rem .5rem', border:'1px solid rgba(255,255,255,.08)' }}>＋</button>
          <button className="nav-item" onClick={() => setScale(s=>Math.max(.35,s-.1))} style={{ padding:'.2rem .5rem', border:'1px solid rgba(255,255,255,.08)' }}>－</button>
          <button className="nav-item" onClick={() => { setScale(1); setPan({x:0,y:0}); }}
            style={{ padding:'.2rem .5rem', fontSize:'.65rem', border:'1px solid rgba(255,255,255,.08)' }}>RESET</button>
        </div>
      </div>

      {/* ── Canvas + SVG container ─────────────────────────────────────────── */}
      <div style={{ background:'rgba(4,6,11,.9)', borderRadius:8, border:'1px solid rgba(255,255,255,.04)', position:'relative', overflow:'hidden' }}>

        {/* Match badge */}
        <div style={{ position:'absolute', top:10, left:10, zIndex:10, pointerEvents:'none',
          background:'rgba(0,0,0,.65)', padding:'.3rem .7rem', borderRadius:4,
          border:`1px solid ${C.cyan}33`, fontSize:'.72rem', color:C.cyan, fontWeight:800 }}>
          {refMatch.team1.toUpperCase()} vs {refMatch.team2.toUpperCase()}
        </div>

        {/* Layer labels */}
        {layers.length > 0 && (() => {
          const xStep = (W - 2*PAD) / Math.max(1, layers.length - 1);
          return layers.map((l, li) => (
            <div key={li} style={{
              position:'absolute', top:8, zIndex:10, pointerEvents:'none',
              left: `calc(${(PAD + li * xStep) / W * 100}%)`,
              transform: 'translateX(-50%)',
              fontSize:'.6rem', color: l.type==='output' ? C.gold : C.text2,
              fontFamily:'var(--font-display)', fontWeight:800, letterSpacing:'.05em',
              background:'rgba(0,0,0,.5)', padding:'.15rem .4rem', borderRadius:3,
            }}>{l.name.toUpperCase()}</div>
          ));
        })()}

        {/* SVG — static connection lines & nodes */}
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`}
          style={{ width:'100%', height:'auto', minHeight:380, display:'block',
            cursor: drag ? 'grabbing' : 'grab', position:'relative', zIndex:2 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleSVGMouseMove}
          onMouseUp={() => setDrag(null)}
          onMouseLeave={() => { setDrag(null); setTooltip(null); }}
        >
          <rect width={W} height={H} fill="transparent" />
          <g transform={`translate(${pan.x},${pan.y}) scale(${scale})`}>

            {/* Static connection lines */}
            <g>
              {connections.map(c => {
                const abs = Math.abs(c.weight);
                const col = contributionColor(c.weight);
                const isSel = selectedConnectionId === c.id;
                const isHov = hovNode === c.srcId || hovNode === c.dstId;
                const alpha = (hovNode && !isHov) ? .05 : (isSel ? .9 : .1 + Math.min(.5, abs*.8));
                return (
                  <line key={c.id}
                    x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2}
                    stroke={colorString(col, alpha)}
                    strokeWidth={isSel ? 3 : Math.max(.4, Math.min(2.5, abs*2))}
                    style={{ cursor:'pointer', transition:'stroke .2s, stroke-width .2s' }}
                    onClick={() => onSelectConnection?.({ id:c.id, src:c.srcId, dst:c.dstId, layerIdx:c.layerIdx, srcIdx:c.srcIdx, dstIdx:c.dstIdx, weight:c.weight, gradient:c.gradient })}
                  />
                );
              })}
            </g>

            {/* Neurons */}
            <g>
              {Object.entries(nodePos).map(([id, np]) => {
                const val  = np.value;
                const isSel = selectedNodeId === id;
                const isOut = id.startsWith('out');
                const isIn  = id.startsWith('in');
                const z     = weightedSums[id] ?? val;
                const prob  = isOut ? outputProbs[np.index] : null;
                const isWin = isOut && np.index === winnerIdx;

                const fillCol = val >= 0
                  ? `rgba(0,242,254,${Math.min(.95,.25+val*.7)})`
                  : `rgba(255,0,127,${Math.min(.95,.25+Math.abs(val)*.7)})`;
                const biasStroke = np.bias > .02 ? C.cyan : (np.bias < -.02 ? C.pink : 'rgba(255,255,255,.1)');

                return (
                  <g key={id} data-node="1"
                    transform={`translate(${np.x},${np.y})`}
                    style={{ cursor:'pointer', transition:'opacity .25s' }}
                    onClick={e => { e.stopPropagation(); onSelectNode?.({ id, label:np.label, value:val, bias:np.bias, layerIdx:np.layerIdx, nodeIdx:np.index }); }}
                    onMouseEnter={() => setHovNode(id)}
                    onMouseLeave={() => setHovNode(null)}
                  >

                    {/* Activation halo */}
                    {Math.abs(val) > .01 && (
                      <circle r={13 + Math.min(8,Math.abs(val)*4)} fill={fillCol} opacity={.12}
                        style={{ filter:'blur(3px)' }} />
                    )}

                    {/* Winner ring */}
                    {isWin && <circle r={20} fill="none" stroke={C.gold} strokeWidth={2} className="pulse-ring" opacity={.7} />}

                    {/* Outer ring (bias colour) */}
                    <circle r={12} fill={C.bg}
                      stroke={isSel ? '#fff' : biasStroke}
                      strokeWidth={isSel ? 3 : 1.5}
                      style={{ transition:'stroke .2s' }}
                    />
                    {/* Inner fill (activation) */}
                    <circle r={7.5}
                      fill={val >= 0 ? C.cyan : C.pink}
                      opacity={Math.max(.15, Math.min(1, .2+Math.abs(val)*.8))}
                      style={{ transition:'fill .25s, opacity .25s' }}
                    />

                    {/* Output: label above + prob */}
                    {isOut ? (
                      <g style={{ pointerEvents:'none', userSelect:'none' }}>
                        <text y={-34} textAnchor="middle"
                          fill={isWin ? C.gold : C.cyan}
                          fontSize="8.5px" fontFamily="var(--font-display)" fontWeight={800}>
                          {np.label}{isWin ? ' 🏆' : ''}
                        </text>
                        <text y={-22} textAnchor="middle" fill={C.text2} fontSize="7.5px">
                          {(prob*100).toFixed(1)}%  logit {val.toFixed(3)}
                        </text>
                      </g>
                    ) : (
                      <text y={isIn ? 22 : -18} textAnchor="middle"
                        fill={C.text2}
                        fontSize={isIn ? '6.5px' : '8px'}
                        fontWeight={isIn ? 800 : 400}
                        fontFamily="var(--font-sans)"
                        style={{ pointerEvents:'none', userSelect:'none' }}
                      >
                        {np.label.length > 14 ? np.label.slice(0,12)+'…' : np.label}
                      </text>
                    )}

                    {/* Activation value below hidden nodes */}
                    {!isOut && !isIn && (
                      <text y={18} textAnchor="middle" fill={C.cyan} fontSize="6.5px" style={{ pointerEvents:'none' }}>
                        a={val.toFixed(3)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>

        {/* Canvas overlay — particles & glows */}
        <canvas ref={canvasRef} width={W} height={H}
          style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%',
            pointerEvents:'none', zIndex:5 }}
        />

        {/* ── Particle hover tooltip ──────────────────────────────────────── */}
        {tooltip && (
          <div style={{
            position:'absolute',
            left: Math.min(tooltip.screenX+16, W*0.55) + 'px',
            top:  tooltip.screenY - 10  + 'px',
            zIndex:200, pointerEvents:'none',
            background:'rgba(7,10,15,.97)',
            border:`1px solid ${C.cyan}55`,
            borderRadius:8, padding:'.7rem .9rem',
            color:'#fff', fontSize:'.7rem', lineHeight:1.55,
            boxShadow:'0 6px 24px rgba(0,0,0,.7)',
            backdropFilter:'blur(6px)',
            minWidth:230,
          }}>
            <div style={{ fontWeight:800, color:C.cyan, borderBottom:`1px solid rgba(255,255,255,.08)`, paddingBottom:'.3rem', marginBottom:'.45rem', fontSize:'.72rem' }}>
              ⚡ SIGNAL DETAILS
            </div>
            <div><span style={{ color:C.text2 }}>Source:</span> {tooltip.srcLabel}</div>
            <div><span style={{ color:C.text2 }}>Destination:</span> {tooltip.dstLabel}</div>
            <div style={{ marginTop:'.4rem', borderTop:`1px solid rgba(255,255,255,.06)`, paddingTop:'.4rem' }}>
              <div><span style={{ color:C.text2 }}>Activation (aᵢ):</span> {tooltip.activation.toFixed(5)}</div>
              <div><span style={{ color:C.text2 }}>Weight (wᵢ):</span> {tooltip.weight.toFixed(5)}</div>
              <div style={{ color: tooltip.contrib>=0 ? C.cyan : C.pink, fontWeight:800 }}>
                Contribution (aᵢ×wᵢ): {tooltip.contrib>=0?'+':''}{tooltip.contrib.toFixed(5)}
              </div>
            </div>
            <div style={{ marginTop:'.4rem', borderTop:`1px solid rgba(255,255,255,.06)`, paddingTop:'.4rem' }}>
              <div><span style={{ color:C.text2 }}>Dest. Bias:</span> {(tooltip.dstBias||0).toFixed(5)}</div>
              <div><span style={{ color:C.text2 }}>Dest. z = Σ(aᵢwᵢ)+b:</span> <span style={{ color:'#fff' }}>{(weightedSums[tooltip.dstId]??0).toFixed(5)}</span></div>
              <div><span style={{ color:C.text2 }}>Dest. activation:</span> <span style={{ color:C.cyan }}>{(nodePos[tooltip.dstId]?.value??0).toFixed(5)}</span></div>
            </div>
          </div>
        )}

        {/* ── Clicked neuron inspector panel ─────────────────────────────── */}
        {neuronPanel && (
          <div className="slide-up" style={{
            position:'absolute', right:12, top:12, zIndex:120,
            background:'rgba(7,10,15,.97)', border:'1px solid rgba(255,255,255,.1)',
            borderRadius:10, padding:'1rem 1.1rem', width:270, maxHeight:400,
            overflowY:'auto', boxShadow:'0 8px 28px rgba(0,0,0,.7)',
            backdropFilter:'blur(6px)', fontSize:'.72rem', lineHeight:1.5,
          }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'.5rem' }}>
              <span style={{ fontWeight:800, color:C.cyan }}>NEURON INSPECTOR</span>
              <button onClick={() => onSelectNode?.(null)}
                style={{ background:'none', border:'none', color:C.text2, cursor:'pointer', fontSize:'.9rem', lineHeight:1 }}>✕</button>
            </div>

            <div style={{ fontWeight:700, marginBottom:'.35rem', color:'#fff' }}>{neuronPanel.label}</div>

            {/* Math card */}
            <div style={{ background:'rgba(0,0,0,.35)', borderRadius:6, padding:'.6rem .7rem', fontFamily:'monospace', fontSize:'.68rem', lineHeight:1.7, marginBottom:'.6rem' }}>
              {neuronPanel.type !== 'input' ? (
                <>
                  <div><span style={{ color:C.text2 }}>z = Σ(aᵢ×wᵢ) + b</span></div>
                  <div style={{ color:'#fff' }}>z = <strong>{neuronPanel.z.toFixed(5)}</strong></div>
                  <div><span style={{ color:C.text2 }}>b = </span>{neuronPanel.bias.toFixed(5)}</div>
                  <div><span style={{ color:C.text2 }}>a = {activationName}(z) = </span>
                    <span style={{ color:C.cyan, fontWeight:800 }}>{neuronPanel.a.toFixed(5)}</span>
                  </div>
                </>
              ) : (
                <div><span style={{ color:C.text2 }}>Input value = </span><span style={{ color:C.cyan }}>{neuronPanel.a.toFixed(5)}</span></div>
              )}
            </div>

            {/* Incoming contributions */}
            {neuronPanel.incoming.length > 0 && (
              <div style={{ marginBottom:'.6rem' }}>
                <div style={{ fontSize:'.63rem', fontWeight:800, color:C.cyan, marginBottom:'.25rem', letterSpacing:'.05em' }}>
                  INCOMING CONTRIBUTIONS ({neuronPanel.incoming.length})
                </div>
                <div style={{ maxHeight:120, overflowY:'auto', display:'flex', flexDirection:'column', gap:'.15rem',
                  background:'rgba(0,0,0,.25)', borderRadius:5, padding:'.4rem .5rem' }}>
                  {neuronPanel.incoming.map((inc, i) => (
                    <div key={i} style={{ display:'flex', justifyContent:'space-between', fontSize:'.63rem' }}>
                      <span style={{ color:C.text2, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:130 }}
                        title={inc.srcLabel}>{inc.srcLabel}</span>
                      <span style={{ color: inc.contrib>=0 ? C.cyan : C.pink, fontWeight:700, flexShrink:0 }}>
                        {inc.contrib>=0?'+':''}{inc.contrib.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Outgoing weights */}
            {neuronPanel.outgoing.length > 0 && (
              <div>
                <div style={{ fontSize:'.63rem', fontWeight:800, color:'rgba(255,193,7,.8)', marginBottom:'.25rem', letterSpacing:'.05em' }}>
                  OUTGOING WEIGHTS ({neuronPanel.outgoing.length})
                </div>
                <div style={{ maxHeight:100, overflowY:'auto', display:'flex', flexDirection:'column', gap:'.15rem',
                  background:'rgba(0,0,0,.25)', borderRadius:5, padding:'.4rem .5rem' }}>
                  {neuronPanel.outgoing.map((out, i) => (
                    <div key={i} style={{ display:'flex', justifyContent:'space-between', fontSize:'.63rem' }}>
                      <span style={{ color:C.text2 }}>{out.dstLabel}</span>
                      <span style={{ color: out.weight>=0 ? C.cyan : C.pink, fontWeight:700 }}>
                        {out.weight.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>

      {/* ── Explain-mode footer bar ─────────────────────────────────────────── */}
      {explain && (
        <div style={{
          background:'rgba(7,10,15,.9)', border:`1px solid rgba(255,193,7,.15)`,
          borderRadius:8, padding:'.7rem 1.1rem',
          display:'flex', alignItems:'center', justifyContent:'space-between', gap:'1rem',
          fontSize:'.78rem',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:'.5rem' }}>
            <span style={{ color:C.gold, fontSize:'1rem' }}>🔬</span>
            {fsmState === 'forward_paused' && (
              <span>
                <strong style={{ color:'#fff' }}>Layer {fsmLayer + 1} complete.</strong>
                <span style={{ color:C.text2 }}> All {layers[fsmLayer+1]?.nodes.length ?? 0} neurons received contributions. Click <strong style={{ color:C.gold }}>NEXT LAYER</strong> to compute gradients.</span>
              </span>
            )}
            {fsmState === 'backward_paused' && (
              <span>
                <strong style={{ color:'#fff' }}>Backpropagation complete.</strong>
                <span style={{ color:C.text2 }}> Gradients updated. Click <strong style={{ color:C.gold }}>NEXT LAYER</strong> to start the next forward pass.</span>
              </span>
            )}
            {fsmState === 'forward_animating' && <span style={{ color:C.text2 }}>Propagating layer {fsmLayer}→{fsmLayer+1}…</span>}
            {fsmState === 'backward_animating' && <span style={{ color:C.text2 }}>Backpropagating gradients L{fsmLayer+1}←L{fsmLayer}…</span>}
            {fsmState === 'idle' && <span style={{ color:C.text2 }}>Waiting for model data…</span>}
          </div>
          <div style={{ display:'flex', gap:'.5rem', flexShrink:0 }}>
            {(fsmState === 'forward_paused' || fsmState === 'backward_paused') && (
              <button onClick={resume} className="nav-item"
                style={{ background:C.gold, color:'#000', fontWeight:800, fontSize:'.75rem', padding:'.3rem .9rem', border:'none' }}>
                NEXT LAYER ➔
              </button>
            )}
            <button onClick={restart} className="nav-item"
              style={{ fontSize:'.72rem', padding:'.3rem .75rem' }}>
              ↺ RESTART
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
