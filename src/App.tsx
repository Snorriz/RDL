import { useEffect, useRef, useState, useCallback } from 'react';
import { Simulation, SimStats } from './Simulation';

function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<Simulation | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showNN, setShowNN] = useState(true);
  const [stats, setStats] = useState<SimStats>({
    episode: 0,
    epsilon: 1.0,
    avgReward: 0,
    lastReward: 0,
    distance: 0,
    memorySize: 0,
    step: 0,
    catchCount: 0,
    ballSpeed: 0,
    lastAction: 0,
  });
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [epsilonHistory, setEpsilonHistory] = useState<number[]>([]);
  const rewardCanvasRef = useRef<HTMLCanvasElement>(null);
  const epsilonCanvasRef = useRef<HTMLCanvasElement>(null);
  const nnCanvasRef = useRef<HTMLCanvasElement>(null);
  const nnAnimRef = useRef<number>(0);

  const handleStatsUpdate = useCallback((newStats: SimStats) => {
    setStats(newStats);
    setRewardHistory(prev => {
      const updated = [...prev, newStats.avgReward];
      return updated.slice(-150);
    });
    setEpsilonHistory(prev => {
      const updated = [...prev, newStats.epsilon];
      return updated.slice(-150);
    });
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const sim = new Simulation(containerRef.current);
    simRef.current = sim;
    sim.onStatsUpdate = handleStatsUpdate;
    return () => { sim.dispose(); };
  }, [handleStatsUpdate]);

  // Neural Network visualization loop
  useEffect(() => {
    if (!showNN) return;
    const drawNN = () => {
      const canvas = nnCanvasRef.current;
      if (!canvas || !simRef.current) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      const agent = simRef.current.agent;
      const network = agent.network;

      ctx.clearRect(0, 0, w, h);

      // Background
      const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
      bgGrad.addColorStop(0, 'rgba(8, 8, 24, 0.95)');
      bgGrad.addColorStop(1, 'rgba(4, 4, 16, 0.98)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Border
      ctx.strokeStyle = 'rgba(100, 100, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, w, h);

      const layers = network.layers;
      const numLayers = layers.length;
      const padding = 35;
      const layerSpacing = (w - padding * 2) / (numLayers - 1);

      // Calculate node positions
      const nodePositions: { x: number; y: number }[][] = [];
      for (let l = 0; l < numLayers; l++) {
        const layerNodes: { x: number; y: number }[] = [];
        const n = layers[l];
        const displayN = Math.min(n, 12); // Show max 12 nodes per layer for clarity
        const nodeSpacing = (h - padding * 2) / (displayN + 1);
        for (let i = 0; i < displayN; i++) {
          layerNodes.push({
            x: padding + l * layerSpacing,
            y: padding + (i + 1) * nodeSpacing,
          });
        }
        nodePositions.push(layerNodes);
      }

      // Get activations
      const forwardResult = agent.lastForwardResult;
      const activations = forwardResult ? forwardResult.outputs : null;

      // Draw connections
      for (let l = 0; l < numLayers - 1; l++) {
        const fromNodes = nodePositions[l];
        const toNodes = nodePositions[l + 1];
        const weights = network.weights[l];

        for (let i = 0; i < fromNodes.length; i++) {
          for (let j = 0; j < toNodes.length; j++) {
            const actualI = Math.floor(i * layers[l] / Math.min(layers[l], 12));
            const actualJ = Math.floor(j * layers[l + 1] / Math.min(layers[l + 1], 12));
            const weight = weights[actualI]?.[actualJ] || 0;

            const absW = Math.min(Math.abs(weight), 2) / 2;
            const alpha = 0.05 + absW * 0.4;

            if (weight > 0) {
              ctx.strokeStyle = `rgba(0, 200, 255, ${alpha})`;
            } else {
              ctx.strokeStyle = `rgba(255, 80, 80, ${alpha})`;
            }
            ctx.lineWidth = 0.5 + absW * 1.5;

            ctx.beginPath();
            ctx.moveTo(fromNodes[i].x, fromNodes[i].y);
            ctx.lineTo(toNodes[j].x, toNodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (let l = 0; l < numLayers; l++) {
        const nodes = nodePositions[l];
        for (let i = 0; i < nodes.length; i++) {
          const node = nodes[i];
          let activation = 0;

          if (activations && activations[l]) {
            const actualI = Math.floor(i * layers[l] / Math.min(layers[l], 12));
            activation = activations[l][actualI] || 0;
          }

          // Node glow
          const glowRadius = 12;
          const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowRadius);
          
          let nodeColor: string;
          let glowColor: string;
          
          if (l === 0) {
            // Input layer - green
            const intensity = Math.min(Math.abs(activation), 1);
            nodeColor = `rgba(0, ${150 + intensity * 105}, ${100 + intensity * 55}, 0.9)`;
            glowColor = `rgba(0, 255, 136, ${0.2 + intensity * 0.3})`;
          } else if (l === numLayers - 1) {
            // Output layer - amber/orange
            const qVal = activation;
            const maxQ = Math.max(...(activations?.[numLayers - 1] || [0]));
            const isMax = Math.abs(qVal - maxQ) < 0.001;
            const norm = (qVal + 2) / 4; // normalize roughly to 0-1
            if (isMax) {
              nodeColor = `rgba(255, 200, 0, 1)`;
              glowColor = `rgba(255, 200, 0, 0.6)`;
            } else {
              nodeColor = `rgba(${180 + norm * 75}, ${100 + norm * 80}, 0, 0.8)`;
              glowColor = `rgba(255, 150, 0, ${0.1 + norm * 0.3})`;
            }
          } else {
            // Hidden layers - blue/purple
            const intensity = Math.min(activation, 2) / 2;
            nodeColor = `rgba(${80 + intensity * 100}, ${50 + intensity * 100}, ${200 + intensity * 55}, 0.9)`;
            glowColor = `rgba(100, 100, 255, ${0.1 + intensity * 0.4})`;
          }

          // Draw glow
          glow.addColorStop(0, glowColor);
          glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
          ctx.fill();

          // Draw node
          ctx.fillStyle = nodeColor;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
          ctx.fill();

          // Node border
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // Layer labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      const layerLabels = ['Input (8)', 'Hidden (64)', 'Hidden (32)', 'Output (9)'];
      for (let l = 0; l < numLayers; l++) {
        ctx.fillText(layerLabels[l] || `L${l}`, padding + l * layerSpacing, h - 8);
      }

      // Title
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('NEURAL NETWORK', 8, 14);

      // Current action indicator
      if (agent.lastAction !== undefined) {
        const actionNames = ['WAIT', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        ctx.fillStyle = agent.lastAction === 0 ? 'rgba(150,150,150,0.7)' : 'rgba(255, 200, 0, 0.9)';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`Action: ${actionNames[agent.lastAction] || '?'}`, w - 8, 14);
      }

      nnAnimRef.current = requestAnimationFrame(drawNN);
    };

    drawNN();
    return () => {
      if (nnAnimRef.current) cancelAnimationFrame(nnAnimRef.current);
    };
  }, [showNN]);

  const toggleSimulation = () => {
    if (!simRef.current) return;
    if (isRunning) simRef.current.stop();
    else simRef.current.start();
    setIsRunning(!isRunning);
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    if (simRef.current) simRef.current.setSpeed(newSpeed);
  };

  const handleReset = () => {
    if (!simRef.current) return;
    simRef.current.reset();
    setRewardHistory([]);
    setEpsilonHistory([]);
  };

  // Draw reward chart
  useEffect(() => {
    const canvas = rewardCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, 'rgba(10, 10, 30, 0.9)');
    bgGrad.addColorStop(1, 'rgba(5, 5, 20, 0.95)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(50, 50, 100, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (rewardHistory.length < 2) return;

    const maxR = Math.max(...rewardHistory, 1);
    const minR = Math.min(...rewardHistory, -5);
    const range = maxR - minR || 1;
    const padding = 10;

    // Gradient fill under curve
    ctx.beginPath();
    for (let i = 0; i < rewardHistory.length; i++) {
      const x = padding + (i / (rewardHistory.length - 1)) * (w - padding * 2);
      const y = h - padding - ((rewardHistory[i] - minR) / range) * (h - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(w - padding, h - padding);
    ctx.lineTo(padding, h - padding);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, 'rgba(0, 255, 136, 0.15)');
    fillGrad.addColorStop(1, 'rgba(0, 255, 136, 0.0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Main line
    ctx.beginPath();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 8;
    for (let i = 0; i < rewardHistory.length; i++) {
      const x = padding + (i / (rewardHistory.length - 1)) * (w - padding * 2);
      const y = h - padding - ((rewardHistory[i] - minR) / range) * (h - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px monospace';
    ctx.fillText(`${maxR.toFixed(1)}`, 4, 14);
    ctx.fillText(`${minR.toFixed(1)}`, 4, h - 4);
  }, [rewardHistory]);

  // Draw epsilon chart
  useEffect(() => {
    const canvas = epsilonCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, 'rgba(10, 10, 30, 0.9)');
    bgGrad.addColorStop(1, 'rgba(5, 5, 20, 0.95)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    if (epsilonHistory.length < 2) return;

    const padding = 10;

    // Fill
    ctx.beginPath();
    for (let i = 0; i < epsilonHistory.length; i++) {
      const x = padding + (i / (epsilonHistory.length - 1)) * (w - padding * 2);
      const y = h - padding - epsilonHistory[i] * (h - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(w - padding, h - padding);
    ctx.lineTo(padding, h - padding);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, 'rgba(168, 85, 247, 0.2)');
    fillGrad.addColorStop(1, 'rgba(168, 85, 247, 0.0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#a855f7';
    ctx.shadowBlur = 6;
    for (let i = 0; i < epsilonHistory.length; i++) {
      const x = padding + (i / (epsilonHistory.length - 1)) * (w - padding * 2);
      const y = h - padding - epsilonHistory[i] * (h - padding * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Current value
    const currentE = epsilonHistory[epsilonHistory.length - 1];
    ctx.fillStyle = '#a855f7';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(`ε = ${currentE.toFixed(3)}`, w - 80, 14);
  }, [epsilonHistory]);

  const actionNames = ['WAIT', '↑N', '↗NE', '→E', '↘SE', '↓S', '↙SW', '←W', '↖NW'];

  return (
    <div className="w-full h-screen flex flex-col bg-[#060612] text-white overflow-hidden font-sans">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-5 py-2.5 bg-[#0a0a1f]/90 backdrop-blur-md border-b border-white/5 z-10">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-emerald-400' : 'bg-gray-500'}`}>
              {isRunning && <div className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-75" />}
            </div>
          </div>
          <h1 className="text-base font-bold tracking-tight">
            <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Deep RL
            </span>
            <span className="text-white/60 ml-1.5 font-light">Physics Simulator</span>
          </h1>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-white/40 border border-white/10 font-mono">
            v3.0
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
            DQN + Physics
          </span>
          <span className="text-[10px] px-2 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
            8→64→32→9
          </span>
          <span className="text-[10px] px-2 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono">
            Impulse Model
          </span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* 3D Viewport */}
        <div className="flex-1 relative">
          <div ref={containerRef} className="w-full h-full" />

          {/* Neural Network Overlay */}
          {showNN && (
            <div className="absolute top-4 right-4 w-[320px] h-[200px] rounded-xl overflow-hidden border border-white/10 shadow-2xl shadow-blue-500/5">
              <canvas
                ref={nnCanvasRef}
                width={640}
                height={400}
                className="w-full h-full"
              />
            </div>
          )}

          {/* Bottom Controls */}
          <div className="absolute bottom-5 left-5 right-5 flex items-end justify-between pointer-events-none">
            <div className="flex gap-2 pointer-events-auto">
              <button
                onClick={toggleSimulation}
                className={`group flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 backdrop-blur-md ${
                  isRunning
                    ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 hover:border-red-500/50 shadow-lg shadow-red-500/5'
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 hover:border-emerald-500/50 shadow-lg shadow-emerald-500/5'
                }`}
              >
                <span className="text-lg">{isRunning ? '⏸' : '▶'}</span>
                {isRunning ? 'Pause' : 'Start'}
              </button>
              <button
                onClick={handleReset}
                className="group flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50 transition-all duration-300 backdrop-blur-md shadow-lg shadow-amber-500/5"
              >
                <span className="text-lg">↺</span>
                Reset
              </button>
              <button
                onClick={() => setShowNN(!showNN)}
                className={`group flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 backdrop-blur-md ${
                  showNN
                    ? 'bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 shadow-lg shadow-blue-500/5'
                    : 'bg-white/5 text-white/40 border border-white/10 hover:bg-white/10'
                }`}
              >
                <span className="text-lg">🧠</span>
                NN View
              </button>
            </div>

            <div className="flex items-center gap-2 pointer-events-auto bg-[#0a0a1f]/70 backdrop-blur-md rounded-xl px-4 py-2.5 border border-white/5">
              <span className="text-[10px] text-white/40 uppercase tracking-wider mr-1">Speed</span>
              {[1, 5, 10, 25].map(s => (
                <button
                  key={s}
                  onClick={() => handleSpeedChange(s)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all duration-200 ${
                    speed === s
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm shadow-cyan-500/20'
                      : 'text-white/30 hover:text-white/60 border border-transparent'
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="absolute top-4 left-4 bg-[#0a0a1f]/60 backdrop-blur-md rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-2.5 mb-2">
              <div className="w-3 h-3 rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50" />
              <span className="text-xs text-white/70">Agent (physics ball)</span>
            </div>
            <div className="flex items-center gap-2.5 mb-2">
              <div className="w-3 h-3 rounded-sm bg-red-400 shadow-lg shadow-red-400/50" />
              <span className="text-xs text-white/70">Target</span>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="w-3 h-3 bg-yellow-400 shadow-lg shadow-yellow-400/50" style={{ clipPath: 'polygon(50% 0%, 100% 100%, 0% 100%)' }} />
              <span className="text-xs text-white/70">Impulse arrow</span>
            </div>
          </div>

          {/* Catch counter + speed */}
          <div className="absolute top-4 left-4 mt-[88px] flex gap-2">
            <div className="bg-[#0a0a1f]/60 backdrop-blur-md rounded-xl px-3 py-2 border border-white/5">
              <div className="text-[10px] text-white/40 uppercase tracking-wider">Catches</div>
              <div className="text-xl font-bold font-mono bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent">
                {stats.catchCount}
              </div>
            </div>
            <div className="bg-[#0a0a1f]/60 backdrop-blur-md rounded-xl px-3 py-2 border border-white/5">
              <div className="text-[10px] text-white/40 uppercase tracking-wider">Speed</div>
              <div className="text-xl font-bold font-mono text-cyan-400">
                {stats.ballSpeed.toFixed(2)}
              </div>
            </div>
            <div className="bg-[#0a0a1f]/60 backdrop-blur-md rounded-xl px-3 py-2 border border-white/5">
              <div className="text-[10px] text-white/40 uppercase tracking-wider">Action</div>
              <div className="text-xl font-bold font-mono text-yellow-400">
                {actionNames[stats.lastAction] || '—'}
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <aside className="w-[300px] bg-[#0a0a1f]/80 backdrop-blur-md border-l border-white/5 overflow-y-auto flex flex-col">
          {/* Stats Grid */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-[10px] font-semibold text-white/30 mb-3 uppercase tracking-widest">
              Training Metrics
            </h2>
            <div className="grid grid-cols-2 gap-2">
              <StatCard label="Episode" value={stats.episode} icon="🎯" color="blue" />
              <StatCard label="Epsilon" value={stats.epsilon.toFixed(3)} icon="🎲" color="purple" />
              <StatCard label="Avg Reward" value={stats.avgReward.toFixed(1)} icon="📈" color="emerald" />
              <StatCard label="Distance" value={stats.distance.toFixed(2)} icon="📏" color="amber" />
              <StatCard label="Memory" value={stats.memorySize} icon="🧠" color="cyan" />
              <StatCard label="Step" value={`${stats.step}/300`} icon="⏱" color="orange" />
            </div>
          </div>

          {/* Reward Chart */}
          <div className="p-4 border-b border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-semibold text-white/30 uppercase tracking-widest">
                Reward History
              </h2>
              <span className="text-[10px] font-mono text-emerald-400/60">
                last 150 ep.
              </span>
            </div>
            <div className="rounded-xl overflow-hidden border border-white/5">
              <canvas
                ref={rewardCanvasRef}
                width={560}
                height={200}
                className="w-full"
              />
            </div>
          </div>

          {/* Epsilon Chart */}
          <div className="p-4 border-b border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-semibold text-white/30 uppercase tracking-widest">
                Exploration Rate
              </h2>
              <span className="text-[10px] font-mono text-purple-400/60">
                ε-greedy
              </span>
            </div>
            <div className="rounded-xl overflow-hidden border border-white/5">
              <canvas
                ref={epsilonCanvasRef}
                width={560}
                height={150}
                className="w-full"
              />
            </div>
          </div>

          {/* Physics Info */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-[10px] font-semibold text-white/30 mb-3 uppercase tracking-widest">
              Physics Model
            </h2>
            <div className="space-y-1.5">
              <PhysicsParam name="Friction" value="0.985" desc="Velocity damping/step" />
              <PhysicsParam name="Impulse" value="0.35" desc="Push magnitude" />
              <PhysicsParam name="Max Vel" value="2.50" desc="Velocity cap" />
              <PhysicsParam name="Bounce" value="0.60" desc="Wall restitution" />
            </div>
          </div>

          {/* Network Architecture */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-[10px] font-semibold text-white/30 mb-3 uppercase tracking-widest">
              Network Architecture
            </h2>
            <div className="space-y-1.5">
              <LayerBar name="Input" size={8} total={64} desc="State [pos, vel, target, dist]" color="emerald" />
              <LayerBar name="Hidden 1" size={64} total={64} desc="ReLU + Dense" color="blue" />
              <LayerBar name="Hidden 2" size={32} total={64} desc="ReLU + Dense" color="purple" />
              <LayerBar name="Output" size={9} total={64} desc="Q-values (9 actions)" color="amber" />
            </div>
            <div className="mt-3 flex items-center gap-2 text-[10px] text-white/30 flex-wrap">
              <span className="px-1.5 py-0.5 rounded bg-white/5">Adam</span>
              <span>lr=0.001</span>
              <span>•</span>
              <span>γ=0.95</span>
              <span>•</span>
              <span>batch=32</span>
            </div>
          </div>

          {/* Actions */}
          <div className="p-4 border-b border-white/5">
            <h2 className="text-[10px] font-semibold text-white/30 mb-3 uppercase tracking-widest">
              Agent Actions
            </h2>
            <div className="grid grid-cols-3 gap-1">
              {actionNames.map((name, i) => (
                <div
                  key={i}
                  className={`text-center py-1.5 rounded text-[10px] font-mono border transition-all ${
                    stats.lastAction === i
                      ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40'
                      : 'bg-white/[0.02] text-white/30 border-white/5'
                  }`}
                >
                  {i}: {name}
                </div>
              ))}
            </div>
          </div>

          {/* Info */}
          <div className="p-4 mt-auto">
            <h2 className="text-[10px] font-semibold text-white/30 mb-2 uppercase tracking-widest">
              About
            </h2>
            <p className="text-[11px] text-white/40 leading-relaxed">
              The agent (<span className="text-emerald-400/70">green sphere</span>) starts from a fixed position. 
              Each step it applies an <span className="text-yellow-400/70">impulse</span> (push) to the ball. 
              Physics then takes over: <span className="text-cyan-400/70">friction</span> slows the ball, 
              <span className="text-blue-400/70"> walls bounce</span> it. The agent must learn to use 
              momentum and timing to reach the <span className="text-red-400/70">target</span>.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon, color }: { label: string; value: string | number; icon: string; color: string }) {
  const colorClasses: Record<string, string> = {
    blue: 'from-blue-500/10 to-blue-500/5 border-blue-500/10 text-blue-400',
    purple: 'from-purple-500/10 to-purple-500/5 border-purple-500/10 text-purple-400',
    emerald: 'from-emerald-500/10 to-emerald-500/5 border-emerald-500/10 text-emerald-400',
    amber: 'from-amber-500/10 to-amber-500/5 border-amber-500/10 text-amber-400',
    cyan: 'from-cyan-500/10 to-cyan-500/5 border-cyan-500/10 text-cyan-400',
    orange: 'from-orange-500/10 to-orange-500/5 border-orange-500/10 text-orange-400',
  };
  const cls = colorClasses[color] || colorClasses.blue;
  return (
    <div className={`bg-gradient-to-br ${cls} rounded-xl p-2.5 border`}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-xs">{icon}</span>
        <span className="text-[10px] text-white/40">{label}</span>
      </div>
      <div className="text-lg font-bold font-mono">{value}</div>
    </div>
  );
}

function LayerBar({ name, size, total, desc, color }: { name: string; size: number; total: number; desc: string; color: string }) {
  const colorMap: Record<string, string> = {
    emerald: 'bg-emerald-400',
    blue: 'bg-blue-400',
    purple: 'bg-purple-400',
    amber: 'bg-amber-400',
  };
  const barColor = colorMap[color] || 'bg-blue-400';
  const width = (size / total) * 100;
  return (
    <div className="flex items-center gap-2 bg-white/[0.02] rounded-lg px-2.5 py-2 border border-white/5">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-medium text-white/70">{name}</span>
          <span className="text-[10px] font-mono text-white/30">{size}</span>
        </div>
        <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div
            className={`h-full ${barColor} rounded-full transition-all duration-500 opacity-60`}
            style={{ width: `${Math.max(width, 3)}%` }}
          />
        </div>
        <span className="text-[9px] text-white/25 mt-0.5 block">{desc}</span>
      </div>
    </div>
  );
}

function PhysicsParam({ name, value, desc }: { name: string; value: string; desc: string }) {
  return (
    <div className="flex items-center justify-between bg-white/[0.02] rounded-lg px-2.5 py-2 border border-white/5">
      <div>
        <span className="text-[11px] font-medium text-white/70">{name}</span>
        <span className="text-[9px] text-white/25 ml-2">{desc}</span>
      </div>
      <span className="text-[11px] font-mono text-cyan-400/70">{value}</span>
    </div>
  );
}

export default App;
