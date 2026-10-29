import { useEffect, useRef, useState, useCallback } from 'react';
import { Simulation, SimStats } from './Simulation';

function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<Simulation | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [stats, setStats] = useState<SimStats>({
    episode: 0,
    epsilon: 1.0,
    avgReward: 0,
    lastReward: 0,
    distance: 0,
    memorySize: 0,
    step: 0,
  });
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [epsilonHistory, setEpsilonHistory] = useState<number[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleStatsUpdate = useCallback((newStats: SimStats) => {
    setStats(newStats);
    setRewardHistory(prev => {
      const updated = [...prev, newStats.avgReward];
      return updated.slice(-100);
    });
    setEpsilonHistory(prev => {
      const updated = [...prev, newStats.epsilon];
      return updated.slice(-100);
    });
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    const sim = new Simulation(containerRef.current);
    simRef.current = sim;
    sim.onStatsUpdate = handleStatsUpdate;

    return () => {
      sim.dispose();
    };
  }, [handleStatsUpdate]);

  const toggleSimulation = () => {
    if (!simRef.current) return;
    if (isRunning) {
      simRef.current.stop();
    } else {
      simRef.current.start();
    }
    setIsRunning(!isRunning);
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    if (simRef.current) {
      simRef.current.setSpeed(newSpeed);
    }
  };

  const handleReset = () => {
    if (!simRef.current) return;
    simRef.current.reset();
    setRewardHistory([]);
    setEpsilonHistory([]);
  };

  // Draw reward chart
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#1a1a3a';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (rewardHistory.length < 2) return;

    const maxReward = Math.max(...rewardHistory, 1);
    const minReward = Math.min(...rewardHistory, -10);
    const range = maxReward - minReward || 1;

    // Draw reward line
    ctx.beginPath();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 5;

    for (let i = 0; i < rewardHistory.length; i++) {
      const x = (i / (rewardHistory.length - 1)) * width;
      const y = height - ((rewardHistory[i] - minReward) / range) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '11px monospace';
    ctx.fillText(`Max: ${maxReward.toFixed(1)}`, 5, 15);
    ctx.fillText(`Min: ${minReward.toFixed(1)}`, 5, height - 5);
    ctx.fillText(`Avg Reward History`, width - 130, 15);
  }, [rewardHistory]);

  return (
    <div className="w-full h-screen flex flex-col bg-[#0a0a1a] text-white overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 bg-[#0d0d24] border-b border-[#1a1a3a]">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-green-400 animate-pulse shadow-lg shadow-green-400/50"></div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
            Deep RL Neural Network Simulator
          </h1>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span className="px-2 py-1 bg-[#1a1a3a] rounded text-green-400 font-mono">
            DQN Agent
          </span>
          <span className="px-2 py-1 bg-[#1a1a3a] rounded text-blue-400 font-mono">
            6→128→64→8
          </span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* 3D Viewport */}
        <div className="flex-1 relative">
          <div ref={containerRef} className="w-full h-full" />
          
          {/* Overlay controls */}
          <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
            <div className="flex gap-2">
              <button
                onClick={toggleSimulation}
                className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
                  isRunning
                    ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30'
                    : 'bg-green-500/20 text-green-400 border border-green-500/50 hover:bg-green-500/30'
                }`}
              >
                {isRunning ? '⏸ Pause' : '▶ Start'}
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-2 rounded-lg font-semibold text-sm bg-yellow-500/20 text-yellow-400 border border-yellow-500/50 hover:bg-yellow-500/30 transition-all"
              >
                ↺ Reset
              </button>
            </div>

            <div className="flex items-center gap-3 bg-[#0d0d24]/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-[#1a1a3a]">
              <span className="text-xs text-gray-400">Speed:</span>
              {[1, 5, 10, 20].map(s => (
                <button
                  key={s}
                  onClick={() => handleSpeedChange(s)}
                  className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                    speed === s
                      ? 'bg-blue-500/30 text-blue-400 border border-blue-500/50'
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="absolute top-4 left-4 bg-[#0d0d24]/80 backdrop-blur-sm rounded-lg p-3 border border-[#1a1a3a]">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-green-400 shadow-lg shadow-green-400/50"></div>
              <span className="text-xs text-gray-300">Agent (Ball)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-400 shadow-lg shadow-red-400/50"></div>
              <span className="text-xs text-gray-300">Target (Cube)</span>
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-[#0d0d24] border-l border-[#1a1a3a] overflow-y-auto flex flex-col">
          {/* Stats */}
          <div className="p-4 border-b border-[#1a1a3a]">
            <h2 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Training Stats
            </h2>
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="Episode" value={stats.episode.toString()} color="text-blue-400" />
              <StatBox label="Epsilon" value={stats.epsilon.toFixed(3)} color="text-purple-400" />
              <StatBox label="Avg Reward" value={stats.avgReward.toFixed(1)} color="text-green-400" />
              <StatBox label="Distance" value={stats.distance.toFixed(2)} color="text-yellow-400" />
              <StatBox label="Memory" value={stats.memorySize.toString()} color="text-cyan-400" />
              <StatBox label="Step" value={stats.step.toString()} color="text-orange-400" />
            </div>
          </div>

          {/* Reward Chart */}
          <div className="p-4 border-b border-[#1a1a3a]">
            <h2 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Reward History
            </h2>
            <canvas
              ref={canvasRef}
              width={280}
              height={150}
              className="w-full rounded-lg border border-[#1a1a3a]"
            />
          </div>

          {/* Epsilon Chart */}
          <div className="p-4 border-b border-[#1a1a3a]">
            <h2 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Exploration (ε)
            </h2>
            <div className="w-full bg-[#1a1a3a] rounded-full h-4 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300 rounded-full"
                style={{ width: `${stats.epsilon * 100}%` }}
              />
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-xs text-gray-500">0 (exploit)</span>
              <span className="text-xs text-gray-500">1 (explore)</span>
            </div>
          </div>

          {/* Network Architecture */}
          <div className="p-4 border-b border-[#1a1a3a]">
            <h2 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              Network Architecture
            </h2>
            <div className="space-y-2">
              <NetworkLayer name="Input" size={6} desc="State vector" />
              <NetworkLayer name="Hidden 1" size={128} desc="ReLU activation" />
              <NetworkLayer name="Hidden 2" size={64} desc="ReLU activation" />
              <NetworkLayer name="Output" size={8} desc="Q-values (8 actions)" />
            </div>
          </div>

          {/* Info */}
          <div className="p-4 mt-auto">
            <h2 className="text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">
              How it works
            </h2>
            <p className="text-xs text-gray-500 leading-relaxed">
              The green ball (agent) uses a Deep Q-Network to learn how to reach the red cube (target).
              When the ball touches the cube, it receives a reward and the cube teleports to a new position.
              The agent improves through experience replay and epsilon-greedy exploration.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#1a1a3a] rounded-lg p-2">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-lg font-mono font-bold ${color}`}>{value}</div>
    </div>
  );
}

function NetworkLayer({ name, size, desc }: { name: string; size: number; desc: string }) {
  return (
    <div className="flex items-center gap-3 bg-[#1a1a3a] rounded-lg p-2">
      <div className="flex gap-0.5">
        {Array.from({ length: Math.min(size, 8) }).map((_, i) => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-blue-400/60"
            style={{ opacity: 0.3 + (i / Math.min(size, 8)) * 0.7 }}
          />
        ))}
        {size > 8 && <span className="text-xs text-gray-500 ml-1">...</span>}
      </div>
      <div className="flex-1">
        <div className="text-xs font-semibold text-gray-300">{name} ({size})</div>
        <div className="text-xs text-gray-500">{desc}</div>
      </div>
    </div>
  );
}

export default App;
