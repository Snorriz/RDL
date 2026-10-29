import { NeuralNetwork } from './NeuralNetwork';

interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export class DQNAgent {
  network: NeuralNetwork;
  targetNetwork: NeuralNetwork;
  gamma: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  memory: Experience[];
  memorySize: number;
  batchSize: number;
  trainStep: number;
  targetUpdateFreq: number;
  totalReward: number;
  episodeCount: number;
  episodeRewards: number[];

  constructor(
    stateSize: number,
    actionSize: number,
    hiddenLayers: number[] = [64, 64]
  ) {
    const layers = [stateSize, ...hiddenLayers, actionSize];
    this.network = new NeuralNetwork(layers, 0.001);
    this.targetNetwork = this.network.clone();
    this.gamma = 0.95;
    this.epsilon = 1.0;
    this.epsilonDecay = 0.995;
    this.epsilonMin = 0.01;
    this.memory = [];
    this.memorySize = 10000;
    this.batchSize = 32;
    this.trainStep = 0;
    this.targetUpdateFreq = 50;
    this.totalReward = 0;
    this.episodeCount = 0;
    this.episodeRewards = [];
  }

  act(state: number[]): number {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.network.layers[this.network.layers.length - 1]);
    }
    const qValues = this.network.predict(state);
    let maxIdx = 0;
    let maxVal = qValues[0];
    for (let i = 1; i < qValues.length; i++) {
      if (qValues[i] > maxVal) {
        maxVal = qValues[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  remember(state: number[], action: number, reward: number, nextState: number[], done: boolean) {
    this.memory.push({ state, action, reward, nextState, done });
    if (this.memory.length > this.memorySize) {
      this.memory.shift();
    }
    this.totalReward += reward;
  }

  replay() {
    if (this.memory.length < this.batchSize) return;

    const batch: Experience[] = [];
    for (let i = 0; i < this.batchSize; i++) {
      const idx = Math.floor(Math.random() * this.memory.length);
      batch.push(this.memory[idx]);
    }

    for (const exp of batch) {
      const qValues = this.network.predict(exp.state);
      let target = exp.reward;

      if (!exp.done) {
        const nextQValues = this.targetNetwork.predict(exp.nextState);
        let maxNextQ = nextQValues[0];
        for (let i = 1; i < nextQValues.length; i++) {
          if (nextQValues[i] > maxNextQ) {
            maxNextQ = nextQValues[i];
          }
        }
        target += this.gamma * maxNextQ;
      }

      qValues[exp.action] = target;
      this.network.train(exp.state, qValues);
    }

    this.trainStep++;
    if (this.trainStep % this.targetUpdateFreq === 0) {
      this.targetNetwork = this.network.clone();
    }
  }

  endEpisode() {
    this.episodeRewards.push(this.totalReward);
    this.totalReward = 0;
    this.episodeCount++;
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }

  getAverageReward(lastN: number = 10): number {
    if (this.episodeRewards.length === 0) return 0;
    const recent = this.episodeRewards.slice(-lastN);
    return recent.reduce((a, b) => a + b, 0) / recent.length;
  }
}
