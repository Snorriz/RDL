// Simple Neural Network implementation for Deep Q-Learning

export class NeuralNetwork {
  layers: number[];
  weights: number[][][];
  biases: number[][];
  learningRate: number;

  constructor(layers: number[], learningRate: number = 0.001) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];

    // Initialize weights and biases
    for (let i = 0; i < layers.length - 1; i++) {
      const w: number[][] = [];
      for (let j = 0; j < layers[i]; j++) {
        const row: number[] = [];
        for (let k = 0; k < layers[i + 1]; k++) {
          row.push((Math.random() - 0.5) * Math.sqrt(2 / layers[i]));
        }
        w.push(row);
      }
      this.weights.push(w);
      this.biases.push(new Array(layers[i + 1]).fill(0));
    }
  }

  relu(x: number): number {
    return Math.max(0, x);
  }

  reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  forward(input: number[]): { outputs: number[][]; preActivations: number[][] } {
    const outputs: number[][] = [input];
    const preActivations: number[][] = [];

    let current = input;
    for (let l = 0; l < this.weights.length; l++) {
      const next: number[] = [];
      const preAct: number[] = [];
      for (let j = 0; j < this.weights[l][0].length; j++) {
        let sum = this.biases[l][j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * this.weights[l][i][j];
        }
        preAct.push(sum);
        // Use ReLU for hidden layers, linear for output
        if (l < this.weights.length - 1) {
          next.push(this.relu(sum));
        } else {
          next.push(sum);
        }
      }
      outputs.push(next);
      preActivations.push(preAct);
      current = next;
    }

    return { outputs, preActivations };
  }

  predict(input: number[]): number[] {
    const { outputs } = this.forward(input);
    return outputs[outputs.length - 1];
  }

  train(input: number[], target: number[]): void {
    const { outputs, preActivations } = this.forward(input);
    
    // Backpropagation
    const deltas: number[][] = [];
    
    // Output layer delta
    const outputLayer = outputs.length - 1;
    const outputDelta: number[] = [];
    for (let j = 0; j < outputs[outputLayer].length; j++) {
      outputDelta.push(target[j] - outputs[outputLayer][j]);
    }
    deltas.unshift(outputDelta);

    // Hidden layers delta
    for (let l = this.weights.length - 2; l >= 0; l--) {
      const delta: number[] = [];
      for (let i = 0; i < this.weights[l][0].length; i++) {
        let sum = 0;
        for (let j = 0; j < this.weights[l + 1][0].length; j++) {
          sum += deltas[0][j] * this.weights[l + 1][i][j];
        }
        delta.push(sum * this.reluDerivative(preActivations[l][i]));
      }
      deltas.unshift(delta);
    }

    // Update weights and biases
    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          this.weights[l][i][j] += this.learningRate * deltas[l][j] * outputs[l][i];
        }
      }
      for (let j = 0; j < this.biases[l].length; j++) {
        this.biases[l][j] += this.learningRate * deltas[l][j];
      }
    }
  }

  clone(): NeuralNetwork {
    const nn = new NeuralNetwork(this.layers, this.learningRate);
    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          nn.weights[l][i][j] = this.weights[l][i][j];
        }
      }
      for (let j = 0; j < this.biases[l].length; j++) {
        nn.biases[l][j] = this.biases[l][j];
      }
    }
    return nn;
  }
}
