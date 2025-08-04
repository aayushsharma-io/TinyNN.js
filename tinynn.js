// tinynn.js [04082025]
// aayush sharma

class NeuralNetwork {
  constructor(options = {}) {
    this.inputNodes = options.inputNodes || 1;
    this.hiddenNodes = options.hiddenNodes || 3;
    this.outputNodes = options.outputNodes || 1;
    this.learningRate = options.learningRate || 0.1;

    // Initialize weights
    this.weightsInputHidden = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weightsHiddenOutput = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weightsInputHidden.randomize();
    this.weightsHiddenOutput.randomize();

    // Initialize biases
    this.biasHidden = new Matrix(this.hiddenNodes, 1);
    this.biasOutput = new Matrix(this.outputNodes, 1);
    this.biasHidden.randomize();
    this.biasOutput.randomize();
  }

  train(inputs, targets) {
    // Feedforward
    let inputsMatrix = Matrix.fromArray(inputs);
    let hiddenInputs = Matrix.multiply(this.weightsInputHidden, inputsMatrix);
    hiddenInputs.add(this.biasHidden);
    let hiddenOutputs = hiddenInputs.map(sigmoid);
    let outputInputs = Matrix.multiply(this.weightsHiddenOutput, hiddenOutputs);
    outputInputs.add(this.biasOutput);
    let outputs = outputInputs.map(sigmoid);

    // Backpropagation
    let targetsMatrix = Matrix.fromArray(targets);
    let outputErrors = Matrix.subtract(targetsMatrix, outputs);
    let gradients = outputs.map(dsigmoid);
    gradients.multiply(outputErrors);
    gradients.multiply(this.learningRate);

    // Adjust output weights and biases
    let hiddenOutputsT = Matrix.transpose(hiddenOutputs);
    let weightsHiddenOutputDeltas = Matrix.multiply(gradients, hiddenOutputsT);
    this.weightsHiddenOutput.add(weightsHiddenOutputDeltas);
    this.biasOutput.add(gradients);

    // Calculate hidden layer errors
    let weightsHiddenOutputT = Matrix.transpose(this.weightsHiddenOutput);
    let hiddenErrors = Matrix.multiply(weightsHiddenOutputT, outputErrors);

    // Adjust hidden weights and biases
    let hiddenGradients = hiddenOutputs.map(dsigmoid);
    hiddenGradients.multiply(hiddenErrors);
    hiddenGradients.multiply(this.learningRate);

    // Adjust hidden weights and biases
    let inputsT = Matrix.transpose(inputsMatrix);
    let weightsInputHiddenDeltas = Matrix.multiply(hiddenGradients, inputsT);
    this.weightsInputHidden.add(weightsInputHiddenDeltas);
    this.biasHidden.add(hiddenGradients);
  }

  predict(inputs) {
    let inputsMatrix = Matrix.fromArray(inputs);
    let hiddenInputs = Matrix.multiply(this.weightsInputHidden, inputsMatrix);
    hiddenInputs.add(this.biasHidden);
    let hiddenOutputs = hiddenInputs.map(sigmoid);
    let outputInputs = Matrix.multiply(this.weightsHiddenOutput, hiddenOutputs);
    outputInputs.add(this.biasOutput);
    let outputs = outputInputs.map(sigmoid);
    return outputs.toArray();
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(rows).fill().map(() => Array(cols).fill(0));
  }

  randomize() {
    this.data = this.data.map(row => row.map(() => Math.random() * 2 - 1));
  }

  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((_, i) => arr[i]);
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  map(func) {
    this.data = this.data.map((row, i) => row.map((col, j) => func(col, i, j)));
    return this;
  }

  static map(matrix, func) {
    return new Matrix(matrix.rows, matrix.cols)
      .map((_, i, j) => func(matrix.data[i][j], i, j));
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error("Columns of A must match rows of B.");
      return undefined;
    }
    return new Matrix(a.rows, b.cols)
      .map((_, i, j) => {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        return sum;
      });
  }

  multiply(n) {
    if (n instanceof Matrix) {
      this.data = this.data.map((row, i) => row.map((col, j) => col * n.data[i][j]));
    } else {
      this.data = this.data.map(row => row.map(col => col * n));
    }
    return this;
  }

  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows)
      .map((_, i, j) => matrix.data[j][i]);
  }

  add(matrix) {
    if (matrix instanceof Matrix) {
      this.data = this.data.map((row, i) => row.map((col, j) => col + matrix.data[i][j]));
    } else {
      this.data = this.data.map(row => row.map(col => col + matrix));
    }
    return this;
  }

  static subtract(a, b) {
    return new Matrix(a.rows, a.cols)
      .map((_, i, j) => a.data[i][j] - b.data[i][j]);
  }
}

module.exports = NeuralNetwork;
