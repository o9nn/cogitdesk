/**
 * Torch7-style neural network layer abstractions
 */

import { Tensor } from '../ml/tensor'

/**
 * Xavier/Glorot initialization scaling factor
 */
const XAVIER_INIT_FACTOR = 2.0

/**
 * Base layer interface
 */
export interface Layer {
  forward(input: Tensor): Tensor
  backward?(gradOutput: Tensor): Tensor
  parameters?(): Tensor[]
  name: string
}

/**
 * Linear (fully connected) layer
 */
export class Linear implements Layer {
  public readonly name = 'Linear'
  private weight: Tensor
  private bias: Tensor
  private lastInput?: Tensor

  constructor(inputSize: number, outputSize: number) {
    // Initialize with Xavier/Glorot initialization
    const scale = Math.sqrt(XAVIER_INIT_FACTOR / (inputSize + outputSize))
    this.weight = Tensor.randn([outputSize, inputSize], 0, scale)
    this.bias = Tensor.zeros([outputSize])
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
      throw new Error('Linear layer expects 2D input (batch, features)')
    }

    const [batch, features] = input.shape
    if (features !== this.weight.shape[1]) {
      throw new Error(
        `Input features ${features} do not match weight shape ${this.weight.shape[1]}`
      )
    }

    this.lastInput = input

    // output = input @ weight.T + bias
    const output = Tensor.zeros([batch, this.weight.shape[0]])

    for (let b = 0; b < batch; b++) {
      for (let o = 0; o < this.weight.shape[0]; o++) {
        let sum = this.bias.data[o]
        for (let i = 0; i < this.weight.shape[1]; i++) {
          sum += input.get(b, i) * this.weight.get(o, i)
        }
        output.set(sum, b, o)
      }
    }

    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.lastInput) {
      throw new Error('Must call forward before backward')
    }

    const [batch, _] = this.lastInput.shape
    const gradInput = Tensor.zeros(this.lastInput.shape)

    // Compute gradient with respect to input
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < this.weight.shape[1]; i++) {
        let sum = 0
        for (let o = 0; o < this.weight.shape[0]; o++) {
          sum += gradOutput.get(b, o) * this.weight.get(o, i)
        }
        gradInput.set(sum, b, i)
      }
    }

    return gradInput
  }

  parameters(): Tensor[] {
    return [this.weight, this.bias]
  }
}

/**
 * ReLU activation layer
 */
export class ReLU implements Layer {
  public readonly name = 'ReLU'
  private lastInput?: Tensor

  forward(input: Tensor): Tensor {
    this.lastInput = input
    const output = Tensor.zeros(input.shape)
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = Math.max(0, input.data[i])
    }
    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.lastInput) {
      throw new Error('Must call forward before backward')
    }

    const gradInput = Tensor.zeros(this.lastInput.shape)
    for (let i = 0; i < this.lastInput.data.length; i++) {
      gradInput.data[i] = this.lastInput.data[i] > 0 ? gradOutput.data[i] : 0
    }
    return gradInput
  }
}

/**
 * Sigmoid activation layer
 */
export class Sigmoid implements Layer {
  public readonly name = 'Sigmoid'
  private lastOutput?: Tensor

  forward(input: Tensor): Tensor {
    const output = Tensor.zeros(input.shape)
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = 1 / (1 + Math.exp(-input.data[i]))
    }
    this.lastOutput = output
    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.lastOutput) {
      throw new Error('Must call forward before backward')
    }

    const gradInput = Tensor.zeros(this.lastOutput.shape)
    for (let i = 0; i < this.lastOutput.data.length; i++) {
      const sig = this.lastOutput.data[i]
      gradInput.data[i] = gradOutput.data[i] * sig * (1 - sig)
    }
    return gradInput
  }
}

/**
 * Tanh activation layer
 */
export class Tanh implements Layer {
  public readonly name = 'Tanh'
  private lastOutput?: Tensor

  forward(input: Tensor): Tensor {
    const output = Tensor.zeros(input.shape)
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = Math.tanh(input.data[i])
    }
    this.lastOutput = output
    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.lastOutput) {
      throw new Error('Must call forward before backward')
    }

    const gradInput = Tensor.zeros(this.lastOutput.shape)
    for (let i = 0; i < this.lastOutput.data.length; i++) {
      const t = this.lastOutput.data[i]
      gradInput.data[i] = gradOutput.data[i] * (1 - t * t)
    }
    return gradInput
  }
}

/**
 * Dropout layer (for regularization)
 */
export class Dropout implements Layer {
  public readonly name = 'Dropout'
  private p: number
  private mask?: Tensor
  private training: boolean = true

  constructor(p: number = 0.5) {
    if (p < 0 || p >= 1) {
      throw new Error('Dropout probability must be in [0, 1)')
    }
    this.p = p
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input.clone()
    }

    const scale = 1 / (1 - this.p)
    this.mask = Tensor.zeros(input.shape)
    const output = Tensor.zeros(input.shape)

    for (let i = 0; i < input.data.length; i++) {
      const keep = Math.random() > this.p
      this.mask.data[i] = keep ? 1 : 0
      output.data[i] = keep ? input.data[i] * scale : 0
    }

    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.mask) {
      throw new Error('Must call forward before backward')
    }

    const scale = 1 / (1 - this.p)
    const gradInput = Tensor.zeros(gradOutput.shape)

    for (let i = 0; i < gradOutput.data.length; i++) {
      gradInput.data[i] = gradOutput.data[i] * this.mask.data[i] * scale
    }

    return gradInput
  }

  train(mode: boolean = true): void {
    this.training = mode
  }

  eval(): void {
    this.training = false
  }
}

/**
 * Softmax layer
 */
export class Softmax implements Layer {
  public readonly name = 'Softmax'
  private lastOutput?: Tensor

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
      throw new Error('Softmax expects 2D input (batch, features)')
    }

    const [batch, features] = input.shape
    const output = Tensor.zeros(input.shape)

    for (let b = 0; b < batch; b++) {
      // Find max for numerical stability
      let max = -Infinity
      for (let f = 0; f < features; f++) {
        max = Math.max(max, input.get(b, f))
      }

      // Compute exp and sum
      let sum = 0
      for (let f = 0; f < features; f++) {
        const exp = Math.exp(input.get(b, f) - max)
        output.set(exp, b, f)
        sum += exp
      }

      // Normalize
      for (let f = 0; f < features; f++) {
        output.set(output.get(b, f) / sum, b, f)
      }
    }

    this.lastOutput = output
    return output
  }

  backward(gradOutput: Tensor): Tensor {
    if (!this.lastOutput) {
      throw new Error('Must call forward before backward')
    }

    // Simplified backward pass for softmax
    // In practice, softmax is usually combined with cross-entropy loss
    const gradInput = Tensor.zeros(gradOutput.shape)
    for (let i = 0; i < gradOutput.data.length; i++) {
      gradInput.data[i] = gradOutput.data[i]
    }

    return gradInput
  }
}
