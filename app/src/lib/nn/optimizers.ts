/**
 * Optimizers for neural network training
 */

import { Tensor } from '../ml/tensor'

/**
 * Base optimizer interface
 */
export interface Optimizer {
  step(parameters: Tensor[], gradients: Tensor[]): void
  zeroGrad(parameters: Tensor[]): void
}

/**
 * Stochastic Gradient Descent optimizer
 */
export class SGD implements Optimizer {
  private learningRate: number
  private momentum: number
  private velocities: Map<Tensor, Tensor> = new Map()

  constructor(learningRate: number = 0.01, momentum: number = 0) {
    this.learningRate = learningRate
    this.momentum = momentum
  }

  step(parameters: Tensor[], gradients: Tensor[]): void {
    if (parameters.length !== gradients.length) {
      throw new Error('Parameters and gradients must have same length')
    }

    for (let i = 0; i < parameters.length; i++) {
      const param = parameters[i]
      const grad = gradients[i]

      if (param.data.length !== grad.data.length) {
        throw new Error('Parameter and gradient shapes must match')
      }

      if (this.momentum > 0) {
        // Momentum update
        let velocity = this.velocities.get(param)
        if (!velocity) {
          velocity = Tensor.zeros(param.shape)
          this.velocities.set(param, velocity)
        }

        for (let j = 0; j < param.data.length; j++) {
          velocity.data[j] =
            this.momentum * velocity.data[j] - this.learningRate * grad.data[j]
          param.data[j] += velocity.data[j]
        }
      } else {
        // Simple SGD update
        for (let j = 0; j < param.data.length; j++) {
          param.data[j] -= this.learningRate * grad.data[j]
        }
      }
    }
  }

  zeroGrad(parameters: Tensor[]): void {
    for (const param of parameters) {
      if (param.grad) {
        param.grad.data.fill(0)
      }
    }
  }
}

/**
 * Adam optimizer (Adaptive Moment Estimation)
 */
export class Adam implements Optimizer {
  private learningRate: number
  private beta1: number
  private beta2: number
  private epsilon: number
  private t: number = 0
  private m: Map<Tensor, Tensor> = new Map() // First moment
  private v: Map<Tensor, Tensor> = new Map() // Second moment

  constructor(
    learningRate: number = 0.001,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    this.learningRate = learningRate
    this.beta1 = beta1
    this.beta2 = beta2
    this.epsilon = epsilon
  }

  step(parameters: Tensor[], gradients: Tensor[]): void {
    if (parameters.length !== gradients.length) {
      throw new Error('Parameters and gradients must have same length')
    }

    this.t++

    for (let i = 0; i < parameters.length; i++) {
      const param = parameters[i]
      const grad = gradients[i]

      if (param.data.length !== grad.data.length) {
        throw new Error('Parameter and gradient shapes must match')
      }

      // Initialize first and second moments if needed
      let m = this.m.get(param)
      let v = this.v.get(param)
      if (!m) {
        m = Tensor.zeros(param.shape)
        this.m.set(param, m)
      }
      if (!v) {
        v = Tensor.zeros(param.shape)
        this.v.set(param, v)
      }

      for (let j = 0; j < param.data.length; j++) {
        // Update biased first moment estimate
        m.data[j] = this.beta1 * m.data[j] + (1 - this.beta1) * grad.data[j]

        // Update biased second moment estimate
        v.data[j] =
          this.beta2 * v.data[j] + (1 - this.beta2) * grad.data[j] * grad.data[j]

        // Compute bias-corrected moments
        const mHat = m.data[j] / (1 - Math.pow(this.beta1, this.t))
        const vHat = v.data[j] / (1 - Math.pow(this.beta2, this.t))

        // Update parameters
        param.data[j] -= (this.learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon)
      }
    }
  }

  zeroGrad(parameters: Tensor[]): void {
    for (const param of parameters) {
      if (param.grad) {
        param.grad.data.fill(0)
      }
    }
  }

  /**
   * Reset optimizer state
   */
  reset(): void {
    this.t = 0
    this.m.clear()
    this.v.clear()
  }
}

/**
 * RMSprop optimizer
 */
export class RMSprop implements Optimizer {
  private learningRate: number
  private alpha: number
  private epsilon: number
  private cache: Map<Tensor, Tensor> = new Map()

  constructor(
    learningRate: number = 0.01,
    alpha: number = 0.99,
    epsilon: number = 1e-8
  ) {
    this.learningRate = learningRate
    this.alpha = alpha
    this.epsilon = epsilon
  }

  step(parameters: Tensor[], gradients: Tensor[]): void {
    if (parameters.length !== gradients.length) {
      throw new Error('Parameters and gradients must have same length')
    }

    for (let i = 0; i < parameters.length; i++) {
      const param = parameters[i]
      const grad = gradients[i]

      if (param.data.length !== grad.data.length) {
        throw new Error('Parameter and gradient shapes must match')
      }

      // Initialize cache if needed
      let c = this.cache.get(param)
      if (!c) {
        c = Tensor.zeros(param.shape)
        this.cache.set(param, c)
      }

      for (let j = 0; j < param.data.length; j++) {
        // Update cache
        c.data[j] =
          this.alpha * c.data[j] + (1 - this.alpha) * grad.data[j] * grad.data[j]

        // Update parameters
        param.data[j] -=
          (this.learningRate * grad.data[j]) / (Math.sqrt(c.data[j]) + this.epsilon)
      }
    }
  }

  zeroGrad(parameters: Tensor[]): void {
    for (const param of parameters) {
      if (param.grad) {
        param.grad.data.fill(0)
      }
    }
  }
}
