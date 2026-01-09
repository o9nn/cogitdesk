/**
 * Torch7-style sequential neural network builder
 */

import { Layer } from './layers'
import { Tensor } from '../ml/tensor'

/**
 * Sequential model container
 */
export class Sequential {
  private layers: Layer[] = []
  private training: boolean = true

  /**
   * Add a layer to the model
   */
  add(layer: Layer): Sequential {
    this.layers.push(layer)
    return this
  }

  /**
   * Forward pass through all layers
   */
  forward(input: Tensor): Tensor {
    let output = input
    for (const layer of this.layers) {
      output = layer.forward(output)
    }
    return output
  }

  /**
   * Backward pass through all layers
   */
  backward(gradOutput: Tensor): Tensor {
    let grad = gradOutput
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      if (layer.backward) {
        grad = layer.backward(grad)
      }
    }
    return grad
  }

  /**
   * Get all trainable parameters
   */
  parameters(): Tensor[] {
    const params: Tensor[] = []
    for (const layer of this.layers) {
      if (layer.parameters) {
        params.push(...layer.parameters())
      }
    }
    return params
  }

  /**
   * Set model to training mode
   */
  train(mode: boolean = true): void {
    this.training = mode
    for (const layer of this.layers) {
      if ('train' in layer && typeof layer.train === 'function') {
        layer.train(mode)
      }
    }
  }

  /**
   * Set model to evaluation mode
   */
  eval(): void {
    this.train(false)
  }

  /**
   * Get summary of the model
   */
  summary(): string {
    const lines: string[] = ['Model: Sequential', '='.repeat(60)]

    let totalParams = 0
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]
      const params = layer.parameters ? layer.parameters() : []
      const numParams = params.reduce((sum, p) => sum + p.size, 0)
      totalParams += numParams

      lines.push(
        `Layer ${i + 1}: ${layer.name} (params: ${numParams.toLocaleString()})`
      )
    }

    lines.push('='.repeat(60))
    lines.push(`Total parameters: ${totalParams.toLocaleString()}`)

    return lines.join('\n')
  }

  /**
   * Get number of layers
   */
  get length(): number {
    return this.layers.length
  }

  /**
   * Get a specific layer by index
   */
  getLayer(index: number): Layer {
    if (index < 0 || index >= this.layers.length) {
      throw new Error(`Invalid layer index: ${index}`)
    }
    return this.layers[index]
  }
}
