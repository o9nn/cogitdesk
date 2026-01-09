/**
 * Tensor data structure for ML operations (GGML-inspired)
 */

export type Shape = number[]

export class Tensor {
  public readonly data: Float32Array
  public readonly shape: Shape
  public readonly strides: number[]
  public grad?: Tensor

  constructor(data: number[] | Float32Array, shape: Shape) {
    if (Array.isArray(data)) {
      this.data = new Float32Array(data)
    } else {
      this.data = data
    }
    this.shape = shape
    this.strides = this.computeStrides(shape)

    const expectedSize = shape.reduce((a, b) => a * b, 1)
    if (this.data.length !== expectedSize) {
      throw new Error(
        `Data size ${this.data.length} does not match shape ${shape}`
      )
    }
  }

  private computeStrides(shape: Shape): number[] {
    const strides = new Array(shape.length)
    let stride = 1
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride
      stride *= shape[i]
    }
    return strides
  }

  /**
   * Get element at given indices
   */
  get(...indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new Error('Invalid number of indices')
    }
    let offset = 0
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(`Index ${indices[i]} out of bounds for dimension ${i}`)
      }
      offset += indices[i] * this.strides[i]
    }
    return this.data[offset]
  }

  /**
   * Set element at given indices
   */
  set(value: number, ...indices: number[]): void {
    if (indices.length !== this.shape.length) {
      throw new Error('Invalid number of indices')
    }
    let offset = 0
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(`Index ${indices[i]} out of bounds for dimension ${i}`)
      }
      offset += indices[i] * this.strides[i]
    }
    this.data[offset] = value
  }

  /**
   * Reshape tensor
   */
  reshape(newShape: Shape): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1)
    const oldSize = this.shape.reduce((a, b) => a * b, 1)
    if (newSize !== oldSize) {
      throw new Error(
        `Cannot reshape tensor of size ${oldSize} to size ${newSize}`
      )
    }
    return new Tensor(this.data, newShape)
  }

  /**
   * Create a copy of this tensor
   */
  clone(): Tensor {
    return new Tensor(new Float32Array(this.data), [...this.shape])
  }

  /**
   * Get the total number of elements
   */
  get size(): number {
    return this.data.length
  }

  /**
   * Create a tensor filled with zeros
   */
  static zeros(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1)
    return new Tensor(new Float32Array(size), shape)
  }

  /**
   * Create a tensor filled with ones
   */
  static ones(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(size).fill(1)
    return new Tensor(data, shape)
  }

  /**
   * Create a tensor with random values
   */
  static random(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(size).map(() => Math.random())
    return new Tensor(data, shape)
  }

  /**
   * Create a tensor with values from a normal distribution
   */
  static randn(shape: Shape, mean: number = 0, std: number = 1): Tensor {
    const size = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(size).map(() => {
      // Box-Muller transform
      const u1 = Math.random()
      const u2 = Math.random()
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      return mean + std * z
    })
    return new Tensor(data, shape)
  }
}
