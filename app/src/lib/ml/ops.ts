/**
 * GGML-like ML operations
 */

import { Tensor, Shape } from './tensor'

/**
 * Element-wise addition
 */
export function add(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== b.shape.length) {
    throw new Error('Tensors must have same number of dimensions')
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      throw new Error('Tensors must have same shape')
    }
  }

  const result = Tensor.zeros(a.shape)
  for (let i = 0; i < a.data.length; i++) {
    result.data[i] = a.data[i] + b.data[i]
  }
  return result
}

/**
 * Element-wise multiplication
 */
export function mul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== b.shape.length) {
    throw new Error('Tensors must have same number of dimensions')
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      throw new Error('Tensors must have same shape')
    }
  }

  const result = Tensor.zeros(a.shape)
  for (let i = 0; i < a.data.length; i++) {
    result.data[i] = a.data[i] * b.data[i]
  }
  return result
}

/**
 * Scalar multiplication
 */
export function scale(tensor: Tensor, scalar: number): Tensor {
  const result = Tensor.zeros(tensor.shape)
  for (let i = 0; i < tensor.data.length; i++) {
    result.data[i] = tensor.data[i] * scalar
  }
  return result
}

/**
 * Matrix multiplication (2D only)
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error('Matrix multiplication requires 2D tensors')
  }

  const [m, k1] = a.shape
  const [k2, n] = b.shape

  if (k1 !== k2) {
    throw new Error(
      `Incompatible shapes for matmul: [${m}, ${k1}] x [${k2}, ${n}]`
    )
  }

  const result = Tensor.zeros([m, n])

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0
      for (let k = 0; k < k1; k++) {
        sum += a.get(i, k) * b.get(k, j)
      }
      result.set(sum, i, j)
    }
  }

  return result
}

/**
 * ReLU activation function
 */
export function relu(tensor: Tensor): Tensor {
  const result = Tensor.zeros(tensor.shape)
  for (let i = 0; i < tensor.data.length; i++) {
    result.data[i] = Math.max(0, tensor.data[i])
  }
  return result
}

/**
 * Sigmoid activation function
 */
export function sigmoid(tensor: Tensor): Tensor {
  const result = Tensor.zeros(tensor.shape)
  for (let i = 0; i < tensor.data.length; i++) {
    result.data[i] = 1 / (1 + Math.exp(-tensor.data[i]))
  }
  return result
}

/**
 * Tanh activation function
 */
export function tanh(tensor: Tensor): Tensor {
  const result = Tensor.zeros(tensor.shape)
  for (let i = 0; i < tensor.data.length; i++) {
    result.data[i] = Math.tanh(tensor.data[i])
  }
  return result
}

/**
 * Softmax activation function
 */
export function softmax(tensor: Tensor): Tensor {
  if (tensor.shape.length !== 2) {
    throw new Error('Softmax only supports 2D tensors (batch, features)')
  }

  const [batch, features] = tensor.shape
  const result = Tensor.zeros(tensor.shape)

  for (let b = 0; b < batch; b++) {
    // Find max for numerical stability
    let max = -Infinity
    for (let f = 0; f < features; f++) {
      max = Math.max(max, tensor.get(b, f))
    }

    // Compute exp and sum
    let sum = 0
    for (let f = 0; f < features; f++) {
      const exp = Math.exp(tensor.get(b, f) - max)
      result.set(exp, b, f)
      sum += exp
    }

    // Normalize
    for (let f = 0; f < features; f++) {
      result.set(result.get(b, f) / sum, b, f)
    }
  }

  return result
}

/**
 * Mean along an axis
 */
export function mean(tensor: Tensor, axis?: number): Tensor {
  if (axis === undefined) {
    // Mean of all elements
    let sum = 0
    for (let i = 0; i < tensor.data.length; i++) {
      sum += tensor.data[i]
    }
    return new Tensor([sum / tensor.data.length], [1])
  }

  if (axis < 0 || axis >= tensor.shape.length) {
    throw new Error(`Invalid axis ${axis}`)
  }

  // Compute new shape
  const newShape = tensor.shape.filter((_, i) => i !== axis)
  const result = Tensor.zeros(newShape)

  // This is a simplified implementation for 2D tensors
  if (tensor.shape.length === 2 && axis === 0) {
    const [rows, cols] = tensor.shape
    for (let j = 0; j < cols; j++) {
      let sum = 0
      for (let i = 0; i < rows; i++) {
        sum += tensor.get(i, j)
      }
      result.data[j] = sum / rows
    }
  } else if (tensor.shape.length === 2 && axis === 1) {
    const [rows, cols] = tensor.shape
    for (let i = 0; i < rows; i++) {
      let sum = 0
      for (let j = 0; j < cols; j++) {
        sum += tensor.get(i, j)
      }
      result.data[i] = sum / cols
    }
  }

  return result
}

/**
 * Sum along an axis
 */
export function sum(tensor: Tensor, axis?: number): Tensor {
  if (axis === undefined) {
    // Sum of all elements
    let total = 0
    for (let i = 0; i < tensor.data.length; i++) {
      total += tensor.data[i]
    }
    return new Tensor([total], [1])
  }

  if (axis < 0 || axis >= tensor.shape.length) {
    throw new Error(`Invalid axis ${axis}`)
  }

  // Compute new shape
  const newShape = tensor.shape.filter((_, i) => i !== axis)
  const result = Tensor.zeros(newShape)

  // Simplified implementation for 2D tensors
  if (tensor.shape.length === 2 && axis === 0) {
    const [rows, cols] = tensor.shape
    for (let j = 0; j < cols; j++) {
      let total = 0
      for (let i = 0; i < rows; i++) {
        total += tensor.get(i, j)
      }
      result.data[j] = total
    }
  } else if (tensor.shape.length === 2 && axis === 1) {
    const [rows, cols] = tensor.shape
    for (let i = 0; i < rows; i++) {
      let total = 0
      for (let j = 0; j < cols; j++) {
        total += tensor.get(i, j)
      }
      result.data[i] = total
    }
  }

  return result
}
