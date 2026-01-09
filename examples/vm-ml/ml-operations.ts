/**
 * Example: ML Operations with Tensors
 * 
 * This example demonstrates various tensor operations and ML functions
 * similar to GGML style.
 */

import { Tensor } from '../../app/src/lib/ml'
import * as ops from '../../app/src/lib/ml/ops'

function main() {
  console.log('=== ML Operations Example ===\n')

  // Example 1: Creating tensors
  console.log('Example 1: Creating tensors')
  const t1 = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
  console.log('Tensor shape:', t1.shape)
  console.log('Tensor data:', Array.from(t1.data))
  console.log('Element at [0, 1]:', t1.get(0, 1))
  console.log()

  // Example 2: Tensor creation methods
  console.log('Example 2: Tensor creation methods')
  const zeros = Tensor.zeros([2, 3])
  console.log('Zeros:', Array.from(zeros.data))
  
  const ones = Tensor.ones([2, 3])
  console.log('Ones:', Array.from(ones.data))
  
  const random = Tensor.random([2, 3])
  console.log('Random:', Array.from(random.data).map(x => x.toFixed(4)))
  console.log()

  // Example 3: Tensor reshaping
  console.log('Example 3: Tensor reshaping')
  const original = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
  console.log('Original shape:', original.shape)
  
  const reshaped = original.reshape([3, 2])
  console.log('Reshaped to:', reshaped.shape)
  console.log('Data (same):', Array.from(reshaped.data))
  console.log()

  // Example 4: Element-wise operations
  console.log('Example 4: Element-wise operations')
  const a = new Tensor([1, 2, 3, 4], [2, 2])
  const b = new Tensor([5, 6, 7, 8], [2, 2])
  
  const sum = ops.add(a, b)
  console.log('A:', Array.from(a.data))
  console.log('B:', Array.from(b.data))
  console.log('A + B:', Array.from(sum.data))
  
  const product = ops.mul(a, b)
  console.log('A * B (element-wise):', Array.from(product.data))
  console.log()

  // Example 5: Matrix multiplication
  console.log('Example 5: Matrix multiplication')
  const m1 = new Tensor([1, 2, 3, 4], [2, 2])
  const m2 = new Tensor([5, 6, 7, 8], [2, 2])
  
  const result = ops.matmul(m1, m2)
  console.log('M1:')
  console.log(`  [${m1.get(0, 0)}, ${m1.get(0, 1)}]`)
  console.log(`  [${m1.get(1, 0)}, ${m1.get(1, 1)}]`)
  console.log('M2:')
  console.log(`  [${m2.get(0, 0)}, ${m2.get(0, 1)}]`)
  console.log(`  [${m2.get(1, 0)}, ${m2.get(1, 1)}]`)
  console.log('M1 @ M2:')
  console.log(`  [${result.get(0, 0)}, ${result.get(0, 1)}]`)
  console.log(`  [${result.get(1, 0)}, ${result.get(1, 1)}]`)
  console.log()

  // Example 6: Activation functions
  console.log('Example 6: Activation functions')
  const input = new Tensor([-2, -1, 0, 1, 2], [5])
  
  const reluResult = ops.relu(input)
  console.log('Input:   ', Array.from(input.data))
  console.log('ReLU:    ', Array.from(reluResult.data))
  
  const sigmoidResult = ops.sigmoid(input)
  console.log('Sigmoid: ', Array.from(sigmoidResult.data).map(x => x.toFixed(4)))
  
  const tanhResult = ops.tanh(input)
  console.log('Tanh:    ', Array.from(tanhResult.data).map(x => x.toFixed(4)))
  console.log()

  // Example 7: Softmax
  console.log('Example 7: Softmax activation')
  const logits = new Tensor([1, 2, 3], [1, 3])
  const softmaxResult = ops.softmax(logits)
  
  console.log('Logits:  ', Array.from(logits.data))
  console.log('Softmax: ', Array.from(softmaxResult.data).map(x => x.toFixed(4)))
  
  // Sum should be 1
  let total = 0
  for (let i = 0; i < softmaxResult.data.length; i++) {
    total += softmaxResult.data[i]
  }
  console.log('Sum:     ', total.toFixed(4), '(should be 1.0)')
  console.log()

  // Example 8: Reduction operations
  console.log('Example 8: Reduction operations')
  const data = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
  console.log('Data:')
  console.log(`  [${data.get(0, 0)}, ${data.get(0, 1)}, ${data.get(0, 2)}]`)
  console.log(`  [${data.get(1, 0)}, ${data.get(1, 1)}, ${data.get(1, 2)}]`)
  
  const totalSum = ops.sum(data)
  console.log('Total sum:', totalSum.data[0])
  
  const colSum = ops.sum(data, 0) // Sum along rows (result: columns)
  console.log('Column sums:', Array.from(colSum.data))
  
  const rowSum = ops.sum(data, 1) // Sum along columns (result: rows)
  console.log('Row sums:', Array.from(rowSum.data))
  
  const totalMean = ops.mean(data)
  console.log('Total mean:', totalMean.data[0])
  console.log()

  // Example 9: Scalar operations
  console.log('Example 9: Scalar operations')
  const tensor = new Tensor([1, 2, 3, 4], [2, 2])
  const scaled = ops.scale(tensor, 2.5)
  
  console.log('Original:', Array.from(tensor.data))
  console.log('Scaled by 2.5:', Array.from(scaled.data))
  console.log()

  // Example 10: Large matrix multiplication
  console.log('Example 10: Large matrix multiplication')
  const large1 = Tensor.randn([100, 50])
  const large2 = Tensor.randn([50, 30])
  
  console.log('Matrix 1 shape:', large1.shape)
  console.log('Matrix 2 shape:', large2.shape)
  
  const start = Date.now()
  const largeResult = ops.matmul(large1, large2)
  const elapsed = Date.now() - start
  
  console.log('Result shape:', largeResult.shape)
  console.log(`Computation time: ${elapsed}ms`)
  console.log()

  console.log('=== Example Complete ===')
}

// Run if executed directly
if (require.main === module) {
  main()
}

export { main }
