import { describe, it } from 'node:test'
import assert from 'node:assert'
import { Tensor } from '../../src/lib/ml/tensor'
import * as ops from '../../src/lib/ml/ops'

describe('ML Operations', () => {
  describe('Tensor', () => {
    it('should create a tensor with correct shape', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])

      assert.deepStrictEqual(tensor.shape, [2, 2])
      assert.strictEqual(tensor.size, 4)
    })

    it('should get element at indices', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])

      assert.strictEqual(tensor.get(0, 0), 1)
      assert.strictEqual(tensor.get(0, 1), 2)
      assert.strictEqual(tensor.get(1, 0), 3)
      assert.strictEqual(tensor.get(1, 1), 4)
    })

    it('should set element at indices', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])

      tensor.set(99, 0, 0)
      assert.strictEqual(tensor.get(0, 0), 99)
    })

    it('should reshape tensor', () => {
      const tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
      const reshaped = tensor.reshape([3, 2])

      assert.deepStrictEqual(reshaped.shape, [3, 2])
      assert.strictEqual(reshaped.get(0, 0), 1)
      assert.strictEqual(reshaped.get(0, 1), 2)
      assert.strictEqual(reshaped.get(1, 0), 3)
    })

    it('should create zeros tensor', () => {
      const tensor = Tensor.zeros([2, 3])

      assert.deepStrictEqual(tensor.shape, [2, 3])
      assert.strictEqual(tensor.size, 6)
      for (let i = 0; i < tensor.data.length; i++) {
        assert.strictEqual(tensor.data[i], 0)
      }
    })

    it('should create ones tensor', () => {
      const tensor = Tensor.ones([2, 3])

      assert.deepStrictEqual(tensor.shape, [2, 3])
      for (let i = 0; i < tensor.data.length; i++) {
        assert.strictEqual(tensor.data[i], 1)
      }
    })

    it('should clone tensor', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const clone = tensor.clone()

      assert.deepStrictEqual(clone.shape, tensor.shape)
      assert.deepStrictEqual(Array.from(clone.data), Array.from(tensor.data))

      // Verify it's a separate copy
      clone.set(99, 0, 0)
      assert.notStrictEqual(clone.get(0, 0), tensor.get(0, 0))
    })

    it('should throw error for mismatched data and shape', () => {
      assert.throws(() => {
        new Tensor([1, 2, 3], [2, 2]) // 3 elements but shape requires 4
      })
    })
  })

  describe('ops.add', () => {
    it('should add two tensors element-wise', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2])
      const b = new Tensor([5, 6, 7, 8], [2, 2])

      const result = ops.add(a, b)

      assert.deepStrictEqual(result.shape, [2, 2])
      assert.strictEqual(result.get(0, 0), 6)
      assert.strictEqual(result.get(0, 1), 8)
      assert.strictEqual(result.get(1, 0), 10)
      assert.strictEqual(result.get(1, 1), 12)
    })

    it('should throw error for mismatched shapes', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2])
      const b = new Tensor([1, 2, 3], [3, 1])

      assert.throws(() => {
        ops.add(a, b)
      })
    })
  })

  describe('ops.mul', () => {
    it('should multiply two tensors element-wise', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2])
      const b = new Tensor([2, 3, 4, 5], [2, 2])

      const result = ops.mul(a, b)

      assert.strictEqual(result.get(0, 0), 2)
      assert.strictEqual(result.get(0, 1), 6)
      assert.strictEqual(result.get(1, 0), 12)
      assert.strictEqual(result.get(1, 1), 20)
    })
  })

  describe('ops.scale', () => {
    it('should scale tensor by scalar', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.scale(tensor, 3)

      assert.strictEqual(result.get(0, 0), 3)
      assert.strictEqual(result.get(0, 1), 6)
      assert.strictEqual(result.get(1, 0), 9)
      assert.strictEqual(result.get(1, 1), 12)
    })
  })

  describe('ops.matmul', () => {
    it('should perform matrix multiplication', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2])
      const b = new Tensor([5, 6, 7, 8], [2, 2])

      const result = ops.matmul(a, b)

      // [1 2] x [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
      // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
      assert.strictEqual(result.get(0, 0), 19)
      assert.strictEqual(result.get(0, 1), 22)
      assert.strictEqual(result.get(1, 0), 43)
      assert.strictEqual(result.get(1, 1), 50)
    })

    it('should handle non-square matrices', () => {
      const a = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
      const b = new Tensor([7, 8, 9, 10, 11, 12], [3, 2])

      const result = ops.matmul(a, b)

      // Result should be 2x2
      assert.deepStrictEqual(result.shape, [2, 2])
    })

    it('should throw error for incompatible shapes', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2])
      const b = new Tensor([1, 2, 3], [3, 1])

      assert.throws(() => {
        ops.matmul(a, b)
      })
    })
  })

  describe('activation functions', () => {
    it('ops.relu should apply ReLU activation', () => {
      const tensor = new Tensor([-2, -1, 0, 1, 2], [5])
      const result = ops.relu(tensor)

      assert.strictEqual(result.data[0], 0)
      assert.strictEqual(result.data[1], 0)
      assert.strictEqual(result.data[2], 0)
      assert.strictEqual(result.data[3], 1)
      assert.strictEqual(result.data[4], 2)
    })

    it('ops.sigmoid should apply sigmoid activation', () => {
      const tensor = new Tensor([0], [1])
      const result = ops.sigmoid(tensor)

      // Sigmoid(0) = 0.5
      assert.ok(Math.abs(result.data[0] - 0.5) < 1e-6)
    })

    it('ops.tanh should apply tanh activation', () => {
      const tensor = new Tensor([0], [1])
      const result = ops.tanh(tensor)

      // Tanh(0) = 0
      assert.ok(Math.abs(result.data[0]) < 1e-6)
    })

    it('ops.softmax should apply softmax activation', () => {
      const tensor = new Tensor([1, 2, 3], [1, 3])
      const result = ops.softmax(tensor)

      // Check that values sum to 1
      let sum = 0
      for (let i = 0; i < result.data.length; i++) {
        sum += result.data[i]
      }
      assert.ok(Math.abs(sum - 1.0) < 1e-6)

      // Check that all values are between 0 and 1
      for (let i = 0; i < result.data.length; i++) {
        assert.ok(result.data[i] >= 0 && result.data[i] <= 1)
      }
    })
  })

  describe('ops.mean', () => {
    it('should compute mean of all elements', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.mean(tensor)

      assert.strictEqual(result.data[0], 2.5)
    })

    it('should compute mean along axis 0', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.mean(tensor, 0)

      // Mean of columns: [(1+3)/2, (2+4)/2] = [2, 3]
      assert.strictEqual(result.data[0], 2)
      assert.strictEqual(result.data[1], 3)
    })

    it('should compute mean along axis 1', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.mean(tensor, 1)

      // Mean of rows: [(1+2)/2, (3+4)/2] = [1.5, 3.5]
      assert.strictEqual(result.data[0], 1.5)
      assert.strictEqual(result.data[1], 3.5)
    })
  })

  describe('ops.sum', () => {
    it('should compute sum of all elements', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.sum(tensor)

      assert.strictEqual(result.data[0], 10)
    })

    it('should compute sum along axis 0', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.sum(tensor, 0)

      // Sum of columns: [1+3, 2+4] = [4, 6]
      assert.strictEqual(result.data[0], 4)
      assert.strictEqual(result.data[1], 6)
    })

    it('should compute sum along axis 1', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2])
      const result = ops.sum(tensor, 1)

      // Sum of rows: [1+2, 3+4] = [3, 7]
      assert.strictEqual(result.data[0], 3)
      assert.strictEqual(result.data[1], 7)
    })
  })
})
