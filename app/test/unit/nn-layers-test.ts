import { describe, it } from 'node:test'
import assert from 'node:assert'
import { Linear, ReLU, Sigmoid, Tanh, Dropout, Softmax } from '../../src/lib/nn/layers'
import { Sequential } from '../../src/lib/nn/sequential'
import { Tensor } from '../../src/lib/ml/tensor'

describe('Neural Network Layers', () => {
  describe('Linear', () => {
    it('should create a linear layer with correct dimensions', () => {
      const layer = new Linear(3, 2)
      const params = layer.parameters()

      assert.strictEqual(params.length, 2) // weight and bias
      assert.deepStrictEqual(params[0].shape, [2, 3]) // weight
      assert.deepStrictEqual(params[1].shape, [2]) // bias
    })

    it('should perform forward pass', () => {
      const layer = new Linear(3, 2)

      // Create a simple input
      const input = new Tensor([1, 2, 3], [1, 3])
      const output = layer.forward(input)

      assert.deepStrictEqual(output.shape, [1, 2])
    })

    it('should throw error for mismatched input size', () => {
      const layer = new Linear(3, 2)
      const wrongInput = new Tensor([1, 2], [1, 2])

      assert.throws(() => {
        layer.forward(wrongInput)
      })
    })

    it('should perform backward pass', () => {
      const layer = new Linear(3, 2)

      // Forward pass
      const input = new Tensor([1, 2, 3], [1, 3])
      const output = layer.forward(input)

      // Backward pass
      const gradOutput = Tensor.ones([1, 2])
      const gradInput = layer.backward(gradOutput)

      assert.deepStrictEqual(gradInput.shape, [1, 3])
    })
  })

  describe('ReLU', () => {
    it('should apply ReLU activation', () => {
      const layer = new ReLU()
      const input = new Tensor([-2, -1, 0, 1, 2], [1, 5])
      const output = layer.forward(input)

      assert.strictEqual(output.data[0], 0)
      assert.strictEqual(output.data[1], 0)
      assert.strictEqual(output.data[2], 0)
      assert.strictEqual(output.data[3], 1)
      assert.strictEqual(output.data[4], 2)
    })

    it('should perform backward pass', () => {
      const layer = new ReLU()
      const input = new Tensor([-2, -1, 0, 1, 2], [1, 5])
      
      layer.forward(input)
      const gradOutput = Tensor.ones([1, 5])
      const gradInput = layer.backward(gradOutput)

      // Gradient should be 0 for negative inputs, 1 for positive
      assert.strictEqual(gradInput.data[0], 0)
      assert.strictEqual(gradInput.data[1], 0)
      assert.strictEqual(gradInput.data[2], 0)
      assert.strictEqual(gradInput.data[3], 1)
      assert.strictEqual(gradInput.data[4], 1)
    })
  })

  describe('Sigmoid', () => {
    it('should apply sigmoid activation', () => {
      const layer = new Sigmoid()
      const input = new Tensor([0], [1, 1])
      const output = layer.forward(input)

      // Sigmoid(0) = 0.5
      assert.ok(Math.abs(output.data[0] - 0.5) < 1e-6)
    })

    it('should output values between 0 and 1', () => {
      const layer = new Sigmoid()
      const input = new Tensor([-10, -5, 0, 5, 10], [1, 5])
      const output = layer.forward(input)

      for (let i = 0; i < output.data.length; i++) {
        assert.ok(output.data[i] >= 0 && output.data[i] <= 1)
      }
    })
  })

  describe('Tanh', () => {
    it('should apply tanh activation', () => {
      const layer = new Tanh()
      const input = new Tensor([0], [1, 1])
      const output = layer.forward(input)

      // Tanh(0) = 0
      assert.ok(Math.abs(output.data[0]) < 1e-6)
    })

    it('should output values between -1 and 1', () => {
      const layer = new Tanh()
      const input = new Tensor([-10, -5, 0, 5, 10], [1, 5])
      const output = layer.forward(input)

      for (let i = 0; i < output.data.length; i++) {
        assert.ok(output.data[i] >= -1 && output.data[i] <= 1)
      }
    })
  })

  describe('Dropout', () => {
    it('should create dropout layer with correct probability', () => {
      const layer = new Dropout(0.3)
      assert.strictEqual(layer.name, 'Dropout')
    })

    it('should pass through in eval mode', () => {
      const layer = new Dropout(0.5)
      layer.eval()

      const input = new Tensor([1, 2, 3, 4], [1, 4])
      const output = layer.forward(input)

      // In eval mode, all values should pass through unchanged
      for (let i = 0; i < input.data.length; i++) {
        assert.strictEqual(output.data[i], input.data[i])
      }
    })

    it('should drop some values in training mode', () => {
      const layer = new Dropout(0.5)
      layer.train()

      const input = Tensor.ones([1, 100])
      const output = layer.forward(input)

      // Some values should be dropped (zero)
      let zeroCount = 0
      for (let i = 0; i < output.data.length; i++) {
        if (output.data[i] === 0) zeroCount++
      }

      // With 0.5 dropout, roughly half should be zero (allow some variance)
      assert.ok(zeroCount > 20 && zeroCount < 80)
    })
  })

  describe('Softmax', () => {
    it('should apply softmax activation', () => {
      const layer = new Softmax()
      const input = new Tensor([1, 2, 3], [1, 3])
      const output = layer.forward(input)

      // Sum should be 1
      let sum = 0
      for (let i = 0; i < output.data.length; i++) {
        sum += output.data[i]
      }
      assert.ok(Math.abs(sum - 1.0) < 1e-6)

      // All values should be between 0 and 1
      for (let i = 0; i < output.data.length; i++) {
        assert.ok(output.data[i] >= 0 && output.data[i] <= 1)
      }
    })

    it('should handle batch inputs', () => {
      const layer = new Softmax()
      const input = new Tensor([1, 2, 3, 4, 5, 6], [2, 3])
      const output = layer.forward(input)

      // Each row should sum to 1
      for (let b = 0; b < 2; b++) {
        let sum = 0
        for (let f = 0; f < 3; f++) {
          sum += output.get(b, f)
        }
        assert.ok(Math.abs(sum - 1.0) < 1e-6)
      }
    })
  })
})

describe('Sequential Model', () => {
  it('should create an empty sequential model', () => {
    const model = new Sequential()
    assert.strictEqual(model.length, 0)
  })

  it('should add layers to the model', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5))
    model.add(new ReLU())
    model.add(new Linear(5, 2))

    assert.strictEqual(model.length, 3)
  })

  it('should perform forward pass through all layers', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5))
    model.add(new ReLU())
    model.add(new Linear(5, 2))

    const input = new Tensor([1, 2, 3], [1, 3])
    const output = model.forward(input)

    assert.deepStrictEqual(output.shape, [1, 2])
  })

  it('should get all parameters', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5)) // 2 parameters (weight, bias)
    model.add(new ReLU()) // No parameters
    model.add(new Linear(5, 2)) // 2 parameters

    const params = model.parameters()
    assert.strictEqual(params.length, 4)
  })

  it('should generate model summary', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5))
    model.add(new ReLU())
    model.add(new Linear(5, 2))

    const summary = model.summary()
    assert.ok(summary.includes('Sequential'))
    assert.ok(summary.includes('Linear'))
    assert.ok(summary.includes('ReLU'))
  })

  it('should get a specific layer by index', () => {
    const model = new Sequential()
    const linear = new Linear(3, 5)
    model.add(linear)
    model.add(new ReLU())

    const layer = model.getLayer(0)
    assert.strictEqual(layer.name, 'Linear')
  })

  it('should throw error for invalid layer index', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5))

    assert.throws(() => {
      model.getLayer(5)
    })
  })

  it('should switch between train and eval modes', () => {
    const model = new Sequential()
    model.add(new Linear(3, 5))
    model.add(new Dropout(0.5))

    model.train()
    // Training mode is set
    
    model.eval()
    // Eval mode is set
  })

  it('should perform end-to-end inference', () => {
    const model = new Sequential()
    model.add(new Linear(2, 3))
    model.add(new ReLU())
    model.add(new Linear(3, 1))
    model.add(new Sigmoid())

    model.eval()

    const input = new Tensor([1, 2], [1, 2])
    const output = model.forward(input)

    assert.deepStrictEqual(output.shape, [1, 1])
    // Output should be between 0 and 1 (sigmoid)
    assert.ok(output.data[0] >= 0 && output.data[0] <= 1)
  })
})
