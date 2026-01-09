/**
 * Example: Neural Network Training
 * 
 * This example demonstrates how to build and train a simple neural network
 * for classification tasks using the Torch7-style API.
 */

import { Sequential, Linear, ReLU, Sigmoid, Dropout } from '../../app/src/lib/nn'
import { Tensor } from '../../app/src/lib/ml'
import { SGD, Adam } from '../../app/src/lib/nn/optimizers'

function main() {
  console.log('=== Neural Network Example ===\n')

  // Example 1: Build a simple model
  console.log('Example 1: Building a simple feedforward network')
  const model = new Sequential()
    .add(new Linear(10, 20))
    .add(new ReLU())
    .add(new Dropout(0.3))
    .add(new Linear(20, 10))
    .add(new ReLU())
    .add(new Linear(10, 1))
    .add(new Sigmoid())

  console.log(model.summary())
  console.log()

  // Example 2: Forward pass
  console.log('Example 2: Forward pass through the network')
  model.eval() // Set to evaluation mode (disables dropout)
  
  const input = Tensor.randn([1, 10]) // Batch of 1, 10 features
  console.log('Input shape:', input.shape)
  
  const output = model.forward(input)
  console.log('Output shape:', output.shape)
  console.log('Output value:', output.data[0].toFixed(4))
  console.log()

  // Example 3: Multi-class classification model
  console.log('Example 3: Multi-class classification network')
  const classifier = new Sequential()
    .add(new Linear(4, 8))
    .add(new ReLU())
    .add(new Linear(8, 3))

  const testInput = new Tensor([1, 2, 3, 4], [1, 4])
  const classScores = classifier.forward(testInput)
  console.log('Class scores:', Array.from(classScores.data))
  console.log()

  // Example 4: Layer inspection
  console.log('Example 4: Inspecting individual layers')
  const linearLayer = new Linear(5, 3)
  console.log(`Layer: ${linearLayer.name}`)
  const params = linearLayer.parameters()
  console.log(`Number of parameters: ${params.length}`)
  console.log(`Weight shape: [${params[0].shape}]`)
  console.log(`Bias shape: [${params[1].shape}]`)
  console.log(`Total parameters: ${params[0].size + params[1].size}`)
  console.log()

  // Example 5: Activation functions comparison
  console.log('Example 5: Comparing activation functions')
  const testData = new Tensor([-2, -1, 0, 1, 2], [1, 5])
  
  const reluLayer = new ReLU()
  const sigmoidLayer = new Sigmoid()
  
  const reluOutput = reluLayer.forward(testData.clone())
  const sigmoidOutput = sigmoidLayer.forward(testData.clone())
  
  console.log('Input:   ', Array.from(testData.data).map(x => x.toFixed(2)))
  console.log('ReLU:    ', Array.from(reluOutput.data).map(x => x.toFixed(2)))
  console.log('Sigmoid: ', Array.from(sigmoidOutput.data).map(x => x.toFixed(4)))
  console.log()

  // Example 6: Optimizer usage
  console.log('Example 6: Using optimizers')
  
  // SGD optimizer
  const sgd = new SGD(0.01, 0.9) // learning rate, momentum
  console.log('Created SGD optimizer with lr=0.01, momentum=0.9')
  
  // Adam optimizer
  const adam = new Adam(0.001)
  console.log('Created Adam optimizer with lr=0.001')
  console.log()

  // Example 7: Simple training simulation
  console.log('Example 7: Simulated training loop')
  const trainModel = new Sequential()
    .add(new Linear(2, 3))
    .add(new ReLU())
    .add(new Linear(3, 1))

  trainModel.train()
  
  // Simulated training data
  const batchSize = 4
  const X = Tensor.randn([batchSize, 2])
  
  console.log('Training for 3 iterations...')
  for (let i = 0; i < 3; i++) {
    const predictions = trainModel.forward(X)
    console.log(`Iteration ${i + 1}: predictions shape = [${predictions.shape}]`)
    
    // In real training, you would:
    // 1. Compute loss
    // 2. Compute gradients via backward pass
    // 3. Update parameters with optimizer
  }
  console.log()

  // Example 8: Model evaluation
  console.log('Example 8: Model evaluation mode')
  const evalModel = new Sequential()
    .add(new Linear(3, 5))
    .add(new Dropout(0.5))
    .add(new Linear(5, 2))

  evalModel.train()
  console.log('Training mode: Dropout is active')
  
  evalModel.eval()
  console.log('Evaluation mode: Dropout is disabled')
  
  const evalInput = Tensor.ones([1, 3])
  const evalOutput = evalModel.forward(evalInput)
  console.log('Evaluation output:', Array.from(evalOutput.data).map(x => x.toFixed(4)))
  console.log()

  console.log('=== Example Complete ===')
}

// Run if executed directly
if (require.main === module) {
  main()
}

export { main }
