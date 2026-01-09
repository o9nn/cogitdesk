/**
 * Example: Simple VM Program Execution
 * 
 * This example demonstrates how to create and execute a simple VM program
 * that performs basic arithmetic operations.
 */

import { VMEngine, OpCode, VMProgram } from '../../app/src/lib/vm'

function main() {
  console.log('=== VM Engine Example ===\n')

  // Example 1: Simple addition
  console.log('Example 1: Computing 42 + 58')
  const engine1 = new VMEngine()
  const program1: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [42] },
      { opcode: OpCode.PUSH, operands: [58] },
      { opcode: OpCode.ADD, operands: [] },
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine1.loadProgram(program1)
  const result1 = engine1.execute()
  console.log(`Result: ${result1.value}`)
  console.log(`Cycles: ${result1.cycles}\n`)

  // Example 2: Expression evaluation (a + b) * c
  console.log('Example 2: Computing (10 + 5) * 3')
  const engine2 = new VMEngine()
  const program2: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [10] },
      { opcode: OpCode.PUSH, operands: [5] },
      { opcode: OpCode.ADD, operands: [] },
      { opcode: OpCode.PUSH, operands: [3] },
      { opcode: OpCode.MUL, operands: [] },
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine2.loadProgram(program2)
  const result2 = engine2.execute()
  console.log(`Result: ${result2.value}`)
  console.log(`Cycles: ${result2.cycles}\n`)

  // Example 3: Using memory
  console.log('Example 3: Using memory to store and load values')
  const engine3 = new VMEngine()
  const program3: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [100] },
      { opcode: OpCode.STORE, operands: [0] }, // Store 100 at address 0
      { opcode: OpCode.PUSH, operands: [200] },
      { opcode: OpCode.STORE, operands: [1] }, // Store 200 at address 1
      { opcode: OpCode.LOAD, operands: [0] },  // Load from address 0
      { opcode: OpCode.LOAD, operands: [1] },  // Load from address 1
      { opcode: OpCode.ADD, operands: [] },    // Add them
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine3.loadProgram(program3)
  const result3 = engine3.execute()
  console.log(`Result: ${result3.value}`)
  console.log(`Cycles: ${result3.cycles}\n`)

  // Example 4: Conditional execution
  console.log('Example 4: Conditional jump')
  const engine4 = new VMEngine()
  const program4: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [1] },   // Condition (true)
      { opcode: OpCode.JMPIF, operands: [3] },  // Jump if true to instruction 3
      { opcode: OpCode.PUSH, operands: [999] }, // Skipped
      { opcode: OpCode.PUSH, operands: [42] },  // Executed
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine4.loadProgram(program4)
  const result4 = engine4.execute()
  console.log(`Result: ${result4.value}`)
  console.log(`Cycles: ${result4.cycles}\n`)

  // Example 5: ML operation - ReLU
  console.log('Example 5: Applying ReLU activation to -5')
  const engine5 = new VMEngine()
  const program5: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [-5] },
      { opcode: OpCode.RELU, operands: [] },
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine5.loadProgram(program5)
  const result5 = engine5.execute()
  console.log(`Result: ${result5.value} (expected: 0)`)
  console.log(`Cycles: ${result5.cycles}\n`)

  // Example 6: Computing factorial 5! = 120
  console.log('Example 6: Computing 5! = 120')
  const engine6 = new VMEngine()
  const program6: VMProgram = {
    instructions: [
      { opcode: OpCode.PUSH, operands: [1] },
      { opcode: OpCode.PUSH, operands: [2] },
      { opcode: OpCode.MUL, operands: [] },
      { opcode: OpCode.PUSH, operands: [3] },
      { opcode: OpCode.MUL, operands: [] },
      { opcode: OpCode.PUSH, operands: [4] },
      { opcode: OpCode.MUL, operands: [] },
      { opcode: OpCode.PUSH, operands: [5] },
      { opcode: OpCode.MUL, operands: [] },
      { opcode: OpCode.HALT, operands: [] },
    ],
    constants: [],
  }
  engine6.loadProgram(program6)
  const result6 = engine6.execute()
  console.log(`Result: ${result6.value}`)
  console.log(`Cycles: ${result6.cycles}\n`)
}

// Run if executed directly
if (require.main === module) {
  main()
}

export { main }
