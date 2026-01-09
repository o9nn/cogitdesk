import { describe, it } from 'node:test'
import assert from 'node:assert'
import { VMEngine, OpCode, VMProgram } from '../../src/lib/vm'

describe('VMEngine', () => {
  describe('basic operations', () => {
    it('should execute a simple addition program', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [5] },
          { opcode: OpCode.PUSH, operands: [3] },
          { opcode: OpCode.ADD, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 8)
    })

    it('should execute a simple subtraction program', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [10] },
          { opcode: OpCode.PUSH, operands: [3] },
          { opcode: OpCode.SUB, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 7)
    })

    it('should execute a simple multiplication program', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [4] },
          { opcode: OpCode.PUSH, operands: [5] },
          { opcode: OpCode.MUL, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 20)
    })

    it('should execute a simple division program', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [20] },
          { opcode: OpCode.PUSH, operands: [4] },
          { opcode: OpCode.DIV, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 5)
    })
  })

  describe('memory operations', () => {
    it('should store and load values from memory', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [42] },
          { opcode: OpCode.STORE, operands: [100] },
          { opcode: OpCode.LOAD, operands: [100] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 42)
    })
  })

  describe('control flow', () => {
    it('should execute unconditional jump', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [1] },
          { opcode: OpCode.JMP, operands: [3] },
          { opcode: OpCode.PUSH, operands: [99] }, // Should be skipped
          { opcode: OpCode.PUSH, operands: [2] },
          { opcode: OpCode.ADD, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 3) // 1 + 2 = 3, not 1 + 99
    })

    it('should execute conditional jump when condition is true', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [1] }, // Condition (non-zero)
          { opcode: OpCode.JMPIF, operands: [3] },
          { opcode: OpCode.PUSH, operands: [99] }, // Should be skipped
          { opcode: OpCode.PUSH, operands: [42] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 42)
    })

    it('should not jump when condition is false', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [0] }, // Condition (zero)
          { opcode: OpCode.JMPIF, operands: [3] },
          { opcode: OpCode.PUSH, operands: [99] }, // Should execute
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 99)
    })
  })

  describe('ML operations', () => {
    it('should apply ReLU activation', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [-5] },
          { opcode: OpCode.RELU, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 0) // ReLU(-5) = 0
    })

    it('should apply Sigmoid activation', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [0] },
          { opcode: OpCode.SIGMOID, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 0.5) // Sigmoid(0) = 0.5
    })
  })

  describe('error handling', () => {
    it('should fail when no program is loaded', () => {
      const engine = new VMEngine()
      const result = engine.execute()

      assert.strictEqual(result.success, false)
      assert.strictEqual(result.error, 'No program loaded')
    })

    it('should fail on stack underflow', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.ADD, operands: [] }, // No values on stack
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, false)
      assert.ok(result.error?.includes('Stack underflow'))
    })

    it('should fail on division by zero', () => {
      const engine = new VMEngine()

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [10] },
          { opcode: OpCode.PUSH, operands: [0] },
          { opcode: OpCode.DIV, operands: [] },
          { opcode: OpCode.HALT, operands: [] },
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, false)
      assert.ok(result.error?.includes('Division by zero'))
    })

    it('should enforce maximum cycles', () => {
      const engine = new VMEngine(10) // Only 10 cycles allowed

      const program: VMProgram = {
        instructions: [
          { opcode: OpCode.PUSH, operands: [1] },
          { opcode: OpCode.JMP, operands: [0] }, // Infinite loop
        ],
        constants: [],
      }

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, false)
      assert.ok(result.error?.includes('Maximum cycles exceeded'))
    })
  })

  describe('complex programs', () => {
    it('should compute factorial 5! = 120', () => {
      const engine = new VMEngine()

      // Compute 5! = 120 by pushing and multiplying 1*2*3*4*5
      const program: VMProgram = {
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

      engine.loadProgram(program)
      const result = engine.execute()

      assert.strictEqual(result.success, true)
      assert.strictEqual(result.value, 120)
    })
  })
})
