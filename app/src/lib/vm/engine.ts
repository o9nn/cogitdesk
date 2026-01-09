/**
 * Inferno-inspired VM deployment engine
 */

import { OpCode, Instruction, VMContext, VMProgram, VMResult } from './types'

/**
 * Virtual Machine executor
 */
export class VMEngine {
  private context: VMContext
  private program: VMProgram | null = null
  private maxCycles: number

  constructor(maxCycles: number = 100000) {
    this.maxCycles = maxCycles
    this.context = this.createContext()
  }

  private createContext(): VMContext {
    return {
      stack: [],
      memory: new Map(),
      pc: 0,
      registers: new Array(16).fill(0),
      callStack: [],
    }
  }

  /**
   * Load a program into the VM
   */
  loadProgram(program: VMProgram): void {
    this.program = program
    this.context = this.createContext()
  }

  /**
   * Execute the loaded program
   */
  execute(): VMResult {
    if (!this.program) {
      return {
        success: false,
        error: 'No program loaded',
        cycles: 0,
      }
    }

    let cycles = 0
    const startTime = Date.now()

    try {
      while (
        this.context.pc < this.program.instructions.length &&
        cycles < this.maxCycles
      ) {
        const instruction = this.program.instructions[this.context.pc]
        this.executeInstruction(instruction)
        cycles++

        if (instruction.opcode === OpCode.HALT) {
          break
        }
      }

      if (cycles >= this.maxCycles) {
        return {
          success: false,
          error: 'Maximum cycles exceeded',
          cycles,
        }
      }

      return {
        success: true,
        value: this.context.stack.length > 0 ? this.context.stack.pop() : 0,
        cycles,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        cycles,
      }
    }
  }

  private executeInstruction(instruction: Instruction): void {
    const { opcode, operands } = instruction

    switch (opcode) {
      // Stack operations
      case OpCode.PUSH:
        this.context.stack.push(operands[0])
        this.context.pc++
        break

      case OpCode.POP:
        if (this.context.stack.length === 0) {
          throw new Error('Stack underflow')
        }
        this.context.stack.pop()
        this.context.pc++
        break

      case OpCode.LOAD:
        const addr = operands[0]
        const value = this.context.memory.get(addr) ?? 0
        this.context.stack.push(value)
        this.context.pc++
        break

      case OpCode.STORE:
        if (this.context.stack.length === 0) {
          throw new Error('Stack underflow')
        }
        const storeAddr = operands[0]
        const storeValue = this.context.stack.pop()!
        this.context.memory.set(storeAddr, storeValue)
        this.context.pc++
        break

      // Arithmetic operations
      case OpCode.ADD:
        this.binaryOp((a, b) => a + b)
        break

      case OpCode.SUB:
        this.binaryOp((a, b) => a - b)
        break

      case OpCode.MUL:
        this.binaryOp((a, b) => a * b)
        break

      case OpCode.DIV:
        this.binaryOp((a, b) => {
          if (b === 0) throw new Error('Division by zero')
          return a / b
        })
        break

      // Control flow
      case OpCode.JMP:
        this.context.pc = operands[0]
        break

      case OpCode.JMPIF:
        if (this.context.stack.length === 0) {
          throw new Error('Stack underflow')
        }
        const condition = this.context.stack.pop()!
        if (condition !== 0) {
          this.context.pc = operands[0]
        } else {
          this.context.pc++
        }
        break

      case OpCode.CALL:
        this.context.callStack.push(this.context.pc + 1)
        this.context.pc = operands[0]
        break

      case OpCode.RET:
        if (this.context.callStack.length === 0) {
          throw new Error('Call stack underflow')
        }
        this.context.pc = this.context.callStack.pop()!
        break

      // ML operations (simplified)
      case OpCode.RELU:
        this.unaryOp((x) => Math.max(0, x))
        break

      case OpCode.SIGMOID:
        this.unaryOp((x) => 1 / (1 + Math.exp(-x)))
        break

      case OpCode.HALT:
        // Do nothing, will break in execute loop
        break

      default:
        throw new Error(`Unknown opcode: ${opcode}`)
    }
  }

  private binaryOp(op: (a: number, b: number) => number): void {
    if (this.context.stack.length < 2) {
      throw new Error('Stack underflow')
    }
    const b = this.context.stack.pop()!
    const a = this.context.stack.pop()!
    this.context.stack.push(op(a, b))
    this.context.pc++
  }

  private unaryOp(op: (x: number) => number): void {
    if (this.context.stack.length === 0) {
      throw new Error('Stack underflow')
    }
    const x = this.context.stack.pop()!
    this.context.stack.push(op(x))
    this.context.pc++
  }

  /**
   * Get the current VM context state
   */
  getContext(): Readonly<VMContext> {
    return this.context
  }

  /**
   * Reset the VM to initial state
   */
  reset(): void {
    this.context = this.createContext()
  }
}
