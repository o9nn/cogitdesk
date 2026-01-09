/**
 * Core types for the Inferno-inspired VM deployment engine
 */

/**
 * VM instruction opcodes
 */
export enum OpCode {
  // Stack operations
  LOAD = 0x00,
  STORE = 0x01,
  PUSH = 0x02,
  POP = 0x03,

  // Arithmetic operations
  ADD = 0x10,
  SUB = 0x11,
  MUL = 0x12,
  DIV = 0x13,

  // Control flow
  JMP = 0x20,
  JMPIF = 0x21,
  CALL = 0x22,
  RET = 0x23,

  // ML operations
  MATMUL = 0x30,
  CONV2D = 0x31,
  POOL = 0x32,
  RELU = 0x33,
  SIGMOID = 0x34,
  SOFTMAX = 0x35,

  // Distributed operations
  DISPATCH = 0x40,
  GATHER = 0x41,
  REDUCE = 0x42,

  // System operations
  HALT = 0xFF,
}

/**
 * VM instruction structure
 */
export interface Instruction {
  opcode: OpCode
  operands: number[]
}

/**
 * VM execution context
 */
export interface VMContext {
  stack: number[]
  memory: Map<number, number>
  pc: number // Program counter
  registers: number[]
  callStack: number[]
}

/**
 * VM program representation
 */
export interface VMProgram {
  instructions: Instruction[]
  constants: number[]
  metadata?: {
    name?: string
    version?: string
  }
}

/**
 * VM execution result
 */
export interface VMResult {
  success: boolean
  value?: number | number[]
  error?: string
  cycles: number
}
