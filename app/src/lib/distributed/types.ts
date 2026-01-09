/**
 * Distributed compute task types and interfaces
 */

import { Tensor } from '../ml/tensor'

/**
 * Task status enumeration
 */
export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Task definition
 */
export interface Task {
  id: string
  type: string
  data: any
  priority: number
  status: TaskStatus
  createdAt: Date
  startedAt?: Date
  completedAt?: Date
  result?: any
  error?: string
}

/**
 * Worker information
 */
export interface Worker {
  id: string
  status: 'idle' | 'busy' | 'offline'
  capacity: number
  currentLoad: number
  lastHeartbeat: Date
}

/**
 * Task result
 */
export interface TaskResult {
  taskId: string
  success: boolean
  result?: any
  error?: string
  executionTime: number
}

/**
 * Distributed operation types
 */
export enum DistributedOpType {
  MAP = 'map',
  REDUCE = 'reduce',
  MATMUL = 'matmul',
  TRAIN_BATCH = 'train_batch',
  INFERENCE = 'inference',
}

/**
 * Message types for worker communication
 */
export enum MessageType {
  TASK_ASSIGN = 'task_assign',
  TASK_COMPLETE = 'task_complete',
  TASK_FAILED = 'task_failed',
  HEARTBEAT = 'heartbeat',
  WORKER_REGISTER = 'worker_register',
  WORKER_UNREGISTER = 'worker_unregister',
}

/**
 * Message structure for worker communication
 */
export interface Message {
  type: MessageType
  workerId?: string
  taskId?: string
  data?: any
  timestamp: Date
}
