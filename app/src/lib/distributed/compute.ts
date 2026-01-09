/**
 * Distributed compute coordinator
 */

import { TaskScheduler } from './scheduler'
import { Task, TaskStatus, DistributedOpType } from './types'
import { Tensor } from '../ml/tensor'

/**
 * Distributed compute coordinator
 */
export class DistributedCompute {
  private scheduler: TaskScheduler

  constructor() {
    this.scheduler = new TaskScheduler()
  }

  /**
   * Register a worker node
   */
  registerWorker(workerId: string, capacity: number = 1): void {
    this.scheduler.registerWorker(workerId, capacity)
  }

  /**
   * Unregister a worker node
   */
  unregisterWorker(workerId: string): void {
    this.scheduler.unregisterWorker(workerId)
  }

  /**
   * Distributed map operation
   */
  async map<T, R>(
    data: T[],
    fn: (item: T) => R,
    priority: number = 0
  ): Promise<R[]> {
    const tasks = data.map((item, index) => {
      return this.scheduler.submitTask({
        type: DistributedOpType.MAP,
        data: { item, fn: fn.toString(), index },
        priority,
      })
    })

    // Wait for all tasks to complete
    const results = await Promise.all(
      tasks.map((taskId) => this.waitForTask(taskId))
    )

    return results.map((task) => task.result)
  }

  /**
   * Distributed reduce operation
   */
  async reduce<T>(
    data: T[],
    fn: (acc: T, item: T) => T,
    initialValue: T,
    priority: number = 0
  ): Promise<T> {
    // Simple sequential reduce for now
    // In practice, you'd want to implement a parallel tree reduction
    let result = initialValue
    for (const item of data) {
      result = fn(result, item)
    }
    return result
  }

  /**
   * Distributed matrix multiplication
   */
  async distributedMatmul(a: Tensor, b: Tensor): Promise<Tensor> {
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

    // Submit row-wise multiplication tasks
    const rowTasks: string[] = []
    for (let i = 0; i < m; i++) {
      const taskId = this.scheduler.submitTask({
        type: DistributedOpType.MATMUL,
        data: {
          row: i,
          aRow: Array.from(a.data.slice(i * k1, (i + 1) * k1)),
          bData: Array.from(b.data),
          bShape: b.shape,
        },
        priority: 1,
      })
      rowTasks.push(taskId)
    }

    // Wait for all row computations
    const results = await Promise.all(
      rowTasks.map((taskId) => this.waitForTask(taskId))
    )

    // Combine results
    const resultData = new Float32Array(m * n)
    for (const task of results) {
      const { row, rowResult } = task.result
      resultData.set(rowResult, row * n)
    }

    return new Tensor(resultData, [m, n])
  }

  /**
   * Get scheduler statistics
   */
  getStats(): {
    workers: ReturnType<TaskScheduler['getWorkerStats']>
    tasks: {
      total: number
      pending: number
      running: number
      completed: number
      failed: number
    }
  } {
    const workerStats = this.scheduler.getWorkerStats()
    const allTasks = this.scheduler.getAllTasks()

    const taskStats = {
      total: allTasks.length,
      pending: allTasks.filter((t) => t.status === TaskStatus.PENDING).length,
      running: allTasks.filter((t) => t.status === TaskStatus.RUNNING).length,
      completed: allTasks.filter((t) => t.status === TaskStatus.COMPLETED)
        .length,
      failed: allTasks.filter((t) => t.status === TaskStatus.FAILED).length,
    }

    return {
      workers: workerStats,
      tasks: taskStats,
    }
  }

  /**
   * Wait for a task to complete
   */
  private async waitForTask(taskId: string): Promise<Task> {
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        const task = this.scheduler.getTaskStatus(taskId)
        if (!task) {
          clearInterval(checkInterval)
          reject(new Error(`Task ${taskId} not found`))
          return
        }

        if (task.status === TaskStatus.COMPLETED) {
          clearInterval(checkInterval)
          resolve(task)
        } else if (task.status === TaskStatus.FAILED) {
          clearInterval(checkInterval)
          reject(new Error(task.error || 'Task failed'))
        }
      }, 100)

      // Timeout after 60 seconds
      setTimeout(() => {
        clearInterval(checkInterval)
        reject(new Error(`Task ${taskId} timed out`))
      }, 60000)
    })
  }

  /**
   * Complete a task (called by worker)
   */
  completeTask(taskId: string, result: any): void {
    this.scheduler.completeTask(taskId, result)
  }

  /**
   * Fail a task (called by worker)
   */
  failTask(taskId: string, error: string): void {
    this.scheduler.failTask(taskId, error)
  }

  /**
   * Update worker heartbeat
   */
  updateWorkerHeartbeat(workerId: string): void {
    this.scheduler.updateHeartbeat(workerId)
  }

  /**
   * Check worker health
   */
  checkWorkerHealth(timeoutMs: number = 30000): void {
    this.scheduler.checkWorkerHealth(timeoutMs)
  }
}
