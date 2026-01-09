/**
 * Task scheduler for distributed compute
 */

import {
  Task,
  Worker,
  TaskStatus,
  TaskResult,
  Message,
  MessageType,
} from './types'

/**
 * Task scheduler manages task distribution across workers
 */
export class TaskScheduler {
  private tasks: Map<string, Task> = new Map()
  private workers: Map<string, Worker> = new Map()
  private pendingQueue: Task[] = []
  private runningTasks: Map<string, string> = new Map() // taskId -> workerId

  /**
   * Register a new worker
   */
  registerWorker(workerId: string, capacity: number = 1): void {
    this.workers.set(workerId, {
      id: workerId,
      status: 'idle',
      capacity,
      currentLoad: 0,
      lastHeartbeat: new Date(),
    })
  }

  /**
   * Unregister a worker
   */
  unregisterWorker(workerId: string): void {
    const worker = this.workers.get(workerId)
    if (worker) {
      // Reassign tasks from this worker
      for (const [taskId, assignedWorkerId] of this.runningTasks) {
        if (assignedWorkerId === workerId) {
          const task = this.tasks.get(taskId)
          if (task) {
            task.status = TaskStatus.PENDING
            task.startedAt = undefined
            this.pendingQueue.push(task)
          }
          this.runningTasks.delete(taskId)
        }
      }
      this.workers.delete(workerId)
    }
  }

  /**
   * Submit a task to the scheduler
   */
  submitTask(task: Omit<Task, 'id' | 'createdAt' | 'status'>): string {
    const taskId = this.generateTaskId()
    const fullTask: Task = {
      ...task,
      id: taskId,
      status: TaskStatus.PENDING,
      createdAt: new Date(),
    }

    this.tasks.set(taskId, fullTask)
    this.pendingQueue.push(fullTask)
    this.pendingQueue.sort((a, b) => b.priority - a.priority)

    // Try to schedule immediately
    this.scheduleTasks()

    return taskId
  }

  /**
   * Schedule pending tasks to available workers
   */
  private scheduleTasks(): void {
    while (this.pendingQueue.length > 0) {
      const worker = this.findAvailableWorker()
      if (!worker) {
        break
      }

      const task = this.pendingQueue.shift()!
      this.assignTaskToWorker(task, worker)
    }
  }

  /**
   * Find an available worker with capacity
   */
  private findAvailableWorker(): Worker | null {
    let bestWorker: Worker | null = null
    let minLoad = Infinity

    for (const worker of this.workers.values()) {
      if (worker.status !== 'offline' && worker.currentLoad < worker.capacity) {
        if (worker.currentLoad < minLoad) {
          minLoad = worker.currentLoad
          bestWorker = worker
        }
      }
    }

    return bestWorker
  }

  /**
   * Assign a task to a worker
   */
  private assignTaskToWorker(task: Task, worker: Worker): void {
    task.status = TaskStatus.RUNNING
    task.startedAt = new Date()

    worker.currentLoad++
    worker.status = worker.currentLoad >= worker.capacity ? 'busy' : 'idle'

    this.runningTasks.set(task.id, worker.id)
  }

  /**
   * Handle task completion
   */
  completeTask(taskId: string, result: any): void {
    const task = this.tasks.get(taskId)
    if (!task) {
      return
    }

    task.status = TaskStatus.COMPLETED
    task.completedAt = new Date()
    task.result = result

    const workerId = this.runningTasks.get(taskId)
    if (workerId) {
      const worker = this.workers.get(workerId)
      if (worker) {
        worker.currentLoad--
        worker.status = worker.currentLoad >= worker.capacity ? 'busy' : 'idle'
      }
      this.runningTasks.delete(taskId)
    }

    // Schedule more tasks
    this.scheduleTasks()
  }

  /**
   * Handle task failure
   */
  failTask(taskId: string, error: string): void {
    const task = this.tasks.get(taskId)
    if (!task) {
      return
    }

    task.status = TaskStatus.FAILED
    task.completedAt = new Date()
    task.error = error

    const workerId = this.runningTasks.get(taskId)
    if (workerId) {
      const worker = this.workers.get(workerId)
      if (worker) {
        worker.currentLoad--
        worker.status = worker.currentLoad >= worker.capacity ? 'busy' : 'idle'
      }
      this.runningTasks.delete(taskId)
    }

    // Schedule more tasks
    this.scheduleTasks()
  }

  /**
   * Get task status
   */
  getTaskStatus(taskId: string): Task | undefined {
    return this.tasks.get(taskId)
  }

  /**
   * Get all tasks
   */
  getAllTasks(): Task[] {
    return Array.from(this.tasks.values())
  }

  /**
   * Get worker statistics
   */
  getWorkerStats(): {
    total: number
    idle: number
    busy: number
    offline: number
  } {
    const stats = { total: 0, idle: 0, busy: 0, offline: 0 }

    for (const worker of this.workers.values()) {
      stats.total++
      if (worker.status === 'idle') stats.idle++
      else if (worker.status === 'busy') stats.busy++
      else if (worker.status === 'offline') stats.offline++
    }

    return stats
  }

  /**
   * Update worker heartbeat
   */
  updateHeartbeat(workerId: string): void {
    const worker = this.workers.get(workerId)
    if (worker) {
      worker.lastHeartbeat = new Date()
    }
  }

  /**
   * Check for stale workers and mark them offline
   */
  checkWorkerHealth(timeoutMs: number = 30000): void {
    const now = new Date()
    for (const worker of this.workers.values()) {
      const timeSinceHeartbeat =
        now.getTime() - worker.lastHeartbeat.getTime()
      if (timeSinceHeartbeat > timeoutMs && worker.status !== 'offline') {
        worker.status = 'offline'
        // Reassign tasks from this worker
        for (const [taskId, workerId] of this.runningTasks) {
          if (workerId === worker.id) {
            const task = this.tasks.get(taskId)
            if (task) {
              task.status = TaskStatus.PENDING
              task.startedAt = undefined
              this.pendingQueue.push(task)
            }
            this.runningTasks.delete(taskId)
          }
        }
      }
    }
  }

  private generateTaskId(): string {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }
}
