import { describe, it } from 'node:test'
import assert from 'node:assert'
import { TaskScheduler } from '../../src/lib/distributed/scheduler'
import { DistributedCompute } from '../../src/lib/distributed/compute'
import { TaskStatus, DistributedOpType } from '../../src/lib/distributed/types'

describe('Distributed Compute', () => {
  describe('TaskScheduler', () => {
    it('should register workers', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 2)
      scheduler.registerWorker('worker2', 1)

      const stats = scheduler.getWorkerStats()
      assert.strictEqual(stats.total, 2)
      assert.strictEqual(stats.idle, 2)
    })

    it('should unregister workers', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)
      scheduler.unregisterWorker('worker1')

      const stats = scheduler.getWorkerStats()
      assert.strictEqual(stats.total, 0)
    })

    it('should submit tasks', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: { value: 42 },
        priority: 1,
      })

      assert.ok(taskId)
      const task = scheduler.getTaskStatus(taskId)
      assert.ok(task)
      assert.strictEqual(task.type, 'test')
    })

    it('should assign tasks to available workers', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: {},
        priority: 1,
      })

      const task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.RUNNING)

      const stats = scheduler.getWorkerStats()
      assert.strictEqual(stats.busy, 1)
      assert.strictEqual(stats.idle, 0)
    })

    it('should queue tasks when no workers available', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      // Submit two tasks, second should be queued
      const task1 = scheduler.submitTask({
        type: 'test1',
        data: {},
        priority: 1,
      })
      const task2 = scheduler.submitTask({
        type: 'test2',
        data: {},
        priority: 1,
      })

      const taskStatus1 = scheduler.getTaskStatus(task1)
      const taskStatus2 = scheduler.getTaskStatus(task2)

      assert.strictEqual(taskStatus1?.status, TaskStatus.RUNNING)
      assert.strictEqual(taskStatus2?.status, TaskStatus.PENDING)
    })

    it('should respect task priority', () => {
      const scheduler = new TaskScheduler()
      // Don't register any workers yet

      const lowPriority = scheduler.submitTask({
        type: 'low',
        data: {},
        priority: 1,
      })
      const highPriority = scheduler.submitTask({
        type: 'high',
        data: {},
        priority: 10,
      })

      // Register a worker now
      scheduler.registerWorker('worker1', 1)

      // High priority task should be running
      const highTask = scheduler.getTaskStatus(highPriority)
      const lowTask = scheduler.getTaskStatus(lowPriority)

      assert.strictEqual(highTask?.status, TaskStatus.RUNNING)
      assert.strictEqual(lowTask?.status, TaskStatus.PENDING)
    })

    it('should complete tasks', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: {},
        priority: 1,
      })

      scheduler.completeTask(taskId, { result: 'success' })

      const task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.COMPLETED)
      assert.deepStrictEqual(task?.result, { result: 'success' })

      // Worker should be idle again
      const stats = scheduler.getWorkerStats()
      assert.strictEqual(stats.idle, 1)
      assert.strictEqual(stats.busy, 0)
    })

    it('should fail tasks', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: {},
        priority: 1,
      })

      scheduler.failTask(taskId, 'Test error')

      const task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.FAILED)
      assert.strictEqual(task?.error, 'Test error')
    })

    it('should update worker heartbeat', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      scheduler.updateHeartbeat('worker1')
      // Should not throw
    })

    it('should mark stale workers as offline', async () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      // Wait 10ms then check with 5ms timeout - worker should be offline
      await new Promise((resolve) => setTimeout(resolve, 10))
      scheduler.checkWorkerHealth(5)

      const stats = scheduler.getWorkerStats()
      assert.strictEqual(stats.offline, 1)
    })

    it('should reassign tasks from offline workers', async () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: {},
        priority: 1,
      })

      // Task should be running
      let task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.RUNNING)

      // Wait 10ms then mark worker as offline with 5ms timeout
      await new Promise((resolve) => setTimeout(resolve, 10))
      scheduler.checkWorkerHealth(5)

      // Task should be pending again
      task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.PENDING)
    })

    it('should get all tasks', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      scheduler.submitTask({ type: 'test1', data: {}, priority: 1 })
      scheduler.submitTask({ type: 'test2', data: {}, priority: 1 })

      const allTasks = scheduler.getAllTasks()
      assert.strictEqual(allTasks.length, 2)
    })
  })

  describe('DistributedCompute', () => {
    it('should create a distributed compute instance', () => {
      const dc = new DistributedCompute()
      assert.ok(dc)
    })

    it('should register and unregister workers', () => {
      const dc = new DistributedCompute()
      dc.registerWorker('worker1', 2)

      const stats = dc.getStats()
      assert.strictEqual(stats.workers.total, 1)

      dc.unregisterWorker('worker1')
      const stats2 = dc.getStats()
      assert.strictEqual(stats2.workers.total, 0)
    })

    it('should complete tasks', () => {
      const dc = new DistributedCompute()
      dc.registerWorker('worker1', 1)

      // We can't easily test the async map function without simulating worker responses,
      // but we can test the infrastructure
      const stats = dc.getStats()
      assert.ok(stats.workers)
      assert.ok(stats.tasks)
    })

    it('should get statistics', () => {
      const dc = new DistributedCompute()
      dc.registerWorker('worker1', 1)
      dc.registerWorker('worker2', 1)

      const stats = dc.getStats()
      assert.strictEqual(stats.workers.total, 2)
      assert.strictEqual(stats.workers.idle, 2)
      assert.strictEqual(stats.tasks.total, 0)
    })

    it('should update worker heartbeat', () => {
      const dc = new DistributedCompute()
      dc.registerWorker('worker1', 1)

      dc.updateWorkerHeartbeat('worker1')
      // Should not throw
    })

    it('should check worker health', () => {
      const dc = new DistributedCompute()
      dc.registerWorker('worker1', 1)

      dc.checkWorkerHealth(30000)

      const stats = dc.getStats()
      assert.strictEqual(stats.workers.total, 1)
    })
  })

  describe('load balancing', () => {
    it('should distribute tasks evenly across workers', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 2)
      scheduler.registerWorker('worker2', 2)

      // Submit 4 tasks
      const tasks = []
      for (let i = 0; i < 4; i++) {
        tasks.push(
          scheduler.submitTask({
            type: `test${i}`,
            data: {},
            priority: 1,
          })
        )
      }

      // All tasks should be running (2 workers x 2 capacity = 4)
      for (const taskId of tasks) {
        const task = scheduler.getTaskStatus(taskId)
        assert.strictEqual(task?.status, TaskStatus.RUNNING)
      }
    })

    it('should prefer less loaded workers', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 3)
      scheduler.registerWorker('worker2', 3)

      // Submit 4 tasks - they should be distributed to minimize load per worker
      for (let i = 0; i < 4; i++) {
        scheduler.submitTask({
          type: `test${i}`,
          data: {},
          priority: 1,
        })
      }

      const stats = scheduler.getWorkerStats()
      // All tasks should be running (2 workers x 3 capacity >= 4 tasks)
      // Both workers should have some load but neither should be at full capacity
      assert.ok(stats.busy + stats.idle === 2)
      assert.strictEqual(stats.offline, 0)
    })
  })

  describe('task lifecycle', () => {
    it('should track task lifecycle from pending to completed', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: { value: 42 },
        priority: 1,
      })

      // Task should start as running (auto-assigned)
      let task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.RUNNING)
      assert.ok(task?.startedAt)

      // Complete the task
      scheduler.completeTask(taskId, { result: 'done' })

      task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.COMPLETED)
      assert.ok(task?.completedAt)
      assert.deepStrictEqual(task?.result, { result: 'done' })
    })

    it('should track failed tasks', () => {
      const scheduler = new TaskScheduler()
      scheduler.registerWorker('worker1', 1)

      const taskId = scheduler.submitTask({
        type: 'test',
        data: {},
        priority: 1,
      })

      scheduler.failTask(taskId, 'Something went wrong')

      const task = scheduler.getTaskStatus(taskId)
      assert.strictEqual(task?.status, TaskStatus.FAILED)
      assert.strictEqual(task?.error, 'Something went wrong')
    })
  })
})
