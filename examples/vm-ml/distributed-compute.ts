/**
 * Example: Distributed Computing
 * 
 * This example demonstrates how to use the distributed compute system
 * for parallel task execution.
 */

import { DistributedCompute, TaskScheduler } from '../../app/src/lib/distributed'
import { Tensor } from '../../app/src/lib/ml'

function main() {
  console.log('=== Distributed Computing Example ===\n')

  // Example 1: Basic task scheduling
  console.log('Example 1: Basic task scheduling')
  const scheduler = new TaskScheduler()
  
  // Register workers
  scheduler.registerWorker('worker1', 2)
  scheduler.registerWorker('worker2', 2)
  
  console.log('Registered 2 workers with capacity 2 each')
  
  // Submit tasks
  const taskIds = []
  for (let i = 0; i < 5; i++) {
    const taskId = scheduler.submitTask({
      type: 'compute',
      data: { value: i * 10 },
      priority: i % 2, // Alternate priorities
    })
    taskIds.push(taskId)
  }
  
  console.log(`Submitted ${taskIds.length} tasks`)
  
  // Check statistics
  const stats1 = scheduler.getWorkerStats()
  console.log('Worker stats:', stats1)
  
  const allTasks = scheduler.getAllTasks()
  console.log('Task statuses:')
  allTasks.forEach(task => {
    console.log(`  Task ${task.id.substring(0, 15)}...: ${task.status}`)
  })
  console.log()

  // Example 2: Task completion and load balancing
  console.log('Example 2: Task completion and load balancing')
  const scheduler2 = new TaskScheduler()
  scheduler2.registerWorker('fast-worker', 3)
  scheduler2.registerWorker('slow-worker', 1)
  
  // Submit multiple tasks
  const tasks = []
  for (let i = 0; i < 6; i++) {
    const id = scheduler2.submitTask({
      type: 'process',
      data: { item: i },
      priority: 1,
    })
    tasks.push(id)
  }
  
  console.log('Submitted 6 tasks to 2 workers')
  const stats2 = scheduler2.getWorkerStats()
  console.log('Initial worker stats:', stats2)
  
  // Complete some tasks
  scheduler2.completeTask(tasks[0], { result: 'done' })
  scheduler2.completeTask(tasks[1], { result: 'done' })
  
  console.log('Completed 2 tasks')
  const stats3 = scheduler2.getWorkerStats()
  console.log('Updated worker stats:', stats3)
  console.log()

  // Example 3: Worker health monitoring
  console.log('Example 3: Worker health monitoring')
  const scheduler3 = new TaskScheduler()
  scheduler3.registerWorker('worker-a', 1)
  scheduler3.registerWorker('worker-b', 1)
  
  console.log('Registered 2 workers')
  
  // Update heartbeat for one worker
  scheduler3.updateHeartbeat('worker-a')
  console.log('Updated heartbeat for worker-a')
  
  // Check health (all workers should be healthy)
  scheduler3.checkWorkerHealth(30000) // 30 second timeout
  const stats4 = scheduler3.getWorkerStats()
  console.log('Worker health check:', stats4)
  console.log()

  // Example 4: Priority-based scheduling
  console.log('Example 4: Priority-based scheduling')
  const scheduler4 = new TaskScheduler()
  
  // Submit high and low priority tasks before registering workers
  const lowPriority = scheduler4.submitTask({
    type: 'low',
    data: {},
    priority: 1,
  })
  
  const highPriority = scheduler4.submitTask({
    type: 'high',
    data: {},
    priority: 10,
  })
  
  console.log('Submitted low priority (1) and high priority (10) tasks')
  
  // Now register a worker
  scheduler4.registerWorker('priority-worker', 1)
  
  const lowTask = scheduler4.getTaskStatus(lowPriority)
  const highTask = scheduler4.getTaskStatus(highPriority)
  
  console.log(`Low priority task status: ${lowTask?.status}`)
  console.log(`High priority task status: ${highTask?.status}`)
  console.log('High priority task should be running first\n')

  // Example 5: Distributed Compute Coordinator
  console.log('Example 5: Using DistributedCompute coordinator')
  const dc = new DistributedCompute()
  
  // Register workers
  dc.registerWorker('compute1', 2)
  dc.registerWorker('compute2', 2)
  dc.registerWorker('compute3', 2)
  
  console.log('Registered 3 workers with capacity 2 each')
  
  // Get statistics
  const dcStats = dc.getStats()
  console.log('Coordinator stats:', {
    totalWorkers: dcStats.workers.total,
    idleWorkers: dcStats.workers.idle,
    totalTasks: dcStats.tasks.total,
  })
  console.log()

  // Example 6: Simulated distributed map operation
  console.log('Example 6: Simulated distributed map operation')
  console.log('Note: This is a simulation as actual async execution requires worker implementation')
  
  const data = [1, 2, 3, 4, 5, 6, 7, 8]
  console.log('Input data:', data)
  console.log('Operation: multiply each by 2')
  
  // In practice, you would use: await dc.map(data, x => x * 2)
  // For this example, we show the structure
  const expectedResult = data.map(x => x * 2)
  console.log('Expected result:', expectedResult)
  console.log()

  // Example 7: Task failure handling
  console.log('Example 7: Task failure handling')
  const scheduler7 = new TaskScheduler()
  scheduler7.registerWorker('reliable-worker', 1)
  
  const successTask = scheduler7.submitTask({
    type: 'stable',
    data: {},
    priority: 1,
  })
  
  const failTask = scheduler7.submitTask({
    type: 'unstable',
    data: {},
    priority: 1,
  })
  
  // Simulate task completion and failure
  scheduler7.completeTask(successTask, { result: 'success' })
  scheduler7.failTask(failTask, 'Simulated error: timeout')
  
  const successStatus = scheduler7.getTaskStatus(successTask)
  const failStatus = scheduler7.getTaskStatus(failTask)
  
  console.log(`Success task status: ${successStatus?.status}`)
  console.log(`Failed task status: ${failStatus?.status}`)
  console.log(`Failed task error: ${failStatus?.error}`)
  console.log()

  // Example 8: Distributed matrix operations concept
  console.log('Example 8: Distributed matrix operations (concept)')
  const matrix1 = Tensor.randn([50, 30])
  const matrix2 = Tensor.randn([30, 20])
  
  console.log(`Matrix 1 shape: [${matrix1.shape}]`)
  console.log(`Matrix 2 shape: [${matrix2.shape}]`)
  console.log('In distributed mode, rows would be processed across workers')
  console.log('Expected result shape: [50, 20]')
  console.log()

  console.log('=== Example Complete ===')
}

// Run if executed directly
if (require.main === module) {
  main()
}

export { main }
