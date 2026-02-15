ger.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)
26/02/13 21:43:56 WARN TaskSetManager: Lost task 0.0 in stage 0.0 (TID 0) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

26/02/13 21:43:56 ERROR TaskSetManager: Task 0 in stage 0.0 failed 1 times; aborting job
2026-02-13 21:43:56,317 - src.comparison_pipeline - WARNING - Spark/JVM 오류, Python 폴백 사용: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 0.0 failed 1 times, most recent failure: Lost task 0.0 in stage 0.0 (TID 0) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

Driver stacktrace:
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$3(DAGScheduler.scala:3122)
        at scala.Option.getOrElse(Option.scala:201)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:3122)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:3114)
        at scala.collection.immutable.List.foreach(List.scala:323)
        at org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:3114)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1$adapted(DAGScheduler.scala:1303)
        at scala.Option.foreach(Option.scala:437)
        at org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:3397)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3328)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3317)
        at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:50)
        at org.apache.spark.scheduler.DAGScheduler.runJob(DAGScheduler.scala:1017)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2496)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2517)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2536)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2561)
        at org.apache.spark.rdd.RDD.$anonfun$collect$1(RDD.scala:1057)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:151)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:112)
        at org.apache.spark.rdd.RDD.withScope(RDD.scala:417)
        at org.apache.spark.rdd.RDD.collect(RDD.scala:1056)
        at org.apache.spark.api.python.PythonRDD$.collectAndServe(PythonRDD.scala:205)
        at org.apache.spark.api.python.PythonRDD.collectAndServe(PythonRDD.scala)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:75)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:52)
        at java.base/java.lang.reflect.Method.invoke(Method.java:580)
        at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
        at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:374)
        at py4j.Gateway.invoke(Gateway.java:282)
        at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
        at py4j.commands.CallCommand.execute(CallCommand.java:79)
        at py4j.ClientServerConnection.waitForCommands(ClientServerConnection.java:184)
        at py4j.ClientServerConnection.run(ClientServerConnection.java:108)
        at java.base/java.lang.Thread.run(Thread.java:1583)
Caused by: org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        ... 1 more

2026-02-13 21:43:56,400 - src.comparison_pipeline - WARNING - SparkSession 초기화 완료 (Py4J 오류 후 재생성용)
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
2026-02-13 21:43:57,427 - src.comparison - INFO - 표본(단일 음식점) restaurant_id=4 service=0.6200, price=0.5000 | lift service=3%, price=175%
2026-02-13 21:43:57,427 - src.comparison_pipeline - INFO - 표본 리뷰수 n=62 >= 50이므로 톤 '좋은 편' 사용
2026-02-13 21:43:57,427 - src.comparison - INFO - comparison 진행: restaurant_id=5, restaurant_name=(이름없음)
2026-02-13 21:43:57,427 - src.comparison - INFO - 전체 평균 ① 파일 시도 (Spark 직접 읽기): path=data/test_data_sample.json
26/02/13 21:43:57 WARN BlockManager: Putting block rdd_9_0 failed due to exception org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.
.
26/02/13 21:43:57 WARN BlockManager: Block rdd_9_0 could not be removed as it was not found on disk or in memory
26/02/13 21:43:57 ERROR Executor: Exception in task 0.0 in stage 1.0 (TID 1)
org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)
26/02/13 21:43:57 WARN TaskSetManager: Lost task 0.0 in stage 1.0 (TID 1) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

26/02/13 21:43:57 ERROR TaskSetManager: Task 0 in stage 1.0 failed 1 times; aborting job
2026-02-13 21:43:57,957 - src.comparison_pipeline - WARNING - Spark/JVM 오류, Python 폴백 시도: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 1.0 failed 1 times, most recent failure: Lost task 0.0 in stage 1.0 (TID 1) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

Driver stacktrace:
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$3(DAGScheduler.scala:3122)
        at scala.Option.getOrElse(Option.scala:201)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:3122)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:3114)
        at scala.collection.immutable.List.foreach(List.scala:323)
        at org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:3114)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1$adapted(DAGScheduler.scala:1303)
        at scala.Option.foreach(Option.scala:437)
        at org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:3397)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3328)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3317)
        at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:50)
        at org.apache.spark.scheduler.DAGScheduler.runJob(DAGScheduler.scala:1017)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2496)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2517)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2536)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2561)
        at org.apache.spark.rdd.RDD.$anonfun$collect$1(RDD.scala:1057)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:151)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:112)
        at org.apache.spark.rdd.RDD.withScope(RDD.scala:417)
        at org.apache.spark.rdd.RDD.collect(RDD.scala:1056)
        at org.apache.spark.api.python.PythonRDD$.collectAndServe(PythonRDD.scala:205)
        at org.apache.spark.api.python.PythonRDD.collectAndServe(PythonRDD.scala)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:75)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:52)
        at java.base/java.lang.reflect.Method.invoke(Method.java:580)
        at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
        at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:374)
        at py4j.Gateway.invoke(Gateway.java:282)
        at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
        at py4j.commands.CallCommand.execute(CallCommand.java:79)
        at py4j.ClientServerConnection.waitForCommands(ClientServerConnection.java:184)
        at py4j.ClientServerConnection.run(ClientServerConnection.java:108)
        at java.base/java.lang.Thread.run(Thread.java:1583)
Caused by: org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        ... 1 more

2026-02-13 21:43:58,461 - src.comparison_pipeline - WARNING - SparkSession 초기화 완료 (Py4J 오류 후 재생성용)
2026-02-13 21:43:58,464 - src.comparison_pipeline - INFO - aspect_data에서 리뷰 142건 로드: /Users/js/tasteam-new-async/data/test_data_sample.json
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
2026-02-13 21:43:59,591 - src.comparison - INFO - 전체 평균 ① 파일 사용 (Spark 직접 읽기): path=data/test_data_sample.json → service=0.6000, price=0.1818
26/02/13 21:44:00 WARN BlockManager: Putting block rdd_1_0 failed due to exception org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.
.
26/02/13 21:44:00 WARN BlockManager: Block rdd_1_0 could not be removed as it was not found on disk or in memory
26/02/13 21:44:00 ERROR Executor: Exception in task 0.0 in stage 0.0 (TID 0)
org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)
26/02/13 21:44:00 WARN TaskSetManager: Lost task 0.0 in stage 0.0 (TID 0) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

26/02/13 21:44:00 ERROR TaskSetManager: Task 0 in stage 0.0 failed 1 times; aborting job
2026-02-13 21:44:00,165 - src.comparison_pipeline - WARNING - Spark/JVM 오류, Python 폴백 사용: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 0.0 failed 1 times, most recent failure: Lost task 0.0 in stage 0.0 (TID 0) (172.20.5.186 executor driver): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        at java.base/java.lang.Thread.run(Thread.java:1583)

Driver stacktrace:
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$3(DAGScheduler.scala:3122)
        at scala.Option.getOrElse(Option.scala:201)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2(DAGScheduler.scala:3122)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$abortStage$2$adapted(DAGScheduler.scala:3114)
        at scala.collection.immutable.List.foreach(List.scala:323)
        at org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:3114)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGScheduler.$anonfun$handleTaskSetFailed$1$adapted(DAGScheduler.scala:1303)
        at scala.Option.foreach(Option.scala:437)
        at org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:1303)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:3397)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3328)
        at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:3317)
        at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:50)
        at org.apache.spark.scheduler.DAGScheduler.runJob(DAGScheduler.scala:1017)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2496)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2517)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2536)
        at org.apache.spark.SparkContext.runJob(SparkContext.scala:2561)
        at org.apache.spark.rdd.RDD.$anonfun$collect$1(RDD.scala:1057)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:151)
        at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:112)
        at org.apache.spark.rdd.RDD.withScope(RDD.scala:417)
        at org.apache.spark.rdd.RDD.collect(RDD.scala:1056)
        at org.apache.spark.api.python.PythonRDD$.collectAndServe(PythonRDD.scala:205)
        at org.apache.spark.api.python.PythonRDD.collectAndServe(PythonRDD.scala)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:75)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:52)
        at java.base/java.lang.reflect.Method.invoke(Method.java:580)
        at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
        at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:374)
        at py4j.Gateway.invoke(Gateway.java:282)
        at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
        at py4j.commands.CallCommand.execute(CallCommand.java:79)
        at py4j.ClientServerConnection.waitForCommands(ClientServerConnection.java:184)
        at py4j.ClientServerConnection.run(ClientServerConnection.java:108)
        at java.base/java.lang.Thread.run(Thread.java:1583)
Caused by: org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker.py", line 3305, in main
    check_python_version(infile)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/opt/apache-spark/libexec/python/lib/pyspark.zip/pyspark/worker_util.py", line 77, in check_python_version
    raise PySparkRuntimeError(
    ...<5 lines>...
    )
pyspark.errors.exceptions.base.PySparkRuntimeError: [PYTHON_VERSION_MISMATCH] Python in worker has different version: 3.13 than that in driver: 3.11, PySpark cannot run with different minor versions.
Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.

        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:645)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1029)
        at org.apache.spark.api.python.PythonRunner$$anon$3.read(PythonRunner.scala:1014)
        at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:596)
        at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
        at org.apache.spark.storage.memory.MemoryStore.putIterator(MemoryStore.scala:230)
        at org.apache.spark.storage.memory.MemoryStore.putIteratorAsBytes(MemoryStore.scala:368)
        at org.apache.spark.storage.BlockManager.$anonfun$doPutIterator$1(BlockManager.scala:1676)
        at org.apache.spark.storage.BlockManager.org$apache$spark$storage$BlockManager$$doPut(BlockManager.scala:1585)
        at org.apache.spark.storage.BlockManager.doPutIterator(BlockManager.scala:1650)
        at org.apache.spark.storage.BlockManager.getOrElseUpdate(BlockManager.scala:1429)
        at org.apache.spark.storage.BlockManager.getOrElseUpdateRDDBlock(BlockManager.scala:1383)
        at org.apache.spark.rdd.RDD.getOrCompute(RDD.scala:386)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:336)
        at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:72)
        at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:374)
        at org.apache.spark.rdd.RDD.iterator(RDD.scala:338)
        at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:93)
        at org.apache.spark.TaskContext.runTaskWithListeners(TaskContext.scala:180)
        at org.apache.spark.scheduler.Task.run(Task.scala:147)
        at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$5(Executor.scala:716)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally(SparkErrorUtils.scala:86)
        at org.apache.spark.util.SparkErrorUtils.tryWithSafeFinally$(SparkErrorUtils.scala:83)
        at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:97)
        at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:719)
        at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
        at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
        ... 1 more

2026-02-13 21:44:00,222 - src.comparison_pipeline - WARNING - SparkSession 초기화 완료 (Py4J 오류 후 재생성용)
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
2026-02-13 21:44:01,296 - src.comparison - INFO - 표본(단일 음식점) restaurant_id=5 service=0.6700, price=0.0700 | lift service=12%, price=-61%
2026-02-13 21:44:01,296 - src.comparison_pipeline - INFO - 표본 리뷰수 20 <= n=20 < 50이므로 톤 '상대적으로 좋은 편' 사용
2026-02-13 21:44:01,673 - openai._base_client - INFO - Retrying request to /chat/completions in 0.425216 seconds
2026-02-13 21:44:01,674 - openai._base_client - INFO - Retrying request to /chat/completions in 0.441007 seconds
2026-02-13 21:44:01,674 - openai._base_client - INFO - Retrying request to /chat/completions in 0.488214 seconds
2026-02-13 21:44:02,353 - openai._base_client - INFO - Retrying request to /chat/completions in 0.878098 seconds
2026-02-13 21:44:02,371 - openai._base_client - INFO - Retrying request to /chat/completions in 0.960420 seconds
2026-02-13 21:44:02,408 - openai._base_client - INFO - Retrying request to /chat/completions in 0.846812 seconds
2026-02-13 21:44:03,487 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 1/3): Connection error.
2026-02-13 21:44:03,513 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 1/3): Connection error.
2026-02-13 21:44:03,588 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 1/3): Connection error.
2026-02-13 21:44:04,745 - openai._base_client - INFO - Retrying request to /chat/completions in 0.401571 seconds
2026-02-13 21:44:04,768 - openai._base_client - INFO - Retrying request to /chat/completions in 0.418967 seconds
2026-02-13 21:44:04,841 - openai._base_client - INFO - Retrying request to /chat/completions in 0.416891 seconds
2026-02-13 21:44:05,404 - openai._base_client - INFO - Retrying request to /chat/completions in 0.852724 seconds
2026-02-13 21:44:05,442 - openai._base_client - INFO - Retrying request to /chat/completions in 0.777316 seconds
2026-02-13 21:44:05,510 - openai._base_client - INFO - Retrying request to /chat/completions in 0.892397 seconds
2026-02-13 21:44:06,476 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 2/3): Connection error.
2026-02-13 21:44:06,510 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 2/3): Connection error.
2026-02-13 21:44:06,656 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 2/3): Connection error.
2026-02-13 21:44:08,733 - openai._base_client - INFO - Retrying request to /chat/completions in 0.478444 seconds
2026-02-13 21:44:08,761 - openai._base_client - INFO - Retrying request to /chat/completions in 0.463405 seconds
2026-02-13 21:44:08,912 - openai._base_client - INFO - Retrying request to /chat/completions in 0.458070 seconds
2026-02-13 21:44:09,464 - openai._base_client - INFO - Retrying request to /chat/completions in 0.886948 seconds
2026-02-13 21:44:09,478 - openai._base_client - INFO - Retrying request to /chat/completions in 0.935824 seconds
2026-02-13 21:44:09,624 - openai._base_client - INFO - Retrying request to /chat/completions in 0.976842 seconds
2026-02-13 21:44:10,605 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 3/3): Connection error.
2026-02-13 21:44:10,605 - src.llm_utils - WARNING - 비교 해석 LLM 실패: Connection error.
2026-02-13 21:44:10,668 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 3/3): Connection error.
2026-02-13 21:44:10,668 - src.llm_utils - WARNING - 비교 해석 LLM 실패: Connection error.
2026-02-13 21:44:10,854 - src.llm_utils - WARNING - RunPod OpenAI API 실패 (시도 3/3): Connection error.
2026-02-13 21:44:10,854 - src.llm_utils - WARNING - 비교 해석 LLM 실패: Connection error.
INFO:     127.0.0.1:51750 - "POST /api/v1/llm/comparison/batch HTTP/1.1" 200 OK
INFO:     127.0.0.1:51800 - "POST /api/v1/vector/search/similar HTTP/1.1" 200 OK

---

로그를 보면 핵심은 이거야:

* `[entrypoint] ... downloading from HuggingFace ...`
* 곧바로 `socket.gaierror: Temporary failure in name resolution`
  → **Hugging Face 도메인(huggingface.co / cdn 등) DNS 해석이 Pod 안에서 실패**한 케이스야. (인터넷 “자체”가 끊긴 게 아니라, *DNS가 안 풀려서* HTTPS 요청을 못 나감)

아래 순서대로 보면 거의 해결돼.

---

## 1) Pod 안에서 “DNS만” 문제인지, “아예 외부 통신” 문제인지 먼저 확인

RunPod “Connect”로 들어가서 아래 실행:

```bash
# 1) IP로는 나가나? (DNS 없이)
curl -I https://1.1.1.1 --max-time 5

# 2) DNS가 풀리나?
getent hosts huggingface.co
getent hosts cdn-lfs.huggingface.co

# 3) 실제로 HF 접속 되나?
curl -I https://huggingface.co --max-time 10
```

* **1)은 되는데 2)~3)만 실패** → DNS resolver 쪽 문제(지금 로그랑 동일)
* **1)도 실패** → Pod의 outbound(egress) 자체가 막혔거나 해당 노드 네트워크 이슈

---

## 2) “일시적 DNS”면: 재시도/지연을 넣는 게 제일 현실적인 해결

RunPod/클라우드 쪽 DNS가 순간 흔들리는 경우가 종종 있어서(특히 막 뜬 직후), **다운로드 로직에 재시도(backoff)** 를 넣으면 안정성이 확 올라가.

entrypoint에서 HF 다운로드 전에:

```bash
# DNS/네트워크 흔들릴 때 대비: 잠깐 대기 + 재시도
for i in 1 2 3 4 5; do
  echo "DNS warmup try=$i"
  getent hosts huggingface.co && break
  sleep $((i*2))
done
```

그리고 `huggingface-cli download` / `snapshot_download`도 실패 시 재시도하도록 래핑.

---

## 3) DNS 설정을 “강제로” 바꿀 수 있으면 (권한 있을 때만)

Pod 쉘에서 `/etc/resolv.conf` 확인:

```bash
cat /etc/resolv.conf
```

만약 컨테이너가 root이고 파일이 쓰기 가능하면(환경에 따라 다름):

```bash
printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions timeout:1 attempts:3\n" | tee /etc/resolv.conf
```

> 안 되는 환경이면(읽기전용/권한 없음) 이 방법은 패스하고, **2) 재시도** + **4) 프리다운로드**로 가는 게 정석.

---

## 4) 가장 확실한 해결: “네트워크 볼륨에 미리 모델을 넣고” Pod는 오프라인처럼 기동

지금처럼 Pod가 뜰 때마다 HF에서 받으려 하면,

* DNS/네트워크 변동에 취약
* 부팅이 느림

RunPod는 **네트워크 볼륨을 S3-compatible API로 직접 업로드/관리**할 수 있어서, **컴퓨트 없이 미리 파일을 넣어둘 수 있어.** ([docs.runpod.io][1])

추천 흐름:

1. 네트워크 볼륨에 `Qwen/Qwen2.5-7B-Instruct`를 미리 저장
2. entrypoint는 “있으면 바로 vLLM 실행 / 없으면 다운로드(재시도)”만 수행
3. 실전에서는 다운로드 분기를 거의 안 타게 됨

---

## 5) (중요) “Pod는 살아있는데 모델 다운로드만 실패”한 상황 정리

두 번째 로그에서 APIServer가 라우트 등록하고 `Application startup complete`인 걸 보면,

* **컨테이너/서버 프로세스는 정상 기동**
* 다만 **모델 다운로드 단계에서 DNS가 터졌던 타이밍**이 있었던 거야

즉, **다운로드 로직을 “서버 기동과 분리”**하거나,
**서버가 뜨기 전에 모델 준비(4번)** 를 하면 안정적이야.

---

## 6) 참고: RunPod API로 Pod 제어는 GraphQL이 맞음

RunPod Pod 제어/관리(GraphQL) 문서/스펙: ([docs.runpod.io][2])
(그래서 너가 앞에서 말한 REST API는 “REST 느낌이지만 실제로는 GraphQL endpoint”로 호출하는 형태가 맞아.)

---

원하면 내가 **지금 네 entrypoint 로그 흐름(“모델 있으면 스킵/없으면 다운로드”)** 기준으로,

* DNS warmup + HF 다운로드 재시도 + 실패 시 종료코드 설계
* (볼륨 경로 `/workspace/llm-models/...` 유지)
  이렇게 **그대로 붙여넣기 가능한 entrypoint 스크립트**로 만들어줄게.
  너 지금 HF 다운로드는 `huggingface-cli`로 하고 있어, 아니면 파이썬 `snapshot_download`야?

[1]: https://docs.runpod.io/storage/s3-api?utm_source=chatgpt.com "S3-compatible API"
[2]: https://docs.runpod.io/sdks/graphql/manage-pods?utm_source=chatgpt.com "Manage Pods - GraphQL API"
