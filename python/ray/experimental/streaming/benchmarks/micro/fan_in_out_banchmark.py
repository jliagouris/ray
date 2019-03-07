from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import statistics
import string
import sys
import time

import ray
from ray.experimental.streaming.batched_queue import BatchedQueue
from ray.experimental.streaming.communication import QueueConfig
from ray.experimental.streaming.streaming import Environment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", default=10,
                    help="the number of experiment rounds")
parser.add_argument("--task-based", default=False,
                    help="task-based execution")
parser.add_argument("--fan-in", default=0,
                    help="the number of input channels to the 2nd stage")
parser.add_argument("--fan-out", default=0,
                    help="the number of output channels from the 1st stage")
parser.add_argument("--latency-file", required=True,
                    help="the file to log per-record latencies")
parser.add_argument("--throughput-file", required=True,
                    help="the file to log actors throughput")
parser.add_argument("--sample-period", default=1,
                    help="every how many input records latency is measured.")
parser.add_argument("--record-type", default="int",
                    choices = ["int","string"],
                    help="the type of records used in the benchmark")
parser.add_argument("--record-size", default=10,
                    help="the size of a record of type string in bytes")
parser.add_argument("--partitioning", default = "round_robin",
                    choices = ["shuffle","round_robin","broadcast"],
                    help="whether to shuffle or balance after each stage")
parser.add_argument("--queue-size", default=10000,
                    help="the queue size in number of batches")
parser.add_argument("--batch-size", default=1000,
                    help="the batch size in number of elements")
parser.add_argument("--flush-timeout", default=0.001,
                    help="the timeout to flush a batch")
parser.add_argument("--prefetch-depth", default=10,
                    help="the number of batches to prefetch from plasma")
parser.add_argument("--background-flush", default=False,
                    help="whether to flush in the backrgound or not")
parser.add_argument("--max-throughput", default="inf",
                    help="maximum read throughput (elements/s)")

# A custom source that periodically assigns timestamps to records
class Source(object):
    def __init__(self, rounds, record_type="int",
                 record_size=None, sample_period=1):
        assert rounds > 0
        self.total_elements = 100000 * rounds
        self.total_count = 0
        self.period = sample_period
        self.count = 0
        self.current_round = 0
        self.record_type = record_type
        self.record_size = record_size
        self.record = -1

    # Returns the next int
    def __get_next_int(self):
        self.record += 1
        return self.record

    # Returns the next random string
    def __get_next_string(self):
        return "".join(random.choice(
                string.ascii_letters + string.digits) for _ in range(
                                                        self.record_size))

    # Generates next record
    def get_next(self):
        record = None
        if self.total_count == self.total_elements:
            return record
        self.record += 1
        record = self.record
        self.total_count += 1
        self.count += 1
        if self.count == self.period:
            self.count = 0
            return (time.time(),record)  # Assign the generation timestamp
        else:
            return (-1,record)

def compute_elapsed_time(record):
    generation_time, _ = record
    if generation_time != -1:
        # TODO (john): Clock skew might distort elapsed time
        return [time.time() - generation_time]
    else:
        return []

# Measures latency
def fan_in_benchmark(rounds, fan_in, partitioning, record_type,
                    record_size, queue_config, sample_period,
                    latency_filename, throughput_filename,
                    task_based):
    # Create streaming environment, construct and run dataflow
    env = Environment()
    env.set_queue_config(queue_config)
    env.enable_logging()
    if task_based:
        env.enable_tasks()
    source_streams = []
    for i in range(fan_in):
        stream = env.source(Source(rounds, record_type,
                                    record_size, sample_period))
        if partitioning == "shuffle":
            stream = stream.shuffle()
        elif partitioning == "broadcast":
            stream = stream.broadcast()
        source_streams.append(stream)
    stream = source_streams.pop()
    stream = stream.union(source_streams)
    # TODO (john): Having one flatmap here might be a bottleneck
    stream = stream.flat_map(compute_elapsed_time)
    # Add one sink per flatmap instance to log the per-record latencies
    _ = stream.sink(Sink(), id="my_sink")
    start = time.time()
    dataflow = env.execute()
    ray.get(dataflow.termination_status())

    latency_filename, throughput_filename = create_filenames(
                        latency_filename, throughput_filename,
                        rounds, sample_period,
                        record_type, record_size, queue_config, 
                        max_reads_per_second, num_stages,
                        partitioning, task_based, fan_in)
    write_log_files(latency_filename,
                    throughput_filename, dataflow)
    logger.info("Elapsed time: {}".format(time.time()-start))

def fan_out_benchmark(rounds, fan_out, partitioning, record_type,
                     record_size, queue_config, sample_period,
                     latency_filename, throughput_filename,
                     task_based):
    # Create streaming environment, construct and run dataflow
    env = Environment()
    env.set_queue_config(queue_config)
    env.enable_logging()
    if task_based:
        env.enable_tasks()
    stream = env.source(Source(rounds, record_type,
                                record_size, sample_period))
    if partitioning == "shuffle":
        stream = stream.shuffle()
    elif partitioning == "broadcast":
        stream = stream.broadcast()
    stream = stream.flat_map(compute_elapsed_time).set_parallelism(fan_out)
    # Add one sink per flatmap instance to log the per-record latencies
    _ = stream.sink(Sink(), id="my_sink").set_parallelism(fan_out)
    start = time.time()
    dataflow = env.execute()
    ray.get(dataflow.termination_status())

    # Write log files
    latency_filename, throughput_filename = create_filenames(
                        latency_filename, throughput_filename,
                        rounds, sample_period,
                        record_type, record_size,
                        queue_config, max_reads_per_second, num_stages,
                        partitioning, task_based, fan_out)
    write_log_files(latency_filename,
                    throughput_filename, dataflow)
    logger.info("Elapsed time: {}".format(time.time()-start))

# Creates the log file names
def create_filenames(latency_filename, throughput_filename,
                    rounds, sample_period,
                    record_type, record_size, queue_config,
                    max_reads_per_second, num_stages,
                    partitioning, task_based, fan_in_out):
    # Create log filenames
    max_queue_size = queue_config.max_size
    max_batch_size = queue_config.max_batch_size
    batch_timeout = queue_config.max_batch_time
    prefetch_depth = queue_config.prefetch_depth
    background_flush = queue_config.background_flush
    all_parameters = "-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        rounds, sample_period,
        record_type, record_size, max_queue_size,
        max_batch_size, batch_timeout, prefetch_depth,
        background_flush, max_reads_per_second, num_stages,
        partitioning, task_based, fan_in_out
    )
    latency_filename = latency_filename + all_parameters
    throughput_filename = throughput_filename + all_parameters
    return latency_filename, throughput_filename

# Collects sampled latencies and throughputs for all actors in the dataflow
def write_log_files(all_parameters, latency_filename,
                    throughput_filename, dataflow):

    # Dump timeline
    dump_file = "dump" + all_parameters
    ray.global_state.chrome_tracing_dump(dump_file)

    # Collect sampled per-record latencies
    local_states = ray.get(dataflow.state_of("my_sink"))
    latencies = [state for state in local_states if state is not None]
    latency_file = latency_file + all_parameters
    with open(latency_file, "w") as tf:
        for _, latency_values in latencies:
            if latency_values is not None:
                for value in latency_values:
                    tf.write(str(value) + "\n")

    # Collect throughputs from all actors (except the sink)
    ids = dataflow.operator_ids()
    throughputs = []
    for id in ids:
        logs = ray.get(dataflow.logs_of(id))
        throughputs.extend(logs)
    throughput_file = throughput_file + all_parameters
    with open(throughput_file, "w") as tf:
        for actor_id, throughput_values in throughputs:
            operator_id, instance_id = actor_id
            for value in throughput_values:
                tf.write(str(operator_id) + " " + str(
                    instance_id) + " " + str(value) + "\n")


if __name__ == "__main__":
    ray.init()
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)

    args = parser.parse_args()

    rounds = int(args.rounds)
    task_based = int(args.task_based)
    latency_filename = str(args.latency_file)
    throughput_filename = str(args.throughput_file)
    sample_period = int(args.sample_period)
    fan_in = int(args.fan_in)
    fan_out = int(args.fan_out)
    record_type = str(args.record_type)
    record_size = int(args.record_size) if record_type == "string" else None
    partitioning = str(args.partitioning)
    max_queue_size = int(args.queue_size)
    max_batch_size = int(args.batch_size)
    batch_timeout = float(args.flush_timeout)
    prefetch_depth = int(args.prefetch_depth)
    background_flush = bool(args.background_flush)
    max_reads_per_second = float(args.max_throughput)

    logger.info("== Parameters ==")
    logger.info("Rounds: {}".format(rounds))
    logger.info("Task-based execution: {}".format(task_based))
    logger.info("Sample period: {}".format(sample_period))
    logger.info("Latency file: {}".format(latency_filename))
    logger.info("Throughput file: {}".format(throughput_filename))
    if fan_in != 0:
        logger.info("Fan-in: {}".format(fan_in))
    if fan_out != 0:
        logger.info("Fan-out: {}".format(fan_out))
    logger.info("Record type: {}".format(record_type))
    if record_type == "string":
        logger.info("Record size: {}".format(record_size))
    logger.info("Partitioning: {}".format(partitioning))
    logger.info("Max queue size: {}".format(max_queue_size))
    logger.info("Max batch size: {}".format(max_batch_size))
    logger.info("Batch timeout: {}".format(batch_timeout))
    logger.info("Prefetch depth: {}".format(prefetch_depth))
    logger.info("Background flush: {}".format(background_flush))
    logger.info("Max read throughput: {}".format(max_reads_per_second))

    # Estimate the ideal throughput
    source = Source(rounds, record_type, record_size, sample_period)
    count = 0
    start = time.time()
    while True:
        next = source.get_next()
        if next is None:
            break
        count += 1
    logger.info("Ideal throughput: {}".format(count / (time.time()-start)))

    queue_config = QueueConfig(max_queue_size,
                        max_batch_size, batch_timeout,
                        prefetch_depth, background_flush)

    if fan_in != 0:
        logger.info("== Testing fan-in ==")
        fan_in_benchmark(rounds, fan_in, partitioning, record_type,
                        record_size, queue_config, sample_period,
                        latency_filename, throughput_filename,
                        task_based)
    if fan_out != 0:
        logger.info("== Testing fan-out ==")
        fan_out_benchmark(rounds, fan_out, partitioning, record_type,
                         record_size, queue_config, sample_period,
                         latency_filename, throughput_filename,
                         task_based)
