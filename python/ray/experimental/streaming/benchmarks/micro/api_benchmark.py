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
parser.add_argument("--num-stages", default=2,
                    help="the number of stages in the chain")
parser.add_argument("--dataflow-parallelism", default=1,
                    help="the number of instances per operator")
parser.add_argument("--record-type", default="int",
                    choices = ["int","string"],
                    help="the number of instances per operator")
parser.add_argument("--log-file", required=True,
                    help="the file to log per-record latencies")
parser.add_argument("--sample-period", default=1,
                    help="every how many input records latency is measured.")
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

def compute_elapsed_time(record):
    try:
        generation_time, data = record
        # TODO (john): Clock skew might distort elapsed time
        return str(time.time() - generation_time) + " " + str(data)
    except TypeError:
        return ""

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
        assert self.record_type in ["int","string"], (self.record_type)
        # Set the right function
        if self.record_type == "int":
            self.record = -1
            self.__get_next_record = self.__get_next_int
        else:
            self.__get_next_record = self.__get_next_string

    # Returns the next int
    def __get_next_int(self):
        self.record += 1
        return self.record

    # Returns the next random string
    def __get_next_string(self):
        return ''.join(random.choice(
                string.ascii_letters + string.digits) for _ in range(
                                                                record_size))

    # Generates next record
    def get_next(self):
        if self.total_count == self.total_elements:
            return None
        record = self.__get_next_record()
        self.total_count += 1
        self.count += 1
        if self.count == self.period:
            self.count = 0
            return (time.time(),record)  # Assign the generation timestamp
        else:
            return self.record

def create_and_run_dataflow(rounds, num_stages, dataflow_parallelism,
                            partitioning, record_type, record_size,
                            queue_config, log_filename, sample_period):
    # Create streaming environment, construct and run dataflow
    env = Environment()
    env.set_queue_config(queue_config)
    env.set_parallelism(dataflow_parallelism)
    stream = env.source(Source(rounds, record_type,
                                record_size, sample_period))
    for stage in range(num_stages):
        if partitioning == "shuffling":
            stream = stream.shuffle()
        elif partitioning == "broadcast":
            stream = stream.broadcast()
        elif partitioning != "round_robin":
            sys.exit("Unrecognized patitioning strategy.")
        stream = stream.map(lambda record: record)
    stream = stream.write_text_file(log_filename,compute_elapsed_time)
    start = time.time()
    dataflow = env.execute()
    ray.get(dataflow.termination_status())
    logger.info("Elapsed time: {}".format(time.time()-start))

if __name__ == "__main__":
    ray.init()
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)

    args = parser.parse_args()

    rounds = int(args.rounds)
    num_stages = int(args.num_stages)
    log_filename = str(args.log_file)
    sample_period = int(args.sample_period)
    dataflow_parallelism = int(args.dataflow_parallelism)
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
    logger.info("Sample period: {}".format(sample_period))
    logger.info("Log file: {}".format(log_filename))
    logger.info("Number of stages: {}".format(num_stages))
    logger.info("Parallelism: {}".format(dataflow_parallelism))
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
    source = Source(rounds, record_type, record_size)
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

    logger.info("== Testing max throughput ==")
    create_and_run_dataflow(rounds, num_stages, dataflow_parallelism,
                            partitioning, record_type, record_size,
                            queue_config, log_filename, sample_period)
