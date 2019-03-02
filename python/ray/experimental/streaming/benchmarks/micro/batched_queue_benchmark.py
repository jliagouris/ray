from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import string
import time

import ray
from ray.experimental.streaming.batched_queue import BatchedQueue

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", default=10,
                    help="the number of experiment rounds")
parser.add_argument("--num-queues", default=1,
                    help="the number of queues in the chain")
parser.add_argument("--record-type", default="int",
                    choices = ["int","string"],
                    help="the number of instances per operator")
parser.add_argument("--record-size", default=10,
                    help="the size of a record of type string in bytes")
parser.add_argument("--latency-file", required=True,
                    help="the file to log per-record latencies")
parser.add_argument("--throughput-file", required=True,
                    help="the file to log actor throughputs")
parser.add_argument("--sample-period", default=1,
                    help="every how many input records latency is measured.")
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

@ray.remote
class Node(object):
    """An actor that reads from an input queue and writes to an output queue.

    Attributes:
        id (int): The id of the actor.
        queue (BatchedQueue): The input queue.
        out_queue (BatchedQueue): The output queue.
        max_reads_per_second (int): The max read throughput (default: inf).
        num_reads (int): Number of elements read.
        num_writes (int): Number of elements written.
    """
    def __init__(self, id, in_queue, out_queue,
                    max_reads_per_second=float("inf"),
                    log_file=None):
        self.id = id
        self.queue = in_queue
        self.out_queue = out_queue
        self.max_reads_per_second = max_reads_per_second
        self.num_reads = 0
        self.num_writes = 0
        self.throughputs = []
        self.start = time.time()
        self.log_file = log_file
        # Open log file for writing
        if self.log_file:
            self.writer = open(self.log_file, "w")


    def read_write(self, rounds):
        debug_log = "[actor {}] Reads throttled to {} reads/s"
        log = ""
        if self.out_queue is not None:
            self.out_queue.enable_writes()
            log += "[actor {}] Reads/Writes per second {}"
        else:   # It's just a reader
            log += "[actor {}] Reads per second {}"
        # Start spinning
        expected_value = 0
        for _ in range(rounds):
            start = time.time()
            N = 100000
            for _ in range(N):
                record = self.queue.read_next()
                try:
                    start_time, value = record
                except TypeError:
                    start_time = -1
                    value = record
                expected_value += 1
                self.num_reads += 1
                if self.out_queue is not None:
                    self.out_queue.put_next((start_time,value))
                    self.num_writes += 1
                elif self.log_file:  # Log end-to-end latency
                    # TODO (john): Clock skew might distort elapsed time
                    if start_time != -1:
                        elapsed_time = time.time() - start_time
                        flog = str(elapsed_time) + " " + str(value) + "\n"
                        self.writer.write(flog)
                while (self.num_reads / (time.time() - self.start) >
                        self.max_reads_per_second):
                    logger.debug(debug_log.format(self.id,
                                        self.max_reads_per_second))
                    time.sleep(0.1)
            self.throughputs.append(N / (time.time() - start))
            logger.info(log.format(self.id,
                                   N / (time.time() - start)))
        # Flush any remaining elements
        if self.out_queue is not None:
            self.out_queue._flush_writes()
        elif self.log_file:  # Close log file
            self.writer.close()

        return (self.id, self.throughputs)

def test_max_throughput(rounds, latency_file, throughput_file,
                        sample_period, max_queue_size,
                        max_batch_size, batch_timeout,
                        prefetch_depth, background_flush,
                        num_queues, max_reads_per_second=float("inf")):
    assert num_queues >= 1
    first_queue = BatchedQueue(
        max_size=max_queue_size,
        max_batch_size=max_batch_size,
        max_batch_time=batch_timeout,
        prefetch_depth=prefetch_depth,
        background_flush=background_flush)
    previous_queue = first_queue
    throughputs = []
    log_file = None
    for i in range(num_queues):
        # Construct the batched queue
        in_queue = previous_queue
        out_queue = None
        if i < num_queues-1:
            out_queue = BatchedQueue(
                max_size=max_queue_size,
                max_batch_size=max_batch_size,
                max_batch_time=batch_timeout,
                prefetch_depth=prefetch_depth,
                background_flush=background_flush)
        else:  # Pass the log filename to the last actor
            log_file = latency_file
        node = Node.remote(i, in_queue, out_queue,
                           max_reads_per_second, log_file)
        # output will eventually store a None value returned by
        # calling read_write() on the last actor
        throughputs.append(node.read_write.remote(rounds))
        previous_queue = out_queue

    value = 0
    count = 0
    source_throughputs = []
    # Feed the chain
    for round in range(rounds):
        logger.info("Round {}".format(round))
        N = 100000
        start = time.time()
        for _ in range(N):
            count += 1
            if count == sample_period:
                count = 0
                first_queue.put_next((time.time(),value))
            else:
                first_queue.put_next(value)
            value += 1
        log = "[writer] Puts per second {}"
        logger.info(log.format(N / (time.time() - start)))
        source_throughputs.append(N / (time.time() - start))
    first_queue._flush_writes()
    result = ray.get(throughputs)
    result.append((-1,source_throughputs))  # Use -1 as the source id
    # Write throughputs to file
    with open(throughput_file, "w") as tf:
        for node_id, throughputs in result:
            for throughput in throughputs:
                tf.write(str(node_id) + " " + str(throughput) + "\n")

def get_next_int(self):
    self.record += 1
    return self.record

# Returns the next random string
def get_next_string(self):
    return ''.join(random.choice(
            string.ascii_letters + string.digits) for _ in range(
                                                            record_size))
if __name__ == "__main__":
    ray.init()
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)

    # Benchmark parameters
    args = parser.parse_args()
    rounds = int(args.rounds)
    latency_filename = str(args.latency_file)
    throughput_filename = str(args.throughput_file)
    sample_period = int(args.sample_period)
    record_type = str(args.record_type)
    record_size = int(args.record_size) if record_type == "string" else None
    max_queue_size = int(args.queue_size)
    max_batch_size = int(args.batch_size)
    batch_timeout = float(args.flush_timeout)
    prefetch_depth = int(args.prefetch_depth)
    background_flush = bool(args.background_flush)
    num_queues = int(args.num_queues)
    max_reads_per_second = float(args.max_throughput)

    logger.info("== Parameters ==")
    logger.info("Rounds: {}".format(rounds))
    logger.info("Sample period: {}".format(sample_period))
    logger.info("Latency file: {}".format(latency_filename))
    logger.info("Throughput file: {}".format(throughput_filename))
    logger.info("Record type: {}".format(record_type))
    if record_type == "string":
        logger.info("Record size: {}".format(record_size))
    logger.info("Max queue size: {}".format(max_queue_size))
    logger.info("Max batch size: {}".format(max_batch_size))
    logger.info("Batch timeout: {}".format(batch_timeout))
    logger.info("Prefetch depth: {}".format(prefetch_depth))
    logger.info("Background flush: {}".format(background_flush))
    logger.info("Max read throughput: {}".format(max_reads_per_second))

    # Estimate the 'ideal' throughput
    value = 0 if record_type == "int" else ""
    total_count = 0
    count = 0
    start = time.time()
    for round in range(rounds):
        N = 100000
        for _ in range(N):
            count += 1
            if count == sample_period:
                count = 0
                time.time()
            if record_type == "int":
                value += 1
            else:
                value = "".join(random.choice(
                        string.ascii_letters + string.digits) for _ in range(
                                                                record_size))
            total_count += 1
    logger.info("Ideal throughput: {}".format(
                                        total_count / (time.time()-start)))

    logger.info("== Testing max throughput ==")
    start = time.time()
    test_max_throughput(rounds, latency_filename, throughput_filename,
                        sample_period, max_queue_size,
                        max_batch_size, batch_timeout,
                        prefetch_depth, background_flush,
                        num_queues, max_reads_per_second)
    logger.info("Elapsed time: {}".format(time.time()-start))
