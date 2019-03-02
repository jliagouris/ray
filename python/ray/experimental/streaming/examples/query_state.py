from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import time
import wikipedia

import ray
import ray.experimental.signal as signal
from ray.experimental.streaming.streaming import Environment
from ray.experimental.streaming.batched_queue import BatchedQueue
from ray.experimental.streaming.operator import OpType, PStrategy

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True,
                    help="the iput text file")

# Splits input line into words and
# outputs records of the form (word,1)
def splitter(line):
    records = []
    words = line.split()
    for w in words:
        records.append((w, 1))
    return records

# Returns the first attribute of a tuple
def key_selector(tuple):
    return tuple[0]

# Returns the second attribute of a tuple
def attribute_selector(tuple):
    return tuple[1]

if __name__ == "__main__":
    # Get program parameters
    args = parser.parse_args()
    input_file = str(args.input_file)

    ray.init()
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)
    ray.register_custom_serializer(OpType, use_pickle=True)
    ray.register_custom_serializer(PStrategy, use_pickle=True)

    # A Ray streaming environment with the default configuration
    env = Environment()
    env.set_parallelism(2)  # Each operator will be executed by two actors

    env.read_text_file(input_file) \
       .flat_map(splitter) \
       .key_by(key_selector) \
       .sum(attribute_selector, id="123") \
       .inspect(print)     # Prints the content of the stream to stdout

    start = time.time()
    # dataflow is a handle to the running dataflow
    dataflow = env.execute()

    # Get a snapshot of rolling sum's state
    sum_state = ray.get(dataflow.state_of("123"))
    print("Sum's state: {}".format(sum_state))

    # Stay alive until execution finishes
    exit_status = ray.get(dataflow.termination_status())
    end = time.time()
    logger.info("Elapsed time: {} secs".format(end - start))
    logger.debug("Exit status: {}".format(exit_status))
