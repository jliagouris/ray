from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
import time
import wikipedia

import ray
from ray.experimental.streaming.streaming import Environment
from ray.experimental.streaming.batched_queue import BatchedQueue
from ray.experimental.streaming.operator import OpType, PStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--num-arrays", required=True,
                    help="the total number of arrays to generate")

# A custom data source that reads articles from wikipedia
# Custom data sources need to implement a get_next() method
# that returns the next data element, in this case sentences
class RandomArrays(object):
    def __init__(self, num_arrays):
        self.num_arrays = num_arrays
        self.counter = 0

    # Returns next numpy array
    def get_next(self):
        if self.counter == self.num_arrays:
            return None     # Source exhausted
        # Generate next random 100x100 array
        rand_array = np.random.rand(100,100)
        self.counter += 1
        return rand_array

# Splits an numpy array into rows
def splitter(array):
    return list(array)

# Returns the sum of the 4th, 12th, and 84th attribute of the row
def key_selector(row):
    indices = [(0,3), (0,11), (0,83)]
    return np.sum(np.take(row,indices))

# Stack rows vertically
def stack(array_1,array_2):
    return np.vstack(array_1, array_2)

if __name__ == "__main__":
    # Get program parameters
    args = parser.parse_args()
    num_arrays = int(args.num_arrays)

    ray.init()
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)
    ray.register_custom_serializer(OpType, use_pickle=True)
    ray.register_custom_serializer(PStrategy, use_pickle=True)

    # A Ray streaming environment with the default configuration
    env = Environment()
    env.set_parallelism(2)  # Each operator will be executed by two actors

    # Stream represents the ouput of the reduce and
    # can be forked into other dataflows
    stream = env.source(RandomArrays(num_arrays)) \
                .flat_map(splitter) \
                .key_by(key_selector) \
                .reduce(stack) \
                .inspect(print)  # Prints the content of the stream

    start = time.time()
    dataflow = env.execute()        # Deploys and executes the dataflow
    # Stay alive until execution finishes
    ray.get(dataflow.termination_status())
    end = time.time()
    logger.info("Elapsed time: {} secs".format(end - start))
    logger.debug("Output stream id: {}".format(stream.id))
