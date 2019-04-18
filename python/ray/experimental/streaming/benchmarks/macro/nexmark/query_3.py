from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import string
import sys
import time

import ray
import ray.experimental.streaming.benchmarks.utils as utils
import ray.experimental.streaming.benchmarks.macro.nexmark.data_generator as dg
from ray.experimental.streaming.batched_queue import BatchedQueue
from ray.experimental.streaming.benchmarks.macro.nexmark.event import Person
from ray.experimental.streaming.benchmarks.macro.nexmark.event import Auction
from ray.experimental.streaming.benchmarks.macro.nexmark.event import Record
from ray.experimental.streaming.communication import QueueConfig
from ray.experimental.streaming.streaming import Environment

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

parser = argparse.ArgumentParser()
parser.add_argument("--pin-processes", default=False,
                    action='store_true',
                    help="whether to pin python processes to cores or not")
parser.add_argument("--simulate-cluster", default=False,
                    action='store_true',
                    help="simulate a Ray cluster on a single machine")
parser.add_argument("--fetch-data", default=False,
                    action='store_true',
                    help="fecth data from S3")
parser.add_argument("--omit-extra", default=False,
                    action='store_true',
                    help="omit extra field from events")
parser.add_argument("--records", default=-1,
                help="maximum number of records to replay from each source.")
parser.add_argument("--nodes", default=1,
                    help="total number of nodes in the cluster")
parser.add_argument("--redis-shards", default=1,
                    help="total number of Redis shards")
parser.add_argument("--redis-max-memory", default=10**9,
                    help="max amount of memory per Redis shard")
parser.add_argument("--plasma-memory", default=10**9,
                    help="amount of memory to start plasma with")
# Dataflow-related parameters
parser.add_argument("--placement-file", default="",
            help="Path to the file containing the explicit actor placement")
parser.add_argument("--bids-file", required=False,
                    help="Path to the bids file")
parser.add_argument("--auctions-file", required=True,
                    help="Path to the auctions file")
parser.add_argument("--persons-file", required=True,
                    help="Path to the persons file")
parser.add_argument("--enable-logging", default=False,
                    action='store_true',
                    help="whether to log actor latency and throughput")
parser.add_argument("--queue-based", default=False,
                    action='store_true',
                    help="queue-based execution")
parser.add_argument("--join-instances", default=1,
                    help="the number of join instances")
parser.add_argument("--latency-file", default="latencies",
                    help="a prefix for the latency log files")
parser.add_argument("--throughput-file", default="throughputs",
                    help="a prefix for the rate log files")
parser.add_argument("--dump-file", default="",
                    help="a prefix for the chrome dump file")
parser.add_argument("--sample-period", default=100,
                    help="every how many input records latency is measured.")
parser.add_argument("--auction-sources", default=1,
                    # TODO (john): Add check
                    help="number of auction sources")
parser.add_argument("--source-rate", default=-1,
                    type=lambda x: float(x) or
                                parser.error("Source rate cannot be zero."),
                    help="source output rate (records/s)")
parser.add_argument("--person-sources", default=1,
                    # TODO (john): Add check
                    help="number of person sources")
parser.add_argument("--persons-rate", default=-1,
                    type=lambda x: float(x) or
                                parser.error("Source rate cannot be zero."),
                    help="source output rate (records/s)")
# Queue-related parameters
parser.add_argument("--queue-size", default=4,
                    help="the queue size in number of batches")
parser.add_argument("--batch-size", default=1000,
                    help="the batch size in number of elements")
parser.add_argument("--flush-timeout", default=0.1,
                    help="the timeout to flush a batch")
parser.add_argument("--prefetch-depth", default=1,
                    help="the number of batches to prefetch from plasma")
parser.add_argument("--background-flush", default=False,
                    help="whether to flush in the backrgound or not")
parser.add_argument("--max-throughput", default="inf",
                    help="maximum read throughput (records/s)")

# Used to join auctions with persons incrementally. An auction has exactly one
# seller, thus, we can remove the auction entry from local state upon a join
class JoinLogic(object):
    def __init__(self):
        # Local state
        self.auctions = {}  # seller -> auctions
        self.persons = {}   # id -> person

    def process_left(self, auction):
        result = []
        person = self.persons.get(auction["seller"])
        if person is None:  # Store auction for future join
            entry = self.auctions.setdefault(auction["seller"], [])
            entry.append(auction)
        else:  # Found a join; emit and do not store auction
            # logger.info("Found a join {} - {}".format(auction, person))
            p_time = person["system_time"]
            a_time = auction["system_time"]

            if p_time is None:
                s_time = a_time
            else:
                if a_time is not None and a_time > p_time:
                    s_time = a_time
                else:
                    s_time = p_time
            # record = Record(name=person.name, city=person.city,
            #                 state=person.state, auction=auction.id,
            #                 system_time=s_time)
            person["auction_id"] = auction["id"]
            person["system_time"] = s_time
            result.append(person)
        return result

    def process_right(self, person):
        result = []
        self.persons.setdefault(person["id"],person)
        auctions = self.auctions.pop(person["id"], None)
        if auctions is not None:
            for auction in auctions:
                # logger.info("Found a join {} - {}".format(auction, person))
                # Remove entry
                p_time = person["system_time"]
                a_time = auction["system_time"]

                if p_time is None:
                    s_time = a_time
                else:
                    if a_time is not None and a_time > p_time:
                        s_time = a_time
                    else:
                        s_time = p_time
                # record = Record(name=person.name, city=person.city,
                #                 state=person.state, auction=auction.id,
                #                 system_time=s_time)
                person["auction_id"] = auction["id"]
                person["system_time"] = s_time
                result.append(person)
        return result


if __name__ == "__main__":

    args = parser.parse_args()

    num_nodes = int(args.nodes)
    simulate_cluster = bool(args.simulate_cluster)
    fetch_data = bool(args.fetch_data)
    omit_extra = bool(args.omit_extra)
    max_records = int(args.records)
    placement_file = str(args.placement_file)
    num_redis_shards = int(args.redis_shards)
    redis_max_memory = int(args.redis_max_memory)
    plasma_memory = int(args.plasma_memory)
    auctions_file = str(args.auctions_file)
    persons_file = str(args.persons_file)
    latency_filename = str(args.latency_file)
    throughput_filename = str(args.throughput_file)
    dump_filename = str(args.dump_file)
    logging = bool(args.enable_logging)
    sample_period = int(args.sample_period)
    task_based = not bool(args.queue_based)
    join_instances = int(args.join_instances)
    max_queue_size = int(args.queue_size)
    max_batch_size = int(args.batch_size)
    batch_timeout = float(args.flush_timeout)
    prefetch_depth = int(args.prefetch_depth)
    background_flush = bool(args.background_flush)
    auction_sources = int(args.auction_sources)
    auctions_rate = float(args.source_rate)
    person_sources = int(args.person_sources)
    persons_rate = float(args.persons_rate)
    pin_processes = bool(args.pin_processes)

    logger.info("== Parameters ==")
    logger.info("Number of nodes: {}".format(num_nodes))
    logger.info("Simulate cluster: {}".format(simulate_cluster))
    logger.info("Fetch data: {}".format(fetch_data))
    logger.info("Omit extra: {}".format(omit_extra))
    logger.info("Maximum number of records: {}".format(max_records))
    logger.info("Placement file: {}".format(placement_file))
    logger.info("Number of Redis shards: {}".format(num_redis_shards))
    logger.info("Max memory per Redis shard: {}".format(redis_max_memory))
    logger.info("Plasma memory: {}".format(plasma_memory))
    logger.info("Logging: {}".format(logging))
    logger.info("Sample period: {}".format(sample_period))
    logger.info("Task-based execution: {}".format(task_based))
    logger.info("Auctions file: {}".format(auctions_file))
    logger.info("Persons file: {}".format(persons_file))
    logger.info("Latency file prefix: {}".format(latency_filename))
    logger.info("Throughput file prefix: {}".format(throughput_filename))
    logger.info("Dump file prefix: {}".format(dump_filename))
    logger.info("Join instances: {}".format(join_instances))
    logger.info("Max queue size: {}".format(max_queue_size))
    logger.info("Max batch size: {}".format(max_batch_size))
    logger.info("Batch timeout: {}".format(batch_timeout))
    logger.info("Prefetch depth: {}".format(prefetch_depth))
    logger.info("Background flush: {}".format(background_flush))
    logger.info("Auction sources: {}".format(auction_sources))
    message = (" (as fast as it gets)") if auctions_rate < 0 else ""
    logger.info("Auctions rate: {}".format(auctions_rate) + message)
    logger.info("Person sources: {}".format(person_sources))
    message = (" (as fast as it gets)") if persons_rate < 0 else ""
    logger.info("Persons rate: {}".format(persons_rate) + message)
    logger.info("Pin processes: {}".format(pin_processes))

    if fetch_data:
        logger.info("Fetching data...")
        s3 = boto3.resource('s3')
        s3.meta.client.download_file('nexmark', "auctions", "auctions.data")
        s3.meta.client.download_file('nexmark', "persons", "persons.data")

    # Total number of source instances
    num_sources = auction_sources + person_sources

    # Number of actors per dataflow stage
    stage_parallelism = [join_instances,
                         join_instances]  # One sink per flatmap instance

    placement = {}  # oparator name -> node ids
    if simulate_cluster:  # Simulate a cluster with the given configuration
        utils.start_virtual_cluster(num_nodes, num_redis_shards,
                                    plasma_memory, redis_max_memory,
                                    stage_parallelism, num_sources,
                                    pin_processes)
        num_stages = 2  # We have a source and a join stage (sinks omitted)
        stages_per_node = math.trunc(math.ceil(num_stages / num_nodes))
        source_node_id  = utils.CLUSTER_NODE_PREFIX + "0"
        placement["Auctions Source"] = [source_node_id] * auction_sources
        placement["Persons Source"] = [source_node_id] * person_sources
        id = 1 // stages_per_node
        join_node_id = utils.CLUSTER_NODE_PREFIX + str(id)
        placement["Join Auctions-Persons"] = [join_node_id] * join_instances
        placement["sink"] = [join_node_id] * join_instances
    else:  # Connect to existing cluster
        if pin_processes:
            utils.pin_processes()
        ray.init(redis_address="localhost:6379")
        if not placement_file:
            sys.exit("No actor placement specified.")
        node_ids = utils.get_cluster_node_ids()
        logger.info("Found {} cluster nodes.".format(len(node_ids)))
        placement = utils.parse_placement(placement_file, node_ids)

    # Use pickle for BatchedQueue
    ray.register_custom_serializer(BatchedQueue, use_pickle=True)

    # Batched queue configuration
    queue_config = QueueConfig(max_queue_size,
                        max_batch_size, batch_timeout,
                        prefetch_depth, background_flush)

    # Create streaming environment, construct and run dataflow
    env = Environment()
    env.set_queue_config(queue_config)
    if logging:
        env.enable_logging()
    if task_based:
        env.enable_tasks()

    # Construct the auction source objects (all read from the same file)
    source_objects = [dg.NexmarkEventGenerator(auctions_file, "Auction",
                                               auctions_rate,
                                               sample_period=sample_period,
                                               max_records=max_records,
                                               omit_extra=omit_extra)
                      for _ in range(auction_sources)]

    # Add auction sources to the dataflow
    auctions_source = env.source(source_objects,
                    name="Auctions Source",
                    batch_size=max_batch_size,
                    placement=placement["Auctions Source"]).set_parallelism(
                                                              auction_sources)
    auctions = auctions_source.partition(lambda auction: auction["seller"])

    # Construct the person source objects (all read from the same file)
    source_objects = [dg.NexmarkEventGenerator(persons_file, "Person",
                                               persons_rate,
                                               sample_period=sample_period,
                                               max_records=max_records,
                                               omit_extra=omit_extra)
                      for _ in range(person_sources)]
    # Add auction sources to the dataflow
    persons_source = env.source(source_objects,
                    name="Persons Source",
                    batch_size=max_batch_size,
                    placement=placement["Persons Source"]).set_parallelism(
                                                              person_sources)
    persons = persons_source.partition(lambda person: person["id"])

    # Add the join
    output = auctions.join(persons,
                           JoinLogic(),  # The custom join logic (see above)
                           name="Join Auctions-Persons",
                           placement=placement[
                                "Join Auctions-Persons"]).set_parallelism(
                                                            join_instances)

    # Add a final custom sink to measure latency if logging is enabled
    output.sink(dg.LatencySink(), name="sink",
                placement=placement["sink"]).set_parallelism(join_instances)

    start = time.time()
    dataflow = env.execute()
    ray.get(dataflow.termination_status())

    # Write log files
    max_queue_size = queue_config.max_size
    max_batch_size = queue_config.max_batch_size
    batch_timeout = queue_config.max_batch_time
    prefetch_depth = queue_config.prefetch_depth
    background_flush = queue_config.background_flush
    auctions_rate = auctions_rate if auctions_rate > 0 else "inf"
    persons_rate = persons_rate if persons_rate > 0 else "inf"
    all = "-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        num_nodes, auctions_rate, auction_sources,
        persons_rate, person_sources,
        num_redis_shards, redis_max_memory, plasma_memory,
        sample_period, logging,
        max_queue_size, max_batch_size, batch_timeout, prefetch_depth,
        background_flush, pin_processes,
        task_based, join_instances
    )
    utils.write_log_files(all, latency_filename,
                          throughput_filename, dump_filename, dataflow)

    logger.info("Elapsed time: {}".format(time.time() - start))

    utils.shutdown_ray(sleep=2)
