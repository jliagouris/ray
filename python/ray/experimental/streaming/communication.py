from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import hashlib
import logging
import sys
import time
import uuid

from ray.experimental.streaming.operator import PStrategy
from ray.experimental.streaming.batched_queue import BatchedQueue
from ray.experimental.streaming.benchmarks.macro.nexmark.event import Record
from ray.experimental.streaming.benchmarks.macro.nexmark.event import Watermark

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

LOGGING_PERIOD = 100000  # Log throughput every 100K records

# Generates UUIDs
def _generate_uuid():
    return str(uuid.uuid4())

# Forward and broadcast stream partitioning strategies
forward_broadcast_strategies = [PStrategy.Forward, PStrategy.Broadcast]

# Round robin and rescale partitioning strategies (default)
round_robin_strategies = [PStrategy.RoundRobin, PStrategy.Rescale]

# Used to choose output channel in case of hash-based shuffling
# TODO (john): Replace pickle
def _hash(value):
    if isinstance(value, int):
        return value
    elif isinstance(value,str):
        return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16)
    else: # All other data types
        try:  # Try hashing the value
            return int(hashlib.sha1(value).hexdigest(), 16)
        except TypeError:  # Serialize object and hash
            pickled = pickle.dumps(value)
            return int(hashlib.sha1(pickled).hexdigest(), 16)


# A data channel is a batched queue between two
# operator instances in a streaming environment
class DataChannel(object):
    """A data channel for actor-to-actor communication.

    Attributes:
         env (Environment): The environment the channel belongs to.
         src_operator_id (UUID): The id of the source operator of the channel.
         dst_operator_id (UUID): The id of the destination operator of the
         channel.
         src_instance_id (int): The id of the source instance.
         dst_instance_id (int): The id of the destination instance.
         queue (BatchedQueue): The batched queue used for data movement.
    """

    def __init__(self, env_config, src_operator_id, dst_operator_id,
                 src_instance_id, dst_instance_id):
        self.queue_config = env_config.queue_config
        self.src_operator_id = src_operator_id
        self.dst_operator_id = dst_operator_id
        self.src_instance_id = src_instance_id
        self.dst_instance_id = dst_instance_id
        self.src_actor_id = (self.src_operator_id, self.src_instance_id)
        self.dst_actor_id = (self.dst_operator_id, self.dst_instance_id)
        self.id = _generate_uuid()
        self.source_actor = None
        self.destination_actor = None
        # Construct queue
        self.queue = BatchedQueue(
            self.id,
            self.src_operator_id, self.src_instance_id,
            self.dst_operator_id, self.dst_instance_id,
            max_size=self.queue_config.max_size,
            max_batch_size=self.queue_config.max_batch_size,
            max_batch_time=self.queue_config.max_batch_time,
            prefetch_depth=self.queue_config.prefetch_depth,
            background_flush=self.queue_config.background_flush,
            task_based=env_config.task_based)

    # Registers source actor handle
    def register_source_actor(self, actor_handle):
        self.source_actor = actor_handle
        self.queue.register_source_actor(actor_handle)

    # Registers destination actor handle
    def register_destination_actor(self, actor_handle):
        self.destination_actor = actor_handle
        self.queue.register_destination_actor(actor_handle)

    def __repr__(self):
        return "({},{},{},{})".format(
            self.src_operator_id, self.dst_operator_id, self.src_instance_id,
            self.dst_instance_id)

# Pulls and merges data from multiple input channels
class DataInput(object):
    """An input gate of an operator instance.

    The input gate pulls records from all input channels in a round-robin
    fashion.

    Attributes:
         input_channels (list): The list of input channels.
         channel_index (int): The index of the next channel to pull from.
         max_index (int): The number of input channels.
         closed (list): A list of flags indicating whether an input channel
         has been marked as 'closed'.
         all_closed (bool): Denotes whether all input channels have been
         closed (True) or not (False).
    """

    def __init__(self, channels):
        self.multiple_inputs = False
        self.input_channels = channels
        self.channel_index = 0
        self.max_index = len(channels)
        self.closed = [False] * len(
            self.input_channels)  # Tracks the channels that have been closed
        self.all_closed = False
        self.closed_channels = [] # Used to terminate a task-based execution

        # Logging-related attributes
        self.logging = False    # Default
        self.records = 0        # Counter
        self.rates = []         # Input rates
        self.start = 0.0        # Start timestamp
        # Measure input rate every RECORDS_PER_ROUND
        self.period = LOGGING_PERIOD

    # Groups input channels per logical stream so that the operator can
    # distinguish between inputs and pull from channels of a particular stream
    # TODO (john): Make this to work for queue-based execution
    def _group_channels_per_stream(self, source_operator_ids):
        self.multiple_inputs = True
        temp_channels = [[None]] * len(source_operator_ids)
        for channel in self.input_channels:
            index = source_operator_ids.index(channel.src_operator_id)
            temp_channels[index].append(channel)
        # Set flags
        self.closed = [None] * len(source_operator_ids)
        for i, channels in enumerate(temp_channels):
            flags = [False] * len(channels)
            self.closed[i] = flags
        # Set input channels
        self.input_channels = temp_channels

    # Fetches records from input channels in a round-robin fashion
    # TODO (john): Make sure the instance is not blocked on any of its input
    # channels
    # TODO (john): In case of input skew, it might be better to pull from
    # the largest queue more often
    def _pull(self, index):
        # TODO (john): Multi-input operators do not work with queues for now
        assert(self.multiple_inputs is False), (self.multiple_inputs)
        while True:
            if self.max_index == 0:
                # TODO (john): We should detect this earlier
                return None
            # Channel to pull from
            channel = self.input_channels[self.channel_index]
            self.channel_index += 1
            self.channel_index %= self.max_index
            if self.closed[self.channel_index -1]:
                continue  # Channel has been 'closed', check next
            record = channel.queue.read_next()
            # logger.debug("Actor ({},{}) pulled '{}' from channel {}.".format(
            #    channel.dst_operator_id, channel.dst_instance_id, record,
            #    channel))
            if record is None:
                # Mark channel as 'closed' and pull from the next open one
                self.closed[self.channel_index - 1] = True
                self.all_closed = True
                for flag in self.closed:
                    if flag is False:
                        self.all_closed = False
                        break
                if not self.all_closed:
                    continue
            if self.logging:
                if record is not None:  # Log throughput
                    self.__log(batch_size=1)
                else:
                    self.__log(batch_size=0, force=True)
            # Returns 'None' iff all input channels are 'closed'
            return record

    # Registers an input channel as 'closed'
    # Returns True if all channels are closed, False otherwise
    def _close_channel(self, channel_id):
        assert channel_id not in self.closed_channels
        self.closed_channels.append(channel_id)
        return len(self.closed_channels) == self.max_index

    # Enables throughput logging on output channels
    def enable_logging(self):
        self.logging = True

    # Logs input rate
    def __log(self, batch_size=0, force=False):
        # Log throughput every N records
        if not self.start:
            self.start = time.time()
            time.sleep(0.001)
        self.records += batch_size
        if self.records >= self.period or (
           force is True and self.records > 0):
            self.rates.append(self.records / (time.time() - self.start))
            self.records = 0
            self.start = time.time()

# Selects output channel(s) and pushes data
class DataOutput(object):
    """An output gate of an operator instance.

    The output gate pushes records to output channels according to the
    user-defined partitioning scheme.

    Attributes:
         partitioning_schemes (dict): A mapping from destination operator ids
         to partitioning schemes (see: PScheme in operator.py).
         forward_channels (list): A list of channels to forward records.
         shuffle_channels (list(list)): A list of output channels to shuffle
         records grouped by destination operator.
         shuffle_key_channels (list(list)): A list of output channels to
         shuffle records by a key grouped by destination operator.
         shuffle_exists (bool): A flag indicating that there exists at least
         one shuffle_channel.
         shuffle_key_exists (bool): A flag indicating that there exists at
         least one shuffle_key_channel.
    """

    def __init__(self, channels, partitioning_schemes):
        self.input_channels = {}  # Channel id -> Watermark
        # TODO (john): Should be equal to the offset
        self.last_emitted_watermark = 0

        self.custom_partitioning_functions = None
        self.channel_index = 0
        self.key_selector = None
        self.partitioning_schemes = partitioning_schemes

        # Prepare output -- collect channels by type
        self.forward_channels = []  # Forward and broadcast channels
        slots = sum(1 for scheme in self.partitioning_schemes.values()
                    if scheme.strategy in round_robin_strategies)
        self.round_robin_channels = [[]] * slots    # RoundRobin channels
        self.round_robin_indexes = [0] * slots
        slots = sum(1 for scheme in self.partitioning_schemes.values()
                    if scheme.strategy == PStrategy.Shuffle)
        # Flag used to avoid hashing when there is no shuffling
        self.shuffle_exists = slots > 0
        self.shuffle_channels = [[]] * slots  # Shuffle channels
        slots = sum(1 for scheme in self.partitioning_schemes.values()
                    if scheme.strategy == PStrategy.ShuffleByKey)
        # Flag used to avoid hashing when there is no shuffling by key
        self.shuffle_key_exists = slots > 0
        self.shuffle_key_channels = [[]] * slots  # Shuffle by key channels
        slots = sum(1 for scheme in self.partitioning_schemes.values()
                    if scheme.strategy == PStrategy.Custom)
        # Flag denoting whether there is a custom defined
        self.custom_partitioning_exists = slots > 0
        # Custom partitioning channels
        self.custom_partitioning_channels = [[]] * slots
        # Custom partitioning functions
        self.custom_partitioning_functions = [
        scheme.partition_fn for scheme in self.partitioning_schemes.values()
                                    if scheme.strategy == PStrategy.Custom]

        # Distinct shuffle destinations
        shuffle_destinations = {}
        # Distinct shuffle by key destinations
        shuffle_by_key_destinations = {}
        # Distinct round robin destinations
        round_robin_destinations = {}
        # Distinct custom partitioning destinations
        custom_partitioning_destinations = {}
        index_1 = 0
        index_2 = 0
        index_3 = 0
        index_4 = 0
        for channel in channels:
            p_scheme = self.partitioning_schemes[channel.dst_operator_id]
            strategy = p_scheme.strategy
            if strategy in forward_broadcast_strategies:
                self.forward_channels.append(channel)
            elif strategy == PStrategy.Shuffle:
                pos = shuffle_destinations.setdefault(channel.dst_operator_id,
                                                      index_1)
                self.shuffle_channels[pos].append(channel)
                if pos == index_1:
                    index_1 += 1
            elif strategy == PStrategy.ShuffleByKey:
                pos = shuffle_by_key_destinations.setdefault(
                    channel.dst_operator_id, index_2)
                self.shuffle_key_channels[pos].append(channel)
                if pos == index_2:
                    index_2 += 1
            elif strategy in round_robin_strategies :
                pos = round_robin_destinations.setdefault(
                    channel.dst_operator_id, index_3)
                self.round_robin_channels[pos].append(channel)
                if pos == index_3:
                    index_3 += 1
            elif strategy == PStrategy.Custom:  # User-defined partitioning
                pos = custom_partitioning_destinations.setdefault(
                                                    channel.dst_operator_id,
                                                    index_4)
                self.custom_partitioning_channels[pos].append(channel)
                if pos == index_4:
                    index_4 += 1
            else:
                sys.exit("Unrecognized or unsupported partitioning strategy.")

        # Change round robin to simple forward if there is only one channel
        slots_to_remove = []
        slot = 0
        for channels in self.round_robin_channels:
            if len(channels) == 1:
                self.forward_channels.extend(channels)
                slots_to_remove.append(slot)
            slot += 1
        for slot in slots_to_remove:
            self.round_robin_channels.pop(slot)
            self.round_robin_indexes.pop(slot)
        # A KeyedDataStream can only be shuffled by key
        assert not (self.shuffle_exists and self.shuffle_key_exists)

        # Logging-related attributes
        self.logging = False    # Default
        self.records = 0        # Counter
        self.rates = []         # Output rates
        self.start = 0.0        # Start timestamp
        # Measure input rate every RECORDS_PER_ROUND
        self.period = LOGGING_PERIOD

    def save(self):
        queue_states = []
        for channel in self.forward_channels:
            queue_states.append(channel.queue.save())
        for channels in self.shuffle_channels:
            for channel in channels:
                queue_states.append(channel.queue.save())
        for channels in self.shuffle_key_channels:
            for channel in channels:
                queue_states.append(channel.queue.save())
        for channels in self.round_robin_channels:
            for channel in channels:
                queue_states.append(channel.queue.save())
            queue_states.append(self.round_robin_indexes)
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                queue_states.append(channel.queue.save())
        return queue_states

    def load(self, queue_states):
        for channel in self.forward_channels:
            queue_state = queue_states.pop(0)
            channel.queue.load(queue_state)
        for channels in self.shuffle_channels:
            for channel in channels:
                queue_state = queue_states.pop(0)
                channel.queue.load(queue_state)
        for channels in self.shuffle_key_channels:
            for channel in channels:
                queue_state = queue_states.pop(0)
                channel.queue.load(queue_state)
        for channels in self.round_robin_channels:
            for channel in channels:
                queue_state = queue_states.pop(0)
                channel.queue.load(queue_state)
            self.round_robin_indexes = queue_states.pop(0)
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                queue_state = queue_states.pop(0)
                channel.queue.load(queue_state)
        assert len(queue_states) == 0

    def set_checkpoint_epoch(self, checkpoint_epoch):
        self.checkpoint_epoch = checkpoint_epoch
        for channel in self.forward_channels:
            channel.queue.set_checkpoint_epoch(checkpoint_epoch)
        for channels in self.shuffle_channels:
            for channel in channels:
                channel.queue.set_checkpoint_epoch(checkpoint_epoch)
        for channels in self.shuffle_key_channels:
            for channel in channels:
                channel.queue.set_checkpoint_epoch(checkpoint_epoch)
        for channels in self.round_robin_channels:
            for channel in channels:
                channel.queue.set_checkpoint_epoch(checkpoint_epoch)

    # Enables rate logging on output channels
    def enable_logging(self):
        self.logging = True

    # Flushes any remaining records in the output channels
    # 'close' indicates whether we should also 'close' the channel (True)
    # by propagating 'None' or just flush the remaining records to plasma
    def _flush(self, close=False, flush_empty=False, event=None):
        """Flushes remaining output records in the output queues to plasma.

        None is used as special type of record that is propagated from sources
        to sink to notify that the end of data in a stream.

        Attributes:
             close (bool): A flag denoting whether the channel should be
             also marked as 'closed' (True) or not (False) after flushing.
        """
        for channel in self.forward_channels:
            if close is True:
                channel.queue.push_next(None)
            channel.queue._flush_writes(flush_empty=flush_empty, event=event)
        for channels in self.shuffle_channels:
            for channel in channels:
                if close is True:
                    channel.queue.push_next(None)
                channel.queue._flush_writes(flush_empty=flush_empty, event=event)
        for channels in self.shuffle_key_channels:
            for channel in channels:
                if close is True:
                    channel.queue.push_next(None)
                channel.queue._flush_writes(flush_empty=flush_empty, event=event)
        for channels in self.round_robin_channels:
            for channel in channels:
                if close is True:
                    channel.queue.push_next(None)
                channel.queue._flush_writes(flush_empty=flush_empty, event=event)
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                if close is True:
                    channel.queue.push_next(None)
                channel.queue._flush_writes(flush_empty=flush_empty, event=event)

        if self.logging:  # Log rate
            self.__log(force=close)  # force=True only on termination

    # Returns all destination actor ids
    def _destination_actor_ids(self):
        destinations = []
        for channel in self.forward_channels:
            destinations.append((channel.dst_operator_id,
                                 channel.dst_instance_id))
        for channels in self.shuffle_channels:
            for channel in channels:
                destinations.append((channel.dst_operator_id,
                                     channel.dst_instance_id))
        for channels in self.shuffle_key_channels:
            for channel in channels:
                destinations.append((channel.dst_operator_id,
                                     channel.dst_instance_id))
        for channels in self.round_robin_channels:
            for channel in channels:
                destinations.append((channel.dst_operator_id,
                                     channel.dst_instance_id))
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                destinations.append((channel.dst_operator_id,
                                     channel.dst_instance_id))
        return destinations

    # Pushes the record to the output
    # Each individual output queue flushes batches to plasma periodically
    # based on 'batch_max_size' and 'batch_max_time'
    def _push(self, record, input_channel_id=None, event=None):
        if record['event_type'] == 'Watermark':
            self.__forward_watermark(record, input_channel_id)
            return
        # Simple forwarding
        for channel in self.forward_channels:
            # logger.debug("Actor ({},{}) pushed '{}' to channel {}.".format(
            #    channel.src_operator_id, channel.src_instance_id, record,
            #    channel))
            channel.queue.push_next(record, event=event)
        # Round-robin forwarding
        for i, channels in enumerate(self.round_robin_channels):
            channel_index = self.round_robin_indexes[i]
            flushed = channels[channel_index].queue.push_next(record, event=event)
            if flushed:  # Round-robin batches, not individual records
                channel_index += 1
                channel_index %= len(channels)
                self.round_robin_indexes[i] = channel_index
        # Hash-based shuffling by key
        if self.shuffle_key_exists:
            key, _ = record
            h = _hash(key)
            for channels in self.shuffle_key_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.push_next(record, event=event)
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            h = _hash(record)
            for channels in self.shuffle_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.push_next(record, event=event)
        elif self.custom_partitioning_exists:
            for i, channels in enumerate(self.custom_partitioning_channels):
                # Set the right function. In general, there might be multiple
                # destinations, each one with a different custom partitioning
                partitioning_fn = self.custom_partitioning_functions[i]
                h = partitioning_fn(record)
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.push_next(record, event=event)

        if self.logging:  # Log rate
            self.__log(batch_size=1)

    def _push_batch(self, record_batch, input_channel_id=None, event=None):
        assert isinstance(record_batch, list)
        # Simple forwarding
        for channel in self.forward_channels:
            # logger.debug("Actor ({},{}) pushed '{}' to channel {}.".format(
            #    channel.src_operator_id, channel.src_instance_id, record,
            #    channel))
            channel.queue.push_next_batch(record_batch, event=event)
        # Round-robin forwarding
        for i, channels in enumerate(self.round_robin_channels):
            channel_index = self.round_robin_indexes[i]
            flushed = channels[channel_index].queue.push_next_batch(
                                                                record_batch,
                                                                event=event)
            if flushed:  # Round-robin batches, not individual records
                channel_index += 1
                channel_index %= len(channels)
                self.round_robin_indexes[i] = channel_index
        # Hash-based shuffling by key
        if self.shuffle_key_exists:
            raise Exception("Shuffle doesn't work with the source")
            key, _ = record
            h = _hash(key)
            for channels in self.shuffle_key_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.push_next(record, event=event)
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            raise Exception("Shuffle doesn't work with the source")
            h = _hash(record)
            for channels in self.shuffle_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.push_next(record, event=event)
        elif self.custom_partitioning_exists:
            # raise Exception("Shuffle doesn't work with the source")
            for record in record_batch:
                for i, channels in enumerate(self.custom_partitioning_channels):
                    # Set the right function. In general, there might be multiple
                    # destinations, each one with a different custom partitioning
                    partitioning_fn = self.custom_partitioning_functions[i]
                    h = partitioning_fn(record)
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.push_next(record, event=event)

        if self.logging:  # Log rate
            self.__log(batch_size=len(record_batch))

    # Pushes the record to the output
    # Each individual output queue flushes batches to plasma periodically
    # based on 'batch_max_size' and 'batch_max_time'
    def _replay_push(self, record, input_channel_id=None, flush=False):
        if record['event_type'] == 'Watermark':
            self.__forward_watermark(record, input_channel_id)
            return
        # Simple forwarding
        for channel in self.forward_channels:
            # logger.debug("Actor ({},{}) pushed '{}' to channel {}.".format(
            #    channel.src_operator_id, channel.src_instance_id, record,
            #    channel))
            channel.queue.put_next(record)
            if flush:
                channel.queue._flush_writes()
        # Round-robin forwarding
        for i, channels in enumerate(self.round_robin_channels):
            channel_index = self.round_robin_indexes[i]
            channels[channel_index].queue.put_next(record)
            if flush:
                channel.queue._flush_writes()
                channel_index += 1
                channel_index %= len(channels)
                self.round_robin_indexes[i] = channel_index
        # Hash-based shuffling by key
        if self.shuffle_key_exists:
            key, _ = record
            h = _hash(key)
            for channels in self.shuffle_key_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.put_next(record)
                if flush:
                    channel.queue._flush_writes()
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            h = _hash(record)
            for channels in self.shuffle_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.put_next(record)
                if flush:
                    channel.queue._flush_writes()
        elif self.custom_partitioning_exists:
            for i, channels in enumerate(self.custom_partitioning_channels):
                # Set the right function. In general, there might be multiple
                # destinations, each one with a different custom partitioning
                partitioning_fn = self.custom_partitioning_functions[i]
                h = partitioning_fn(record)
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel.queue.put_next(record)
                if flush:
                    channel.queue._flush_writes()

        if self.logging:  # Log rate
            self.__log(batch_size=1)

    # Maintains the last received watermark per input channel
    def __forward_watermark(self, watermark, input_channel_id):
        # logger.info("Previous watermark: {}".format(
        #             self.last_emitted_watermark))
        if input_channel_id is None: # Must be a source
            self.last_emitted_watermark = watermark['event_time']
            # logger.info("Source watermark: {}".format(
            #             self.last_emitted_watermark))
            self.__broadcast(watermark)
        else:  # It is an operator with at least one input
            self.input_channels[input_channel_id] = watermark['event_time']
            # Find minimum watermark
            minimum_watermark = min(self.input_channels.values())
            if minimum_watermark > self.last_emitted_watermark:  # Forward
                self.last_emitted_watermark = minimum_watermark
                watermark['event_time'] = minimum_watermark
                # logger.info("Emitting watermark: {}".format(
                #             self.last_emitted_watermark))
                self.__broadcast(watermark)

    # Broadcasts a watermark to all output channels
    def __broadcast(self, watermark):
        for channel in self.forward_channels:
            channel.queue.push_next(watermark)
        for channels in self.shuffle_channels:
            for channel in channels:
                channel.queue.push_next(watermark)
        for channels in self.shuffle_key_channels:
            for channel in channels:
                channel.queue.push_next(watermark)
        for channels in self.round_robin_channels:
            for channel in channels:
                channel.queue.push_next(watermark)
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                channel.queue.push_next(watermark)

    # Pushes a list of records to the output
    # Each individual output queue flushes batches to plasma periodically
    # based on the configured batch size and timeout
    def _push_all(self, records, input_channel_id=None, event=None):
        assert isinstance(records, list)
        # Forward records
        for record in records:
            if record['event_type'] == "Watermark":
                self.__forward_watermark(record, input_channel_id)
                return
            for channel in self.forward_channels:
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #      record, channel))
                channel.queue.push_next(record, event=event)
            # Round-robin forwarding
            for i, channels in enumerate(self.round_robin_channels):
                channel_index = self.round_robin_indexes[i]
                flushed = channels[channel_index].queue.push_next(record, event=event)
                if flushed:  # Round-robin batches, not individual records
                    channel_index += 1
                    channel_index %= len(channels)
                    self.round_robin_indexes[i] = channel_index
        # Hash-based shuffling by key per destination
        if self.shuffle_key_exists:
            for record in records:
                if record['event_type'] == "Watermark":
                    self.__forward_watermark(record, input_channel_id)
                    return
                key, _ = record
                h = _hash(key)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    # "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.push_next(record, event=event)
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            for record in records:
                if record['event_type'] == 'Watermark':
                    self.__forward_watermark(record, input_channel_id)
                    return
                h = _hash(record)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.push_next(record, event=event)
        elif self.custom_partitioning_exists:
            for record in records:
                if record['event_type'] == 'Watermark':
                    self.__forward_watermark(record, input_channel_id)
                    return
                for i, channels in enumerate(
                                    self.custom_partitioning_channels):
                    # Set the right function. In general, there might be
                    # multiple destinations, each one with a different custom
                    # partitioning
                    partitioning_fn = self.custom_partitioning_functions[i]
                    h = partitioning_fn(record)
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.push_next(record, event=event)

        if self.logging:  # Log rate
            self.__log(batch_size=len(records))

    # Pushes a list of records to the output
    # Each individual output queue flushes batches to plasma periodically
    # based on the configured batch size and timeout
    def _replay_push_all(self, records, input_channel_id=None, event=None):
        assert isinstance(records, list)
        # Forward records
        for record in records:
            if record['event_type'] == 'Watermark':
                self.__forward_watermark(record, input_channel_id)
                return
            for channel in self.forward_channels:
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #      record, channel))
                channel.queue.put_next(record)
                if flush:
                    channel.queue._flush_writes()
            # Round-robin forwarding
            for i, channels in enumerate(self.round_robin_channels):
                channel_index = self.round_robin_indexes[i]
                channels[channel_index].queue.put_next(record)
                if flush:
                    channel.queue._flush_writes()
                    channel_index += 1
                    channel_index %= len(channels)
                    self.round_robin_indexes[i] = channel_index
        # Hash-based shuffling by key per destination
        if self.shuffle_key_exists:
            for record in records:
                if record['event_type'] == 'Watermark':
                    self.__forward_watermark(record, input_channel_id)
                    return
                key, _ = record
                h = _hash(key)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    # "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.put_next(record)
                    if flush:
                        channel.queue._flush_writes()
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            for record in records:
                if record['event_type'] == 'Watermark':
                    self.__forward_watermark(record, input_channel_id)
                    return
                h = _hash(record)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.put_next(record)
                    if flush:
                        channel.queue._flush_writes()
        elif self.custom_partitioning_exists:
            for record in records:
                if record['event_type'] == 'Watermark':
                    self.__forward_watermark(record, input_channel_id)
                    return
                for i, channels in enumerate(
                                    self.custom_partitioning_channels):
                    # Set the right function. In general, there might be
                    # multiple destinations, each one with a different custom
                    # partitioning
                    partitioning_fn = self.custom_partitioning_functions[i]
                    h = partitioning_fn(record)
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # logger.debug(
                    #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                    #     channel.src_operator_id, channel.src_instance_id,
                    #     record, channel))
                    channel.queue.put_next(record)
                    if flush:
                        channel.queue._flush_writes()

        if self.logging:  # Log rate
            self.__log(batch_size=len(records))

    # Logs output rate
    def __log(self, batch_size=0, force=False):
        # Log throughput every N records
        if not self.start:
            self.start = time.time()
            time.sleep(0.001)
        self.records += batch_size
        if self.records >= self.period or (
           force is True and self.records > 0):
            rate = self.records / (time.time() - self.start)
            self.rates.append(rate)
            logger.info("THROUGHPUT:%f", rate)
            self.records = 0
            self.start = time.time()


# Batched queue configuration
class QueueConfig(object):
    """The configuration of a batched queue.

    Attributes:
         max_size (int): The maximum size of the queue in number of batches
         (if exceeded, backpressure kicks in).
         max_batch_size (int): The size of each batch in number of records.
         max_batch_time (float): The flush timeout per batch.
         prefetch_depth (int): The  number of batches to prefetch from plasma.
         background_flush (bool): Denotes whether a daemon flush thread should
         be used (True) to flush batches to plasma.
    """

    def __init__(self,
                 max_size=1000,
                 max_batch_size=100,
                 max_batch_time=0.1,
                 prefetch_depth=1,
                 background_flush=False):
        self.max_size = max_size
        self.max_batch_size = max_batch_size
        self.max_batch_time = max_batch_time
        self.prefetch_depth = prefetch_depth
        self.background_flush = background_flush
