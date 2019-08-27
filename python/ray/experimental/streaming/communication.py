from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import sys
import time
import uuid

from ray.experimental.streaming.operator import PStrategy

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Round robin and rescale partitioning strategies (default)
round_robin_strategies = [PStrategy.RoundRobin, PStrategy.Rescale]

# Forward and broadcast stream partitioning strategies
forward_broadcast_strategies = [PStrategy.Forward, PStrategy.Broadcast]


# Generates UUIDs
def _generate_uuid():
    return str(uuid.uuid4())


# Used to choose output channel in case of hash-based shuffling
def _hash(value):
    if isinstance(value, int):
        return value
    try:
        return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16)
    except AttributeError:
        return int(hashlib.sha1(value).hexdigest(), 16)


# Virtual queue configuration
class QueueConfig(object):
    """The configuration of a virtual queue between two operator instances.

    Attributes:
         max_size (int): The maximum size of the queue in number of batches
         (if exceeded, backpressure kicks in).
         max_batch_size (int): The maximum size of each batch in number of
         records.
         max_batch_time (float): The flush timeout per batch. Note that if
         there is no single record added to the queue for a time period larger
         than the timeout, then the timeout can be violated, as there is no
         background flush thread).
    """

    def __init__(
            self,
            max_size=100,         # Size in number of batches
            max_batch_size=1000,  # Size in number of records
            max_batch_time=0.1):  # Time in secs
        self.max_size = max_size
        self.max_batch_size = max_batch_size
        self.max_batch_time = max_batch_time


# A data channel between two operator instances in a streaming environment
class DataChannel(object):
    """A virtual data channel for actor-to-actor communication.

    Attributes:
         env (Environment): The streaming environment the channel belongs to.
         src_operator_id (str(UUID)): The id of the source operator of the
         channel.
         dst_operator_id (str(UUID)): The id of the destination operator of
         the channel.
         src_instance_id (int): The local id of the source instance.
         dst_instance_id (int): The local id of the destination instance.
    """

    def __init__(self, env_config, src_operator_id, dst_operator_id,
                 src_instance_id, dst_instance_id):
        self.queue_config = env_config.queue_config
        # Actor handles
        self.source_actor = None
        self.destination_actor = None
        # IDs
        self.src_operator_id = src_operator_id
        self.dst_operator_id = dst_operator_id
        self.src_instance_id = src_instance_id
        self.dst_instance_id = dst_instance_id
        self.src_actor_id = (self.src_operator_id, self.src_instance_id)
        self.dst_actor_id = (self.dst_operator_id, self.dst_instance_id)
        self.id = _generate_uuid()

        self.last_flush_time = 0
        self.num_batches_sent = 0
        self.task_queue = []    # A list of pending downstream tasks (futures)
        self.write_buffer = []  # A list of records to push downstream

    # For pretty print
    def __repr__(self):
        return "DataChannel({},{},{},{})".format(
            self.src_operator_id, self.dst_operator_id, self.src_instance_id,
            self.dst_instance_id)

    # Registers source actor handle (for recovery purposes)
    def _register_source_actor(self, actor_handle):
        self.source_actor = actor_handle

    # Registers destination actor handle (to schedule tasks at destination)
    def _register_destination_actor(self, actor_handle):
        self.destination_actor = actor_handle

    # Pushes a record to the output buffer
    def _push_next(self, record):
        """Pushes a record into the output buffer (and flushes if needed)."""
        if not self.write_buffer:  # Reset last flush time for the new batch
            self.last_flush_time = time.time()
        self.write_buffer.append(record)
        # If buffer is full...
        if (len(self.write_buffer) >= self.queue_config.max_batch_size):
            self._flush_writes()
            return True
        # ...or if the timeout expired
        delay = time.time() - self.last_flush_time
        if delay >= self.queue_config.max_batch_time:
            self._flush_writes()
            return True
        return False

    # Pushes a batch of records to the destination
    def _push_next_batch(self, batch):
        """Pushes a batch of records to the destination operator instance."""
        args = [batch, self.id]
        obj_id = self.destination_actor._apply._remote(
                    args=args, kwargs={}, num_return_vals=1)
        self.task_queue.append(obj_id)
        self._wait_for_consumer()
        self.last_flush_time = time.time()
        return True

    # Pushes all pending data to the destination
    def _flush_writes(self):
        """Pushes data to the destination operator instance via a new task."""
        if not self.write_buffer:
            return
        # Schedule a new task at the destination
        args = [self.write_buffer, self.id]
        obj_id = self.destination_actor._apply._remote(
                    args=args, kwargs={}, num_return_vals=1)
        self.task_queue.append(obj_id)
        self.write_buffer = []
        self._wait_for_consumer()
        self.last_flush_time = time.time()

    # Simulates backpressure based on the virtual queue configuration
    def _wait_for_consumer(self):
        """Simulates backpressure by the destination operator instance."""
        if self.queue_config.max_size <= 0:  # Unlimited queue
            return
        if len(self.task_queue) <= self.queue_config.max_size:
            return  # Has not exceeded max size
        # Check pending downstream tasks
        _, self.task_queue = ray.wait(
            self.task_queue, num_returns=len(self.task_queue),
            timeout=0)  # Return immediately
        # Wait for downstream operator instance to catch up
        while len(self.task_queue) > self.queue_config.max_size:
            # logger.debug("Waiting for ({},{}) to catch up".format(
            #              self.dst_operator_id, self.dst_instance_id))
            _, self.task_queue = ray.wait(
                self.task_queue,
                num_returns=len(self.task_queue),
                timeout=0.01)  # Wait for 10ms before calling wait() again

    # Closes the channel at the destination. Channels in a dataflow are
    # closed from sources to sinks when the input is exhuasted.
    def _close(self):
        """Schedules a task at the destination to close its input."""
        args = [self.id]
        obj_id = self.destination_actor._close_input._remote(
            args=args, kwargs={}, num_return_vals=1)
        assert obj_id is not None


# An input gate that maintains information about input channels
class DataInput(object):
    """An input gate of an operator instance.

    The input gate contains information about all input channels of an
    operator instance.

    Attributes:
         channels (list): The list of input channels.
    """

    def __init__(self, channels):
        self.input_channels = channels
        self.closed_channels = []  # Drained input data streams

    # Registers an input channel as closed and returns True
    # if all input channels are closed, False otherwise
    def _close_channel(self, channel_id):
        assert channel_id not in self.closed_channels
        self.closed_channels.append(channel_id)
        return len(self.closed_channels) == len(self.input_channels)


# An input gate that maintains information about input channels. It selects
# output channel(s) to push data according to the partitioning strategy
class DataOutput(object):
    """An output gate of an operator instance.

    The output gate pushes records to output channels according to the
    user-defined partitioning scheme.

    Upon initialization of the output gate, the output channels are grouped by
    type as follows:
        - forward_channels (list): A list of output channels to simply forward
          records.
        - shuffle_channels (list(list)): A list of output channels to shuffle
          records grouped by destination operator.
        - shuffle_key_channels (list(list)): A list of output channels to
          shuffle records by a key grouped by destination operator.

    Attributes:
         channels (list): The list of all output channels
         partitioning_schemes (dict): A mapping from destination operator ids
         to partitioning schemes (see: PScheme in operator.py).
    """

    def __init__(self, channels, partitioning_schemes):
        self.key_selector = None
        self.partitioning_schemes = partitioning_schemes

        # Prepare output -- collect channels by type
        self.forward_channels = []  # Forward and broadcast channels
        slots = sum(1 for scheme in self.partitioning_schemes.values()
                    if scheme.strategy in round_robin_strategies)
        self.round_robin_channels = [[]] * slots  # Round-robin channels
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

        # Distinct round robin destinations
        round_robin_destinations = {}
        # Distinct shuffle destinations
        shuffle_destinations = {}
        # Distinct shuffle by key destinations
        shuffle_by_key_destinations = {}
        index_1 = 0
        index_2 = 0
        index_3 = 0
        # Group output channels by type
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
            elif strategy in round_robin_strategies:
                pos = round_robin_destinations.setdefault(
                    channel.dst_operator_id, index_3)
                self.round_robin_channels[pos].append(channel)
                if pos == index_3:
                    index_3 += 1
            else:  # TODO (john): Handle custom partitioning
                message = "Unrecognized or unsupported partitioning strategy."
                raise Exception(message)
        # Change round-robin strategy to simple
        # forward if there is only one channel
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
        # A keyed data stream can only be shuffled by key
        assert not (self.shuffle_key_exists and self.shuffle_exists)

    # Returns all destination actor ids
    def _destination_actor_ids(self):
        """Returns all destination actor (operator instance) ids.

        Actor ids have the form (operator id, local instance id), where
        'local instance id' is in [0...number of operator instances).
        """
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
        # TODO (john): Add custom partitioning channels
        return destinations

    # Flushes any remaining records and closes all output channels
    def _close(self):
        """Flushes remaining output records in the virtual output queues and
        marks all output channels as closed.

        This method schedules new tasks at the destination instances to close
        their respective inputs.
        """
        for channel in self.forward_channels:
            channel._flush_writes()
            channel._close()
        for channels in self.shuffle_channels:
            for channel in channels:
                channel._flush_writes()
                channel._close()
        for channels in self.shuffle_key_channels:
            for channel in channels:
                channel._flush_writes()
                channel._close()
        for channels in self.round_robin_channels:
            for channel in channels:
                channel._flush_writes()
                channel._close()
        for channels in self.custom_partitioning_channels:
            for channel in channels:
                channel._flush_writes()
                channel._close()

    # Used for record-at-a-time operators
    def _push(self, record):
        """Pushes a record to the output buffers."""
        # Simple forwarding
        for channel in self.forward_channels:
            # logger.debug("Actor ({},{}) pushed '{}' to channel {}.".format(
            #    channel.src_operator_id, channel.src_instance_id, record,
            #    channel))
            channel._push_next(record)
        # Round-robin forwarding
        for i, channels in enumerate(self.round_robin_channels):
            channel_index = self.round_robin_indexes[i]
            flushed = channels[channel_index]._push_next(record)
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
                channel._push_next(record)
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            h = _hash(record)
            for channels in self.shuffle_channels:
                num_instances = len(channels)  # Downstream instances
                channel = channels[h % num_instances]
                # logger.debug(
                #     "Actor ({},{}) pushed '{}' to channel {}.".format(
                #     channel.src_operator_id, channel.src_instance_id,
                #     record, channel))
                channel._push_next(record)
        # TODO (john): Add custom partitioning channels

    # Pushes a list of records to the output
    def _push_all(self, records):
        # Forward records
        for record in records:
            for channel in self.forward_channels:
                # log_message = "Actor ({},{}) pushed '{}' to channel {}."
                # logger.debug(log_message.format(
                #    channel.src_operator_id, channel.src_instance_id, record,
                #    channel))
                channel.put_next(record)
        # Hash-based shuffling by key
        if self.shuffle_key_exists:
            for record in records:
                key, _ = record
                h = _hash(key)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # log_message = "Actor ({},{}) pushed '{}' to channel {}."
                    # logger.debug(log_message.format(
                    #    channel.src_operator_id, channel.src_instance_id,
                    #    record, channel))
                    channel.put_next(record)
        elif self.shuffle_exists:  # Hash-based shuffling per destination
            for record in records:
                h = _hash(record)
                for channels in self.shuffle_channels:
                    num_instances = len(channels)  # Downstream instances
                    channel = channels[h % num_instances]
                    # log_message = "Actor ({},{}) pushed '{}' to channel {}."
                    # logger.debug(log_message.format(
                    #    channel.src_operator_id, channel.src_instance_id,
                    #    record, channel))
                    channel.put_next(record)
        else:  # TODO (john): Handle custom partitioning channels
            pass
