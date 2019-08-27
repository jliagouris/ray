from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import time
import types

import ray

import ray.experimental.signal as signal


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

#
# Each Ray actor corresponds to an operator instance in the physical dataflow
# Actors communicate using batched queues as data channels (no standing TCP
# connections)
# Currently, batched queues are based on Eric's implementation (see:
# batched_queue.py)


def _identity(element):
    return element

# Signal denoting that a streaming actor finished processing
# and returned after all its input channels have been closed
class ActorExit(signal.Signal):
    def __init__(self, value=None):
        self.value = value


# Signal denoting that a streaming data source has started emitting records
class ActorStart(signal.Signal):
    def __init__(self, value=None):
        self.value = value


class OperatorInstance(object):
    """A streaming operator instance.

    Attributes:
        instance_id (str(UUID)): The id of the instance.
        operator_metadata (Operator): The operator metadata (see: operator.py)
        input (DataInput): The input gate that manages input channels of
        the instance (see: DataInput in communication.py).
        output (DataOutput): The output gate that manages output channels of
        the instance (see: DataOutput in communication.py).
        checkpoint_dir (str): The checkpoints directory
    """

    def __init__(self, instance_id, operator_metadata, input_gate, output_gate,
                 checkpoint_dir):
        self.instance_id = instance_id  # (Operator id, local instance id)
        self.metadata = operator_metadata
        self.input = input_gate
        self.output = output_gate
        self.this_actor = None  # Owns actor handle

        self.key_index = None  # Index for key selection
        self.key_attribute = None  # Attribute name for key selection

        self.num_records_seen = 0  # Number of records seen by the actor

    # Used for index-based key extraction, e.g. for tuples
    def _index_based_selector(self, record):
        return record[self.key_index]

    # Used for attribute-based key extraction, e.g. for classes
    def _attribute_based_selector(self, record):
        return record[self.key_attribute]

    # Registers own actor handle
    def _register_handle(self, actor_handle):
        self.this_actor = actor_handle

    # Registers the handle of a destination actor to an output channel
    def _register_destination_handle(self, actor_handle, channel_id):
        for channel in self.output.forward_channels:
            if channel.id == channel_id:
                channel._register_destination_actor(actor_handle)
                return
        for channels in self.output.shuffle_channels:
            for channel in channels:
                if channel.id == channel_id:
                    channel._register_destination_actor(actor_handle)
                    return
        for channels in self.output.shuffle_key_channels:
            for channel in channels:
                if channel.id == channel_id:
                    channel._register_destination_actor(actor_handle)
                    return
        for channels in self.output.round_robin_channels:
            for channel in channels:
                if channel.id == channel_id:
                    channel._register_destination_actor(actor_handle)
                    return
        # TODO (john): Handle custom partitioning channels

    # Closes an input channel (called by the upstream operator instance)
    def _close_input(self, channel_id):
        """ Closes an input channel and exits if all inputs have been
        closed.

        Attributes:
            channel_id (str(UUID)): The id of the input 'channel' to close.
        """
        if self.input._close_channel(channel_id):
            # logger.debug("Closing channel {}".format(channel_id))
            self.output._close()
            signal.send(ActorExit(self.instance_id))

    # This method must be implemented by the subclasses
    def _apply(self, batch, input_channel_id=None):
        """ Applies the user-defined operator logic to a batch of records.

        Attributes:
            batch (list): The batch of input records
            input_channel_id (str(UUID)): The id of the input 'channel' the
            batch comes from.
        """
        raise Exception("OperatorInstances must implement _apply()")

    # Used for index-based key extraction, e.g. for tuples
    def index_based_selector(self, record):
        return record[self.key_index]

    # Used for attribute-based key extraction, e.g. for classes
    def attribute_based_selector(self, record):
        return vars(record)[self.key_attribute]


# Map actor
@ray.remote
class Map(OperatorInstance):
    """A map operator instance that applies a user-defined
    stream transformation.

    A map produces exactly one output record for each record in
    the input stream.

    Attributes:
        map_fn (function): The user-defined function.
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        self.map_fn = operator_metadata.logic

    # Applies map logic on a batch of records (one record at a time)
    def _apply(self, batch, input_channel_id=None):
        for record in batch:
            self.output._push(self.map_fn(record))


# Flatmap actor
@ray.remote
class FlatMap(OperatorInstance):
    """A map operator instance that applies a user-defined
    stream transformation.

    A flatmap produces zero or more output records for each record in
    the input stream.
    """

    def __init__(self, instance_id, operator_metadata, input_gate, output_gate,
                 checkpoint_dir):
        OperatorInstance.__init__(self, instance_id, operator_metadata,
                                  input_gate, output_gate, checkpoint_dir)
        # The user-defined flatmap function
        self.flatmap_fn = operator_metadata.logic
        self.max_batch_size = input_gate[0].queue_config.max_batch_size

    # TODO (john): Batches generated by the flatmap may exceed max_batch_size
    # and should be pushed to the output gradually, as the records are being
    # produced
    
    # Applies flatmap logic on a batch of records
    def _apply(self, batch, input_channel_id=None):
        for record in batch:
            self.output._push_batch(self.flatmap_fn(record))


@ray.remote
class KeyBy(OperatorInstance):
    """A key_by operator instance that physically partitions the
    stream based on a key.

    The key_by actor transforms the input data stream or records into a new
    stream of tuples of the form (key, record).

    Attributes:
        key_attribute (int): The index of the value to reduce
        (assuming tuple records).
    """

    def __init__(self, instance_id, operator_metadata, input_gate, output_gate,
                 checkpoint_dir):
        OperatorInstance.__init__(self, instance_id, operator_metadata,
                                  input_gate, output_gate, checkpoint_dir)
        # Set the key selector
        self.key_selector = operator_metadata.key_selector
        if isinstance(self.key_selector, int):
            self.key_index = self.key_selector
            self.key_selector = self._index_based_selector
        elif isinstance(self.key_selector, str):
            self.key_attribute = self.key_selector
            self.key_selector = self._attribute_based_selector
        elif not isinstance(self.key_selector, types.FunctionType):
            raise Exception("Unrecognized or unsupported key selector.")

    # Extracts the shuffling key from each record in a batch
    def _apply(self, batch, input_channel_id=None):
        batch[:] = [(self.key_selector(record), record) for record in batch]
        self.output._push_batch(batch)


# A custom source actor
@ray.remote
class Source(OperatorInstance):
    """A source emitting records.

    A user-defined source object must implement the following methods:
        - init()
        - get_next(batch_size)
        - close()
    """

    def __init__(self, instance_id, operator_metadata, input_gate, output_gate,
                 checkpoint_dir):
        OperatorInstance.__init__(self, instance_id, operator_metadata,
                                  input_gate, output_gate, checkpoint_dir)
        # The user-defined source
        self.source = operator_metadata.source
        self.source.init()  # Initialize the source

    # Starts the source
    def start(self):
        signal.send(ActorStart(self.instance_id))
        self.start_time = time.time()
        batch_size = self.output.max_batch_size
        while True:
            record_batch = self.source.get_next(batch_size)
            if record_batch is None:  # Source is exhausted
                logger.info("Source throuhgput: {}".format(
                    self.num_records_seen / (time.time() - self.start_time)))
                self.output._close()  # Flush and close output
                signal.send(ActorExit(self.instance_id))  # Send exit signal
                self.source.close()
                return
            self.output._push_batch(record_batch)
