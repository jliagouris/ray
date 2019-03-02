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
logging.basicConfig(level=logging.INFO)

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

class OperatorInstance(object):
    """A streaming operator instance.

    Attributes:
        instance_id (UUID): The id of the instance.
        input (DataInput): The input gate that manages input channels of
        the instance (see: DataInput in communication.py).
        input (DataOutput): The output gate that manages output channels of
        the instance (see: DataOutput in communication.py).
        config (EnvironmentConfig): The environment's configuration.
    """
    # TODO (john): Keep all operator metadata and environment's configuration
    # for debugging purposes
    def __init__(self,
                 instance_id,
                 input_gate,
                 output_gate,
                 config=None):
        self.key_index = None       # Index for key selection
        self.key_attribute = None   # Attribute name for key selection
        self.instance_id = instance_id
        self.input = input_gate
        self.output = output_gate
        # Parameters for periodic rescheduling
        if not config:  # Actor will spin continuously until termination
            self.records_limit = float("inf")   # Unlimited
            self.scheduling_timeout = None      # No timeout
        else:  # Actor will be rescheduled periodically
            self.records_limit = config.scheduling_period_in_records
            self.scheduling_timeout = config.scheduling_timeout
        self.records_processed = 0
        self.previous_scheduling_time = time.time()
        # Enable writes to all output channels
        for channel in self.output.forward_channels:
            channel.queue.enable_writes()
        for channels in self.output.shuffle_channels:
            for channel in channels:
                channel.queue.enable_writes()
        for channels in self.output.shuffle_key_channels:
            for channel in channels:
                channel.queue.enable_writes()
        for channels in self.output.round_robin_channels:
            for channel in channels:
                channel.queue.enable_writes()
        # TODO (john): Add more channel types here

    # Used for index-based key extraction, e.g. for tuples
    def index_based_selector(self,record):
        return record[self.key_index]

    # Used for attribute-based key extraction, e.g. for classes
    def attribute_based_selector(self,record):
        return vars(record)[self.key_attribute]

    # Used to register own handle so that the actor can schedule itself
    def register_handle(self,actor_handle):
        self.this_actor = actor_handle

    # Used to periodically stop spinning and reschedule an actor
    # in order to call other methods on it from the outside world
    # Returns True if the actor should stop spinning, False otherwise
    def _reschedule(self):
        reschedule = False
        # Check if number of processed records reached the limit
        if self.records_processed >= self.records_limit:
            reschedule = True
        # Check if there is a timeout
        if self.scheduling_timeout:
            elapsed_time = time.time() - self.previous_scheduling_time
            if elapsed_time >= self.scheduling_timeout:
                reschedule = True
        if reschedule:
            self.this_actor.start.remote()  # Submit task to scheduler
            self.previous_scheduling_time = time.time()
            self.records_processed = 0
        return reschedule

    # Starts the actor
    def start(self):
        pass

# A monitoring actor used to keep track of the execution's progress
@ray.remote
class ProgressMonitor(OperatorInstance):
    """A monitoring actor used to track the progress of
    the dataflow execution.

    Attributes:
        running_actors list(actor handles): A list of handles to all actors
        executing the physical dataflow.
    """

    def __init__(self,running_actors):
        self.running_actors = running_actors
        self.exit_signals = []
        logger.debug("Running actors: {}".format(self.running_actors))

    # Returns all ActorExit signals when the dataflow execution is over
    def all_exit_signals(self):
        while True:
            # Block until the ActorExit signal has been
            # received from each actor executing the dataflow
            signals = signal.receive(self.running_actors)
            if len(signals) > 0:
                self.exit_signals.extend(signals)
            if len(self.exit_signals) == len(self.running_actors):
                return

# A source actor that reads a text file line by line
@ray.remote
class ReadTextFile(OperatorInstance):
    """A source operator instance that reads a text file line by line.

    Attributes:
        filepath (string): The path to the input file.
    """

    def __init__(self,
                 instance_id,
                 operator_metadata,
                 input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        self.filepath = operator_metadata.other_args
        # TODO (john): Handle possible exception here
        self.reader = open(self.filepath, "r")

    # Read input file line by line
    def start(self):
        while True:
            record = self.reader.readline()
            # Reader returns empty string ('') on EOF
            if not record:
                # Flush any remaining records to plasma and close the file
                self.output._flush(close=True)
                self.reader.close()
                signal.send(ActorExit())
                return
            self.output._push(
                record[:-1])  # Push after removing newline characters


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

    # Applies the mapper each record of the input stream(s)
    # and pushes resulting records to the output stream(s)
    def start(self):
        start = time.time()
        elements = 0
        while True:
            record = self.input._pull()
            if record is None:
                self.output._flush(close=True)
                logger.debug("[map {}] read/writes per second: {}".format(
                                        self.instance_id,
                                        elements / (time.time()-start)))
                signal.send(ActorExit())
                return
            self.output._push(self.map_fn(record))
            elements += 1

# Flatmap actor
@ray.remote
class FlatMap(OperatorInstance):
    """A map operator instance that applies a user-defined
    stream transformation.

    A flatmap produces one or more output records for each record in
    the input stream.

    Attributes:
        flatmap_fn (function): The user-defined function.
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        self.flatmap_fn = operator_metadata.logic

    # Applies the splitter to the records of the input stream(s)
    # and pushes resulting records to the output stream(s)
    def start(self):
        while True:
            record = self.input._pull()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            self.output._push_all(self.flatmap_fn(record))


# Filter actor
@ray.remote
class Filter(OperatorInstance):
    """A filter operator instance that applies a user-defined filter to
    each record of the stream.

    Output records are those that pass the filter, i.e. those for which
    the filter function returns True.

    Attributes:
        filter_fn (function): The user-defined boolean function.
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        self.filter_fn = operator_metadata.logic

    # Applies the filter to the records of the input stream(s)
    # and pushes resulting records to the output stream(s)
    def start(self):
        while True:
            record = self.input._pull()
            if record is None:  # Close channel and return
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            if self.filter_fn(record):
                self.output._push(record)


# Union actor
@ray.remote
class Union(OperatorInstance):
    """A union operator instance that concatenates two or more streams."""

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)

    # Simply moves records from input to output repeatedly
    def start(self):
        while True:
            record = self.input._pull()
            if record is None:  # Close channel and return
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            self.output._push(record)


# Inspect actor
@ray.remote
class Inspect(OperatorInstance):
    """A inspect operator instance that inspects the content of the stream.

    Inspect is useful for printing the records in the stream.

    Attributes:
         inspect_fn (function): The user-defined inspect logic.
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        self.inspect_fn = operator_metadata.logic

        # Applies the inspect logic (e.g. print) to the records of
        # the input stream(s)
        # and leaves stream unaffected by simply pushing the records to
        # the output stream(s)
        while True:
            record = self.input._pull()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            self.inspect_fn(record)
            self.output._push(record)

# Reduce actor
@ray.remote
class Reduce(OperatorInstance):
    """A reduce operator instance that combines a new value for a key
    with the last reduced one according to a user-defined logic.

    Attributes:
        reduce_fn (function): The user-defined reduce logic.
        value_attribute (int): The index of the value to reduce
        (assuming tuple records).
        state (dict): A mapping from keys to values.
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate, config=None):
        OperatorInstance.__init__(self, instance_id, input_gate,
                                 output_gate, config)
        self.state = {}                             # key -> value
        self.reduce_fn = operator_metadata.logic    # Reduce function
        # Set the attribute selector
        self.attribute_selector = operator_metadata.other_args
        if self.attribute_selector is None:
            self.attribute_selector = _identity
        elif isinstance(self.attribute_selector, int):
            self.key_index = self.attribute_selector
            self.attribute_selector = self.index_based_selector
        elif isinstance(self.attribute_selector, str):
            self.key_attribute = self.attribute_selector
            self.attribute_selector = self.attribute_based_selector
        elif not isinstance(self.attribute_selector, types.FunctionType):
            sys.exit("Unrecognized or unsupported key selector.")

    # Combines the input value for a key with the last reduced
    # value for that key to produce a new value.
    # Outputs the result as a tuple of the form (key,new value)
    def start(self):
        while True:
            if self._reschedule():
                # Stop spinning so that other methods can be
                # called on this actor from the outside world
                return
            record = self.input._pull()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            key, rest = record
            new_value = self.attribute_selector(rest)
            # TODO (john): Is there a way to update state with
            # a single dictionary lookup?
            try:
                old_value = self.state[key]
                new_value = self.reduce_fn(old_value, new_value)
                self.state[key] = new_value
            except KeyError:  # Key does not exist in state
                self.state.setdefault(key, new_value)
            self.output._push((key, new_value))
            self.records_processed += 1

    # Returns the local state of the actor
    def state(self):
        return self.state

@ray.remote
class KeyBy(OperatorInstance):
    """A key_by operator instance that physically partitions the
    stream based on a key.

    Attributes:
        key_attribute (int): The index of the value to reduce
        (assuming tuple records).
    """

    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        # Set the key selector
        self.key_selector = operator_metadata.other_args
        if isinstance(self.key_selector, int):
            self.key_index = self.key_selector
            self.key_selector = self.index_based_selector
        elif isinstance(self.key_selector, str):
            self.key_attribute = self.key_selector
            self.key_selector = self.attribute_based_selector
        elif not isinstance(self.key_selector, types.FunctionType):
            sys.exit("Unrecognized or unsupported key selector.")

    # The actual partitioning is done by the output gate
    def start(self):
        while True:
            record = self.input._pull()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            key = self.key_selector(record)
            self.output._push((key,record))


# A custom source actor
@ray.remote
class Source(OperatorInstance):
    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        # The user-defined source with a get_next() method
        self.source = operator_metadata.other_args

    # Starts the source by calling get_next() repeatedly
    def start(self):
        start = time.time()
        records = 0
        while True:
            record = self.source.get_next()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                logger.debug("[writer {}] puts per second: {}".format(
                                            self.instance_id,
                                            records / (time.time()-start)))
                return
            self.output._push(record)
            records += 1

# A sink actor that writes records to a distributed text file
@ray.remote
class WriteTextFile(OperatorInstance):
    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        # Common prefix for all instances of the sink as given by the user
        prefix = operator_metadata.other_args[0]
        # Suffix of self's filename
        operator_id, local_instance_id = instance_id
        suffix = str(operator_id) + "_" + str(local_instance_id)
        self.filename = prefix + "_" + suffix
        # TODO (john): Handle possible exception here
        self.writer = open(self.filename, "w")
        # User-defined logic applied to each record
        user_logic = operator_metadata.other_args[1]
        self.logic = user_logic if user_logic else _identity

    # Applies logic (if any) and writes result to file
    def _put_next(self, record):
        self.writer.write(str(self.logic(record)) + "\n")

    # Starts the sink by calling put_next() repeatedly
    def start(self):
        while True:
            record = self.input._pull()
            if record is None:
                self.writer.close()
                signal.send(ActorExit())
                return
            self._put_next(record)

# Time window actor (uses system time)
@ray.remote
class TimeWindow(OperatorInstance):
    def __init__(self, instance_id, operator_metadata, input_gate,
                 output_gate):
        OperatorInstance.__init__(self, instance_id, input_gate, output_gate)
        # The length of the window in ms
        self.length = operator_metadata.other_args
        self.window_state = []
        self.start = time.time()

    def start(self):
        while True:
            if self._reschedule():
                # Stop spinning so that other methods can be
                # called on this actor from the outside world
                return
            record = self.source.get_next()
            if record is None:
                self.output._flush(close=True)
                signal.send(ActorExit())
                return
            self.window_state.append(record)
            elapsed_time = time.time() - self.start
            if elapsed_time >= self.length:
                self.output._push_all(self.window_state)
                self.window_state.clear()
                self.start = time.time()
