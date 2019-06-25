from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import msgpack
import numpy as np
import threading
import time

import ray
from ray.experimental import internal_kv

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def plasma_prefetch(object_id):
    """Tells plasma to prefetch the given object_id."""
    local_sched_client = ray.worker.global_worker.raylet_client
    ray_obj_id = ray.ObjectID(object_id)
    local_sched_client.fetch_or_reconstruct([ray_obj_id], True)

def plasma_get(object_id):
    """Get an object directly from plasma without going through object table.

    Precondition: plasma_prefetch(object_id) has been called before.
    """
    client = ray.worker.global_worker.plasma_client
    plasma_id = ray.pyarrow.plasma.ObjectID(object_id)
    while not client.contains(plasma_id):
        pass
    return client.get(plasma_id)

class BatchedQueue(object):
    """A batched queue for actor to actor communication.

    Attributes:
         max_size (int): The maximum size of the queue in number of batches
         (if exceeded, backpressure kicks in)
         max_batch_size (int): The size of each batch in number of records.
         max_batch_time (float): The flush timeout per batch.
         prefetch_depth (int): The  number of batches to prefetch from plasma.
         background_flush (bool): Denotes whether a daemon flush thread should
         be used (True) to flush batches to plasma.
         base (ndarray): A unique signature for the queue.
         read_ack_key (bytes): The signature of the queue in bytes.
         prefetch_batch_offset (int): The number of the last read prefetched
         batch.
         read_batch_offset (int): The number of the last read batch.
         read_item_offset (int): The number of the last read record inside a
         batch.
         write_batch_offset (int): The number of the last written batch.
         write_item_offset (int): The numebr of the last written item inside a
         batch.
         write_buffer (list): The write buffer, i.e. an in-memory batch.
         cached_remote_offset (int): The number of the last read record as
         recorded by the writer after the previous flush.
    """

    def __init__(self,
                 channel_id,
                 src_operator_id, src_instance_id,
                 dst_operator_id, dst_instance_id,
                 max_size=999999,
                 max_batch_size=99999,
                 max_batch_time=0.01,
                 prefetch_depth=10,
                 background_flush=True,
                 task_based=True):
        self.src_operator_id = src_operator_id
        self.src_instance_id = src_instance_id
        self.dst_operator_id = dst_operator_id
        self.dst_instance_id = dst_instance_id
        self.channel_id = channel_id
        self.max_size_batches = max_size
        self.max_batch_size = max_batch_size
        self.max_size = max_size * max_batch_size  # In number of records
        self.max_batch_time = max_batch_time  # Flushing timeout
        self.prefetch_depth = prefetch_depth
        self.background_flush = False # background_flush
        self.task_based = task_based  # True for task-based data exchange

        # Common queue metadata -- This serves as the unique id of the queue
        self.base = np.random.randint(0, 2**32 - 1, size=5, dtype="uint32")
        self.base[-2] = 0
        self.base[-1] = 0
        self.read_ack_key = np.ndarray.tobytes(self.base)

        # Reader state
        self.prefetch_batch_offset = 0
        self.read_item_offset = 0
        self.read_batch_offset = 0
        self.read_buffer = []

        # Writer state
        self.write_item_offset = 0
        self.write_batch_offset = 0
        self.write_buffer = []
        self.cached_remote_offset = 0
        self.task_queue = []

        self.source_actor = None
        self.destination_actor = None

        # Used to simulate backpressure in task-based execution
        self.records_sent = 0
        self.records_per_task = {}
        self.num_batches_sent = 0

        # TODO (john): Use a buffer pool and a separate thread for flushing.
        # Currently, the timeout is checked within put_next(). This means
        # that the timeout might be violated in case the actor does not push
        # any output record to this channel for a time period equal to the
        # timeout
        self.last_flush_time = 0.0  # Used to flush based on a timeout
        self.checkpoint_epoch = 0

        self.checkpoint_epoch = 0

    def save(self):
        assert len(self.write_buffer) == 0, len(self.write_buffer)
        self.records_sent
        return self.records_sent

    def load(self, records_sent):
        self.records_sent = records_sent

    def set_checkpoint_epoch(self, checkpoint_epoch):
        self.checkpoint_epoch = checkpoint_epoch

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["write_buffer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Batch ids consist of a unique queue id used as prefix along with
    # two numbers generated using the batch offset in the queue
    def _batch_id(self, batch_offset):
        oid = self.base.copy()
        oid[-2] = batch_offset // 2**32
        oid[-1] = batch_offset % 2**32
        return np.ndarray.tobytes(oid)

    def push_next_batch(self, record_batch, event=None):
        if self.task_based:  # Submit a new downstream task
            args = [[record_batch], self.channel_id,
                                self.src_operator_id,
                                self.checkpoint_epoch]
            if event is not None:
                event = str(event).encode('ascii')

            num_return_vals = 0
            if self.num_batches_sent % self.max_size_batches == 0:
                num_return_vals = 1

            obj_id = self.destination_actor.apply._remote(
                    args=args,
                    kwargs={},
                    num_return_vals=num_return_vals,
                    nondeterministic_event=event)
            # logger.debug("Flushed task %s %d, event:%s", obj_id.hex(),
            #                                     self.checkpoint_epoch, event)

            num_records = len(record_batch)
            self.records_sent += num_records
            #self.records_per_task[obj_id] = num_records
            if obj_id:
                self.task_queue.append(obj_id)
            self.write_batch_offset += (num_records / self.max_batch_size)
            self._wait_for_task_reader()
            self.last_flush_time = time.time()

            self.num_batches_sent += 1
            return True

    def _flush_writes(self, event=None, flush_empty=False):
        # TODO: This forces a flush.
        if not self.write_buffer and not flush_empty:
            return
        if self.task_based:  # Submit a new downstream task
            args = [[self.write_buffer], self.channel_id,
                                self.src_operator_id,
                                self.checkpoint_epoch]
            if event is not None:
                event = str(event).encode('ascii')

            num_return_vals = 0
            if self.num_batches_sent % self.max_size_batches == 0:
                num_return_vals = 1

            obj_id = self.destination_actor.apply._remote(
                    args=args,
                    kwargs={},
                    num_return_vals=num_return_vals,
                    nondeterministic_event=event)
            # logger.debug("Flushed task %s %d, event:%s", obj_id.hex(), self.checkpoint_epoch, event)

            num_records = len(self.write_buffer)
            self.records_sent += num_records
            #self.records_per_task[obj_id] = num_records
            if obj_id:
                self.task_queue.append(obj_id)
            self.num_batches_sent += 1
        else:  # Flush batch to plasma
            # with ray.profiling.profile("flush_batch"):
            batch_id = self._batch_id(self.write_batch_offset)
            ray.worker.global_worker.put_object(
                    ray.ObjectID(batch_id), self.write_buffer)
        # logger.debug("[writer] Flush batch {} offset {} size {}".format(
        #              self.write_batch_offset, self.write_item_offset,
        #              len(self.write_buffer)))
        self.write_buffer = []
        self.write_batch_offset += 1
        # Check for backpressure
        # with ray.profiling.profile("wait_for_reader"):
        if self.task_based:
            self._wait_for_task_reader()
        else:
            self._wait_for_reader()
        self.last_flush_time = time.time()

    # Currently, the 'queue' size in both task- and queue-based execution is
    # estimated based on the number of unprocessed records, which in turn
    # is estimated based on the number of pending tasks and the number of
    # records assigned to each one of them
    def _wait_for_task_reader(self):
        """Checks for backpressure by the downstream task-based reader."""
        if self.max_size <= 0:  # Unlimited queue
            return
        if len(self.task_queue) <= 1:
            return  # Hasn't exceeded max size
        # Check pending downstream tasks
        finished_tasks, self.task_queue = ray.wait(
                self.task_queue,
                num_returns=len(self.task_queue),
                timeout=0)
        #for task_id in finished_tasks:
        #    self.records_sent -= self.records_per_task.pop(task_id)
        while len(self.task_queue) > 1:
            # logger.debug("Waiting for ({},{}) to catch up".format(
            #              self.dst_operator_id, self.dst_instance_id))
            _, self.task_queue = ray.wait(
                    self.task_queue,
                    num_returns=len(self.task_queue),
                    timeout=0.01)
            #for task_id in finished_tasks:
            #    self.records_sent -= self.records_per_task.pop(task_id)

    def _wait_for_reader(self):
        """Checks for backpressure by the downstream reader."""
        if self.max_size <= 0:  # Unlimited queue
            return
        if self.write_item_offset - self.cached_remote_offset <= self.max_size:
            return  # Hasn't exceeded max size
        remote_offset = internal_kv._internal_kv_get(self.read_ack_key)
        if remote_offset is None:
            # logger.debug("[writer] Waiting for reader to start...")
            while remote_offset is None:
                time.sleep(0.01)
                remote_offset = internal_kv._internal_kv_get(self.read_ack_key)
        # logger.debug("Remote offset: {}".format(int(remote_offset)))
        remote_offset = int(remote_offset)
        if self.write_item_offset - remote_offset > self.max_size:
            # logger.debug(
            #     "Waiting for ({},{}) to catch up {} to {} - {}".format(
            #         self.dst_operator_id, self.dst_instance_id,
            #         remote_offset, self.write_item_offset, self.max_size))
            while self.write_item_offset - remote_offset > self.max_size:
                time.sleep(0.01)
                remote_offset = int(
                    internal_kv._internal_kv_get(self.read_ack_key))
        self.cached_remote_offset = remote_offset

    def _read_next_batch(self):
        while (self.prefetch_batch_offset <
               self.read_batch_offset + self.prefetch_depth):
            plasma_prefetch(self._batch_id(self.prefetch_batch_offset))
            self.prefetch_batch_offset += 1
        self.read_buffer = plasma_get(self._batch_id(self.read_batch_offset))
        self.read_batch_offset += 1
        # logger.debug("[reader] Fetched batch {} offset {} size {}".format(
        #              self.read_batch_offset, self.read_item_offset,
        #              len(self.read_buffer)))
        self._ack_reads(self.read_item_offset + len(self.read_buffer))

    # Reader acks the key it reads so that writer knows reader's offset.
    # This is to cap queue size and simulate backpressure
    def _ack_reads(self, offset):
        if self.max_size > 0:
            internal_kv._internal_kv_put(
                self.read_ack_key, offset, overwrite=True)

    # This is to enable writing functionality in
    # case the queue is not created by the writer
    # The reason is that python locks cannot be serialized
    def enable_writes(self):
        """Restores the state of the batched queue for writing."""
        self.write_buffer = []

    # Registers source actor handle
    def register_source_actor(self, actor_handle):
        logger.debug("Registered source {} at channel {}".format(
                     actor_handle, self.channel_id))
        self.source_actor = actor_handle

    # Registers destination actor handle
    def register_destination_actor(self, actor_handle):
        logger.debug("Registered destination {} at channel {}".format(
                     actor_handle, self.channel_id))
        self.destination_actor = actor_handle

    def put_next(self, item):
        if not self.write_buffer:  # Reset last flush time for the new batch
            self.last_flush_time = time.time()
        self.write_buffer.append(item)
        self.write_item_offset += 1

    def try_flush(self, event=None):
        # If buffer is full...
        if (len(self.write_buffer) >= self.max_batch_size):
            self._flush_writes(event=event)
            return True
        # ...or if the timeout expired
        delay = time.time() - self.last_flush_time
        if delay >= self.max_batch_time:
            self._flush_writes(event=event)
            return True
        return False

    def push_next(self, item, event=None):
        self.put_next(item)
        return self.try_flush(event=event)

    def read_next(self):
        # Actors never pull in task-based execution
        assert self.task_based is False
        if not self.read_buffer:
            # with ray.profiling.profile("read_batch"):
            self._read_next_batch()
            assert self.read_buffer
        self.read_item_offset += 1
        return self.read_buffer.pop(0)
