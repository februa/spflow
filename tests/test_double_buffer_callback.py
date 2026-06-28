import numpy as np

from spflow import DoubleBufferCallback, StepScheduler


class SumCallback(DoubleBufferCallback):
    def signature(self, inputs):
        return tuple(inputs["items"])

    def make_initial_output(self, inputs):
        return np.zeros(1, dtype=np.int64)

    def make_work_buffer(self, inputs):
        return np.zeros_like(self.prev)

    def make_items(self, inputs):
        return inputs["items"]

    def update_item(self, item, inputs):
        self.work[0] += item


def test_double_buffer_callback_publishes_only_when_done():
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=2)

    out1 = scheduler.process({"items": [1, 2, 3]})
    out2 = scheduler.process({"items": [1, 2, 3]})

    np.testing.assert_array_equal(out1, np.array([0]))
    np.testing.assert_array_equal(out2, np.array([6]))


def test_double_buffer_callback_restarts_on_signature_change():
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=1)

    out1 = scheduler.process({"items": [1, 2]})
    out2 = scheduler.process({"items": [10, 20]})
    out3 = scheduler.process({"items": [10, 20]})

    np.testing.assert_array_equal(out1, np.array([0]))
    np.testing.assert_array_equal(out2, np.array([0]))
    np.testing.assert_array_equal(out3, np.array([30]))


def test_double_buffer_callback_publish_copies_work():
    callback = SumCallback()
    scheduler = StepScheduler(callback=callback, items_per_cycle=None)

    out = scheduler.process({"items": [1, 2]})
    callback.prev[0] = 999

    np.testing.assert_array_equal(out, np.array([3]))
