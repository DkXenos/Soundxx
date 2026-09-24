"""Run with: venv/bin/python -m unittest discover -s tests"""
import threading
import time
import unittest

import numpy as np

from focus_ear.audio_io import DeviceError, RingBuffer, check_device_pair, find_device


class RingBufferTest(unittest.TestCase):
    def test_capacity_rounds_up_to_power_of_two(self):
        self.assertEqual(RingBuffer(1000).capacity, 1024)

    def test_wraparound_preserves_order(self):
        rb = RingBuffer(8)
        out = np.zeros(5, np.float32)
        rb.write(np.arange(6, dtype=np.float32))
        rb.read_into(out)
        rb.write(np.arange(6, 12, dtype=np.float32))  # wraps past the end
        got = np.zeros(7, np.float32)
        self.assertEqual(rb.read_into(got), 7)
        np.testing.assert_array_equal(got, np.arange(5, 12))

    def test_write_is_truncated_when_full(self):
        rb = RingBuffer(8)
        self.assertEqual(rb.write(np.ones(10, np.float32)), 8)
        self.assertEqual(rb.write(np.ones(1, np.float32)), 0)

    def test_short_read_and_skip(self):
        rb = RingBuffer(16)
        rb.write(np.arange(4, dtype=np.float32))
        self.assertEqual(rb.skip(1), 1)
        out = np.zeros(8, np.float32)
        self.assertEqual(rb.read_into(out), 3)
        np.testing.assert_array_equal(out[:3], [1, 2, 3])
        self.assertEqual(rb.skip(5), 0)

    def test_concurrent_producer_consumer_keeps_every_sample_in_order(self):
        rb = RingBuffer(64)
        total = 100_000
        received = []

        def produce():
            sent = 0
            while sent < total:
                n = rb.write(np.arange(sent, min(sent + 37, total), dtype=np.float32))
                sent += n
                if not n:
                    time.sleep(0)  # yield the GIL instead of spinning

        def consume():
            buf = np.zeros(29, np.float32)
            got = 0
            while got < total:
                n = rb.read_into(buf)
                received.append(buf[:n].copy())
                got += n
                if not n:
                    time.sleep(0)

        threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        np.testing.assert_array_equal(np.concatenate(received), np.arange(total, dtype=np.float32))


DEVICES = [
    {"name": "MacBook Pro Microphone", "inputs": 1, "outputs": 0},
    {"name": "MacBook Pro Speakers", "inputs": 0, "outputs": 2},
    {"name": "Jaysn's Galaxy Buds2 Pro", "inputs": 1, "outputs": 2},
]


class DeviceTest(unittest.TestCase):
    def test_default_input_is_builtin_mic_not_headset(self):
        self.assertEqual(find_device(DEVICES, None, "input"), 0)

    def test_substring_is_case_insensitive(self):
        self.assertEqual(find_device(DEVICES, "galaxy buds", "output"), 2)

    def test_index_query_must_have_the_right_direction(self):
        self.assertEqual(find_device(DEVICES, "1", "output"), 1)
        with self.assertRaises(DeviceError):
            find_device(DEVICES, "1", "input")

    def test_missing_device_lists_alternatives(self):
        with self.assertRaisesRegex(DeviceError, "MacBook Pro Speakers"):
            find_device(DEVICES, "AirPods", "output")

    def test_rejects_feedback_and_headset_mic(self):
        with self.assertRaises(DeviceError):
            check_device_pair("MacBook Pro Microphone", "MacBook Pro Speakers", allow_speakers=False)
        check_device_pair("MacBook Pro Microphone", "MacBook Pro Speakers", allow_speakers=True)
        with self.assertRaises(DeviceError):
            check_device_pair("Jaysn's Galaxy Buds2 Pro", "Jaysn's Galaxy Buds2 Pro", allow_speakers=False)


if __name__ == "__main__":
    unittest.main()
