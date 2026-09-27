"""Run with: venv/bin/python -m unittest discover -s tests"""
import logging
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from focus_ear import audio_io
from focus_ear.audio_io import AudioEngine, DeviceError, RingBuffer, check_device_pair, find_device
from focus_ear.config import Config
from focus_ear.pipeline import Pipeline


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


class FakeSoundDevice:
    """Just enough of sounddevice for AudioEngine: devices come and go, streams tick on threads."""

    class PortAudioError(Exception):
        pass

    def __init__(self):
        self.buds_awake = True
        self.fail_next_open = False
        self.reinits = 0
        self.streams = []
        fake = self

        class Stream:
            def __init__(self, device, samplerate, blocksize, channels, dtype, latency, callback):
                if fake.fail_next_open and self.kind == "output":
                    fake.fail_next_open = False
                    raise fake.PortAudioError("Bluetooth link not ready")
                self.frames = blocksize or 512
                self.channels, self.callback = channels, callback
                self.active, self.closed, self.latency = False, False, 0.01
                self.device_name = fake.query_devices()[device]["name"]
                fake.streams.append(self)

            def start(self):
                self.active = True
                threading.Thread(target=self._tick, daemon=True).start()

            def _tick(self):
                info = types.SimpleNamespace(inputBufferAdcTime=0, outputBufferDacTime=0, currentTime=0)
                status = types.SimpleNamespace(input_overflow=False, output_underflow=False)
                buf = np.zeros((self.frames, self.channels), np.float32)
                while not self.closed:
                    time.sleep(self.frames / 16000)
                    # A sleeping Bluetooth device just stops calling back; the stream still looks active.
                    if "Buds" in self.device_name and not fake.buds_awake:
                        continue
                    if self.kind == "input":
                        buf[:] = 0.01
                    self.callback(buf, self.frames, info, status)

            def close(self, ignore_errors=True):
                self.closed, self.active = True, False

        self.InputStream = type("InputStream", (Stream,), {"kind": "input"})
        self.OutputStream = type("OutputStream", (Stream,), {"kind": "output"})

    def query_devices(self):
        devices = [{"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0,
                    "default_samplerate": 16000.0}]
        if self.buds_awake:
            devices.append({"name": "Mewo Buds", "max_input_channels": 1, "max_output_channels": 2,
                            "default_samplerate": 16000.0})
        return devices

    def _terminate(self):
        pass

    def _initialize(self):
        self.reinits += 1


class FastEngine(AudioEngine):
    STALE_S, OPEN_GRACE_S, PROBE_INTERVAL_S, POLL_S = 0.15, 0.1, 0.05, 0.02


class ReconnectTest(unittest.TestCase):
    """The Buds sleeping in their case is routine: the engine waits, then resumes, every time."""

    def wait_for(self, cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not cond() and time.monotonic() < deadline:
            time.sleep(0.01)
        return cond()

    def test_buds_sleep_and_wake_repeatedly_without_crashing(self):
        sd = FakeSoundDevice()
        probe = lambda timeout=5.0: [{"name": d["name"], "inputs": d["max_input_channels"],  # noqa: E731
                                      "outputs": d["max_output_channels"]} for d in sd.query_devices()]
        with mock.patch.object(audio_io, "sd", sd), mock.patch.object(audio_io, "probe_devices", probe), \
                self.assertLogs("focus_ear.audio_io", logging.INFO) as logs:
            engine = FastEngine(Config(output_device="Mewo"))
            engine.start()
            pipe = Pipeline(engine, Config())
            pipe.start()
            try:
                self.assertTrue(self.wait_for(lambda: engine.state == "running"))
                threads = threading.active_count()
                for cycle in range(10):
                    sd.buds_awake = False
                    self.assertTrue(self.wait_for(lambda: engine.state == "waiting"), f"cycle {cycle}")
                    blocks = pipe.blocks
                    self.assertTrue(self.wait_for(lambda: pipe.blocks > blocks + 3),
                                    "the mic keeps feeding the analysis while the Buds are away")
                    sd.fail_next_open = cycle == 3  # listed before the link is ready: retried
                    sd.buds_awake = True
                    self.assertTrue(self.wait_for(lambda: engine.state == "running"), f"cycle {cycle}")
                    self.assertTrue(self.wait_for(lambda: engine.stats.out_heartbeat > time.monotonic() - 0.05))
                self.assertIsNone(engine.error)
                self.assertEqual(engine.stats.reconnects, 10)
                self.assertLessEqual(threading.active_count(), threads + 2)  # no thread pile-up
            finally:
                pipe.stop()
                engine.stop()
        text = "\n".join(logs.output)
        self.assertIn("is back after", text)
        self.assertIn("will retry", text)
        live = [s for s in sd.streams if not s.closed]
        self.assertEqual(live, [])  # every stream was closed


if __name__ == "__main__":
    unittest.main()
