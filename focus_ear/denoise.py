"""Stage 0: noise suppression with DeepFilterNet3, streamed one 10 ms frame at a time.

The deepfilternet package only enhances whole recordings (df.enhance). This
module runs the same trained model incrementally: every causal convolution
keeps the input frames it still needs, every GRU keeps its hidden state, the
feature normalisations keep their running means, and libdf's STFT keeps its
overlap buffers. The output equals df.enhance() on the whole signal, delayed
by a fixed 1440 samples at 48 kHz (30 ms: 10 ms STFT overlap plus the
model's two frames of lookahead); tests/test_denoise.py checks this.

Two compatibility patches, both only for importing the package:
  * deepfilternet 0.5.6 imports torchaudio.backend.common, which torchaudio
    2.1+ removed. It is only used for a type annotation, so a stand-in
    module is registered before import.
  * Its metadata pins numpy<2 and packaging<24, which would downgrade the
    whole stack. It works with numpy 2 (verified by the test above), so it is
    installed with --no-deps (see requirements.txt).

Everything here runs on the pipeline worker thread.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

DF_RATE = 48_000


def _import_df():
    if "torchaudio.backend.common" not in sys.modules:
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = object  # only ever used as a type annotation
        sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
        sys.modules["torchaudio.backend.common"] = common
    import importlib
    # Not "import df.enhance": df/__init__.py re-exports a function of that name, shadowing the module.
    return importlib.import_module("df.enhance")


def load_model():
    """(model, ModelParams) for DeepFilterNet3; downloads it (~8 MB) on first use."""
    import torch
    enhance = _import_df()
    from df.deepfilternet3 import ModelParams

    from loguru import logger

    try:
        # log_file=None: otherwise it writes enhance.log into the model cache.
        model, _, _ = enhance.init_df(log_level="WARNING", log_file=None)
    except SystemExit as exc:  # init_df calls exit(1) when the checkpoint is missing
        raise RuntimeError("DeepFilterNet checkpoint not found") from exc
    finally:
        # init_df points loguru at stdout/stderr, which would scribble over the UI.
        logger.remove()
        logger.add(lambda msg: log.warning("deepfilternet: %s", msg.record["message"]), level="WARNING")
    model = model.to("cpu").eval()
    torch.set_grad_enabled(False)
    return model, ModelParams()


class DeepFilterStream:
    """48 kHz in, enhanced 48 kHz out, any chunk size, fixed ``delay`` samples of latency.

    ``mix`` blends enhanced with original in the STFT domain before
    synthesis (DeepFilterNet's own attenuation-limit mechanism), so a partial
    mix is phase-aligned: no comb filtering. ``bypass`` skips the model but
    keeps every buffer and delay, for calibration and tests.
    """

    def __init__(self, model, params, mix: float = 1.0, bypass: bool = False):
        import torch
        from libdf import DF

        self._torch = torch
        self.model = model
        p = params
        self.mix = float(mix)
        self.bypass = bypass
        self.hop = p.hop_size
        self.nb_df = p.nb_df
        self.lookahead = p.conv_lookahead
        assert p.conv_lookahead == p.df_lookahead == 2 and p.df_order == 5, "written for DeepFilterNet3"
        assert not getattr(model, "lsnr_droput", False), "lsnr dropout changes the GRU sequence; not supported"
        self.delay = p.fft_size - p.hop_size + self.lookahead * p.hop_size
        self.df = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb,
                     min_nb_erb_freqs=p.min_nb_freqs)
        self._erb_widths = self.df.erb_widths()
        self._alpha = self._norm_alpha(p)
        self.reset()
        self.lsnr = 0.0          # the model's local SNR estimate for the last frame, dB
        self.frames = 0

    @staticmethod
    def _norm_alpha(p) -> float:
        # df.utils.get_norm_alpha, without its dependency on the global config.
        a_ = float(np.exp(-p.hop_size / p.sr / p.norm_tau))
        precision, a = 3, 1.0
        while a >= 1.0:
            a = round(a_, precision)
            precision += 1
        return a

    def reset(self) -> None:
        from libdf import unit_norm_init

        self.df.reset()
        nb_erb = len(self._erb_widths)
        self._erb_state = np.linspace(-60.0, -90.0, nb_erb, dtype=np.float32)   # libdf's MEAN_NORM_INIT
        self._unit_state = unit_norm_init(self.nb_df)[0].astype(np.float32)
        self._pending = np.zeros(0, np.float32)
        self._skip = self.lookahead      # feature frames the encoder never sees (DfNet.pad_feat crops them)
        self._spec_hist = np.zeros((4, self.df.fft_size() // 2 + 1), np.complex64)  # frames t-2 .. t+1
        # Input frames kept for the causal convolutions with a time kernel > 1
        # (erb_conv0, df_conv0: 3; df_convp: 5). Sized on first use.
        self._erb_cache = self._cplx_cache = self._c0_cache = None
        self._h_enc = self._h_erb = self._h_df = None

    # Features ------------------------------------------------------------
    def _features(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalised ERB and complex features for [T, F] spec, advancing the running means."""
        from libdf import erb

        a = self._alpha
        e = erb(spec[None], self._erb_widths)[0]            # [T, E] in dB, stateless
        erb_feat = np.empty_like(e)
        cplx = spec[:, :self.nb_df]
        cplx_feat = np.empty_like(cplx)
        for i in range(len(spec)):                           # libdf's band_mean_norm_erb / band_unit_norm
            self._erb_state = e[i] * (1 - a) + self._erb_state * a
            erb_feat[i] = (e[i] - self._erb_state) / 40.0
            self._unit_state = np.abs(cplx[i]) * (1 - a) + self._unit_state * a
            cplx_feat[i] = cplx[i] / np.sqrt(self._unit_state)
        return erb_feat, cplx_feat

    def _causal(self, layer, cache, x):
        """Run a Conv2dNormAct whose first module is causal time padding, on cached frames instead."""
        mods = list(layer)
        assert isinstance(mods[0], self._torch.nn.ConstantPad2d), "expected causal time padding first"
        if cache is None:
            left, right = mods[0].padding[2:]
            assert right == 0, "lookahead is handled on the feature side"
            cache = x.new_zeros(x.shape[0], x.shape[1], left, x.shape[3])
        full = self._torch.cat((cache, x), dim=2)
        y = full
        for mod in mods[1:]:
            y = mod(y)
        return y, full[:, :, -cache.shape[2]:]

    # Model ---------------------------------------------------------------
    def _enhance(self, spec_t: np.ndarray, erb_feat: np.ndarray, cplx_feat: np.ndarray) -> np.ndarray:
        """Enhanced spectrum for the N positions t whose lookahead frames just arrived.

        ``spec_t``: [N+4, F] complex, frames t-2 .. t+2 for every t. Mirrors DfNet.forward.
        """
        torch = self._torch
        m = self.model
        n = len(erb_feat)
        feat_erb = torch.from_numpy(erb_feat)[None, None]                        # [1, 1, N, E]
        feat_spec = torch.from_numpy(np.stack((cplx_feat.real, cplx_feat.imag)))[None]  # [1, 2, N, Fdf]

        enc = m.enc
        e0, self._erb_cache = self._causal(enc.erb_conv0, self._erb_cache, feat_erb)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0, self._cplx_cache = self._causal(enc.df_conv0, self._cplx_cache, feat_spec)
        c1 = enc.df_conv1(c0)
        cemb = enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = enc.combine(e3.permute(0, 2, 3, 1).flatten(2), cemb)
        emb, self._h_enc = enc.emb_gru(emb, self._h_enc)
        lsnr = enc.lsnr_fc(emb) * enc.lsnr_scale + enc.lsnr_offset
        self.lsnr = float(lsnr[0, -1, 0])

        dec = m.erb_dec
        f8 = e3.shape[-1]
        x, self._h_erb = dec.emb_gru(emb, self._h_erb)
        x = x.view(1, n, f8, -1).permute(0, 3, 1, 2)
        x = dec.convt3(dec.conv3p(e3) + x)
        x = dec.convt2(dec.conv2p(e2) + x)
        x = dec.convt1(dec.conv1p(e1) + x)
        mask = dec.conv0_out(dec.conv0p(e0) + x)                                   # [1, 1, N, E]

        dfd = m.df_dec
        c, self._h_df = dfd.df_gru(emb, self._h_df)
        if dfd.df_skip is not None:
            c = c + dfd.df_skip(emb)
        cp, self._c0_cache = self._causal(dfd.df_convp, self._c0_cache, c0)
        coefs = dfd.df_out(c).view(1, n, self.nb_df, dfd.df_out_ch) + cp.permute(0, 2, 3, 1)
        coefs = coefs.view(n, self.nb_df, 5, 2).numpy()                            # [N, Fdf, order, re/im]
        coefs = coefs[..., 0] + 1j * coefs[..., 1]

        spec = spec_t[2:n + 2]                                                     # the frames being enhanced
        gains = mask[0, 0].numpy() @ m.mask.erb_inv_fb.numpy()                    # ERB mask -> per bin
        out = spec * gains
        # Deep filter over the lowest bins: sum over the 5 frames t-2 .. t+2.
        window = np.stack([spec_t[i:i + n, :self.nb_df] for i in range(5)], axis=-1)  # [N, Fdf, 5]
        out[:, :self.nb_df] = np.einsum("tfo,tfo->tf", window, coefs)
        if m.post_filter:
            beta, eps = m.post_filter_beta, 1e-12
            ratio = np.clip(np.abs(out) / (np.abs(spec) + eps), eps, 1)
            ratio_sin = ratio * np.maximum(np.sin(np.pi * ratio / 2), eps)
            out *= (1 + beta) / (1 + beta * (ratio / ratio_sin) ** 2)
        return out.astype(np.complex64)

    # Streaming -----------------------------------------------------------
    def process(self, x: np.ndarray, bypass: bool | None = None) -> np.ndarray:
        """Push 48 kHz samples; returns the enhanced samples completed so far (multiples of the hop).

        ``bypass=True`` passes just these samples through unenhanced, for a
        fraction of the cost: the STFT and every delay keep running, so the
        output stays in step, and synthesis overlap-adds the switch in and
        out over one hop instead of clicking. The model's own state doesn't
        advance over the skipped frames.
        """
        bypass = self.bypass if bypass is None else bypass
        buf = np.concatenate((self._pending, x.astype(np.float32, copy=False)))
        n = len(buf) // self.hop
        self._pending = buf[n * self.hop:]
        if n == 0:
            return np.zeros(0, np.float32)
        spec = self.df.analysis(buf[None, :n * self.hop], reset=False)[0]         # [n, F]
        erb_feat, cplx_feat = self._features(spec)
        hist = np.concatenate((self._spec_hist, spec))                           # frames j0-4 .. j0+n-1
        self._spec_hist = hist[-4:]
        skip = min(self._skip, n)
        self._skip -= skip
        k = n - skip                                                             # positions t = j - 2
        out = np.zeros((n, spec.shape[1]), np.complex64)                        # warm-up frames stay silent
        if k:
            spec_t = hist[skip:]                                                 # frames t-2 .. t+2
            if bypass:
                enhanced = spec_t[2:k + 2]
            else:
                # Grad mode is per thread: disabling it at load time doesn't cover the worker.
                with self._torch.inference_mode():
                    enhanced = self._enhance(spec_t, erb_feat[skip:], cplx_feat[skip:])
                if self.mix < 1.0:
                    enhanced = self.mix * enhanced + (1.0 - self.mix) * spec_t[2:k + 2]
            out[skip:] = enhanced
        self.frames += n
        return self.df.synthesis(out[None], reset=False)[0].astype(np.float32, copy=False)


@dataclass
class _Job:
    block: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    out: np.ndarray | None = None
    abandoned: bool = False     # the worker gave up waiting and played it undenoised


class Denoiser:
    """DeepFilterStream at the stream rate: resample to 48 kHz and back, fixed delay, same-size blocks.

    ``process(block)`` always returns ``len(block)`` samples, ``delay_samples``
    behind the input. The resamplers and the model emit in bursts, so a small
    FIFO primed with silence absorbs the jitter; its size is worked out once
    from the actual block size, and the total delay measured, in __init__.

    The model runs with one torch thread: 2.7 ms per 32 ms block on an M3
    back to back, against 5.8 ms with torch's default of 4 (the frames are
    too small to split). Live, at one block every 32 ms, it's 6-9 ms, and
    about once a minute a single block took 70-77 ms (measured; the cause is
    outside the model). The setting is per thread (OpenMP), so nothing else
    is affected.

    ``threaded=True`` (live) runs the model on a thread of its own, and
    process() takes a ``deadline``: if the block isn't denoised by then, the
    caller gets the raw block, delayed by exactly ``delay_samples`` so it
    stays in step, instead of the output running dry. The model still sees
    every block in order (the abandoned ones through the cheap STFT-only
    path, which also lets it catch up), so its delay never shifts. Such
    blocks are counted in ``late_blocks``.
    """

    name = "denoise"
    WARMUP_BLOCKS = 10

    def __init__(self, rate: int, blocksize: int, model, params, mix: float = 1.0, bypass: bool = False,
                 threaded: bool = False):
        self.rate = rate
        self.blocksize = blocksize
        self._make = lambda byp: _Chain(rate, DeepFilterStream(model, params, mix, bypass=byp))
        self._prime = self._fifo_prime()
        self.delay_samples = self._measure_delay()
        self.chain = self._make(bypass)
        self._fifo = np.zeros(self._prime, np.float32)
        self._fifo_cap = self._prime + 4 * blocksize  # never reached; bounds memory regardless
        self._threads_set: set[int] = set()
        self.underflows = 0
        self.last_ms = 0.0
        self.late_blocks = 0    # played undenoised because the model missed the deadline; only increases
        self.errors = 0         # blocks the model failed on (threaded), played undenoised
        self.warmup_ms = self._time_model(model, params, mix)

        self._thread: threading.Thread | None = None
        if threaded:
            from .gain import DelayLine
            self._raw = DelayLine(self.delay_samples)
            self._jobs: deque[_Job] = deque()
            self._cv = threading.Condition()   # worker <-> model thread; never taken by an audio callback
            self._closing = False
            ready = threading.Event()
            self._thread = threading.Thread(target=self._serve, args=(ready,), name="denoise", daemon=True)
            self._thread.start()
            ready.wait()

    @property
    def lsnr(self) -> float:
        return self.chain.dfn.lsnr

    @property
    def delay_ms(self) -> float:
        return 1e3 * self.delay_samples / self.rate

    def _fifo_prime(self) -> int:
        """Smallest silence prefill for which the FIFO never runs short with this block size."""
        chain = self._make(True)
        level = low = 0
        block = np.zeros(self.blocksize, np.float32)
        # 60 s: the pattern of block size vs resampling ratio vs 480-sample hops
        # repeats only every few hundred blocks (441 blocks, 14 s, at 44.1 kHz
        # with 1411-sample blocks); a 3 s run missed the worst case.
        for _ in range(int(60 * self.rate / self.blocksize)):
            level += len(chain.process(block)) - self.blocksize
            low = min(low, level)
        return -low

    def _measure_delay(self) -> int:
        """Input-to-output delay in samples, measured by cross-correlation through a bypassed chain."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.rate).astype(np.float32) * 0.1
        chain, fifo, out = self._make(True), np.zeros(self._prime, np.float32), []
        for i in range(0, len(x) - self.blocksize + 1, self.blocksize):
            fifo = np.concatenate((fifo, chain.process(x[i:i + self.blocksize])))
            out.append(fifo[:self.blocksize])
            fifo = fifo[self.blocksize:]
        y = np.concatenate(out)
        corr = np.correlate(y, x[:len(y) // 2], mode="valid")
        return int(np.argmax(corr))

    def _time_model(self, model, params, mix: float) -> float:
        """Mean ms per block with the real model, measured on 1 s of noise (also warms torch up)."""
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            chain = _Chain(self.rate, DeepFilterStream(model, params, mix))
            x = np.random.default_rng(1).standard_normal(self.rate).astype(np.float32) * 0.05
            times = []
            for i in range(0, len(x) - self.blocksize + 1, self.blocksize):
                t = time.perf_counter()
                chain.process(x[i:i + self.blocksize])
                times.append(time.perf_counter() - t)
        finally:
            torch.set_num_threads(threads)
        return 1e3 * float(np.median(times[5:]))

    def process(self, block: np.ndarray, ctx=None, deadline: float | None = None) -> np.ndarray:
        """The denoised block, ``delay_samples`` late.

        Threaded: waits at most ``deadline`` seconds (None: as long as it
        takes), then settles for the raw block, equally late.
        """
        if self._thread is None:
            return self._process_now(block)
        fallback = self._raw.process(block)  # every block, so it's in step whenever it's needed
        job = _Job(block.copy())
        with self._cv:
            self._jobs.append(job)
            self._cv.notify()
        if job.done.wait(deadline):
            return job.out if job.out is not None else fallback  # None: the model failed (logged)
        job.abandoned = True
        self.late_blocks += 1
        return fallback

    def close(self) -> None:
        if self._thread is not None:
            with self._cv:
                self._closing = True
                self._cv.notify()
            self._thread.join(timeout=2)
            self._thread = None

    def _serve(self, ready: threading.Event) -> None:
        # Warm torch up on this thread (thread count, first-call costs) before the first real block.
        try:
            warm = self._make(False)
            for _ in range(self.WARMUP_BLOCKS):
                self._process_now(np.zeros(self.blocksize, np.float32), chain=warm)
        finally:
            ready.set()
        while True:
            with self._cv:
                while not self._jobs and not self._closing:
                    self._cv.wait()
                if not self._jobs:
                    return
                job = self._jobs.popleft()
            try:
                # Given up on already: keep the stream in step at the STFT-only cost, and catch up.
                job.out = self._process_now(job.block, bypass=job.abandoned)
            except Exception:  # noqa: BLE001 - the worker falls back to the raw block
                self.errors += 1
                if self.errors == 1:
                    log.exception("denoising failed; playing undenoised while it keeps failing")
            job.done.set()

    def _process_now(self, block: np.ndarray, bypass: bool = False, chain: _Chain | None = None) -> np.ndarray:
        """``bypass=True`` skips the model for this block only (see DeepFilterStream.process)."""
        tid = threading.get_ident()
        if tid not in self._threads_set:
            import torch
            torch.set_num_threads(1)
            self._threads_set.add(tid)
        if chain is not None:  # warm-up: a throwaway chain, no FIFO
            chain.process(block)
            return block
        t = time.perf_counter()
        fifo = np.concatenate((self._fifo, self.chain.process(block, bypass or None)))
        n = len(block)
        if len(fifo) < n:  # never happens with the computed prefill; kept as a safety net
            self.underflows += 1
            fifo = np.concatenate((np.zeros(n - len(fifo), np.float32), fifo))
        out, self._fifo = fifo[:n], fifo[n:]
        if len(self._fifo) > self._fifo_cap:
            self._fifo = self._fifo[-self._fifo_cap:]
        self.last_ms = 1e3 * (time.perf_counter() - t)
        return out

    def reset(self) -> None:
        self.chain = self._make(self.chain.dfn.bypass)
        self._fifo = np.zeros(self._prime, np.float32)


class PolyphaseResampler:
    """Streaming rational resampler (``up``/``down``) with an exactly fixed delay and output count.

    soxr's ResampleStream holds back a slowly wandering amount of input (its
    44.1<->48 kHz round trip drifted between 46 and 465 samples short over an
    hour, in bursts every ~14 s), so no fixed FIFO prefill in the denoiser is
    ever guaranteed: each shortfall would click and shift the delay. Here
    output sample m is always computed as soon as its last input sample
    arrives, so the counts follow the ratio exactly.

    Kaiser-windowed sinc, 32 taps per phase: flat to ~16 kHz, >= 80 dB
    rejection from ~23.5 kHz, 16 input samples of delay (0.4 ms at 44.1 kHz).
    """

    def __init__(self, rate_in: int, rate_out: int, taps: int = 32, beta: float = 8.0):
        from math import gcd
        from scipy.signal import firwin

        g = gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        L, M, K = self.up, self.down, taps
        # Prototype at the upsampled rate; cutoff just under the lower Nyquist.
        h = firwin(L * K, 0.92 / max(L, M), window=("kaiser", beta)) * L
        self._phases = h.reshape(K, L).T[:, ::-1].astype(np.float32)   # [phase, tap], oldest tap first
        self.taps = K
        self.delay_in = (L * K - 1) / (2 * L)      # input samples
        self._hist = np.zeros(K - 1, np.float32)   # last K-1 input samples
        self._n_in = 0                             # input samples consumed
        self._m = 0                                # next output sample index

    def process(self, x: np.ndarray) -> np.ndarray:
        L, M, K = self.up, self.down, self.taps
        buf = np.concatenate((self._hist, x.astype(np.float32, copy=False)))  # buf[K-1] = input self._n_in
        n_in = self._n_in + len(x)
        # Output m needs inputs up to floor(m*M/L) <= n_in - 1, i.e. m*M < n_in*L.
        m_end = -(-n_in * L // M)
        m = np.arange(self._m, m_end)
        base = (m * M) // L                                             # newest input index used
        idx = (base - self._n_in + K - 1)[:, None] + np.arange(-K + 1, 1)  # [n_out, K] into buf
        y = np.einsum("nk,nk->n", buf[idx], self._phases[(m * M) % L])
        self._hist = buf[len(buf) - (K - 1):]
        self._n_in, self._m = n_in, m_end
        return y.astype(np.float32)


class _Chain:
    """stream rate -> 48 kHz -> DeepFilterStream -> stream rate, all streaming, fixed delay."""

    def __init__(self, rate: int, dfn: DeepFilterStream):
        self.dfn = dfn
        self.up = self.down = None
        if rate != DF_RATE:
            self.up = PolyphaseResampler(rate, DF_RATE)
            self.down = PolyphaseResampler(DF_RATE, rate)

    def process(self, x: np.ndarray, bypass: bool | None = None) -> np.ndarray:
        if self.up is not None:
            x = self.up.process(x)
        y = self.dfn.process(x, bypass)
        if self.down is not None:
            y = self.down.process(y)
        return y
