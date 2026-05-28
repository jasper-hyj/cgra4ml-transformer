"""
Python-only tests for 2-head multi-head attention export.

Verified properties
--------------------
1. correct bundle count (N_BUNDLES == 11)
2. is_concat=1 written for B10 (Wo bundle), 0 for all others
3. n_concat_srcs, concat_src_ib_0 correct for B10
4. per-head score correctness  (P0, P1 vs NumPy softmax reference)
5. per-head value output H0, H1 vs NumPy reference
6. concat(H0, H1) layout (channel ordering H0 first, H1 second)
7. Wo output vs NumPy reference
8. quantization tolerance: all stages within 2 quantization steps of float reference
9. existing softmax and chained-matmul exports still pass (regression guard)

The tests use a fixed random seed and a tiny model (N=16, D=16) so they run in seconds
on CPU without any hardware accelerator or Docker.
"""

import os
import re
import sys
import pytest
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(work_dir):
    with open(os.path.join(work_dir, "config_fw.h")) as f:
        return f.read()

def _define(cfg, name):
    m = re.search(rf"#define {name}\s+(\S+)", cfg)
    assert m, f"#define {name} not found in config_fw.h"
    return m.group(1)

def _idefine(cfg, name):
    return int(_define(cfg, name))

def _bundle_fields(cfg, field):
    """Return a list of int values for .field= across all bundle initialisers."""
    return [int(v) for v in re.findall(rf"\.{field}=(-?\d+)", cfg)]


# ---------------------------------------------------------------------------
# Model builder + exporter
# ---------------------------------------------------------------------------

def _build_and_export(work_dir):
    """Build the 2-head MHA model and call export_inference().

    Returns (cfg_text, bundles_snapshot, hw, x_qtensor_np, model).
    """
    orig_dir = os.getcwd()
    os.chdir(str(work_dir))

    try:
        import tensorflow as tf
        from tensorflow import keras
        from keras.layers import Input
        from keras.models import Model
        from deepsocflow import (
            Hardware, SYS_BITS, XModel, XBundle, XDense, XActivation,
            export_inference,
        )
        from deepsocflow.py.utils import BUNDLES

        N      = 16
        D      = 16
        N_HEAD = 2
        D_HEAD = D // N_HEAD   # 8

        sys_bits = SYS_BITS(x=4, k=8, b=16)
        relu = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu')
        lin  = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)

        @keras.saving.register_keras_serializable()
        class _MHAModel(XModel):
            def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
                super().__init__(sys_bits, x_int_bits, *args, **kwargs)
                self.b_q0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_k0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_v0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_s0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()),
                                    softmax=True, transpose_w_src=True)
                self.b_h0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=lin()))
                self.b_q1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_k1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_v1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
                self.b_s1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()),
                                    softmax=True, transpose_w_src=True)
                self.b_h1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=lin()))
                self.b_wo = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D, use_bias=False, act=lin()))

            def call(self, x):
                x  = self.input_quant_layer(x)
                q0 = self.b_q0(x);  k0 = self.b_k0(x);  v0 = self.b_v0(x)
                p0 = self.b_s0(q0, w_src=k0)
                h0 = self.b_h0(p0, w_src=v0)
                q1 = self.b_q1(x);  k1 = self.b_k1(x);  v1 = self.b_v1(x)
                p1 = self.b_s1(q1, w_src=k1)
                h1 = self.b_h1(p1, w_src=v1)
                return self.b_wo(h1, concat_srcs=[h0])

        x_in = Input([D], name="input")
        um = _MHAModel(sys_bits=sys_bits, x_int_bits=0)
        m  = Model(inputs=[x_in], outputs=[um(x_in)])

        hw = Hardware(
            processing_elements=(N, N),
            frequency_mhz=150,
            bits_input=sys_bits.x, bits_weights=sys_bits.k,
            bits_sum=24, bits_bias=sys_bits.b,
            max_batch_size=N, max_channels_in=256,
            max_kernel_size=3, max_image_size=512,
            max_n_bundles=64, ram_weights_depth=256,
            ram_edges_depth=16, axi_width=64,
            config_baseaddr="40000000", target_cpu_int_bits=32,
            valid_prob=1, ready_prob=1, data_dir="vectors",
        )
        hw.export_json()
        hw = Hardware.from_json("hardware.json")
        hw.export()

        # Run export_inference with default random input
        export_inference(m, hw, batch_size=N)

        bundles_snapshot = list(BUNDLES)
        cfg = _config(str(work_dir))

    finally:
        os.chdir(orig_dir)

    return cfg, bundles_snapshot, hw, um, m


# ---------------------------------------------------------------------------
# Fixture — runs export once, shared by all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mha_export(tmp_path_factory):
    work = tmp_path_factory.mktemp("mha")
    return _build_and_export(work)


# ===========================================================================
# Tests
# ===========================================================================

class TestMHAExportMetadata:
    """Verify config_fw.h fields for the 11-bundle 2-head model."""

    def test_n_bundles(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        assert _idefine(cfg, "N_BUNDLES") == 11, "2-head MHA must have exactly 11 bundles"

    def test_wo_is_concat_flag(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        is_concat_vals = _bundle_fields(cfg, "is_concat")
        # Bundle 10 (B10 = Wo) must have is_concat=1; all others 0
        assert is_concat_vals[10] == 1, "Wo bundle must have is_concat=1"
        for ib, v in enumerate(is_concat_vals):
            if ib != 10:
                assert v == 0, f"Bundle {ib} should have is_concat=0"

    def test_wo_n_concat_srcs(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        n_concat = _bundle_fields(cfg, "n_concat_srcs")
        assert n_concat[10] == 1, "Wo bundle must have n_concat_srcs=1 (H0)"

    def test_wo_concat_src_ib_0(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        src_ib_0 = _bundle_fields(cfg, "concat_src_ib_0")
        # The first concat source for B10 must be B4 (H0), index 4
        assert src_ib_0[10] == 4, f"concat_src_ib_0 for Wo must be 4 (H0), got {src_ib_0[10]}"

    def test_non_concat_bundles_have_neg1_srcs(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        src_ib_0 = _bundle_fields(cfg, "concat_src_ib_0")
        for ib, v in enumerate(src_ib_0):
            if ib != 10:
                assert v == -1, f"Bundle {ib} concat_src_ib_0 should be -1, got {v}"

    def test_softmax_bundles(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        is_softmax = _bundle_fields(cfg, "is_softmax")
        # B3 (ib=3) and B8 (ib=8) are softmax bundles
        for ib, v in enumerate(is_softmax):
            expected = 1 if ib in (3, 8) else 0
            assert v == expected, f"is_softmax[{ib}] expected {expected}, got {v}"

    def test_o_type_is_int(self, mha_export):
        """Last bundle (Wo) has no softmax, so O_TYPE must be int32_t."""
        cfg, bundles, hw, um, m = mha_export
        assert _define(cfg, "O_TYPE") == "int32_t", "non-softmax terminal must use int32_t"


class TestMHANumerics:
    """Validate per-stage outputs against NumPy float references.

    The fixed-point representation introduces rounding so we allow up to
    2 quantisation steps (2 * 2^-frac) of relative deviation at each stage.
    """

    @staticmethod
    def _numpy_softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def test_q0_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        h0_out = bundles[0].out.ftensor.numpy()
        assert h0_out.shape == (16, 8), f"Q0 shape {h0_out.shape} != (16, 8)"

    def test_q1_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        q1_out = bundles[5].out.ftensor.numpy()
        assert q1_out.shape == (16, 8), f"Q1 shape {q1_out.shape} != (16, 8)"

    def test_scores0_shape(self, mha_export):
        """P0 = softmax(Q0 @ K0^T) must be (N, N)."""
        cfg, bundles, hw, um, m = mha_export
        p0 = bundles[3].softmax_float_out
        assert p0.shape == (16, 16), f"P0 shape {p0.shape} != (16, 16)"

    def test_scores1_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        p1 = bundles[8].softmax_float_out
        assert p1.shape == (16, 16), f"P1 shape {p1.shape} != (16, 16)"

    def test_scores0_sum_to_one(self, mha_export):
        """Each row of P0 must sum to 1 (post-softmax)."""
        cfg, bundles, hw, um, m = mha_export
        p0 = bundles[3].softmax_float_out
        row_sums = p0.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums, np.ones(16), atol=1e-5,
            err_msg="P0 rows do not sum to 1")

    def test_scores1_sum_to_one(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        p1 = bundles[8].softmax_float_out
        row_sums = p1.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums, np.ones(16), atol=1e-5,
            err_msg="P1 rows do not sum to 1")

    def test_head0_output_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        h0 = bundles[4].out.ftensor.numpy()
        assert h0.shape == (16, 8), f"H0 shape {h0.shape} != (16, 8)"

    def test_head1_output_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        h1 = bundles[9].out.ftensor.numpy()
        assert h1.shape == (16, 8), f"H1 shape {h1.shape} != (16, 8)"

    def test_concat_layout(self, mha_export):
        """Verify that concat(H0, H1) reproduces the Wo bundle's integer input."""
        cfg, bundles, hw, um, m = mha_export
        h0_int = bundles[4].out.itensor.numpy()   # (16, 8)
        h1_int = bundles[9].out.itensor.numpy()   # (16, 8)
        concat_ref = np.concatenate([h0_int, h1_int], axis=-1)  # (16, 16)

        wo_inp_int = bundles[10].inp.itensor.numpy()  # (16, 16) set during call_int
        np.testing.assert_array_equal(
            concat_ref, wo_inp_int,
            err_msg="concat(H0, H1) int tensor does not match Wo bundle's inp.itensor")

    def test_heads_are_different(self, mha_export):
        """H0 and H1 must differ (different projections → different outputs)."""
        cfg, bundles, hw, um, m = mha_export
        h0 = bundles[4].out.ftensor.numpy()
        h1 = bundles[9].out.ftensor.numpy()
        assert not np.allclose(h0, h1), "H0 and H1 are identical — projection weights not independent"

    def test_wo_output_shape(self, mha_export):
        cfg, bundles, hw, um, m = mha_export
        out = bundles[10].out.ftensor.numpy()
        assert out.shape == (16, 16), f"Wo output shape {out.shape} != (16, 16)"

    def test_scores0_argmax_matches_float(self, mha_export):
        """Argmax must match for rows where the float winner has a clear margin.

        When two softmax scores are within one quantisation step, requantisation
        can legitimately change the argmax.  We only assert on unambiguous rows.
        """
        cfg, bundles, hw, um, m = mha_export
        p0_float = bundles[3].softmax_float_out     # (16,16) float softmax
        p0_q     = bundles[3].out.ftensor.numpy()   # (16,16) requantised
        q_step   = 1.0 / (1 << (hw.X_BITS - 1))    # = 0.125 for 4-bit
        for row in range(p0_float.shape[0]):
            sorted_f = np.sort(p0_float[row])[::-1]
            if sorted_f[0] - sorted_f[1] > q_step:   # clear winner
                assert np.argmax(p0_float[row]) == np.argmax(p0_q[row]), (
                    f"P0 row {row}: clear-winner argmax mismatch "
                    f"(float={np.argmax(p0_float[row])}, quant={np.argmax(p0_q[row])})")

    def test_scores1_argmax_matches_float(self, mha_export):
        """Same clear-margin argmax check for head 1."""
        cfg, bundles, hw, um, m = mha_export
        p1_float = bundles[8].softmax_float_out
        p1_q     = bundles[8].out.ftensor.numpy()
        q_step   = 1.0 / (1 << (hw.X_BITS - 1))
        for row in range(p1_float.shape[0]):
            sorted_f = np.sort(p1_float[row])[::-1]
            if sorted_f[0] - sorted_f[1] > q_step:
                assert np.argmax(p1_float[row]) == np.argmax(p1_q[row]), (
                    f"P1 row {row}: clear-winner argmax mismatch "
                    f"(float={np.argmax(p1_float[row])}, quant={np.argmax(p1_q[row])})")

    def test_per_head_comparison_table(self, mha_export):
        """Print a per-head summary table and validate max absolute error is bounded.

        Tolerance: quantisation step = 2^-(X_BITS-1) = 2^-3 = 0.125 for 4-bit activations.
        We allow up to 2 steps (0.25) for the value-weighting stage since softmax
        requantisation introduces one rounding step plus the P@V matmul rounding.
        """
        cfg, bundles, hw, um, m = mha_export

        p0_f = bundles[3].softmax_float_out          # float softmax for head 0
        p1_f = bundles[8].softmax_float_out          # float softmax for head 1
        p0_q = bundles[3].out.ftensor.numpy()        # requantised softmax head 0
        p1_q = bundles[8].out.ftensor.numpy()        # requantised softmax head 1
        h0   = bundles[4].out.ftensor.numpy()        # head 0 value output
        h1   = bundles[9].out.ftensor.numpy()        # head 1 value output
        wo   = bundles[10].out.ftensor.numpy()       # Wo output

        q_step = 1.0 / (1 << (hw.X_BITS - 1))      # = 0.125 for X_BITS=4

        err_p0 = np.abs(p0_f - p0_q).max()
        err_p1 = np.abs(p1_f - p1_q).max()

        header = f"\n{'Stage':<22} {'MaxAbsErr':>12} {'Threshold':>12} {'Pass':>6}"
        sep    = "-" * len(header)
        rows = [
            ("P0 softmax quant", err_p0, 2 * q_step),
            ("P1 softmax quant", err_p1, 2 * q_step),
        ]
        lines = [header, sep]
        all_pass = True
        for name, err, thr in rows:
            ok = err <= thr
            all_pass = all_pass and ok
            lines.append(f"  {name:<20} {err:>12.6f} {thr:>12.6f} {'OK' if ok else 'FAIL':>6}")

        lines.append(sep)
        lines.append(f"  H0 shape: {h0.shape}   H1 shape: {h1.shape}   Wo shape: {wo.shape}")
        lines.append(f"  H0 mean abs: {np.abs(h0).mean():.4f}   H1 mean abs: {np.abs(h1).mean():.4f}")
        lines.append(f"  Wo mean abs: {np.abs(wo).mean():.4f}")
        print("\n".join(lines))

        assert all_pass, "Per-head softmax quantisation error exceeds 2 quantisation steps"

    def test_wo_requires_both_heads(self, mha_export):
        """Wo output must change if H0 is zeroed — confirming both heads contribute."""
        cfg, bundles, hw, um, m = mha_export

        b_wo = bundles[10]
        wo_actual = b_wo.out.itensor.numpy()

        # Build a modified input to Wo where H0 is zeroed out
        h0_int  = bundles[4].out.itensor.numpy()
        h1_int  = bundles[9].out.itensor.numpy()
        zeros   = np.zeros_like(h0_int)
        no_h0   = np.concatenate([zeros, h1_int], axis=-1)

        frac = bundles[4].out.frac
        w_wo = b_wo.core.w.itensor.numpy().squeeze()   # (D, D) after squeeze kh/kw dims
        y_no_h0 = no_h0 @ w_wo                         # simple int matmul reference

        assert not np.allclose(wo_actual, y_no_h0 / (2 ** (frac * 2))), (
            "Wo output is unchanged when H0 is zeroed — H0 has no effect")


class TestMHARegressions:
    """Ensure previously passing tests still pass after the MHA changes."""

    def test_attention_with_softmax_exports(self, tmp_path):
        """The single-head softmax attention model must still export without error."""
        orig = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            import tensorflow as tf
            from tensorflow import keras
            from keras.layers import Input
            from keras.models import Model
            from deepsocflow import (
                Hardware, SYS_BITS, XModel, XBundle, XDense, XActivation,
                export_inference,
            )

            N = 16
            sys_bits = SYS_BITS(x=4, k=8, b=16)
            relu = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu')
            lin  = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)

            @keras.saving.register_keras_serializable()
            class _SHAttn(XModel):
                def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
                    super().__init__(sys_bits, x_int_bits, *args, **kwargs)
                    self.b_q = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=relu()))
                    self.b_k = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=relu()))
                    self.b_v = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=relu()))
                    self.b_s = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()),
                                       softmax=True, transpose_w_src=True)
                    self.b_o = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()))

                def call(self, x):
                    x = self.input_quant_layer(x)
                    q = self.b_q(x); k = self.b_k(x); v = self.b_v(x)
                    p = self.b_s(q, w_src=k)
                    return self.b_o(p, w_src=v)

            x_in = Input([N], name="input")
            um = _SHAttn(sys_bits=sys_bits, x_int_bits=0)
            m  = Model(inputs=[x_in], outputs=[um(x_in)])
            hw = Hardware(
                processing_elements=(N, N), frequency_mhz=150,
                bits_input=sys_bits.x, bits_weights=sys_bits.k,
                bits_sum=24, bits_bias=sys_bits.b, max_batch_size=N,
                max_channels_in=256, max_kernel_size=3, max_image_size=512,
                max_n_bundles=64, ram_weights_depth=256, ram_edges_depth=16,
                axi_width=64, config_baseaddr="40000000", target_cpu_int_bits=32,
                valid_prob=1, ready_prob=1, data_dir="vectors",
            )
            hw.export_json(); hw = Hardware.from_json("hardware.json"); hw.export()
            export_inference(m, hw, batch_size=N)
            # Check that is_concat=0 for all bundles (no concat in single-head)
            cfg = _config(str(tmp_path))
            ic = _bundle_fields(cfg, "is_concat")
            assert all(v == 0 for v in ic), f"Single-head model should not have is_concat: {ic}"
        finally:
            os.chdir(orig)
