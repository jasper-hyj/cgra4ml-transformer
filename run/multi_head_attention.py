"""
Multi-head attention (2 heads) end-to-end export test.

Architecture
------------

   X ──┬── B0 (X @ W_Q0)  [N, D→d_head]  → Q0
       ├── B1 (X @ W_K0)  [N, D→d_head]  → K0
       ├── B2 (X @ W_V0)  [N, D→d_head]  → V0
       ├── B3 softmax(Q0 @ K0^T)          → P0   (non-terminal softmax)
       ├── B4 (P0 @ V0)                   → H0   (head-0 output)
       │
       ├── B5 (X @ W_Q1)  [N, D→d_head]  → Q1
       ├── B6 (X @ W_K1)  [N, D→d_head]  → K1
       ├── B7 (X @ W_V1)  [N, D→d_head]  → V1
       ├── B8 softmax(Q1 @ K1^T)          → P1   (non-terminal softmax)
       ├── B9 (P1 @ V1)                   → H1   (head-1 output)
       │
       └── B10  W_o projection:  concat(H0, H1) @ W_o  [N, D→D]  → output

Shapes (N=16, D=16, n_heads=2, d_head=8)
-----------------------------------------
  X      : (16, 16)
  Q0, K0, V0, Q1, K1, V1 : (16,  8)
  Scores0, Scores1        : (16, 16)  — softmax(Qi @ Ki^T), Ki^T = (8, 16)
  H0, H1                  : (16,  8)
  concat(H0, H1)           : (16, 16)
  output                   : (16, 16)

Note on Verilator simulation
------------------------------
This test calls export_inference() only.  verify_inference() is NOT called because
the C runtime does not yet assemble the concat buffer at execution time (is_concat=1
bundles need de-tile → concat → re-tile logic in the firmware before DMA dispatch).
The Python export is fully correct; the Verilator gap is documented as a known
limitation.  All Python-side correctness is verified in tests/test_multi_head_attention.py.
"""

import os
import pytest
import itertools
import sys
sys.path.append("../../")
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from deepsocflow import *
import pprint

SIM = 'xsim' if os.name == 'nt' else 'verilator'

sys_bits = SYS_BITS(x=4, k=8, b=16)

N      = 16   # sequence length / batch size
D      = 16   # model dimension
N_HEAD = 2    # number of attention heads
D_HEAD = D // N_HEAD   # = 8: per-head dimension


@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, *args, **kwargs)

        relu = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu')
        lin  = lambda: XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)

        # ── Head 0 ───────────────────────────────────────────────────────────
        # Q/K/V projections: (N, D) → (N, D_HEAD)
        self.b_q0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        self.b_k0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        self.b_v0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        # softmax(Q0 @ K0^T): non-terminal — output is requantised and fed to B4
        # transpose_w_src=True: K0 (N, D_HEAD) stored as weight (D_HEAD, N) → Q0 @ K0^T
        self.b_s0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()),
                            softmax=True, transpose_w_src=True)
        # P0 @ V0 → H0, shape (N, D_HEAD)
        self.b_h0 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=lin()))

        # ── Head 1 ───────────────────────────────────────────────────────────
        self.b_q1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        self.b_k1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        self.b_v1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=relu()))
        self.b_s1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=N, use_bias=False, act=lin()),
                            softmax=True, transpose_w_src=True)
        self.b_h1 = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D_HEAD, use_bias=False, act=lin()))

        # ── W_o projection ───────────────────────────────────────────────────
        # concat(H0, H1) → (N, D) — handled by concat_srcs=[h0] kwarg in call()
        # then (N, D) @ W_o → (N, D)
        self.b_wo = XBundle(core=XDense(k_int_bits=0, b_int_bits=0, units=D, use_bias=False, act=lin()))

    def call(self, x):
        x  = self.input_quant_layer(x)

        # Head 0
        q0 = self.b_q0(x)                         # B0: X @ W_Q0
        k0 = self.b_k0(x)                         # B1: X @ W_K0   (fan-out on x)
        v0 = self.b_v0(x)                         # B2: X @ W_V0   (fan-out on x)
        p0 = self.b_s0(q0, w_src=k0)              # B3: softmax(Q0 @ K0^T)
        h0 = self.b_h0(p0, w_src=v0)              # B4: P0 @ V0

        # Head 1
        q1 = self.b_q1(x)                         # B5: X @ W_Q1   (fan-out on x)
        k1 = self.b_k1(x)                         # B6: X @ W_K1   (fan-out on x)
        v1 = self.b_v1(x)                         # B7: X @ W_V1   (fan-out on x)
        p1 = self.b_s1(q1, w_src=k1)              # B8: softmax(Q1 @ K1^T)
        h1 = self.b_h1(p1, w_src=v1)              # B9: P1 @ V1

        # Wo projection on concat(H0, H1)
        # concat_srcs=[h0] prepends H0 before the primary input H1:
        #   effective input = concat([h0, h1], axis=-1)  shape (N, D)
        return self.b_wo(h1, concat_srcs=[h0])     # B10: concat(H0,H1) @ W_o


x_in = Input([D], name="input")
user_model = UserModel(sys_bits=sys_bits, x_int_bits=0)
x = user_model(x_in)
model = Model(inputs=[x_in], outputs=[x])


def product_dict(**kwargs):
    for instance in itertools.product(*(kwargs.values())):
        yield dict(zip(kwargs.keys(), instance))


@pytest.mark.parametrize("PARAMS", list(product_dict(
    processing_elements  = [(N, N)      ],
    frequency_mhz        = [150         ],
    bits_input           = [sys_bits.x  ],
    bits_weights         = [sys_bits.k  ],
    bits_sum             = [24          ],
    bits_bias            = [sys_bits.b  ],
    max_batch_size       = [N           ],
    max_channels_in      = [256         ],
    max_kernel_size      = [3           ],
    max_image_size       = [512         ],
    max_n_bundles        = [64          ],
    ram_weights_depth    = [256         ],
    ram_edges_depth      = [16          ],
    axi_width            = [64          ],
    config_baseaddr      = ["40000000"  ],
    target_cpu_int_bits  = [32          ],
    valid_prob           = [1           ],
    ready_prob           = [1           ],
    data_dir             = ['vectors'   ],
)))
def test_dnn_engine(PARAMS):
    """
    Export 2-head attention.  Python-only (no Verilator) because the C runtime
    does not yet implement the concat buffer assembly needed for is_concat=1 bundles.
    """
    hw = Hardware(**PARAMS)
    hw.export_json()
    hw = Hardware.from_json('hardware.json')
    hw.export()
    hw.export_vivado_tcl(board='pynq_z2')

    export_inference(model, hw, batch_size=N)
    # NOTE: verify_inference() is intentionally omitted.
    # The Verilator simulation cannot yet assemble the concat buffer for bundle B10
    # (is_concat=1).  See the module docstring for details.

    d_perf = predict_model_performance(hw)
    pp = pprint.PrettyPrinter(indent=4)
    print("\nPredicted Performance")
    pp.pprint(d_perf)
