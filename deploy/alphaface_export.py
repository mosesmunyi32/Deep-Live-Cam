"""Rebuild AlphaFace from its published checkpoint and export it to ONNX.

Runs at image build time. The result, models/alphaface_256.onnx, then loads like
every other model here - onnxruntime, the card-aware provider options, and the
build's own session check - rather than being the one swapper that needs torch
and a different code path at request time.

Two things are folded into the exported graph so the runtime adapter stays thin:

* The identity projection. AlphaFace's identity encoder is w600k_r50 followed by
  a 512x512 linear layer loaded from emp.npy, then L2 normalisation - and emp.npy
  is byte-for-byte inswapper's own emap matrix. insightface already computes that
  w600k_r50 embedding for every face this server sees, so the graph takes the
  normalised embedding and does the projection itself. No second recognition
  network is loaded. (Normalising before or after a linear map changes nothing
  once the result is normalised again, which is why the normed embedding is a
  valid input.)

* The input range. The checkpoint was trained on RGB in [0, 1] and ends in a
  tanh, so the export clamps to [0, 1] rather than leaving that to every caller.

    python deploy/alphaface_export.py alphaface_demo.pt emp.npy models/alphaface_256.onnx
"""

import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphaface_net"))
from Swapper_Units_AdIN import Decoder, Encoder  # noqa: E402


class Swapper(nn.Module):
    """The published Swapper, minus the training-only attributes."""

    def __init__(self, source_dim=512):
        super().__init__()
        self.E = Encoder(source_dim)
        self.G = Decoder(1024, 3)

    def forward(self, target, source):
        return self.G(self.E(target, source))


class Exportable(nn.Module):
    def __init__(self, swapper, emap):
        super().__init__()
        self.swapper = swapper
        self.register_buffer("emap", torch.from_numpy(emap.astype(np.float32)))

    def forward(self, target, embedding):
        code = embedding @ self.emap
        code = code / torch.linalg.norm(code, dim=1, keepdim=True)
        return torch.clamp(self.swapper(target, code), 0.0, 1.0)


def main(checkpoint, emp_path, out_path):
    # mmap: the published checkpoint is 1.9 GB, most of it the discriminator
    # this never uses. Mapping the file rather than reading it keeps a build
    # machine from paging the whole thing into memory to get at the swapper.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = ck["swapper"] if isinstance(ck, dict) and "swapper" in ck else ck
    del ck
    swapper = Swapper(512)
    missing, unexpected = swapper.load_state_dict(state, strict=False)
    del state
    # Strict in spirit: every learned weight in the checkpoint must land. A few
    # buffers the network rebuilds itself are tolerated; missing learned
    # parameters are not, because they would export as random weights and the
    # model would produce noise without ever raising.
    learned_missing = [k for k in missing if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    if learned_missing or unexpected:
        raise SystemExit(f"checkpoint does not match the network: missing={learned_missing[:5]} unexpected={unexpected[:5]}")

    model = Exportable(swapper, np.load(emp_path)).eval()
    target = torch.rand(1, 3, 256, 256)
    embedding = torch.nn.functional.normalize(torch.randn(1, 512), dim=1)
    with torch.no_grad():
        reference = model(target, embedding)
        torch.onnx.export(
            model, (target, embedding), out_path,
            input_names=["target", "source"], output_names=["output"],
            opset_version=17, do_constant_folding=True, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"target": target.numpy(), "source": embedding.numpy()})[0]
    err = float(np.abs(got - reference.numpy()).max())
    print(f"exported {out_path}: {os.path.getsize(out_path) / 1e6:.0f} MB, onnx vs torch max diff {err:.2e}")
    if err > 1e-3:
        raise SystemExit("ONNX output diverges from torch - refusing to ship it")


if __name__ == "__main__":
    main(*sys.argv[1:4])
