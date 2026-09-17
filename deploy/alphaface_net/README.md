# AlphaFace network definition (vendored)

`Swapper_Units_AdIN.py` is copied unmodified from
[andrewyu90/Alphaface_Official](https://github.com/andrewyu90/Alphaface_Official),
licensed MIT in that repository (see `LICENSE`). It is used only at image build
time, by `deploy/alphaface_export.py`, to rebuild the network and export it to
ONNX. Nothing here runs at request time.

The paper, *AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial
Pose* (Yu et al., arXiv:2601.16429), states **CC BY-NC-SA 4.0**. The two licences
disagree; treat the model as non-commercial.
