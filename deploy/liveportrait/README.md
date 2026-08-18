# Vendored: FasterLivePortrait

Source: https://github.com/warmshao/FasterLivePortrait (MIT, © 2025 warmshao)
Upstream model: https://github.com/KlingAIResearch/LivePortrait (MIT, © 2024 Kuaishou)

Vendored rather than reimplemented: the implicit-keypoint transformation and
stitching maths is easy to get subtly wrong, and a wrong port produces plausible
but degraded output rather than an obvious failure.

Removed from the copy: `kokoro` (TTS), `JoyVASA` (audio-driven motion) and
`XPose` (animal landmarks) — none are used by the human portrait path, and
XPose alone pulls a large transformer stack.

Licences retained in `LICENSE.FasterLivePortrait`.
