"""StereoLite_yolo_ctx design package.

StereoLite with a dedicated context-encoder stream inspired by RAFT-Stereo
(Fig 1, bottom stream). The context encoder runs on the LEFT image only,
produces features at 1/4 resolution, and those features are:

  1. Used to initialise the GRU hidden state (the tile `feat` slot) at the
     coarsest scale via interpolation to 1/16.
  2. Projected to the matching-feature resolution at each scale and
     concatenated into the per-iteration GRU input context.

This is the genuinely new axis vs the existing `raftlike` and `tilegru`
siblings: those re-use the matching features (fL) as the GRU context.
RAFT-Stereo uses a separate encoder because the hidden state should
represent long-range image structure (occlusions, edges, repeated
texture) that is independent of the current disparity hypothesis.
"""