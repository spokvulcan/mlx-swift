// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

extension GPU {
    /// Runtime override of the Metal command-buffer commit limits
    /// (tesseract C7). The decode regimes that matter — MoE
    /// boundary-limited vs dense starvation-limited — are
    /// indistinguishable from the GPU side but known to the app, which
    /// selects the model; MoE loads opt into the relaxed input cap.
    ///
    /// Commit points are scheduling boundaries only; results are
    /// commit-point invariant. Call around model load, not during
    /// in-flight eval.
    /// - Parameters:
    ///   - maxMBPerBuffer: unique input MB that forces a commit (0 = keep).
    ///   - maxMBOutputPerBuffer: output MB that forces a commit (0 = keep).
    ///   - maxOpsPerBuffer: op count that forces a commit (0 = keep).
    public static func setCommitLimits(
        maxMBPerBuffer: Int = 0,
        maxMBOutputPerBuffer: Int = 0,
        maxOpsPerBuffer: Int = 0
    ) {
        mlx_metal_set_commit_limits(
            Int32(maxMBPerBuffer),
            Int32(maxMBOutputPerBuffer),
            Int32(maxOpsPerBuffer))
    }
}
