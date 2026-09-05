// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Update a slice of `array` starting at dynamic (array-valued) indices
/// (tesseract C8). Wraps `mlx_slice_update_dynamic`, which the indexing
/// layer does not reach: subscript slice-assign requires host-integer
/// bounds. The dynamic form lets the start offset be a lazy scalar so
/// graphs that write into a rolling buffer (speculative-decode KV
/// staging) can be built before the write position is known on the host.
///
/// - Parameters:
///   - array: the destination array.
///   - update: the update values; its shape must fit within `array`
///     starting at `start` along `axes` (leading singleton broadcast
///     follows `mlx.core.slice_update` semantics).
///   - start: integer array with one element per entry in `axes`.
///   - axes: the axes `start` applies to; unnamed axes start at 0.
/// - Returns: a new array with the slice replaced.
public func dynamicSliceUpdated(
    _ array: MLXArray, update: MLXArray, start: MLXArray, axes: [Int],
    stream: StreamOrDevice = .default
) -> MLXArray {
    let axes32 = axes.map(Int32.init)
    var result = mlx_array_new()
    mlx_slice_update_dynamic(
        &result, array.ctx, update.ctx, start.ctx, axes32, axes32.count, stream.ctx)
    return MLXArray(result)
}

/// Read a slice of `array` starting at dynamic (array-valued) indices
/// (tesseract C8). Wraps `mlx_slice_dynamic`: `start` is an int array with
/// one element per entry in `axes`, `sliceSize` the full output shape. The
/// result is a fresh contiguous array, so a window whose position is a lazy
/// scalar copies once instead of gathering.
public func dynamicSlice(
    _ array: MLXArray, start: MLXArray, axes: [Int], sliceSize: [Int],
    stream: StreamOrDevice = .default
) -> MLXArray {
    let axes32 = axes.map(Int32.init)
    let size32 = sliceSize.map(Int32.init)
    var result = mlx_array_new()
    mlx_slice_dynamic(
        &result, array.ctx, start.ctx, axes32, axes32.count, size32, size32.count, stream.ctx)
    return MLXArray(result)
}
