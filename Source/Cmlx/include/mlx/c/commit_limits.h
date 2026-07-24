/* Copyright © 2024 Apple Inc. */

#ifndef MLX_COMMIT_LIMITS_H
#define MLX_COMMIT_LIMITS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C7 (tesseract): runtime Metal commit-policy override on the GPU device.
 * A value of 0 leaves that leg unchanged. Call around model load, not
 * during in-flight eval. Commit points are scheduling boundaries only;
 * results are commit-point invariant. */
void mlx_metal_set_commit_limits(
    int32_t max_mb_per_buffer,
    int32_t max_mb_output_per_buffer,
    int32_t max_ops_per_buffer);

#ifdef __cplusplus
}
#endif

#endif
