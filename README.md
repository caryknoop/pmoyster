# Poor Man's Oyster

Modeled after https://github.com/IFeelBloated/Oyster

Using CUDA wherever possible.  Levels 0..4 allows for different processing speeds (0 the slowest level)

Currrently in development (Alpha)

# License

LGPL v3.0

# Usage example

    # clip has to be float 32 
    level = 2   # set level from 0..4 (0 being the slowest)

    # Grey only!
    clip_y, chroma = PMOyster.ChromaSave(clip)   # we do not want to deal with UV stuff Y only!  
    
    super = PMOyster.Super(clip_y, pel=4)
    ref = PMOyster.Basic(clip_y, super=super, radius=3, sad=1000.0, pel=4)
    clip_y = PMOyster.Deringing(clip_y, ref=ref, sigma=10.0, level=level)
    clip_y = PMOyster.Destaircase(clip_y, ref, radius=3, sigma=12.0, thr=0.03,
                                   elast=0.015, lowpass=None, level=level)
    
    # Restore chroma once at the end
    clip = PMOyster.ChromaRestore(clip_y, chroma) # restore the original UV

# Required Plugins

- bm3dcuda_rtc - CUDA-accelerated BM3D denoising
- nlm_cuda - CUDA-accelerated Non-Local Means denoising
- dfttest2 (with CUDA support) - Frequency domain filtering using cuFFT backend
- akarin (vs-akarin) - Required for optimized Expr operations
- nnedi3cl - OpenCL-accelerated NNEDI3 interpolation
- mvsf - MVTools Super Fast (motion compensation functions)

# Level comparison table

| PMOyster v2 | NL Iters | Recalc Steps | BM3D Params | BM3D Passes (by function) |
|-------------|----------|--------------|-------------|---------------------------|
| **Level 0** | 8        | 6            | [1,48,6,12] | Deringing: 3 passes<br>Destaircase: 2 passes<br>Deblocking: 2 passes |
| **Level 1** | 6        | 6            | [2,32,4,8]  | Deringing: 3 passes<br>Destaircase: 2 passes<br>Deblocking: 2 passes |
| **Level 2** | 4        | 4            | [4,16,2,4]  | Deringing: 2 passes<br>Destaircase: 2 passes<br>Deblocking: 2 passes |
| **Level 3** | 2        | 3            | [4,8,1,4]   | Deringing: 1 pass<br>Destaircase: 2 passes<br>Deblocking: 1 pass |
| **Level 4** | 1        | 2            | [8,4,1,2]   | Deringing: 1 pass<br>Destaircase: 1 pass<br>Deblocking: 1 pass |

# Notes

- NL Iters: Only applies to Deringing function
- Recalc Steps: Only applies to Basic (motion estimation) function
- BM3D passes vary by function:
  * Deringing has 3 NL refinement loops at levels 0-1
  * Destaircase has 2 BM3D passes at levels 0-2, 1 pass at levels 3-4
  * Deblocking has 2 passes at levels 0-2, 1 pass at levels 3-4 (NEW: difference-based processing)
- Level 3+ uses simplified processing with early returns

# FEATURES vs Original Oyster

- Uses cuFFT backend for DFTTest2 (faster than nvrtc/cuda fallback chain)
- Shared DFTTest backend reused across calls (eliminates recreation overhead)
- Direct function calls (no class wrapper indirection)
- Optimized bitdepth conversions (only when needed)
- Level 0-1 include triple BM3D pass for ultimate quality
- Level 0 uses larger NLMeans windows (64 vs 32) and stronger sigma scaling
- Adaptive NLMeans parameters at level 0 (a=6, s=3 vs 8,4) to preserve fine detail
- Content-aware presets with automatic detection based on framerate
- Deblocking uses difference-based processing like original Oyster (better artifact removal)

# PERFORMANCE OPTIMIZATIONS

- sosize=0 for faster cuFFT operation
- _ensure_float32() helper reduces redundant conversions
- Faster Expr with akarin requirement
- Reduced redundant CopyFrameProps calls
- Fixed VAggregate syntax (bm3d.VAggregate vs direct call)

# PRESET FREQUENCY CUTOFFS

- auto (default): fps > 30 uses "video" preset, fps ≤ 30 uses "film" preset
- film: 0.65 (Deringing/Destaircase), 0.18 (Deblocking) - preserves film grain
- balanced: 0.48 (Deringing/Destaircase), 0.12 (Deblocking) - middle ground
- video: 0.35 (Deringing/Destaircase), 0.08 (Deblocking) - aggressive filtering

BM3D Params Format: [block_step, bm_range, ps_num, ps_range]
