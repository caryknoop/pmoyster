# Poor Man's Oyster

Modeled after https://github.com/IFeelBloated/Oyster

Using CUDA wherever possible.  Levels 0..4 allows for different processing speeds (0 the slowest level)

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

PMOyster   | NL Iters | Recalc Steps | BM3D Params  | Special Features
-----------|----------|--------------|--------------|------------------
Level 0    | 8        | 6            | [1,48,6,12]  | Triple BM3D pass
Level 1    | 6        | 6            | [2,32,4,8]   | Triple BM3D pass
Level 2    | 4        | 4            | [4,16,2,4]   | Double BM3D pass
Level 3    | 2        | 3            | [4,8,1,4]    | Double BM3D pass
Level 4    | 1        | 2            | [8,4,1,2]    | Single BM3D pass

Classic Oyster uses 4 NL iters, 5 recalc steps, double BM3D

# FEATURES vs Original Oyster:
- Uses cuFFT backend for DFTTest2 )
- Shared DFTTest backend reused across calls (eliminates recreation overhead)
- Direct function calls (no class wrapper indirection)
- Optimized bitdepth conversions (only when needed)
- Explicit GRAY input/output with ChromaSave/ChromaRestore helpers
- Level 0-1 include triple BM3D pass for ultimate quality
- Level 0 uses larger NLMeans windows (64 vs 32) and stronger sigma scaling
- Content-aware presets with automatic detection based on framerate

# PERFORMANCE OPTIMIZATIONS:
- sosize=0 for faster cuFFT operation
- _ensure_float32() helper reduces redundant conversions
- Faster Expr with akarin requirement
- Reduced redundant CopyFrameProps calls
- Fixed VAggregate syntax (bm3d.VAggregate vs direct call)

# PRESET FREQUENCY CUTOFFS:
- auto (default): fps > 30 uses "video" preset, fps ≤ 30 uses "film" preset
- film: 0.65 (Deringing/Destaircase), 0.18 (Deblocking) - preserves film grain
- balanced: 0.48 (Deringing/Destaircase), 0.12 (Deblocking) - middle ground
- video: 0.35 (Deringing/Destaircase), 0.08 (Deblocking) - aggressive filtering

BM3D Params Format: [block_step, bm_range, ps_num, ps_range]
