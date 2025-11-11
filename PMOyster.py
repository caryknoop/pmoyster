"""
PMOyster - Poor Man's Oyster with Level Shift and Optimizations

LEVEL COMPARISON TABLE:
=========================================================================
PMOyster v2  | NL Iters | Recalc Steps | BM3D Params  | BM3D Passes (by function)
-------------|----------|--------------|--------------|-------------------------
Level 0      | 8        | 6            | [1,48,6,12]  | Deringing: 3 passes
             |          |              |              | Destaircase: 2 passes
             |          |              |              | Deblocking: 2 passes
-------------|----------|--------------|--------------|-------------------------
Level 1      | 6        | 6            | [2,32,4,8]   | Deringing: 3 passes
             |          |              |              | Destaircase: 2 passes
             |          |              |              | Deblocking: 2 passes
-------------|----------|--------------|--------------|-------------------------
Level 2      | 4        | 4            | [4,16,2,4]   | Deringing: 2 passes
             |          |              |              | Destaircase: 2 passes
             |          |              |              | Deblocking: 2 passes
-------------|----------|--------------|--------------|-------------------------
Level 3      | 2        | 3            | [4,8,1,4]    | Deringing: 1 pass
             |          |              |              | Destaircase: 2 passes
             |          |              |              | Deblocking: 1 pass
-------------|----------|--------------|--------------|-------------------------
Level 4      | 1        | 2            | [8,4,1,2]    | Deringing: 1 pass
             |          |              |              | Destaircase: 1 pass
             |          |              |              | Deblocking: 1 pass
=========================================================================

Notes:
- NL Iters: Only applies to Deringing function
- Recalc Steps: Only applies to Basic (motion estimation) function
- BM3D passes vary by function:
  * Deringing has 3 NL refinement loops at levels 0-1
  * Destaircase has 2 BM3D passes at levels 0-2, 1 pass at levels 3-4
  * Deblocking has 2 passes at levels 0-2, 1 pass at levels 3-4 (NEW: difference-based processing)
- Level 3+ uses simplified processing with early returns

FEATURES vs Original Oyster:
- Uses cuFFT backend for DFTTest2 (faster than nvrtc/cuda fallback chain)
- Shared DFTTest backend reused across calls (eliminates recreation overhead)
- Direct function calls (no class wrapper indirection)
- Optimized bitdepth conversions (only when needed)
- Level 0-1 include triple BM3D pass for ultimate quality
- Level 0 uses larger NLMeans windows (64 vs 32) and stronger sigma scaling
- Adaptive NLMeans parameters at level 0 (a=6, s=3 vs 8,4) to preserve fine detail
- Content-aware presets with automatic detection based on framerate
- Deblocking uses difference-based processing like original Oyster (better artifact removal)
- All temporal BM3D calls now use BM3Dv2 with integrated aggregation

PERFORMANCE OPTIMIZATIONS:
- sosize=0 for faster cuFFT operation
- _ensure_float32() helper reduces redundant conversions
- Faster Expr with akarin requirement
- Reduced redundant CopyFrameProps calls

PRESET FREQUENCY CUTOFFS:
- auto (default): fps > 25 uses "video" preset, fps ≤ 25 uses "film" preset
- film: 0.65 (Deringing/Destaircase), 0.18 (Deblocking) - preserves film grain
- balanced: 0.48 (Deringing/Destaircase), 0.12 (Deblocking) - middle ground
- video: 0.35 (Deringing/Destaircase), 0.08 (Deblocking) - aggressive filtering

BM3D Params Format: [block_step, bm_range, ps_num, ps_range]
"""

import vapoursynth as vs
import math

# --- CORE ---
core = vs.core

# === PLUGINS ===
if not hasattr(core, 'bm3dcuda_rtc'):
    raise RuntimeError("PMOyster Error: bm3dcuda_rtc plugin not found.")
BM3Dv2 = core.bm3dcuda_rtc.BM3Dv2   # ONLY V2 - NO MORE BM3D + VAggregate

if not hasattr(core, 'nlm_cuda'):
    raise RuntimeError("PMOyster Error: nlm_cuda plugin not found.")
NLMeans = core.nlm_cuda.NLMeans

# DFTTest2 with shared backend
try:
    from dfttest2 import DFTTest as DFTTest2, Backend
    _DFTTEST_BACKEND = Backend.cuFFT(device_id=0, in_place=True)
except ImportError:
    raise RuntimeError("PMOyster Error: dfttest2 not found. Please install vs-dfttest2 and enable dfttest2_cuda.dll")

if not hasattr(core, 'akarin'):
    raise RuntimeError("PMOyster Error: akarin plugin not found. Please install vs-akarin for optimal performance.")
Expr = core.akarin.Expr

# Faster NNEDI fallback
if hasattr(core, 'nnedi3cl'):
    NNEDI = core.nnedi3cl.NNEDI3CL
else:
    raise RuntimeError("PMOyster Error: Need NNEDI3CL")
    
MSuper = core.mvsf.Super
MAnalyze = core.mvsf.Analyze
MRecalculate = core.mvsf.Recalculate
MDegrain = core.mvsf.Degrain
Resample = core.fmtc.resample

MakeDiff = core.std.MakeDiff
MergeDiff = core.std.MergeDiff
CropRel = core.std.CropRel
Transpose = core.std.Transpose
BlankClip = core.std.BlankClip
AddBorders = core.std.AddBorders
MaskedMerge = core.std.MaskedMerge
ShufflePlanes = core.std.ShufflePlanes

fmtc_args = dict(fulls=True, fulld=True)
bitdepth_args = dict(bits=32, flt=1, fulls=True, fulld=True, dmode=1)
msuper_args = dict(hpad=0, vpad=0, sharp=2, levels=0)
manalyze_args = dict(search=3, truemotion=False, trymany=True, levels=0, badrange=-24, divide=0, dct=0)
mrecalculate_args = dict(truemotion=False, search=3, smooth=1, divide=0, dct=0)
mdegrain_args = dict(thscd1=16711680.0, thscd2=255.0)
nnedi_args = dict(field=1, dh=True, nns=4, qual=2, etype=1, nsize=0)

# ============================================================================
# NEW FORMAT & QUALITY SCALING LOGIC
# ============================================================================

SCALING = {
    "sd": {
        "film":     { "sigma": 1.20, "h": 1.05 },
        "video":    { "sigma": 1.35, "h": 1.15 },
        "balanced": { "sigma": 1.27, "h": 1.10 },
    },
    "hd": {
        "film":     { "sigma": 0.95, "h": 1.00 },
        "video":    { "sigma": 1.10, "h": 1.05 },
        "balanced": { "sigma": 1.03, "h": 1.02 },
    }
}

QUALITY = {
    "low":    { "sigma": 0.85, "h": 0.90, "thr": 0.85 },
    "medium": { "sigma": 1.00, "h": 1.00, "thr": 1.00 },
    "good":   { "sigma": 1.17, "h": 1.12, "thr": 1.10 },
}

TEMPORAL = {
    "sd": {
        "film":     { "sad":  800, "radius": 2 },
        "video":    { "sad": 1200, "radius": 3 },
        "balanced": { "sad": 1000, "radius": 2 },
    },
    "hd": {
        "film":     { "sad":  600, "radius": 2 },
        "video":    { "sad":  900, "radius": 3 },
        "balanced": { "sad":  750, "radius": 2 },
    }
}

QUALITY_TEMP = {
    "low":    { "sad_mul": 0.85 },
    "medium": { "sad_mul": 1.00 },
    "good":   { "sad_mul": 1.15 },
}

def _resolve_scale(src, source_format, source_quality):
    # Resolution class
    res = "hd" if src.height >= 720 else "sd"

    # Auto mode resolves based on fps
    if source_format == "auto":
        fps = src.fps.numerator / src.fps.denominator
        source_format = "video" if fps > 25 else "film"

    fmt = SCALING[res][source_format]
    q = QUALITY[source_quality]

    return fmt, q

def _resolve_temporal(src, source_format, source_quality):
    res = "hd" if src.height >= 720 else "sd"

    if source_format == "auto":
        fps = src.fps.numerator / src.fps.denominator
        source_format = "video" if fps > 25 else "film"

    base = TEMPORAL[res][source_format]
    q = QUALITY_TEMP[source_quality]

    sad = base["sad"] * q["sad_mul"]
    radius = base["radius"]

    return sad, radius
    
# ============================================================================
# INTERNAL HELPER FUNCTIONS 
# ============================================================================

def _ensure_float32(clip):
    """Only convert if needed"""
    if clip.format.bits_per_sample != 32 or clip.format.sample_type != vs.FLOAT:
        return core.fmtc.bitdepth(clip, **bitdepth_args)
    return clip

def _freq_merge(low, hi, sbsize, slocation):
    """Reuse shared cuFFT backend"""
    sosize = 0  # cuFFT is faster with sosize=0

    hi_filtered = DFTTest2(hi, sbsize=sbsize, sosize=sosize, slocation=slocation,
                           smode=1, tbsize=1, backend=_DFTTEST_BACKEND)
    hif = MakeDiff(hi, hi_filtered)
    hif = _ensure_float32(hif)

    low_filtered = DFTTest2(low, sbsize=sbsize, sosize=sosize, slocation=slocation,
                            smode=1, tbsize=1, backend=_DFTTEST_BACKEND)

    result = MergeDiff(low_filtered, hif)
    return _ensure_float32(result)

def _safe_pad_size(size, a, s):
    total = a + s
    if total % 2 != 0:
        total += 1
    return total

def _nl_means(src, d, a, s, h, rclip, level):    
    # Adaptive search area for ultra-quality levels
    a_adaptive = [8, a, a, a, a][level]
    s_adaptive = [4, s, s, s, s][level]
    
    pad_total = _safe_pad_size(0, a_adaptive, s_adaptive)
    
    def duplicate(clip):
        """Add temporal padding by duplicating first frame d times"""
        if level <= 2 and d > 0:
            blank = Expr(clip[0], "0.0") * d
            return blank + clip + blank
        return clip
    
    pad = AddBorders(src, pad_total, pad_total, pad_total, pad_total)
    pad = duplicate(pad)
    
    if rclip:
        rclip = AddBorders(rclip, pad_total, pad_total, pad_total, pad_total)
        rclip = duplicate(rclip)
    
    nlm = NLMeans(pad, d=d, a=a_adaptive, s=s_adaptive, h=h, wref=1.0, rclip=rclip)
    crop = CropRel(nlm, pad_total, pad_total, pad_total, pad_total)
    return crop[d:crop.num_frames - d] if level <= 2 and d > 0 else crop

def _gen_block_mask(src):
    """Generate 8x8 block edge mask - works on any format (GRAY/YUV/RGB)"""
    luma = ShufflePlanes(src, 0, vs.GRAY)
    
    pattern = BlankClip(luma, 32, 32, color=0.0)
    inner = BlankClip(luma, 24, 24, color=0.0)
    inner = AddBorders(inner, 4, 4, 4, 4, color=1.0)
    pattern = MaskedMerge(pattern, inner, Expr(inner, "x 0.0 > 1.0 0.0 ?"))
    
    tiled = Resample(pattern, luma.width, luma.height, kernel="point", **fmtc_args)
    mask_luma = Expr(tiled, "x 0.0 > 1.0 0.0 ?")
    
    if src.format.num_planes == 1:
        return mask_luma
    else:
        u = ShufflePlanes(src, 1, vs.GRAY)
        v = ShufflePlanes(src, 2, vs.GRAY)
        return ShufflePlanes([mask_luma, u, v], [0, 0, 0], vs.YUV)

def _super(src, pel):
    src_pad = AddBorders(src, 128, 128, 128, 128)
    clip = Transpose(NNEDI(Transpose(NNEDI(src_pad, **nnedi_args)), **nnedi_args))
    if pel == 4:
        clip = Transpose(NNEDI(Transpose(NNEDI(clip, **nnedi_args)), **nnedi_args))
    return clip

def _basic(src, super_clip, radius, pel, sad, short_time, level):
    src_pad = AddBorders(src, 128, 128, 128, 128)
    supersoft = MSuper(src_pad, pelclip=super_clip, rfilter=4, pel=pel, **msuper_args)
    supersharp = MSuper(src_pad, pelclip=super_clip, rfilter=2, pel=pel, **msuper_args)
    recalc_steps = [6, 6, 4, 3, 2][level] if not short_time else [3, 3, 2, 2, 1][level]
    if short_time:
        c = 0.000198976
        me_sad_list = [c * (sad**2) * math.log(1.0 + 1.0/(c*sad)), sad]
        vmulti = MAnalyze(supersoft, radius=radius, overlap=4, blksize=8, **manalyze_args)
        for i in range(recalc_steps):
            ovlp = 4 // (2**i) if i < 2 else 1
            blksz = 8 // (2**i) if i < 2 else 2
            th = me_sad_list[min(i,1)]
            vmulti = MRecalculate(supersoft, vmulti, overlap=ovlp, blksize=blksz, thsad=th, **mrecalculate_args)
    else:
        c = 0.000013914
        me_sad = c * (sad**2) * math.log(1.0 + 1.0/(c*sad))
        vmulti = MAnalyze(supersoft, radius=radius, overlap=64, blksize=128, **manalyze_args)
        for i in range(recalc_steps):
            ovlp = 64 // (2**i)
            blksz = 128 // (2**i)
            vmulti = MRecalculate(supersoft, vmulti, overlap=ovlp, blksize=blksz, thsad=me_sad, **mrecalculate_args)
    degrained = MDegrain(src_pad, supersharp, vmulti, thsad=sad, **mdegrain_args)
    return CropRel(degrained, 128, 128, 128, 128)

# ============================================================================
# DERINGING
# ============================================================================
def _deringing(src, ref, radius, h, sigma, lowpass, level):
    c1, c2 = 0.113414, 2.8623
    strength = [h, h * (c1*h)**c2 * math.log(1.0 + 1.0/(c1*h)**c2), None]
    nl_iters = [5, 6, 4, 2, 1][level]
    if level == 4: strength[0] *= 1.2
    params = [[1,32,4,8],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    sigma_scaled = sigma * [1.4, 1.0, 1.0, 1.0, 1.0][level]

    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)

    bm3d_basic = BM3Dv2(src, ref=ref_prefiltered, sigma=sigma_scaled,
                        block_step=block_step, bm_range=bm_range,
                        ps_num=ps_num, ps_range=ps_range, radius=radius)
    bm3d_basic = core.std.CopyFrameProps(bm3d_basic, src)

    if level >= 3:
        return _freq_merge(src, bm3d_basic, sbsize, lowpass)

    def nl_loop(flt, src_clip, n):
        for i in range(n + 1):
            window = max(4, 64 // (2**i)) if level == 0 else max(4, 32 // (2**i))
            strength[2] = i * strength[0] / 4 + strength[1] * (1 - i/4)
            dif = MakeDiff(src_clip, flt)
            dif = _nl_means(dif, 0, window, 1, strength[2], flt, level)
            flt = MergeDiff(flt, dif)
        return flt

    refined = nl_loop(bm3d_basic, src, nl_iters - 1)
    sigma_final = sigma_scaled * (0.8 if level == 2 else 0.75 if level < 2 else 1.0)
    
    bm3d_final = BM3Dv2(refined, ref=bm3d_basic, sigma=sigma_final,
                        block_step=block_step, bm_range=bm_range,
                        ps_num=ps_num, ps_range=ps_range, radius=radius)
    bm3d_final = _freq_merge(refined, bm3d_final, sbsize, lowpass)
    
    if level <= 1:
        refined2 = nl_loop(bm3d_final, refined, nl_iters - 1)
        bm3d_ultra = BM3Dv2(refined2, ref=bm3d_final, sigma=sigma_final * 0.85,
                            block_step=block_step, bm_range=bm_range,
                            ps_num=ps_num, ps_range=ps_range, radius=radius)
        bm3d_ultra = _freq_merge(refined2, bm3d_ultra, sbsize, lowpass)
        return nl_loop(bm3d_ultra, refined2, nl_iters - 1)
    
    return nl_loop(bm3d_final, refined, nl_iters - 1)

# ============================================================================
# DESTAIRCASE
# ============================================================================
def _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level):
    mask = _gen_block_mask(src)
    
    params = [[1,32,4,8],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    
    sigma_scaled = sigma * [1.2, 1.0, 0.9, 0.85, 0.8][level]

    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)
    ref_prefiltered = Expr([src, ref_prefiltered], f"x y - abs {thr} > y x ?")
    ref_prefiltered = core.std.CopyFrameProps(ref_prefiltered, src)

    bm3d_basic = BM3Dv2(src, ref=ref_prefiltered, sigma=sigma_scaled,
                        block_step=block_step, bm_range=bm_range,
                        ps_num=ps_num, ps_range=ps_range, radius=radius)

    if level >= 3:
        return MaskedMerge(src, bm3d_basic, mask)
    else:
        sigma_final = sigma_scaled * [0.7, 0.75, 0.8, 0.85, 1.0][level]
        bm3d_final = BM3Dv2(src, ref=bm3d_basic, sigma=sigma_final,
                            block_step=block_step, bm_range=bm_range,
                            ps_num=ps_num, ps_range=ps_range, radius=0)
        return MaskedMerge(src, bm3d_final, mask)

# ============================================================================
# DEBLOCKING
# ============================================================================
def _deblocking(src, ref, radius, h, sigma, lowpass, level):
    mask = _gen_block_mask(src)
    
    params = [[1,32,4,8],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    
    h_scaled = h * [1.3, 1.0, 0.75, 0.7, 0.65][level]
    sigma_scaled = sigma * [1.2, 1.0, 0.9, 0.8, 0.75][level]

    cleansed = _nl_means(ref, radius, 8, 4, h_scaled, ref, level) if level <= 2 else ref
    
    dif = MakeDiff(ref, cleansed)
    dif = BM3Dv2(dif, ref=cleansed, sigma=sigma_scaled,
                 block_step=block_step, bm_range=bm_range,
                 ps_num=ps_num, ps_range=ps_range, radius=radius)
    cleansed = MergeDiff(cleansed, dif)
    
    if level < 3:
        dif = MakeDiff(ref, cleansed)
        dif = BM3Dv2(dif, ref=cleansed, sigma=sigma_scaled*0.75,
                     block_step=block_step, bm_range=bm_range,
                     ps_num=ps_num, ps_range=ps_range, radius=0)
        cleansed = MergeDiff(cleansed, dif)
    
    ref_final = _freq_merge(cleansed, ref, sbsize, lowpass)
    src_final = _freq_merge(cleansed, src, sbsize, lowpass)
    
    return MaskedMerge(src_final, ref_final, mask)

# ============================================================================
# PUBLIC API
# ============================================================================

def Super(src, pel=2):
    """Create super clip for motion estimation.
    
    Args:
        src: Input clip (YUV or GRAY)
        pel: Pixel accuracy (2 or 4)
    
    Returns:
        Super clip for motion analysis
    """
    return _super(src, pel)

def Basic(
    src, super=None, *,
    radius=None, pel=2, sad=None, short_time=False, level=0,
    source_format="balanced", source_quality="medium"
):
    # Resolve temporal scaling
    auto_sad, auto_radius = _resolve_temporal(src, source_format, source_quality)

    # Respect user overrides
    if sad is None:
        sad = auto_sad
    if radius is None:
        radius = auto_radius

    super_clip = super if super is not None else _super(src, pel)
    return _basic(src, super_clip, radius, pel, sad, short_time, level)

    
def Deringing(
    src, ref=None, *,
    radius=3, h=6.4, sigma=10.0, lowpass=None, level=2,
    source_format="balanced", source_quality="medium"
):
    if ref is None:
        ref = src

    fmt, q = _resolve_scale(src, source_format, source_quality)

    # Apply scaling unless user explicitly passed a different value
    sigma *= fmt["sigma"] * q["sigma"]
    h     *= fmt["h"]     * q["h"]

    # Lowpass cutoff selection (film / balanced / video)
    if lowpass is None:
        if source_format == "film" or (source_format == "auto" and sigma <= 10):
            cutoff = 0.65
        elif source_format == "video" or (source_format == "auto" and sigma > 10):
            cutoff = 0.35
        else:
            cutoff = 0.48  # balanced

        lowpass = [0.0, sigma, cutoff, 1024.0, 1.0, 1024.0]

    return _deringing(src, ref, radius, h, sigma, lowpass, level)


def Destaircase(
    src, ref=None, *,
    radius=3, sigma=12.0, thr=0.03, elast=0.015, lowpass=None, level=2,
    source_format="balanced", source_quality="medium"
):
    if ref is None:
        ref = src

    fmt, q = _resolve_scale(src, source_format, source_quality)

    sigma *= fmt["sigma"] * q["sigma"]
    thr   *= q["thr"]
    elast *= q["thr"]

    # lowpass selection
    if lowpass is None:
        if source_format == "film":
            cutoff = 0.65
        elif source_format == "video":
            cutoff = 0.35
        else:
            cutoff = 0.48
        lowpass = [0.0, sigma, cutoff, 1024.0, 1.0, 1024.0]

    return _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level)


def Deblocking(
    src, ref=None, *,
    radius=3, h=6.4, sigma=16.0, lowpass=None, level=2,
    source_format="balanced", source_quality="medium"
):
    if ref is None:
        ref = src

    fmt, q = _resolve_scale(src, source_format, source_quality)

    sigma *= fmt["sigma"] * q["sigma"]
    h     *= fmt["h"]     * q["h"]

    # lowpass selection specifically for blocking
    if lowpass is None:
        if source_format == "film":
            cutoff = 0.22
        elif source_format == "video":
            cutoff = 0.08
        else:
            cutoff = 0.12
        lowpass = [0.0, 0.0, cutoff, 1024.0, 1.0, 1024.0]

    return _deblocking(src, ref, radius, h, sigma, lowpass, level)
