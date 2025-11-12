"""
Poor Man's Oyster with Redesigned Scaling System


CRITICAL FIXES IN v0.25:
- Fixed ThrMerge parameter order (flt/src were swapped)
- Fixed NL loop window calculation (now matches Blueprint's exponential decay)
- Store intermediate refs in Destaircase (no more inline MergeDiff in BM3D calls)
- Destaircase defaults now match Blueprint (radius=6, sigma=16.0, thr=0.03125)
- Reduced Deringing sigma scaling (was too aggressive at 2.0x, now 1.5x max)
- NL loop now correctly uses min window of 2 (not 4)

LEVEL COMPARISON TABLE:
=========================================================================
PMOyster v2  | NL Iters | Recalc Steps | BM3D Params  | BM3D Passes (by function)
-------------|----------|--------------|--------------|-------------------------
Level 0      | 8        | 6            | [1,32,4,8]  | Deringing: 3 passes
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

FEATURES vs Original Oyster:
- Uses cuFFT backend for DFTTest2 (faster than nvrtc/cuda fallback chain)
- Shared DFTTest backend reused across calls (eliminates recreation overhead)
- Direct function calls (no class wrapper indirection)
- Optimized bitdepth conversions (only when needed)
- Level 0-1 include triple BM3D pass for ultimate quality
- Content-aware presets with automatic detection based on framerate
- Difference-based processing matching original  architecture
- All temporal BM3D calls use BM3Dv2 with integrated aggregation
- Elastic threshold blending (ThrMerge) restored and corrected

ARCHITECTURAL ALIGNMENT:
- Deringing: Difference-domain BM3D → NL refinement loops → FreqMerge
- Destaircase: FreqMerge → ThrMerge → Diff-domain double BM3D → MaskedMerge
- Deblocking: NLMeans pre-clean → Diff-domain double BM3D → FreqMerge → MaskedMerge

PRESET FREQUENCY CUTOFFS:
- auto (default): fps > 25 uses "video" preset, fps ≤ 25 uses "film" preset
- film: 0.65 (Deringing/Destaircase), 0.22 (Deblocking) - preserves film grain
- balanced: 0.48 (Deringing/Destaircase), 0.12 (Deblocking) - middle ground
- video: 0.35 (Deringing/Destaircase), 0.08 (Deblocking) - aggressive filtering

BM3D Params Format: [block_step, bm_range, ps_num, ps_range]
"""

import vapoursynth as vs
import math

core = vs.core

# === PLUGINS ===
if not hasattr(core, 'bm3dcuda_rtc'):
    raise RuntimeError("PMOyster Error: bm3dcuda_rtc plugin not found.")
BM3Dv2 = core.bm3dcuda_rtc.BM3Dv2

if not hasattr(core, 'nlm_cuda'):
    raise RuntimeError("PMOyster Error: nlm_cuda plugin not found.")
NLMeans = core.nlm_cuda.NLMeans

try:
    from dfttest2 import DFTTest as DFTTest2, Backend
    _DFTTEST_BACKEND = Backend.cuFFT(device_id=0, in_place=True)
except ImportError:
    raise RuntimeError("PMOyster Error: dfttest2 not found.")

if not hasattr(core, 'akarin'):
    raise RuntimeError("PMOyster Error: akarin plugin not found.")
Expr = core.akarin.Expr

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

# === SCALING & QUALITY ===
# IMPORTANT: format and correction are INDEPENDENT concepts
#
# format: Type of source material (affects grain preservation)
#   'film' = Film source with grain → GENTLER filtering to preserve grain
#   'video' = Video source (camcorder, digital) → Can use STRONGER filtering
#   'balanced' = Mixed or unknown source
#
# correction: How aggressively to process (independent of film/video)
#   'low' = Gentle processing (preserve maximum detail)
#   'medium' = Moderate processing (balanced approach)
#   'good' = Aggressive processing (maximum artifact removal)
#

SCALING = {
    "sd": {
        "film":     { "sigma": 0.75, "h": 0.85 },   # Film with grain: gentle to preserve grain
        "video":    { "sigma": 1.25, "h": 1.30 },   # Video source: can be more aggressive
        "balanced": { "sigma": 1.00, "h": 1.10 },   # Unknown/mixed: moderate
    },
    "hd": {
        "film":     { "sigma": 0.70, "h": 0.80 },   # HD film: even gentler (more grain visible)
        "video":    { "sigma": 1.20, "h": 1.25 },   # HD video: aggressive is safe
        "balanced": { "sigma": 0.95, "h": 1.05 },   # HD balanced: slightly gentle
    }
}

# Quality levels: How hard to process (INDEPENDENT of film/video type)
# These multiply the format-adjusted base values
CORRECTION = {
    "low":    { "sigma": 0.50, "h": 0.60, "thr": 0.70, "sad_mul": 0.75, "elast_mul": 0.6 },   # Gentle approach
    "medium": { "sigma": 1.00, "h": 1.00, "thr": 1.00, "sad_mul": 1.00, "elast_mul": 1.0 },   # Moderate approach
    "good":   { "sigma": 1.80, "h": 1.70, "thr": 1.40, "sad_mul": 1.30, "elast_mul": 1.5 },   # Aggressive approach
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
    """NLMeans with adaptive search area for level 0"""
    a_adaptive = 8 if level == 0 else a
    s_adaptive = 4 if level == 0 else s
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
    """
    Generate 8x8 block edge mask (Blueprint algorithm)
    Creates a 32x32 tile with 4px borders, then tiles it across the frame
    Mask = 1.0 at block edges, 0.0 in block centers
    """
    luma = ShufflePlanes(src, 0, vs.GRAY)
    
    # Step 1: Create 24x24 black square
    base = BlankClip(luma, 24, 24, color=0.0)
    
    # Step 2: Add 4px white borders to make 32x32 tile
    tile = AddBorders(base, 4, 4, 4, 4, color=1.0)
    
    # Step 3: Stack to create 4x4 grid (128x128)
    row = core.std.StackHorizontal([tile, tile, tile, tile])
    grid_4x4 = core.std.StackVertical([row, row, row, row])
    
    # Step 4: Resample 128x128 → 32x32 to anti-alias the pattern
    # This creates smooth edge detection instead of hard edges
    pattern_32 = Resample(grid_4x4, 32, 32, kernel="point", **fmtc_args)
    pattern_32 = Expr(pattern_32, "x 0.0 > 1.0 0.0 ?")
    
    # Step 5: Tile the 32x32 pattern across large area via stacking
    # Horizontal expansion: 32 → 256 (8x)
    h1 = core.std.StackHorizontal([pattern_32, pattern_32, pattern_32, pattern_32, 
                                    pattern_32, pattern_32, pattern_32, pattern_32])
    # Vertical expansion: 32 → 192 (6x)
    v1 = core.std.StackVertical([h1, h1, h1, h1, h1, h1])
    
    # Continue tiling to cover huge resolutions
    # Horizontal: 256 → 1536 (6x)
    h2 = core.std.StackHorizontal([v1, v1, v1, v1, v1, v1])
    # Vertical: 192 → 960 (5x)
    v2 = core.std.StackVertical([h2, h2, h2, h2, h2])
    
    # One more round to handle 4K
    # Horizontal: 1536 → 9216 (6x)
    h3 = core.std.StackHorizontal([v2, v2, v2, v2, v2, v2])
    # Vertical: 960 → 4800 (5x)
    mask_tiled = core.std.StackVertical([h3, h3, h3, h3, h3])
    
    # Step 6: Crop to actual frame size
    mask_luma = core.std.CropAbs(mask_tiled, luma.width, luma.height, 0, 0)
    
    # Step 7: Extend to YUV if needed
    if src.format.num_planes == 1:
        return mask_luma
    else:
        u = ShufflePlanes(src, 1, vs.GRAY)
        v = ShufflePlanes(src, 2, vs.GRAY)
        return ShufflePlanes([mask_luma, u, v], [0, 0, 0], vs.YUV)

def _super(src, pel):
    """Create super clip for motion estimation"""
    src_pad = AddBorders(src, 128, 128, 128, 128)
    clip = Transpose(NNEDI(Transpose(NNEDI(src_pad, **nnedi_args)), **nnedi_args))
    if pel == 4:
        clip = Transpose(NNEDI(Transpose(NNEDI(clip, **nnedi_args)), **nnedi_args))
    return clip

def _basic(src, super_clip, radius, pel, sad, short_time, level):
    """Motion-compensated temporal denoising"""
    src_pad = AddBorders(src, 128, 128, 128, 128)
    supersoft = MSuper(src_pad, pelclip=super_clip, rfilter=4, pel=pel, **msuper_args)
    supersharp = MSuper(src_pad, pelclip=super_clip, rfilter=2, pel=pel, **msuper_args)
    recalc_steps = [6, 6, 4, 3, 2][level] if not short_time else [3, 3, 2, 2, 1][level]
    
    if short_time:
        c = 0.0001989762736579584832432989326
        me_sad_list = [c * (sad**2) * math.log(1.0 + 1.0/(c*sad)), sad]
        vmulti = MAnalyze(supersoft, radius=radius, overlap=4, blksize=8, **manalyze_args)
        for i in range(recalc_steps):
            ovlp = 4 // (2**i) if i < 2 else 1
            blksz = 8 // (2**i) if i < 2 else 2
            th = me_sad_list[min(i, 1)]
            vmulti = MRecalculate(supersoft, vmulti, overlap=ovlp, blksize=blksz, thsad=th, **mrecalculate_args)
    else:
        c = 0.0000139144247313257680589719533
        me_sad = c * (sad**2) * math.log(1.0 + 1.0/(c*sad))
        vmulti = MAnalyze(supersoft, radius=radius, overlap=64, blksize=128, **manalyze_args)
        for i in range(recalc_steps):
            ovlp = 64 // (2**i)
            blksz = 128 // (2**i)
            vmulti = MRecalculate(supersoft, vmulti, overlap=ovlp, blksize=blksz, thsad=me_sad, **mrecalculate_args)
    
    degrained = MDegrain(src_pad, supersharp, vmulti, thsad=sad, **mdegrain_args)
    return CropRel(degrained, 128, 128, 128, 128)

# ============================================================================
# THR MERGE
# ============================================================================
def _thr_merge(flt, src, ref=None, thr=0.03125, elast=None, fast=False):
    """
    Elastic threshold merging - Blueprint implementation
    
    Args:
        flt: Filtered clip (to use if difference exceeds threshold)
        src: Source clip (to fall back to if difference is small)
        ref: Reference clip for difference calculation (defaults to src)
        thr: Threshold for difference detection
        elast: Elastic zone width (defaults to thr/2)
        fast: If True, use simple binary threshold (for speed)
    
    Returns:
        Merged clip with elastic transitions
    """
    ref = src if ref is None else ref
    elast = thr / 2.0 if elast is None else elast
    
    if fast or elast <= 0 or thr <= 0:
        # Simple binary threshold (fast path)
        return Expr([flt, src, ref], f"x z - abs {thr} > x y ?")
    
    # Full elastic blending (Blueprint algorithm)
    # BExp formula: x * ((thr+elast - z) / (2*elast)) + y * ((elast + z - thr) / (2*elast))
    tep = thr + elast
    te2 = 2.0 * elast
    BExp = f"x {tep} z - {te2} / * y {elast} z + {thr} - {te2} / * +"
    
    BDif = Expr(src, "0.0")
    PDif = Expr([flt, src], "x y - 0 max")
    PRef = Expr([flt, ref], "x y - 0 max")
    PBLD = Expr([PDif, BDif, PRef], BExp)
    
    NDif = Expr([flt, src], "y x - 0 max")
    NRef = Expr([flt, ref], "y x - 0 max")
    NBLD = Expr([NDif, BDif, NRef], BExp)
    
    BLDD = MakeDiff(PBLD, NBLD)
    BLD = MergeDiff(src, BLDD)
    
    # UDN = (|flt-ref| > thr - elast) ? BLD : flt
    UDN = Expr([flt, ref, BLD], f"x y - abs {thr} {elast} - > z x ?")
    # out = (|flt-ref| < thr + elast) ? UDN : src
    out = Expr([flt, ref, UDN, src], f"x y - abs {thr} {elast} + < z a ?")
    return out

# ============================================================================
# DESTAIRCASE - Staircase Artifact Removal
# ============================================================================
def _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level):
    """
    Remove staircase artifacts using Blueprint architecture:
    FreqMerge → ThrMerge → Diff-domain double BM3D → MaskedMerge
    """
    mask = _gen_block_mask(src)
    params = [[1,32,4,8], [2,32,4,8], [4,16,2,4], [4,8,1,4], [8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    
    # Conservative sigma scaling (Blueprint doesn't scale aggressively)
    sigma_scaled = sigma * [1.2, 1.1, 1.0, 0.95, 0.9][level]
    
    # Step 1: Frequency merge to prefilter reference
    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)
    
    # Step 2: Elastic threshold merge (CORRECTED parameter order)
    ref_prefiltered = _thr_merge(
        flt=ref_prefiltered,  # Use prefiltered if diff is large
        src=src,              # Fall back to source if diff is small
        ref=ref,              # Compare against original ref
        thr=thr,
        elast=elast,
        fast=(level >= 3)
    )
    
    # Step 3: First BM3D pass on difference signal
    dif = MakeDiff(src, ref_prefiltered)
    dif = BM3Dv2(dif, ref=ref_prefiltered, sigma=sigma_scaled,
                 block_step=block_step, bm_range=bm_range,
                 ps_num=ps_num, ps_range=ps_range, radius=radius)
    
    # Reconstruct intermediate reference
    ref_intermediate = MergeDiff(ref_prefiltered, dif)
    
    # Step 4: Second BM3D pass (for levels 0-2)
    if level < 3:
        dif2 = MakeDiff(src, ref_intermediate)
        sigma_final = sigma_scaled * 0.75
        dif2 = BM3Dv2(dif2, ref=ref_intermediate, sigma=sigma_final,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=0)
        cleaned_ref = MergeDiff(ref_intermediate, dif2)
    else:
        cleaned_ref = ref_intermediate
    
    # Step 5: Apply only to block boundaries
    return MaskedMerge(src, cleaned_ref, mask)

# ============================================================================
# DERINGING - Ringing Artifact Removal
# ============================================================================
def _deringing(src, ref, radius, h, sigma, lowpass, level):
    """
    Remove ringing artifacts using difference-domain BM3D + NL refinement
    
    """
    c1 = 0.1134141984932795312503328847998
    c2 = 2.8623043756241389436528021745239
    
    base_h = h
    h_curve = h * math.pow(c1 * h, c2) * math.log(1.0 + 1.0 / math.pow(c1 * h, c2))
    strength = [base_h, h_curve, None]
    
    # NL iterations per level
    nl_iters = [8, 6, 4, 2, 1][level]
    
    # BM3D search params per level
    params = [[1,32,4,8], [2,32,4,8], [4,16,2,4], [4,8,1,4], [8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    
    # Level-based sigma scaling: REDUCED from previous version
    # Previous: [1.5, 1.3, 1.1, 1.0, 0.95] was compounding the API scaling
    # New: Minimal scaling, let API quality settings do the work
    # Level 0-1 get slight boost for ultimate quality, others are neutral/reduced
    sigma_scaled = sigma * [1.15, 1.05, 1.00, 0.90, 0.85][level]
    
    # Prefilter reference with frequency merge
    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)
    
    # First BM3D pass on difference signal
    dif = MakeDiff(src, ref_prefiltered)
    dif = BM3Dv2(dif, ref=ref_prefiltered, sigma=sigma_scaled,
                 block_step=block_step, bm_range=bm_range,
                 ps_num=ps_num, ps_range=ps_range, radius=radius)
    ref_intermediate = MergeDiff(ref_prefiltered, dif)
    
    # Second BM3D pass (radius=0) for levels 0-2
    if level <= 2:
        dif2 = MakeDiff(src, ref_intermediate)
        sigma_final = sigma_scaled * [0.75, 0.78, 0.82][level]
        dif2 = BM3Dv2(dif2, ref=ref_intermediate, sigma=sigma_final,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=0)
        bm_ref = MergeDiff(ref_intermediate, dif2)
    else:
        bm_ref = ref_intermediate
    
    # NL refinement loops (CORRECTED window calculation)
    def nl_loop(init_clip, src_clip, iters):
        """
        NL refinement with exponentially decreasing window size

        """
        flt = init_clip
        for i in range(iters):
            # Exponential window decay: 64→32→16→8→4→2 or 32→16→8→4→2
            base_window = 64 if level == 0 else 32
            window = base_window >> i  # Equivalent to base_window / (2^i)
            window = max(2, window)    #
            
            # Strength interpolation
            if iters > 1:
                t = i / (iters - 1)
                strength[2] = t * strength[0] + (1 - t) * strength[1]
            else:
                strength[2] = strength[0]
            
            dif_nl = MakeDiff(src_clip, flt)
            dif_nl = _nl_means(dif_nl, 0, window, 1, strength[2], flt, level)
            flt = MergeDiff(flt, dif_nl)
        return flt
    
    # First NL refinement pass
    refined = nl_loop(bm_ref, src, nl_iters)
    
    # Ultra third pass for levels 0-1 (triple BM3D)
    if level <= 1:
        sigma_ultra = sigma_scaled * [0.65, 0.70][level]
        dif_ultra = MakeDiff(src, refined)
        dif_ultra = BM3Dv2(dif_ultra, ref=refined, sigma=sigma_ultra,
                           block_step=block_step, bm_range=bm_range,
                           ps_num=ps_num, ps_range=ps_range, radius=radius)
        ultra_ref = MergeDiff(refined, dif_ultra)
        ultra_ref = _freq_merge(refined, ultra_ref, sbsize, lowpass)
        refined = nl_loop(ultra_ref, refined, nl_iters)
    
    # Final frequency merge
    final = _freq_merge(src, refined, sbsize, lowpass)
    return final

# ============================================================================
# DEBLOCKING - Block Artifact Removal
# ============================================================================
def _deblocking(src, ref, radius, h, sigma, lowpass, level):
    """
    Remove blocking artifacts
    Architecture: NLMeans pre-clean → Diff-domain double BM3D → FreqMerge
    """
    mask = _gen_block_mask(src)
    params = [[1,32,4,8], [2,32,4,8], [4,16,2,4], [4,8,1,4], [8,4,1,2]]
    block_step, bm_range, ps_num, ps_range = params[level]
    sbsize = block_step * 2 + 1
    
    h_scaled = h * [1.3, 1.0, 0.75, 0.7, 0.65][level]
    sigma_scaled = sigma * [1.2, 1.0, 0.9, 0.8, 0.75][level]
    
    # Pre-clean with NLMeans (levels 0-2 only)
    cleansed = _nl_means(ref, radius, 8, 4, h_scaled, ref, level) if level <= 2 else ref
    
    # First BM3D pass on difference
    dif = MakeDiff(ref, cleansed)
    dif = BM3Dv2(dif, ref=cleansed, sigma=sigma_scaled,
                 block_step=block_step, bm_range=bm_range,
                 ps_num=ps_num, ps_range=ps_range, radius=radius)
    cleansed = MergeDiff(cleansed, dif)
    
    # Second BM3D pass (levels 0-2)
    if level < 3:
        dif = MakeDiff(ref, cleansed)
        dif = BM3Dv2(dif, ref=cleansed, sigma=sigma_scaled * 0.75,
                     block_step=block_step, bm_range=bm_range,
                     ps_num=ps_num, ps_range=ps_range, radius=0)
        cleansed = MergeDiff(cleansed, dif)
    
    # Frequency merge both ref and src
    ref_final = _freq_merge(cleansed, ref, sbsize, lowpass)
    src_final = _freq_merge(cleansed, src, sbsize, lowpass)
    
    # Apply only to block boundaries
    return MaskedMerge(src_final, ref_final, mask)

# ============================================================================
# PUBLIC API
# ============================================================================

def _resolve_scale(src, format, correction):
    """Resolve format and quality scaling factors"""
    res = "hd" if src.height >= 720 else "sd"
    if format == "auto":
        fps = src.fps.numerator / src.fps.denominator
        format = "video" if fps > 25 else "film"
    fmt = SCALING[res][format]
    q = CORRECTION[correction]
    return fmt, q

def _resolve_temporal(src, format, correction):
    """Resolve temporal filtering parameters"""
    res = "hd" if src.height >= 720 else "sd"
    if format == "auto":
        fps = src.fps.numerator / src.fps.denominator
        format = "video" if fps > 25 else "film"
    base = TEMPORAL[res][format]
    q = CORRECTION[correction]
    sad = base["sad"] * q["sad_mul"]
    radius = base["radius"]
    return sad, radius

def Super(src, pel=2):
    """
    Create super clip for motion estimation
    
    Args:
        src: Input clip (YUV or GRAY)
        pel: Pixel accuracy (2 or 4)
    
    Returns:
        Super clip for motion analysis
    """
    return _super(src, pel)

def Basic(src, super=None, *, radius=None, pel=2, sad=None, short_time=False, level=0,
          format="balanced", correction="medium"):
    """
    Motion-compensated temporal denoising
    
    Args:
        src: Input clip
        super: Pre-computed super clip (optional)
        radius: Temporal radius (auto-detected if None)
        pel: Pixel accuracy (2 or 4)
        sad: SAD threshold (auto-detected if None)
        short_time: Use short-time motion estimation
        level: Processing level (0-4, lower = higher quality)
        format: "film", "video", "balanced", or "auto"
        correction: "low", "medium", or "good"
    
    Returns:
        Temporally denoised clip
    """
    auto_sad, auto_radius = _resolve_temporal(src, format, correction)
    sad = auto_sad if sad is None else sad
    radius = auto_radius if radius is None else radius
    super_clip = super if super is not None else _super(src, pel)
    return _basic(src, super_clip, radius, pel, sad, short_time, level)

def Deringing(src, ref=None, *, radius=3, h=6.4, sigma=10.0, lowpass=None, level=2,
              format="balanced", correction="medium"):
    """
    Remove ringing artifacts
    
    Args:
        src: Input clip
        ref: Reference clip (defaults to src)
        radius: Temporal radius for BM3D
        h: NLMeans strength
        sigma: BM3D noise estimation (NOTE: CUDA BM3D values differ from CPU)
        lowpass: DFTTest frequency cutoff (auto if None)
        level: Processing level (0-4, lower=higher quality)
        format: Source material type (affects grain preservation)
            "film" - Film source with grain (gentler to preserve grain)
            "video" - Video source/camcorder (can be more aggressive)
            "balanced" - Unknown/mixed source
            "auto" - Auto-detect: fps ≤ 25 = film, fps > 25 = video
        correction: Processing aggressiveness (independent of format)
            "low" - Gentle processing (preserve maximum detail)
            "medium" - Moderate processing (balanced approach)
            "good" - Aggressive processing (maximum artifact removal)
    
    Returns:
        Deringed clip
    
    Notes:
        - format and correction are INDEPENDENT settings
        - Film sources have grain that should be preserved (use 'film' format)
        - Video sources can handle more aggressive filtering (use 'video' format)
        - Both film and video can have MPEG-2 compression artifacts
        - Use correction to control how hard to process regardless of type
    """
    if ref is None:
        ref = src
    
    fmt, q = _resolve_scale(src, format, correction)
    sigma *= fmt["sigma"] * q["sigma"]
    h *= fmt["h"] * q["h"]
    
    if lowpass is None:
        if format == "film" or (format == "auto" and sigma <= 10):
            cutoff = 0.65
        elif format == "video" or (format == "auto" and sigma > 10):
            cutoff = 0.35
        else:
            cutoff = 0.48
        lowpass = [0.0, sigma, cutoff, 1024.0, 1.0, 1024.0]
    
    return _deringing(src, ref, radius, h, sigma, lowpass, level)

def Destaircase(src, ref=None, *, radius=6, sigma=16.0, thr=0.03125, elast=0.015625, 
                lowpass=None, level=2, format="balanced", correction="medium"):
    """
    Remove staircase artifacts (DEFAULTS NOW MATCH BLUEPRINT)
    
    Args:
        src: Input clip
        ref: Reference clip (defaults to src)
        radius: Temporal radius for BM3D (Blueprint default: 6)
        sigma: BM3D noise estimation (Blueprint default: 16.0)
        thr: Threshold for artifact detection (Blueprint default: 0.03125)
        elast: Elastic zone width (Blueprint default: 0.015625)
        lowpass: DFTTest frequency cutoff (auto if None)
        level: Processing level (0-4)
        format: "film", "video", "balanced", or "auto"
        correction: "low", "medium", or "good"
    
    Returns:
        Destaircased clip
    """
    if ref is None:
        ref = src
    
    fmt, q = _resolve_scale(src, format, correction)
    sigma *= fmt["sigma"] * q["sigma"]
    thr *= q["thr"]
    elast *= q["thr"] * q["elast_mul"]
    
    if lowpass is None:
        if format == "film":
            cutoff = 0.65
        elif format == "video":
            cutoff = 0.35
        else:
            cutoff = 0.48
        lowpass = [0.0, sigma, cutoff, 1024.0, 1.0, 1024.0]
    
    return _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level)

def Deblocking(src, ref=None, *, radius=3, h=6.4, sigma=16.0, lowpass=None, level=2,
               format="balanced", correction="medium"):
    """
    Remove blocking artifacts
    
    Args:
        src: Input clip
        ref: Reference clip (defaults to src)
        radius: Temporal radius for BM3D
        h: NLMeans strength
        sigma: BM3D noise estimation
        lowpass: DFTTest frequency cutoff (auto if None)
        level: Processing level (0-4)
        format: "film", "video", "balanced", or "auto"
        correction: "low", "medium", or "good"
    
    Returns:
        Deblocked clip
    """
    if ref is None:
        ref = src
    
    fmt, q = _resolve_scale(src, format, correction)
    sigma *= fmt["sigma"] * q["sigma"]
    h *= fmt["h"] * q["h"]
    
    if lowpass is None:
        if format == "film":
            cutoff = 0.22
        elif format == "video":
            cutoff = 0.08
        else:
            cutoff = 0.12
        lowpass = [0.0, 0.0, cutoff, 1024.0, 1.0, 1024.0]
    
    return _deblocking(src, ref, radius, h, sigma, lowpass, level)
