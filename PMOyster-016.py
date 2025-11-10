"""
PMOyster v2 - Poor Man's Oyster with Level Shift and Optimizations

LEVEL COMPARISON TABLE:
=========================================================================
PMOyster v2  | NL Iters | Recalc Steps | BM3D Params  | Special Features
-------------|----------|--------------|--------------|------------------
Level 0      | 8        | 6            | [1,48,6,12]  | Triple BM3D pass
Level 1      | 6        | 6            | [2,32,4,8]   | Triple BM3D pass
Level 2      | 4        | 4            | [4,16,2,4]   | Double BM3D pass
Level 3      | 2        | 3            | [4,8,1,4]    | Double BM3D pass
Level 4      | 1        | 2            | [8,4,1,2]    | Single BM3D pass
=========================================================================



FEATURES vs Original Oyster:
- Uses cuFFT backend for DFTTest2 (faster than nvrtc/cuda fallback chain)
- Shared DFTTest backend reused across calls (eliminates recreation overhead)
- Direct function calls (no class wrapper indirection)
- Optimized bitdepth conversions (only when needed)
- Explicit GRAY input/output with ChromaSave/ChromaRestore helpers
- Level 0-1 include triple BM3D pass for ultimate quality
- Level 0 uses larger NLMeans windows (64 vs 32) and stronger sigma scaling
- Content-aware presets with automatic detection based on framerate

PERFORMANCE OPTIMIZATIONS:
- sosize=0 for faster cuFFT operation
- _ensure_float32() helper reduces redundant conversions
- Faster Expr with akarin requirement
- Reduced redundant CopyFrameProps calls
- Fixed VAggregate syntax (bm3d.VAggregate vs direct call)

PRESET FREQUENCY CUTOFFS:
- auto (default): fps > 30 uses "video" preset, fps ≤ 30 uses "film" preset
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
BM3D = core.bm3dcuda_rtc.BM3D

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
# CHROMA MANAGEMENT
# ============================================================================

def ChromaSave(clip):
    """
    Extract luma plane for processing and save chroma reference.
    
    Args:
        clip: YUV420PS input clip
    
    Returns:
        tuple: (gray_y_clip, chroma_reference)
    """
    if (clip.format.id != vs.YUV420PS or
        clip.format.sample_type != vs.FLOAT or
        clip.format.bits_per_sample != 32):
        raise TypeError("PMOyster: Input must be YUV420PS (32-bit float)")
    
    y_plane = ShufflePlanes(clip, 0, vs.GRAY)
    return y_plane, clip

def ChromaRestore(processed_y, chroma_source):
    """
    Restore chroma from saved reference to processed luma.
    
    Args:
        processed_y: GRAY processed luma plane
        chroma_source: Original YUV420PS clip containing chroma
    
    Returns:
        YUV420PS clip with processed luma and original chroma
    """
    if (chroma_source.format.id != vs.YUV420PS or
        chroma_source.format.sample_type != vs.FLOAT or
        chroma_source.format.bits_per_sample != 32):
        raise TypeError("PMOyster: Chroma source must be YUV420PS (32-bit float)")
    
    if processed_y.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Processed clip must be GRAY format")
    
    # Ensure processed_y is 32-bit float GRAYS
    if processed_y.format.bits_per_sample != 32 or processed_y.format.sample_type != vs.FLOAT:
        processed_y = core.fmtc.bitdepth(processed_y, **bitdepth_args)
    
    # Extract chroma planes from original
    u_plane = ShufflePlanes(chroma_source, 1, vs.GRAY)
    v_plane = ShufflePlanes(chroma_source, 2, vs.GRAY)
    
    # Combine planes
    result = ShufflePlanes([processed_y, u_plane, v_plane], [0, 0, 0], vs.YUV)
    
    # Ensure output is YUV420PS
    if result.format.id != vs.YUV420PS:
        result = core.fmtc.resample(result, css="420", **fmtc_args)
        result = core.fmtc.bitdepth(result, **bitdepth_args)
    
    return result

# ============================================================================
# INTERNAL HELPER FUNCTIONS (All expect GRAY input)
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

def _nl_means(src, d, a, s, h, rclip, level):
    def duplicate(clip):
        """Add temporal padding by duplicating first frame d times"""
        if level <= 2 and d > 0:
            blank = Expr(clip[0], "0.0") * d
            return blank + clip + blank
        return clip
    
    pad = AddBorders(src, a+s, a+s, a+s, a+s)
    pad = duplicate(pad)
    
    if rclip:
        rclip = AddBorders(rclip, a+s, a+s, a+s, a+s)
        rclip = duplicate(rclip)
    
    nlm = NLMeans(pad, d=d, a=a, s=s, h=h, channels="Y", wref=1.0, rclip=rclip)
    crop = CropRel(nlm, a+s, a+s, a+s, a+s)
    return crop[d:crop.num_frames - d] if level <= 2 and d > 0 else crop

def _gen_block_mask(src):
    pattern = BlankClip(src, 32, 32, color=0.0)
    inner = BlankClip(src, 24, 24, color=0.0)
    inner = AddBorders(inner, 4, 4, 4, 4, color=1.0)
    pattern = MaskedMerge(pattern, inner, Expr(inner, "x 0.0 > 1.0 0.0 ?"))
    tiled = Resample(pattern, src.width, src.height, kernel="point", **fmtc_args)
    return Expr(tiled, "x 0.0 > 1.0 0.0 ?")

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

    # LEVEL SHIFTED: old level 0-3 becomes 1-4, new level 0 added
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

    degrained = MDegrain(src_pad, supersharp, vmulti, thsad=sad, plane=0, **mdegrain_args)
    return CropRel(degrained, 128, 128, 128, 128)

def _deringing(src, ref, radius, h, sigma, lowpass, level):
    c1, c2 = 0.113414, 2.8623
    strength = [h, h * (c1*h)**c2 * math.log(1.0 + 1.0/(c1*h)**c2), None]
    
    # LEVEL SHIFTED: old [6,4,2,1] becomes [8,6,4,2,1]
    nl_iters = [8, 6, 4, 2, 1][level]
    if level == 4: strength[0] *= 1.2

    # LEVEL SHIFTED: old params become indices 1-4
    params = [[1,48,6,12],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]][level]
    block_step, bm_range, ps_num, ps_range = params
    sbsize = block_step * 2 + 1

    # Sigma scaling for new level 0
    sigma_scaled = sigma * [1.4, 1.0, 1.0, 1.0, 1.0][level]

    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)
    bm3d_basic = BM3D(src, ref=ref_prefiltered, sigma=sigma_scaled,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=radius)
    bm3d_basic = core.std.CopyFrameProps(bm3d_basic, src)

    if radius > 0:
        bm3d_basic = bm3d_basic.bm3d.VAggregate(radius=radius)

    bm3d_basic = _ensure_float32(bm3d_basic)

    if level >= 3:
        return _freq_merge(src, bm3d_basic, sbsize, lowpass)

    def nl_loop(flt, src_clip, n):
        for i in range(n + 1):
            window = max(4, 64 // (2**i)) if level == 0 else max(4, 32 // (2**i))  # Minimum window=4
            strength[2] = i * strength[0] / 4 + strength[1] * (1 - i/4)
            dif = MakeDiff(src_clip, flt)
            dif = _nl_means(dif, 0, window, 1, strength[2], flt, level)
            flt = MergeDiff(flt, dif)
        return flt

    refined = nl_loop(bm3d_basic, src, nl_iters - 1)

    # LEVEL SHIFTED: old level==1 becomes level==2
    sigma_final = sigma_scaled * (0.8 if level == 2 else 0.75 if level < 2 else 1.0)
    bm3d_final = BM3D(refined, ref=bm3d_basic, sigma=sigma_final,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=0)
    bm3d_final = _freq_merge(refined, bm3d_final, sbsize, lowpass)
    
    # For ultra-quality level 0-1, add third pass
    if level <= 1:
        refined2 = nl_loop(bm3d_final, refined, nl_iters - 1)
        bm3d_ultra = BM3D(refined2, ref=bm3d_final, sigma=sigma_final * 0.85,
                          block_step=block_step, bm_range=bm_range,
                          ps_num=ps_num, ps_range=ps_range, radius=0)
        bm3d_ultra = _freq_merge(refined2, bm3d_ultra, sbsize, lowpass)
        return nl_loop(bm3d_ultra, refined2, nl_iters - 1)
    
    return nl_loop(bm3d_final, refined, nl_iters - 1)

def _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level):
    """Internal destaircase - expects GRAY input"""
    mask = _gen_block_mask(src)
    
    # LEVEL SHIFTED: old params become indices 1-4
    params = [[1,48,6,12],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]][level]
    block_step, bm_range, ps_num, ps_range = params
    sbsize = block_step * 2 + 1
    
    # LEVEL SHIFTED: add level 0 scaling
    sigma_scaled = sigma * [1.2, 1.0, 0.9, 0.85, 0.8][level]

    ref_prefiltered = _freq_merge(src, ref, sbsize, lowpass)
    ref_prefiltered = Expr([src, ref_prefiltered], f"x y - abs {thr} > y x ?")
    ref_prefiltered = core.std.CopyFrameProps(ref_prefiltered, src)

    bm3d_basic = BM3D(src, ref=ref_prefiltered, sigma=sigma_scaled,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=radius)

    if radius > 0:
        bm3d_basic = bm3d_basic.bm3d.VAggregate(radius=radius)

    bm3d_basic = _ensure_float32(bm3d_basic)

    # LEVEL SHIFTED: old level>=2 becomes level>=3
    if level >= 3:
        return MaskedMerge(src, bm3d_basic, mask)
    else:
        sigma_final = sigma_scaled * [0.7, 0.75, 0.8, 0.85, 1.0][level]
        bm3d_final = BM3D(src, ref=bm3d_basic, sigma=sigma_final,
                          block_step=block_step, bm_range=bm_range,
                          ps_num=ps_num, ps_range=ps_range, radius=0)
        bm3d_final = _ensure_float32(bm3d_final)
        return MaskedMerge(src, bm3d_final, mask)

def _deblocking(src, ref, radius, h, sigma, lowpass, level):
    mask = _gen_block_mask(src)
    
    # LEVEL SHIFTED: old params become indices 1-4
    params = [[1,48,6,12],[2,32,4,8],[4,16,2,4],[4,8,1,4],[8,4,1,2]][level]
    block_step, bm_range, ps_num, ps_range = params
    sbsize = block_step * 2 + 1
    
    # LEVEL SHIFTED: add level 0 scaling
    h_scaled = h * [1.3, 1.0, 0.75, 0.7, 0.65][level]
    sigma_scaled = sigma * [1.2, 1.0, 0.9, 0.8, 0.75][level]

    # LEVEL SHIFTED: old level<=1 becomes level<=2
    cleansed = _nl_means(ref, radius, 8, 4, h_scaled, ref, level) if level <= 2 else ref

    bm3d_basic = BM3D(ref, ref=cleansed, sigma=sigma_scaled,
                      block_step=block_step, bm_range=bm_range,
                      ps_num=ps_num, ps_range=ps_range, radius=radius)

    if radius > 0:
        bm3d_basic = bm3d_basic.bm3d.VAggregate(radius=radius)

    bm3d_basic = _ensure_float32(bm3d_basic)

    # LEVEL SHIFTED: old level<2 becomes level<3
    cleansed_final = (BM3D(ref, ref=bm3d_basic, sigma=sigma_scaled*0.75,
                           block_step=block_step, bm_range=bm_range,
                           ps_num=ps_num, ps_range=ps_range, radius=0)
                      if level < 3 else bm3d_basic)
    cleansed_final = _ensure_float32(cleansed_final)

    ref_final = _freq_merge(cleansed_final, ref, sbsize, lowpass)
    src_final = _freq_merge(cleansed_final, src, sbsize, lowpass)

    return MaskedMerge(src_final, ref_final, mask)

# ============================================================================
# PUBLIC API - All functions now expect GRAY input and return GRAY output
# ============================================================================

def Super(src, pel=2):
    """
    Create super clip for motion estimation.
    
    Args:
        src: GRAY luma plane
        pel: Pixel accuracy (2 or 4)
    
    Returns:
        Super clip for motion analysis
    """
    if src.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Super() expects GRAY input. Use ChromaSave() first.")
    return _super(src, pel)

def Basic(src, super=None, radius=3, pel=2, sad=1000.0, short_time=False, level=0):
    """
    Basic temporal denoising using motion compensation.
    
    Args:
        src: GRAY luma plane
        super: Pre-computed super clip (optional)
        radius: Temporal radius
        pel: Pixel accuracy (2 or 4)
        sad: Sum of absolute differences threshold
        short_time: Use short-time mode
        level: Quality level (0-4, where 0 is slowest/best)
    
    Returns:
        GRAY denoised luma
    """
    if src.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Basic() expects GRAY input. Use ChromaSave() first.")
    
    super_clip = super if super else _super(src, pel)
    return _basic(src, super_clip, radius, pel, sad, short_time, level)

def Deringing(src, ref, radius=3, h=6.4, sigma=10.0, lowpass=None, level=0, preset="auto"):
    """
    Remove ringing artifacts.
    
    Args:
        src: GRAY luma plane
        ref: GRAY reference luma plane
        radius: Temporal radius
        h: NLMeans strength
        sigma: BM3D sigma
        lowpass: Frequency merge parameters (overrides preset if provided)
        level: Quality level (0-4, where 0 is slowest/best)
        preset: Frequency filtering preset - "auto" (default, fps>30 uses video preset),
                "film" (preserve grain, 0.65 cutoff), "balanced" (0.48 cutoff),
                or "video" (aggressive, 0.35 cutoff)
    
    Returns:
        GRAY derung luma
    """
    if src.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Deringing() expects GRAY input. Use ChromaSave() first.")
    if ref.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Deringing() ref must be GRAY.")
    
    if lowpass is None:
        if preset == "auto":
            fps = src.fps.numerator / src.fps.denominator
            preset = "video" if fps > 30 else "film"
        
        if preset == "film":
            lowpass = [0.0, sigma, 0.65, 1024.0, 1.0, 1024.0]
        elif preset == "video":
            lowpass = [0.0, sigma, 0.35, 1024.0, 1.0, 1024.0]
        else:  # "balanced"
            lowpass = [0.0, sigma, 0.48, 1024.0, 1.0, 1024.0]
    
    return _deringing(src, ref, radius, h, sigma, lowpass, level)

def Destaircase(src, ref, radius=3, sigma=12.0, thr=0.03, elast=0.015, lowpass=None, level=0, preset="auto"):
    """
    Remove staircase artifacts from block boundaries.
    
    Args:
        src: GRAY luma plane
        ref: GRAY reference luma plane
        radius: Temporal radius
        sigma: BM3D sigma
        thr: Threshold for artifact detection
        elast: Elasticity parameter
        lowpass: Frequency merge parameters (overrides preset if provided)
        level: Quality level (0-4, where 0 is slowest/best)
        preset: Frequency filtering preset - "auto" (default, fps>30 uses video preset),
                "film" (preserve grain, 0.65 cutoff), "balanced" (0.48 cutoff),
                or "video" (aggressive, 0.35 cutoff)
    
    Returns:
        GRAY destaircased luma
    """
    if src.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Destaircase() expects GRAY input. Use ChromaSave() first.")
    if ref.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Destaircase() ref must be GRAY.")
    
    if lowpass is None:
        if preset == "auto":
            fps = src.fps.numerator / src.fps.denominator
            preset = "video" if fps > 30 else "film"
        
        if preset == "film":
            lowpass = [0.0, sigma, 0.65, 1024.0, 1.0, 1024.0]
        elif preset == "video":
            lowpass = [0.0, sigma, 0.35, 1024.0, 1.0, 1024.0]
        else:  # "balanced"
            lowpass = [0.0, sigma, 0.48, 1024.0, 1.0, 1024.0]
    
    return _destaircase(src, ref, radius, sigma, thr, elast, lowpass, level)

def Deblocking(src, ref, radius=3, h=6.4, sigma=16.0, lowpass=None, level=0, preset="auto"):
    """
    Remove blocking artifacts.
    
    Args:
        src: GRAY luma plane
        ref: GRAY reference luma plane
        radius: Temporal radius
        h: NLMeans strength
        sigma: BM3D sigma
        lowpass: Frequency merge parameters (overrides preset if provided)
        level: Quality level (0-4, where 0 is slowest/best)
        preset: Frequency filtering preset - "auto" (default, fps>30 uses video preset),
                "film" (preserve grain, 0.18 cutoff), "balanced" (0.12 cutoff),
                or "video" (aggressive, 0.08 cutoff)
    
    Returns:
        GRAY deblocked luma
    """
    if src.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Deblocking() expects GRAY input. Use ChromaSave() first.")
    if ref.format.color_family != vs.GRAY:
        raise TypeError("PMOyster: Deblocking() ref must be GRAY.")
    
    if lowpass is None:
        if preset == "auto":
            fps = src.fps.numerator / src.fps.denominator
            preset = "video" if fps > 30 else "film"
        
        if preset == "film":
            lowpass = [0.0, 0.0, 0.22, 1024.0, 1.0, 1024.0]
        elif preset == "video":
            lowpass = [0.0, 0.0, 0.08, 1024.0, 1.0, 1024.0]
        else:  # "balanced"
            lowpass = [0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]
    
    return _deblocking(src, ref, radius, h, sigma, lowpass, level)
