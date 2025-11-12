# PMOyster (Poor Man's Oyster)

PMOyster is a GPU-accelerated implementation of the Oyster restoration pipeline for VapourSynth. 

Modeled after the original Oyster:
https://github.com/IFeelBloated/Oyster


## Requirements

| Dependency        | Purpose                                                |
|------------------|--------------------------------------------------------|
| `bm3dcuda_rtc`   | CUDA-accelerated BM3D filtering                        |
| `nlm_cuda`       | CUDA-accelerated Non-Local Means                       |
| `dfttest2`       | cuFFT-accelerated frequency-domain filtering           |
| `akarin`         | Optimized expression evaluation (required)             |
| `nnedi3cl`       | High-quality neural interpolation                      |
| `mvsf`           | Accelerated motion estimation / compensation (MVTools) |


## Quality Level (`level`)

| Level | Speed    | Deringing       | Destaircase | Deblocking | Notes                       |
|------:|----------|-----------------|-------------|------------|-----------------------------|
| **0** | Slowest  | Strong refine    | 2-3 passes  | 2 passes   | Max quality, slow           |
| **1** | Slow     | Strong           | 2 passes    | 2 passes   | Stable high quality         |
| **2** | Medium   | Medium           | 2 passes    | 2 passes   | **Recommended default**     |
| **3** | Fast     | Light            | 1 pass      | 1 pass     | Good for previewing         |
| **4** | Fastest  | Minimal          | Minimal     | Minimal    | Fast cleanup / diagnostics  |

`level` controls internal processing complexity and is **independent** of the source-adaptation parameters below.


## Source-Dependent Strength Controls

### `format`
Describes the **type** of source material (affects grain preservation):

| Value      | Intended Use Case                         | Behavior                                |
|-----------:|-------------------------------------------|-----------------------------------------|
| `film`     | Film scans with **grain** to preserve    | **Gentler** filtering (preserves grain) |
| `video`    | Video/digital sources **without grain**   | **Stronger** filtering (no grain concern)|
| `balanced` | Unknown or mixed source                   | Moderate approach                       |
| `auto`     | Auto-detect: `fps > 25 -> video`, else film| Safe for NTSC; PAL is mixed             |

**Key concept**: Film sources have **grain** (from film stock) that should be **preserved** as part of the aesthetic. Video sources (camcorder, digital video) have no grain structure, so more aggressive filtering is acceptable. Both can have MPEG-2 compression artifacts (ringing, blocking, mosquito noise).

**IMPORTANT**: `source_format` and `source_quality` are **INDEPENDENT** settings. You can have:
- Film source with gentle processing: `format='film', quality='low'`
- Film source with aggressive processing: `format='film', quality='good'`
- Video source with any quality level

### `correction`
Overall correction strength (**independent** of film/video type):

| Value    | Effect                | Use When                          |
|---------:|-----------------------|-----------------------------------|
| `low`    | Gentle (50% strength) | High-quality sources, preserve detail |
| `medium` | Balanced (100%)       | Typical sources, good default     |
| `good`   | Aggressive (180%)     | Damaged/low-bitrate sources       |

### Resolution class (automatic)
```
if src.height >= 720 -> HD scaling profile
else                 -> SD scaling profile
```

### Strength scaling tables (v0.27)

**Format × Resolution -> BM3D σ & NLMeans h multipliers**

| Resolution | Format     | BM3D σ | NLMeans h | Notes                          |
|-----------|------------|-------:|----------:|--------------------------------|
| **SD**    | film       | 0.75   | 0.85      | Gentle to preserve grain       |
| **SD**    | video      | 1.25   | 1.30      | Stronger (no grain concern)    |
| **SD**    | balanced   | 1.00   | 1.10      | Moderate                       |
| **HD**    | film       | 0.70   | 0.80      | Even gentler (grain more visible) |
| **HD**    | video      | 1.20   | 1.25      | Aggressive is safe             |
| **HD**    | balanced   | 0.95   | 1.05      | Moderate                       |

**Quality multipliers** (applied on top of format multipliers):

| Quality | BM3D σ | NLMeans h | Destaircase `thr`/`elast` |
|--------:|-------:|----------:|---------------------------:|
| low     | 0.50   | 0.60      | 0.70                       |
| medium  | 1.00   | 1.00      | 1.00                       |
| good    | 1.80   | 1.70      | 1.40                       |

**Example calculations** (base sigma=10.0, level=2):
- Film + medium + SD = 10.0 × 0.75 × 1.00 × 1.00 = **7.5**  (gentle, preserves grain)
- Video + medium + SD = 10.0 × 1.25 × 1.00 × 1.00 = **12.5** (stronger, no grain concern)
- Balanced + medium + SD = 10.0 × 1.00 × 1.00 × 1.00 = **10.0** (matches original Oyster)


## Public API

```python
Basic(
    src, super=None, *,
    radius=None, sad=None, pel=2, short_time=False, level=0,
    format="balanced", quality="medium"
)
```
Motion-compensated temporal denoising. If `radius` or `sad` is explicitly provided, the given value is used; otherwise they are chosen from `format` + `quality`.

---

```python
Deringing(
    src, ref=None, *,
    radius=3, h=6.4, sigma=10.0, lowpass=None, level=2,
    format="balanced", correction="medium"
)
```
Ringing removal with BM3D and NLMeans refinement. `h` and `sigma` are scaled by tables above **unless** explicitly provided.

**Note**: `sigma` values assume CUDA BM3D behavior. Currently calibrated to match original Oyster at `balanced + medium + level=2` (sigma=10.0).

---

```python
Destaircase(
    src, ref=None, *,
    radius=6, sigma=16.0, thr=0.03125, elast=0.015625, lowpass=None, level=2,
    format="balanced", correction="medium"
)
```
Reduces block-edge staircasing. `sigma`, `thr`, and `elast` are scaled **unless** explicitly provided.

**Note**: Default parameters now match original Oyster (radius=6, sigma=16.0).

---

```python
Deblocking(
    src, ref=None, *,
    radius=3, h=6.4, sigma=16.0, lowpass=None, level=2,
    format="balanced", correction="medium"
)
```
Removes DCT blocking using difference-based filtering and frequency reintegration. The block mask behavior is fixed; only strengths scale.


## Parameter override rule

> **Any parameter explicitly provided by the caller is respected as-is and is not scaled.**

Examples:
```python
# Fully automatic scaling from tables:
clip = PMOyster.Deringing(clip)

# h is not scaled (explicit override):
clip = PMOyster.Deringing(clip, h=3.0)

# sigma is not scaled (explicit override):
clip = PMOyster.Destaircase(clip, sigma=8.5)
```


## Example Workflows

### Typical Film DVD (with grain)
```python
level = 2  # recommended starting point

super = PMOyster.Super(clip, pel=2)
ref   = PMOyster.Basic(clip, super=super, 
                       format="film",      # Has grain to preserve
                       correction="medium",   # Moderate processing
                       level=level)

clip = PMOyster.Deringing(clip, ref, 
                          format="film",
                          correction="medium",
                          level=level)
# Result: sigma=7.5, gentler to preserve grain
```



## License

LGPL-3.0
