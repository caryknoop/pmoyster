# PMOyster (Poor Man’s Oyster)

PMOyster is a GPU‑accelerated implementation of the Oyster restoration pipeline for VapourSynth. 

Modeled after the original Oyster:
https://github.com/IFeelBloated/Oyster


## Requirements

| Dependency        | Purpose                                                |
|------------------|--------------------------------------------------------|
| `bm3dcuda_rtc`   | CUDA‑accelerated BM3D filtering                        |
| `nlm_cuda`       | CUDA‑accelerated Non‑Local Means                       |
| `dfttest2`       | cuFFT‑accelerated frequency‑domain filtering           |
| `akarin`         | Optimized expression evaluation (required)             |
| `nnedi3cl`       | High‑quality neural interpolation                      |
| `mvsf`           | Accelerated motion estimation / compensation (MVTools) |


## Quality Level (`level`)

| Level | Speed    | Deringing       | Destaircase | Deblocking | Notes                       |
|------:|----------|-----------------|-------------|------------|-----------------------------|
| **0** | Slowest  | Strong refine    | 2–3 passes  | 2 passes   | Max quality, slow           |
| **1** | Slow     | Strong           | 2 passes    | 2 passes   | Stable high quality         |
| **2** | Medium   | Medium           | 2 passes    | 2 passes   | **Recommended default**     |
| **3** | Fast     | Light            | 1 pass      | 1 pass     | Good for previewing         |
| **4** | Fastest  | Minimal          | Minimal     | Minimal    | Fast cleanup / diagnostics  |

`level` controls internal processing complexity and is **independent** of the input‑adaptation parameters below.


## Source‑Dependent Strength Controls

### `source_format`
Describes the origin of the material:

| Value      | Intended Use Case                    | Behavior                     |
|-----------:|--------------------------------------|------------------------------|
| `film`     | Film scans, CGI, cinematic masters   | Grain‑preserving             |
| `video`    | Broadcast / tape / web compression   | Stronger artifact suppression|
| `balanced` | Neutral default                      | Middle ground                |
| `auto`     | Auto: `fps > 25 → video`, else film  | Safe for NTSC; PAL is mixed  |


### `source_quality`
Overall correction strength (not speed):

| Value    | Effect                |
|---------:|-----------------------|
| `low`    | Minimal correction    |
| `medium` | Balanced cleanup      |
| `good`   | More aggressive       |

### Resolution class (automatic)
```
if src.height >= 720 → HD scaling profile
else                 → SD scaling profile
```

### Strength scaling tables

**Format × Resolution → BM3D σ & NLMeans h multipliers**

| Resolution | Format     | BM3D σ | NLMeans h |
|-----------|------------|-------:|----------:|
| **SD**    | film       | 1.20   | 1.05      |
| **SD**    | video      | 1.35   | 1.15      |
| **SD**    | balanced   | 1.27   | 1.10      |
| **HD**    | film       | 0.95   | 1.00      |
| **HD**    | video      | 1.10   | 1.05      |
| **HD**    | balanced   | 1.03   | 1.02      |

**Quality multipliers** (applied on top):

| Quality | BM3D σ | NLMeans h | Destaircase `thr`/`elast` |
|--------:|-------:|----------:|---------------------------:|
| low     | 0.85   | 0.90      | 0.85                       |
| medium  | 1.00   | 1.00      | 1.00                       |
| good    | 1.17   | 1.12      | 1.10                       |


## Public API

```python
Basic(
    src, super=None, *,
    radius=None, sad=None, pel=2, short_time=False, level=0,
    source_format="balanced", source_quality="medium"
)
```
Motion‑compensated temporal denoising. If `radius` or `sad` is explicitly provided, the given value is used; otherwise they are chosen from `source_format` + `source_quality`.

---

```python
Deringing(
    src, ref=None, *,
    radius=3, h=6.4, sigma=10.0, lowpass=None, level=0,
    source_format="balanced", source_quality="medium"
)
```
Ringing removal with BM3D and NLMeans refinement. `h` and `sigma` are scaled by tables above **unless** explicitly provided.

---

```python
Destaircase(
    src, ref=None, *,
    radius=3, sigma=12.0, thr=0.03, elast=0.015, lowpass=None, level=0,
    source_format="balanced", source_quality="medium"
)
```
Reduces block‑edge staircasing. `sigma`, `thr`, and `elast` are scaled **unless** explicitly provided.

---

```python
Deblocking(
    src, ref=None, *,
    radius=3, h=6.4, sigma=16.0, lowpass=None, level=0,
    source_format="balanced", source_quality="medium"
)
```
Removes DCT blocking using difference‑based filtering and frequency reintegration. The block mask behavior is fixed; only strengths scale.


## Parameter override rule

> **Any parameter explicitly provided by the caller is respected as‑is and is not scaled.**

Examples:
```python
# Fully automatic scaling from tables:
clip = PMOyster.Deringing(clip)

# h is not scaled (explicit override):
clip = PMOyster.Deringing(clip, h=3.0)

# sigma is not scaled (explicit override):
clip = PMOyster.Destaircase(clip, sigma=8.5)
```


## Example

```python
level = 2  # recommended starting point

super = PMOyster.Super(clip, pel=2)
ref   = PMOyster.Basic(clip, super=super, source_format="video", source_quality="good")

clip = PMOyster.Deringing(clip, ref, level=level)
#clip = PMOyster.Destaircase(clip, ref, level=level)
#clip = PMOyster.Deblocking(clip, ref, level=level)
```


## License

LGPL‑3.0
