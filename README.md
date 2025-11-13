# PMOyster (Poor Man's Oyster)

PMOyster is a GPU-accelerated implementation of the Oyster restoration pipeline for VapourSynth. 

Modeled after the original Oyster:
https://github.com/IFeelBloated/Oyster

License: LGPL-3.0

**Development version - interface is still subject to change!**

## Table of Contents

- [Public Functions](#public-functions)
  - [Process](#process)
  - [Basic](#basic)
  - [Deringing](#deringing)
  - [Destaircase](#destaircase)
  - [Deblocking](#deblocking)
  - [ProcessInScenes](#processinscenes)
  - [SCDetect](#scdetect)
- [Parameters Reference](#parameters-reference)
  - [Format Presets](#format-presets)
  - [Correction Levels](#correction-levels)
  - [Processing Levels](#processing-levels)
- [Usage Examples](#usage-examples)

---

## Public Functions

### Process

**One-line full pipeline with scene-change chunking and selectable processing modes.**

```python
Process(src, process="combined", format="balanced", correction="medium", 
        short_time=False, pel=2, level=2, use_sc=True, sc_threshold=0.140)
```

**Description:**  
All-in-one ultra-fast Oyster implementation that automatically handles scene-change detection and applies the complete denoising pipeline. This is the recommended entry point for most users.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip to process |
| `process` | str | `"combined"` | Processing mode: `"deblock"`, `"dering"`, `"destaircase"`, or `"combined"` |
| `format` | str | `"balanced"` | Content type: `"film"`, `"video"`, `"balanced"`, or `"auto"` |
| `correction` | str | `"medium"` | Strength preset: `"low"`, `"medium"`, or `"good"` |
| `short_time` | bool | `False` | Use short-time motion estimation (faster, less accurate) |
| `pel` | int | `2` | Pixel accuracy for motion estimation (2 or 4, higher = slower but more accurate) |
| `level` | int | `2` | Processing intensity level (0-4, lower = stronger) |
| `use_sc` | bool | `True` | Enable scene-change detection and chunking |
| `sc_threshold` | float | `0.140` | Scene change detection sensitivity (only used if use_sc=True) |

**Process Modes:**
- `"deblock"` - Only deblocking (removes compression blocking artifacts)
- `"dering"` - Only deringing (removes edge ringing artifacts)
- `"destaircase"` - Only destaircase (removes staircase/gradient banding)
- `"combined"` - Deringing + Destaircase (default, recommended for most content)

**Returns:** VideoNode

**Example:**
```python
# Simple usage with defaults (pel=2, scene detection enabled)
filtered = Process(src)

# Video content with aggressive denoising
filtered = Process(src, format="video", correction="good")

# Film content with deblocking only and higher motion accuracy
filtered = Process(src, process="deblock", format="film", pel=4, level=1)

# Disable scene detection for very stable content (faster)
filtered = Process(src, use_sc=False)

# Custom scene detection sensitivity with higher pel
filtered = Process(src, pel=4, use_sc=True, sc_threshold=0.120)
```

---

### Basic

**Temporal motion-compensated denoising.**

```python
Basic(src, *, radius=None, pel=2, sad=None, short_time=False, 
      level=0, format="balanced", correction="medium")
```

**Description:**  
High-quality temporal denoising using MDegrain with sophisticated motion estimation. Automatically generates high-quality superclips using NNEDI3CL upsampling internally. Super() functionality is now integrated and always enabled within Basic().

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip to denoise |
| `radius` | int | `None` | Temporal radius (auto-determined if None) |
| `pel` | int | `2` | Pixel accuracy (2 or 4, higher = slower but more accurate) |
| `sad` | float | `None` | SAD threshold (auto-determined if None) |
| `short_time` | bool | `False` | Use faster short-time motion estimation |
| `level` | int | `0` | Processing level (0-4) |
| `format` | str | `"balanced"` | Content type: `"film"`, `"video"`, `"balanced"`, or `"auto"` |
| `correction` | str | `"medium"` | Strength: `"low"`, `"medium"`, or `"good"` |

**Returns:** VideoNode

**Example:**
```python
# Auto parameters with integrated Super()
den = Basic(src)

# Manual control with higher pel accuracy
den = Basic(src, pel=4, radius=3, sad=800, level=1)
```

---

### Deringing

**Remove edge ringing artifacts from compression.**

```python
Deringing(src, ref=None, *, radius=3, h=6.4, sigma=10.0, lowpass=None, level=2,
          format="balanced", correction="medium", chroma=True)
```

**Description:**  
Specialized filter for removing ringing artifacts around edges caused by lossy compression. Uses a combination of BM3D and NLMeans with frequency-domain merging.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip |
| `ref` | VideoNode | `None` | Reference clip (uses src if None) |
| `radius` | int | `3` | Temporal radius for BM3D |
| `h` | float | `6.4` | NLMeans filtering strength |
| `sigma` | float | `10.0` | BM3D denoising strength |
| `lowpass` | list | `None` | DFTTest frequency curve (auto if None) |
| `level` | int | `2` | Processing level (0-4, lower = stronger) |
| `format` | str | `"balanced"` | Content type: `"film"`, `"video"`, `"balanced"`, or `"auto"` |
| `correction` | str | `"medium"` | Strength: `"low"`, `"medium"`, or `"good"` |
| `chroma` | bool | `True` | Process chroma channels |

**Returns:** VideoNode

**Example:**
```python
# Basic deringing
dering = Deringing(src)

# Aggressive deringing for film content
dering = Deringing(src, format="film", correction="good", level=0)

# Luma-only processing
dering = Deringing(src, chroma=False)
```

---

### Destaircase

**Remove staircase/banding artifacts from gradients.**

```python
Destaircase(src, ref=None, *, radius=6, sigma=16.0, thr=0.03125, elast=0.015625,
            lowpass=None, level=2, format="balanced", correction="medium", chroma=True)
```

**Description:**  
Specialized filter for smoothing gradient banding and staircase artifacts. Uses threshold-based merging with elastic falloff for natural-looking results.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip |
| `ref` | VideoNode | `None` | Reference clip (uses src if None) |
| `radius` | int | `6` | Temporal radius for BM3D |
| `sigma` | float | `16.0` | BM3D denoising strength |
| `thr` | float | `0.03125` | Threshold for difference merging |
| `elast` | float | `0.015625` | Elastic falloff range |
| `lowpass` | list | `None` | DFTTest frequency curve (auto if None) |
| `level` | int | `2` | Processing level (0-4, lower = stronger) |
| `format` | str | `"balanced"` | Content type: `"film"`, `"video"`, `"balanced"`, or `"auto"` |
| `correction` | str | `"medium"` | Strength: `"low"`, `"medium"`, or `"good"` |
| `chroma` | bool | `True` | Process chroma channels |

**Returns:** VideoNode

**Example:**
```python
# Basic destaircase
debanded = Destaircase(src)

# Strong debanding for video content
debanded = Destaircase(src, format="video", correction="good", level=1)

# Custom threshold and elasticity
debanded = Destaircase(src, thr=0.05, elast=0.025)
```

---

### Deblocking

**Remove blocking artifacts from DCT compression.**

```python
Deblocking(src, ref=None, *, radius=3, h=6.4, sigma=16.0, lowpass=None, level=2,
           format="balanced", correction="medium", chroma=True)
```

**Description:**  
Specialized filter for removing DCT blocking artifacts. Uses block-aware masking to selectively smooth block boundaries while preserving detail.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip |
| `ref` | VideoNode | `None` | Reference clip (uses src if None) |
| `radius` | int | `3` | Temporal radius for BM3D |
| `h` | float | `6.4` | NLMeans filtering strength |
| `sigma` | float | `16.0` | BM3D denoising strength |
| `lowpass` | list | `None` | DFTTest frequency curve (auto if None) |
| `level` | int | `2` | Processing level (0-4, lower = stronger) |
| `format` | str | `"balanced"` | Content type: `"film"`, `"video"`, `"balanced"`, or `"auto"` |
| `correction` | str | `"medium"` | Strength: `"low"`, `"medium"`, or `"good"` |
| `chroma` | bool | `True` | Process chroma channels |

**Returns:** VideoNode

**Example:**
```python
# Basic deblocking
deblocked = Deblocking(src)

# Aggressive deblocking for heavily compressed video
deblocked = Deblocking(src, format="video", correction="good", level=0)
```

---

### ProcessInScenes

**Apply a processing function with automatic scene-change chunking.**

```python
ProcessInScenes(src, func, **kwargs)
```

**Description:**  
Infrastructure function that splits the clip at scene changes and processes each scene independently. This prevents temporal filters from bleeding artifacts across cuts. Used internally by Process() but available for custom workflows.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | VideoNode | *required* | Source clip to process |
| `func` | function | *required* | Function to apply to each scene chunk |
| `**kwargs` | dict | - | Additional arguments passed to func |

**Returns:** VideoNode

**Example:**
```python
# Custom processing with scene chunking
def my_filter(clip, strength=1.0):
    return some_processing(clip, strength)

filtered = ProcessInScenes(src, my_filter, strength=2.0)
```

---

### SCDetect

**Detect scene changes in a clip.**

```python
SCDetect(clip, threshold=0.140)
```

**Description:**  
Wrapper around misc.SCDetect for scene change detection. Used internally by ProcessInScenes() but available for manual scene analysis.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | VideoNode | *required* | Clip to analyze |
| `threshold` | float | `0.140` | Scene change detection sensitivity (0.0-1.0) |

**Returns:** VideoNode (with _SceneChange frame property)

**Example:**
```python
# Detect scenes with custom threshold
scenes = SCDetect(src, threshold=0.120)
```

---

## Parameters Reference

### Format Presets

The `format` parameter adapts processing strength based on content type:

| Format | Description | Best For |
|--------|-------------|----------|
| `"film"` | Lower noise, finer grain | Film sources, animation |
| `"video"` | Higher noise, more artifacts | Interlaced video, live action TV |
| `"balanced"` | Moderate settings | General purpose, mixed content |
| `"auto"` | Detects based on frame rate | Automatic selection (>25fps = video) |

### Correction Levels

The `correction` parameter controls overall processing strength:

| Level | Sigma Scale | H Scale | Threshold Scale | SAD Multiplier | Description |
|-------|-------------|---------|-----------------|----------------|-------------|
| `"low"` | 0.50× | 0.60× | 0.70× | 0.75× | Light cleanup, preserve grain |
| `"medium"` | 1.00× | 1.00× | 1.00× | 1.00× | Balanced (default) |
| `"good"` | 1.80× | 1.70× | 1.40× | 1.30× | Aggressive, heavy cleanup |

### Processing Levels

The `level` parameter controls processing intensity (0-4):

| Level | Speed      | Deringing       | Destaircase | Deblocking | Notes                       |
|------:|------------|-----------------|-------------|------------|-----------------------------|
| **0** | Glacial    | Strong refine    | 2-3 passes  | 2 passes   | Max quality, slow           |
| **1** | Slower     | Strong           | 2 passes    | 2 passes   | Stable high quality         |
| **2** | Still slow | Medium           | 2 passes    | 2 passes   | **Recommended default**     |
| **3** | Slowish    | Light            | 1 pass      | 1 pass     | Good for previewing         |
| **4** | Less slow  | Minimal          | Minimal     | Minimal    | Fast cleanup / diagnostics  |

---

## Usage Examples

### Quick Start (Recommended)

```python
import vapoursynth as vs
from PMOyster import Process

core = vs.core

# read your deinterlaced source 

# One-line processing with defaults
filtered = Process(src)

filtered.set_output()
```

### Content-Specific Processing

```python
# Film content with light cleanup
filtered = Process(src, format="film", correction="low", level=3)

# Noisy video content with aggressive cleanup
filtered = Process(src, format="video", correction="good", level=1)

# Auto-detect content type
filtered = Process(src, format="auto", correction="medium")
```

### Selective Artifact Removal

```python
# Only remove blocking artifacts
filtered = Process(src, process="deblock")

# Only remove ringing
filtered = Process(src, process="dering")

# Only remove banding
filtered = Process(src, process="destaircase")

# Combined deringing + debanding (default)
filtered = Process(src, process="combined")
```

### Manual Pipeline Construction

```python
from PMOyster import Basic, Deringing, Destaircase

# Build custom pipeline
src_32 = core.fmtc.bitdepth(src, bits=32)

# Temporal denoising (Super() is now automatically integrated)
den = Basic(src_32, level=1, format="film")

# Artifact removal
dering = Deringing(den, den, level=1, format="film")
debanded = Destaircase(dering, dering, level=2, format="film")

# Convert back to original format
output = core.fmtc.bitdepth(debanded, bits=8)
```

### Advanced Scene-Aware Processing

```python
from PMOyster import ProcessInScenes, Basic, Deringing

def custom_processing(chunk, strength=1.0):
    """Custom per-scene processing"""
    # Super() is now automatically integrated in Basic()
    den = Basic(chunk, pel=4, level=1)
    return Deringing(den, den, level=2, sigma=10.0*strength)

# Apply with scene chunking
filtered = ProcessInScenes(src, custom_processing, strength=1.5)
```

---

## Performance Notes

- **Scene-change chunking** prevents temporal bleed across cuts (zero cut-bleed guarantee) - enabled by default with `use_sc=True`
- Disabling scene detection with `use_sc=False` provides minor speed improvement for content with no cuts
- **CUDA acceleration** provides 15× speedup over CPU implementations
- **RAM usage** is 70% lower than original Oyster through efficient chunking
- **pel=2** (default) provides good speed/quality balance; **pel=4** is ~3× slower but gives better motion estimation (especially for complex motion)
- **short_time=True** trades accuracy for speed (useful for previews)
- Lower **level** values are slower but provide stronger cleanup

---


## Requirements

PMOyster requires the following VapourSynth plugins:

| Dependency        | Purpose                                                |
|------------------|--------------------------------------------------------|
| `bm3dcuda_rtc`   | CUDA-accelerated BM3D filtering                        |
| `nlm_cuda`       | CUDA-accelerated Non-Local Means                       |
| `dfttest2`       | cuFFT-accelerated frequency-domain filtering           |
| `akarin`         | Optimized expression evaluation (required)             |
| `nnedi3cl`       | High-quality neural interpolation                      |
| `mvsf`           | Accelerated motion estimation / compensation (MVTools) |

---

## Quality Level (`level`)

The `level` parameter controls internal processing complexity and is **independent** of the source-adaptation parameters below.

| Level | Speed      | Deringing       | Destaircase | Deblocking | Notes                       |
|------:|------------|-----------------|-------------|------------|-----------------------------|
| **0** | Glacial    | Strong refine    | 2-3 passes  | 2 passes   | Max quality, slow           |
| **1** | Slower     | Strong           | 2 passes    | 2 passes   | Stable high quality         |
| **2** | Still slow | Medium           | 2 passes    | 2 passes   | **Recommended default**     |
| **3** | Slowish    | Light            | 1 pass      | 1 pass     | Good for previewing         |
| **4** | Less slow  | Minimal          | Minimal     | Minimal    | Fast cleanup / diagnostics  |

---

## Source-Dependent Strength Controls

### `format` Parameter

Describes the **type** of source material (affects grain preservation):

| Value      | Intended Use Case                         | Behavior                                |
|-----------:|-------------------------------------------|-----------------------------------------|
| `film`     | Film scans with **grain** to preserve    | **Gentler** filtering (preserves grain) |
| `video`    | Video/digital sources **without grain**   | **Stronger** filtering (no grain concern)|
| `balanced` | Unknown or mixed source                   | Moderate approach                       |
| `auto`     | Auto-detect: `fps > 25 -> video`, else film| Safe for NTSC; PAL is mixed             |

**Key concept**: Film sources have **grain** (from film stock) that should be **preserved** as part of the aesthetic. Video sources (camcorder, digital video) have no grain structure, so more aggressive filtering is acceptable. Both can have MPEG-2 compression artifacts (ringing, blocking, mosquito noise).

**IMPORTANT**: `format` and `correction` are **INDEPENDENT** settings. You can have:
- Film source with gentle processing: `format='film', correction='low'`
- Film source with aggressive processing: `format='film', correction='good'`
- Video source with any quality level

### `correction` Parameter

Overall correction strength (**independent** of film/video type):

| Value    | Effect                | Use When                          |
|---------:|-----------------------|-----------------------------------|
| `low`    | Gentle (50% strength) | High-quality sources, preserve detail |
| `medium` | Balanced (100%)       | Typical sources, good default     |
| `good`   | Aggressive (180%)     | Damaged/low-bitrate sources       |

### Resolution Class (Automatic)

PMOyster automatically detects resolution class:

```
if src.height >= 720 -> HD scaling profile
else                 -> SD scaling profile
```

---

## Strength Scaling Tables

### Format × Resolution → BM3D σ & NLMeans h Multipliers

| Resolution | Format     | BM3D σ | NLMeans h | Notes                          |
|-----------|------------|-------:|----------:|--------------------------------|
| **SD**    | film       | 0.75   | 0.85      | Gentle to preserve grain       |
| **SD**    | video      | 1.25   | 1.30      | Stronger (no grain concern)    |
| **SD**    | balanced   | 1.00   | 1.10      | Moderate                       |
| **HD**    | film       | 0.70   | 0.80      | Even gentler (grain more visible) |
| **HD**    | video      | 1.20   | 1.25      | Aggressive is safe             |
| **HD**    | balanced   | 0.95   | 1.05      | Moderate                       |

### Quality Multipliers

Applied on top of format multipliers:

| Quality | BM3D σ | NLMeans h | Destaircase `thr`/`elast` | SAD Multiplier | Elasticity |
|--------:|-------:|----------:|---------------------------:|---------------:|-----------:|
| low     | 0.50   | 0.60      | 0.70                       | 0.75           | 0.6        |
| medium  | 1.00   | 1.00      | 1.00                       | 1.00           | 1.0        |
| good    | 1.80   | 1.70      | 1.40                       | 1.30           | 1.5        |

### Example Calculations

Base sigma = 10.0, level = 2:

- **Film + medium + SD** = 10.0 × 0.75 × 1.00 × 1.00 = **7.5** (gentle, preserves grain)
- **Video + medium + SD** = 10.0 × 1.25 × 1.00 × 1.00 = **12.5** (stronger, no grain concern)
- **Balanced + medium + SD** = 10.0 × 1.00 × 1.00 × 1.00 = **10.0** (matches original Oyster)

---


