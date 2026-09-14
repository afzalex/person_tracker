# Areas for Enhancement

## 1. Refactor pipeline.py into submodules
**Current state**: Completed. The monolithic pipeline has been split into focused modules with clear responsibilities.

**Improvement**: Split functionality into logical submodules:
- `detection.py` — Scene detection, signatures, cut refinement
- `tracking.py` — PersonTrackingSession, track ranking, scene summarization
- `cropping.py` — Crop planning, smoothing, interpolation logic
- `rendering.py` — Video rendering, frame extraction, audio muxing
- `widgets.py` — Jupyter interactive widgets and visualization
- `io.py` — Cache serialization, report generation, metadata validation

**Benefits**: Easier maintenance, better testability, clearer separation of concerns.

---

## 2. Add explicit type hints to functions
**Current state**: Completed. Comprehensive type annotations were added across the package.

**Improvement**: Add comprehensive type annotations to all function signatures and return types.

**Benefits**: Better IDE autocompletion, static type checking with mypy, improved code documentation.

---

## 3. Implement per-scene checkpointing for render
**Current state**: Completed. Rendering can now be split into resumable segments with a manifest, so completed chunks are saved and reused instead of restarting from scratch.

**Improvement**: Save intermediate rendered scenes to a cache directory. Resume rendering by skipping already-completed scenes.

**Benefits**: Graceful failure recovery, resume capability, better fault tolerance on long videos.

---

## 4. Add GPU memory adaptive batching
**Current state**: No explicit GPU memory monitoring or adaptive batch sizing.

**Improvement**: Check available VRAM before tracking. Dynamically reduce frame-stride or image-size if memory is constrained.

**Benefits**: Support for limited VRAM environments (RTX 2060, RTX 3070, etc.), prevent OOM crashes.

---

## 5. Create unit and integration tests
**Current state**: No visible test suite; workflow validation only via notebook manual review.

**Improvement**: Add pytest tests covering:
- Scene detection edge cases (no cuts, rapid cuts, fades)
- Track ranking scoring algorithm
- Crop plan smoothing and interpolation
- JSON cache validation and corruption handling
- Widget async state transitions

**Benefits**: Regression prevention, confidence in refactoring, faster iteration cycles.

---
