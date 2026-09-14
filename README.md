# Person Tracker

Notebook-driven person tracking and identity analysis for video. The recovery
priority is the `person_tracker` package; input media, generated output, model
weights, and caches are intentionally excluded from version control.

## Canonical workflow

Use `notebooks/person_tracker.ipynb` as the primary notebook. It defaults to a
short five-second `ANALYSIS` run so that the pipeline can be checked before a
long GPU job. Change `mode` to `PRODUCTION` and adjust the processing range in
the first configuration cell for a full run.

For a reusable two-stage workflow, use:

1. `notebooks/01_collect_observations.ipynb` to run video decoding, YOLO,
   Deep OC-SORT, and InsightFace once and save the observations under `runs/`.
2. `notebooks/02_resolve_identities.ipynb` to load that run and resolve logical
   person IDs. Identity thresholds can then be changed without repeating model
   inference.

The run directory is the contract between the notebooks. It contains a source
and settings manifest, frame-level tracks, face metadata, numeric embeddings,
optional face crops, and the resolved identity files.

`notebooks/person_tracker_improved.ipynb` is an experimental variant for more
aggressive identity-switch handling. `notebooks/persistent_person_tracking.ipynb`
belongs to the separate `subject_reframe` pipeline and is retained as recovery
evidence, but that pipeline is not the primary supported workflow.

## Setup

Python 3.12 was used for the recovered environment.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

Open the project directory or its `notebooks` directory in Jupyter. The
notebooks add `src` to Python's import path automatically.

Place source videos in `input/`. Model weights downloaded by Ultralytics and
InsightFace are local runtime assets and should not be committed.

## Tracking configuration

The recovered Ultralytics environment supports BoT-SORT, ByteTrack, OC-SORT,
Deep OC-SORT, FastTrack, and TrackTrack. Deep OC-SORT is the default because it
supports the identity-continuity workflow used by this project.

`config/subject_strongsort.yaml` is preserved for reference, but StrongSORT is
not supported by the pinned Ultralytics build and should not be selected unless
a compatible backend is added.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The test suite covers scene-cache reuse and resumable rendering. Package import
and notebook execution remain useful smoke tests for the tracking workflow,
which requires model assets and suitable hardware.

## Recovery note

The source files in `src/person_tracker` were checked against the surviving
CPython 3.12 bytecode in the old project. All modules were recovered, with only
a harmless explicit `None` return difference in the notebook cell magics.
