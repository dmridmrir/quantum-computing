# Third-party source revisions

## Tesseract Decoder

- Upstream: https://github.com/quantumlib/tesseract-decoder
- Revision: `54e2668fafebb413c6090ed3e8c127c24902942f`
- Local path: `third_party/tesseract-decoder`
- Role: pinned CPU reference decoder and candidate-trace producer

Local integration changes are intentionally small and isolated:

- `TESSERACT_DPU_TRACE_PATH` enables a versioned candidate trace from the real
  search path; `TESSERACT_DPU_TRACE_Q_BITS` selects fixed-point precision.
- The trace records producer-side expected candidate costs so the producer,
  Go oracle, and DPU output can be checked independently.
- `TESSERACT_BUILD_PYTHON=OFF` permits a C++-only CMake build without Python
  development headers. The default remains `ON` for upstream compatibility.

The revision is pinned for reproducible CPU-only versus CPU+DPU experiments.

## Magic State Cultivation

- Upstream: https://github.com/Strilanc/magic-state-cultivation
- Revision: `871e68ff6df2f75190b1bfd6351459d1b5a037e3`
- Local path: `third_party/magic-state-cultivation`
- Role: official MSC grafted-code circuit generator and postselection reference

The checked-in workload uses the upstream `end2end-inplace-distillation`
generator. Its DEM and syndrome files are prepared with the same detector
postselection rules used by the upstream desaturation sampler.
