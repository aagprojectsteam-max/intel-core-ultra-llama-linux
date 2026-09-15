# Engineering Handoff — Intel Core Ultra llama.cpp on Linux

## Purpose

This document preserves the optimization and operational history behind the repository so the launcher profiles are not treated as arbitrary constants. It distinguishes measured local-hardware results from portable defaults and explains how to re-benchmark safely.

## Goal and hardware context

The project was built to make llama.cpp practical on Intel Core Ultra Linux hardware using the integrated Intel Arc GPU through SYCL/Level Zero, with repeatable server/CLI launch profiles and model-aware capability handling. The development machine used an Intel Core Ultra 7 155H with integrated Arc graphics and large system memory; performance numbers below are measurements from that class of setup, not universal guarantees.

## Optimization program

The work compared CPU/GPU/runtime settings, batch/ubatch sizes, KV choices, context, prompt-cache behavior and speculative/MTP behavior. The accepted SYCL path was retained after comparison with alternatives; an OpenVINO experiment did not replace the production configuration.

For the Gemma 12B optimization track, measured token generation improved from roughly **7.782 tok/s** baseline to about **10.246 tok/s** at the best measured point (about +31.7%). The accepted fast configuration used batch/ubatch 512/512, 8 threads and f16/f16 KV for that track. Prompt-cache work reduced a measured cached TTFT from roughly 4.920 s to 0.700 s in the documented benchmark. MTP/speculative behavior was evaluated separately; acceptance rate and model support must be detected rather than assumed.

The repository documentation under `docs/BENCHMARK-FINDINGS.md`, `docs/PERFORMANCE-PROFILES.md`, `docs/INTEL-SYCL.md` and `docs/MMPROJ-MTP.md` is the evidence-oriented reference for those results.

## Production control layer

`bin/aag-llama-control` is the canonical control surface. The start/status/stop command names are symlinks to the same controller so behavior remains centralized. Model inspection lives under `lib/`, including GGUF metadata and capability logic. Profiles are configurable rather than hard-coded to private model paths.

The accepted profile family distinguishes FAST, MEDIUM and SLOW rather than pretending one set of parameters is optimal for every model/workload. The historical production values included FAST 512/512 with 8 threads for the Gemma 12B track, MEDIUM 512/128 with 8 threads, and SLOW 512/128 with 4 threads. Re-benchmark before treating these as optimal on different hardware or llama.cpp revisions.

## SYCL / Intel GPU

The project uses Intel SYCL/Level Zero rather than NVIDIA/CUDA assumptions. `lib/intel-sycl.sh` contains environment setup. A working build/runtime must be verified against the actual Intel driver stack. GPU availability, memory pressure and model offload should be checked at runtime; failure to initialize the GPU must not be hidden as a successful accelerated run.

## Model capability handling

Not every GGUF/model supports the same features. The control/model libraries inspect metadata and separate text, vision/mmproj and MTP-related capability. Do not attach an mmproj or speculative path solely because another model with a similar name used one successfully.

## AnythingLLM integration

`docs/ANYTHINGLLM.md` documents the local server integration boundary. The controller should expose stable localhost endpoints while keeping model/runtime ownership outside AnythingLLM. Integration tests should distinguish “server reachable” from “model answers correctly” and from “accelerated path actually active.”

## Testing

The repository includes Python tests for model logic, server flow, fast-text behavior and related controller functions, plus GitHub Actions in `.github/workflows/test.yml`. The publication baseline historically passed **72 tests**. Hosted CI validates controller logic without reproducing the exact Intel GPU benchmark machine.

Performance changes require local benchmark evidence in addition to unit tests. Record model identity/quantization, llama.cpp build/revision, backend, context, batch/ubatch, threads, KV types, prompt length, cache state, TTFT and generation throughput. Compare repeated runs rather than one warm/cold outlier.

## Release/publication

The project was published as **v1.0.0** after the accepted SYCL optimization and controller work. `CHANGELOG.md` records concise version changes; Git history remains the implementation chronology. Third-party components retain their own licensing terms; see `THIRD-PARTY-NOTICES.md`.

## Repository map

- `README.md` — public overview and usage.
- `docs/QUICKSTART.md` — setup/launch path.
- `docs/ARCHITECTURE.md` — controller architecture.
- `docs/BENCHMARK-FINDINGS.md` — measured results.
- `docs/PERFORMANCE-PROFILES.md` — profile rationale.
- `docs/INTEL-SYCL.md` — Intel backend setup.
- `docs/MODEL-CAPABILITIES.md` — feature detection.
- `docs/MMPROJ-MTP.md` — multimodal/speculative notes.
- `docs/TROUBLESHOOTING.md` — runtime diagnosis.
- `bin/` — control commands.
- `lib/` — model/backend helpers.
- `tests/` — regression suite.

## Maintenance and re-benchmark procedure

For a new llama.cpp revision or model: preserve the existing known-good profile; verify syntax/tests; establish a repeated baseline; change one important dimension at a time; record cold and cached TTFT separately; record generation throughput separately; verify output correctness and model capabilities; reject a backend/profile that is faster only because features silently failed; update benchmark findings and this handoff; then change defaults only when evidence supports it.

## Historical integrity rule

Do not publish the best single number as a universal speed claim. Preserve rejected experiments such as the OpenVINO path and explain why they were not selected. Keep benchmark machine/model/build metadata with results so future maintainers can tell optimization from environmental drift.

## Current handoff status

As of the documentation audit on 2026-09-15, the repository contains README, quick start, architecture, benchmark/profile/SYCL/model-capability/MTP/AnythingLLM/troubleshooting documentation, tests, CI, changelog, third-party notices, configurable control scripts, and this long-form handoff.