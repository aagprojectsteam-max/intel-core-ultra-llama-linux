# Projector and MTP companions

Keep companions next to the main model. Expected names identify `mmproj`, `mtp`, or `dflash` as a separated filename component. DFlash files are excluded from model discovery but are not enabled by this project.

For example, a main `Qwen3.5-4B-Q4_K_M.gguf` may have `mmproj-Qwen3.5-4B-BF16.gguf` and a corresponding `mtp-Qwen3.5-4B.gguf`, if supplied by the model publisher. Naming alone is insufficient.

Matching checks normalized identity, conflicting family metadata, architecture, projection/embedding dimensions, and MTP vocabulary size and trained prediction layers. A known Pixtral projection-metadata inconsistency is checked against the output tensor's shape without reading tensor data. Gemma assistant matching handles its distinct assistant architecture.

One confident projector offers load/model-only. Several matches require explicit selection. No match skips the question. MTP has ON/OFF with OFF as its default, and multiple candidates require selection. Noninteractive ambiguity fails without attaching an arbitrary file.

`--mtp auto` remains limited to an explicitly approved model/draft pair: configure `APPROVED_MTP_TARGET`, `APPROVED_MTP_DRAFT`, and both corresponding `_SHA256` values. Checksums and metadata must match. `--mtp on` requests a single metadata-matched companion without granting it the MTP3 optimization. `--mmproj auto` uses one confident match only.

Matching reduces accidental attachment; it does not prove tokenizer identity, correctness, or semantic quality. Use companions intended by the model publisher and validate your actual workload before relying on them.
