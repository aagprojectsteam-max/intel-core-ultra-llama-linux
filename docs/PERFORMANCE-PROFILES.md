# Performance profiles

| Mode | Batch | Microbatch | Threads / batch threads |
| --- | --- | --- | --- |
| FAST, Gemma 4 12B metadata | 512 | 512 | 8 / 8 |
| FAST, other models | 512 | 128 | 8 / 8 |
| MEDIUM | 512 | 128 | 8 / 8 |
| SLOW / CONSERVATIVE | 512 | 128 | 4 / 4 |

All modes use Intel SYCL, full GPU offload, f16/f16 KV, and Flash Attention ON by default. Configure `FLASH_ATTN=off` or `auto` if your runtime/model does not support ON. An incompatible runtime fails rather than silently selecting a different backend.

FAST favors generation speed using measured reference settings. MEDIUM uses the smaller microbatch. SLOW reduces thread pressure without artificial delays or a forced CPU fallback. On models without evidence for a larger microbatch, FAST and MEDIUM intentionally coincide. Conservative values combine separately exercised lower-thread and smaller-microbatch settings; they are not a new universal benchmark result.

Safe defaults keep DNN ON and graph OFF. The rejected graph experiment is not a selectable profile. Balanced is the tested power mode. Shared GPU memory still limits model/context choices; reduce context or use a smaller model if allocation fails.

Context choices include 1024, 4096, 8192, 16384, 32768, and Custom. Values must be integers from 512 through the smaller of model metadata capacity and 32768. Unknown capacity fails clearly. Unattended legacy profiles are clamped to a smaller model capacity. These are capacities, not expected prompt sizes; occupied context affects speed.

## FAST TEXT mapping

FAST TEXT is an opt-in text-only mapping, with 32768 recommended capacity (bounded by the model). Example user configuration:

```bash
FAST_TEXT_TARGET="$MODEL_ROOT/gemma-4-12b-it-qat-q4_0.gguf"
```

Then `aag-llama-server-start --profile fast-text --mtp off`. Use `--context 8192` or `--interactive` to select another capacity. Vision is disabled for the explicit FAST TEXT profile; ordinary model selection still allows vision.

The tested reference was Official Gemma 4 12B instruction-tuned QAT Q4_0, 512/512 batches, eight threads, f16 KV, full GPU offload. A different explicitly mapped model uses its metadata-based microbatch and does not inherit model-specific MTP settings.

For the corresponding tested Gemma assistant only:

```bash
FAST_TEXT_MTP="$MODEL_ROOT/mtp-gemma-4-12B-it.gguf"
FAST_TEXT_MTP3=1
```

The main model and companion must share a directory and pass conservative matching. MTP3 additionally requires the mapped Q4_0 Gemma 4 12B metadata and explicit companion mapping. It sets draft-mtp, max=3, min=0, p_min=0, GPU draft sampling. Other MTP configurations use max=2. Mapping expresses your compatibility decision; it is not a model download or guarantee. MTP is never forced.

Legacy `--profile fast|normal|long` selects context; new `--performance fast|medium|slow` selects resources.
