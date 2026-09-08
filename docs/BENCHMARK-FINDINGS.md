# Benchmark findings

**Results are hardware/model/runtime specific.**

These conclusions summarize development experiments on an Intel Core Ultra 7 155H / Meteor Lake Arc iGPU Linux system. They are not claims about every Intel GPU, runtime version, quantization, or prompt distribution. Raw logs, prompts, model files, and private evidence are intentionally not distributed.

| Observation | Scope |
| --- | --- |
| SYCL was the best production backend tested | Throughput, quality, and stability considered together |
| Vulkan worked but did not outperform SYCL in final matched testing | Tested models and runtime builds only |
| OpenVINO showed fast first-token latency in experiments | Failed throughput/quality/stability requirements; not a supported launcher backend |
| Official Gemma QAT Q4_0 configuration significantly outperformed the alternative Gemma Q4_K_M configuration | The optimized comparison also used different MTP settings; this is not an isolated quantization-only effect |
| More occupied context reduced generation speed | Capacity alone is different from the number of tokens actually processed |
| Reusing stable prompt prefixes greatly improved repeat latency | Cache reuse must be measured; it cannot be assumed for every conversation |
| True streaming made generated text visible earlier | It did not create an equivalent increase in model throughput |
| Balanced was preferred for sustained operation | Laptop thermal and power limits remain relevant |
| Graph-ON finalist was rejected | Insufficient matched gain plus a generation exception on a longer quality case |

One reference campaign measured 10.246 tokens/s with Official QAT and MTP3 versus 7.782 tokens/s for the matched alternative configuration with MTP2. Those are configuration comparisons under that campaign's conditions. Later rebooted matched controls produced different absolute rates. Do not combine cohorts or interpret the best individual sample as a universal production speed.

The graph-ON screen initially looked promising with only two samples. Five-sample post-reboot qualification produced only about 1.4% matched throughput gain and then failed a longer quality check. Repeated earlier GPU fence timeouts had already required stopping experiments and rebooting. Clean runs did not erase those failures.

Quality checks are finite and model outputs can differ between speculative and ordinary decoding. Thermal throttling was observed; no unsafe power or cooling change was used to hide it. Reproduce matched prompts, token counts, cache state, repetitions, thermal state, and runtime versions on your own hardware.
