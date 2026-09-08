# Model capabilities

| Badge | Meaning |
| --- | --- |
| [V] | Vision metadata/template or confidently matching projector |
| [T] | Tool schema/call syntax in the embedded chat template |
| [R] | Reasoning/thinking syntax in the embedded chat template |
| [M] | Matching trained MTP companion is present |

A badge is not a quality certification. Ordinary chat generation does not establish reasoning capability. Missing badges appear as empty brackets to keep the list aligned. No model brand is removed; HauhauCS and other distributions are discovered under the configured model root.

Reasoning control is conservative:

- A verified `enable_thinking` branch: OFF/ON.
- GPT-OSS with an effort template field: LOW/MEDIUM/HIGH; no invented OFF option.
- Granite with its `thinking` branch: OFF/ON, applicable only without client system messages, tools, or documents for the recognized template.
- Reasoning syntax without a recognized control: an informational line.
- No reasoning capability: no question.

The mapping uses `--reasoning` and `--chat-template-kwargs`. It does not claim that every model supports four effort levels. Client template overrides or future template changes can alter behavior. Provider references: [GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) and [Granite 3.3](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct).

GGUF v2/v3 metadata is read with bounded standard-library code. Tensor weights are never read. Malformed headers fail closed. Templates and filenames are not executed. Discovery excludes sidecar names, projector/assistant architectures, secondary shards, control-character paths, hidden directories, and symlinks escaping the configured model root. Directory symlinks are not traversed; point MODEL_ROOT at your actual model directory.
