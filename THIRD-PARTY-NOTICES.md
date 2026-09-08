# Third-party notices

This package contains launcher/helper code developed for the project and a new standard-library GGUF metadata reader. It does not embed llama.cpp, ggml, Intel libraries, chat templates, or model weights.

llama.cpp and ggml are external MIT-licensed dependencies, copyright 2023-2026 The ggml authors. See [the upstream license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). The launcher invokes their command-line interface; the metadata reader implements the documented GGUF format.

Intel oneAPI, GPU drivers, and model artifacts have their own licenses. Obtain and comply with them separately. Naming a tested model or hardware vendor does not grant rights to distribute its artifacts and does not imply endorsement.

The package's original launcher code had no conflicting third-party license header. The public adaptation is released under the accompanying MIT license by the project contributors.
