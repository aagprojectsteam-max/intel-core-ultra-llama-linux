# Architecture

The Bash launcher handles model menus, command construction, runtime loading, and managed process lifecycle. Six symlink entry points select CLI/server start/stop/status behavior.

`aag_llama_models.py` implements the existing conservative capability and companion logic. `gguf_metadata.py` reads bounded GGUF headers with Python's standard library, eliminating the metadata UI's dependency on loading a native SYCL-linked library. It never reads tensor payloads.

`intel-sycl.sh` loads the oneAPI environment for the launcher process, selects Intel Level Zero, and reports missing libraries. Safe DNN/graph defaults are passed to the managed child.

Actual server launches use transient systemd user units. Stop/status validate user ID, exact executable, process start time, model argument, unit name, and invocation identity. The public project's runtime/state namespace is separate from older local launchers. It does not manage an AnythingLLM service.

The installer copies only an explicit source allowlist, writes per-file fingerprints and backup locations into a local receipt, and creates entry wrappers pointing at the configured installation. Config is not overwritten on upgrades. The uninstaller removes only unchanged receipt-listed files, restores original backups, and preserves later edits.

The release contains source, tests, documentation, and configuration examples. Runtime libraries, llama.cpp binaries, model weights, and inference evidence are external.
