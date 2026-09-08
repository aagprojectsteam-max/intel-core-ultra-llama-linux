# Quick start

1. Obtain a compatible SYCL llama.cpp build and the matching Intel runtime.
2. Place your licensed GGUF models under a directory you own.
3. Run `./install.sh --llama-root "$HOME/llama.cpp" --model-root "$HOME/Models"`.
4. Check `aag-llama-server-start --show-config`, then `--check-runtime`.
5. Run `aag-llama-server-start --dry-run` and inspect the command.
6. Run without `--dry-run` when ready.

The installer accepts `--prefix`, `--llama-root`, `--build-dir`, `--model-root`, and `--oneapi-env`. It discovers a llama-server on PATH when possible, otherwise suggests `$HOME/llama.cpp`. Missing hardware/runtime does not block installing the UI; inference will fail with setup guidance until prerequisites are ready.

Configuration is user-owned shell code. Do not source configuration from an untrusted source. A generated config is preserved on upgrades. Compare it with `config/profiles.example.conf`.

Environment overrides take precedence over configured paths: `AAG_LLAMA_CONFIG`, `AAG_LLAMA_ROOT`, `AAG_LLAMA_BUILD_DIR`, `AAG_LLAMA_MODEL_ROOT`, `AAG_LLAMA_ONEAPI_ENV`, `AAG_LLAMA_FAST_TEXT_MODEL`, `AAG_LLAMA_HOST`, and `AAG_LLAMA_PORT`. `AAG_LLAMA_HEALTH_TIMEOUT` controls startup waiting.

Interactive order: model, vision, MTP, reasoning, context, performance, summary. Unsupported questions disappear. Explicit CLI values skip their corresponding question. With an explicit legacy profile, automatic preset defaults remain; add `--interactive` to request the complete wizard.

No model is selected automatically for noninteractive use unless FAST TEXT is explicitly mapped and requested. Provide `--model` in scripts. Use `--mmproj off --mtp off` for deterministic companion-free starts; `--mtp on` requests a matching companion, while `--mtp ask` deliberately permits a prompt.

The six commands are `aag-llama-{server,cli}-{start,stop,status}`. The server uses transient systemd user units. CLI mode retains interactive conversation and one-shot generation; server mode provides the five-step controls.
