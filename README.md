# Intel Core Ultra llama.cpp Launcher for Linux

A user-local launcher for Intel SYCL builds of llama.cpp. Choose a model, attach only matching vision/MTP companions, select supported reasoning behavior, and set context and performance before starting a managed server.

For Ubuntu/Linux users with Intel Core Ultra or Intel Arc GPUs. Tested optimization reference: Core Ultra 7 155H (Meteor Lake), integrated Arc graphics. Other Intel devices are targets, not certified configurations. This is an independent community project, not an Intel or llama.cpp product.

## What it includes

- Compact `[V][T][R][M]` model badges: Vision, Tools, Reasoning, matching MTP.
- Conditional projector/MTP menus, template-aware reasoning, validated context choices.
- FAST, MEDIUM, SLOW / CONSERVATIVE, and configurable FAST TEXT.
- Process-local oneAPI setup, Level Zero checks, graph OFF, f16 KV, and no automatic power-mode change.
- Separate server/CLI start, stop, and status commands with process ownership checks.
- Installer, guarded uninstaller, CPU-only fixture tests, and practical documentation.

## Prerequisites

Linux with Bash, Python 3.10+, GNU coreutils, util-linux (`flock`), iproute2 (`ss`), curl, and a working systemd user session for actual server management. Inference requires an Intel GPU driver, an Intel oneAPI/SYCL runtime matching your llama.cpp build, and your own compatible GGUF models. Tests do not require a GPU, model download, or oneAPI.

See [Intel SYCL setup](docs/INTEL-SYCL.md). The installer detects the environment and explains missing components; it never installs system packages, changes drivers, adds repositories, or downloads models.

## Install

Clone the release, then run the one-command installer:

```bash
git clone --branch v1.0.0 https://github.com/aagprojectsteam-max/intel-core-ultra-llama-linux.git
cd intel-core-ultra-llama-linux
./install.sh --llama-root "$HOME/llama.cpp" --model-root "$HOME/Models"
```

Commands go under `~/.local/bin`; configuration goes under `~/.config/intel-core-ultra-llama-linux/config.conf` (or your XDG config directory). Set `BUILD_DIR` if your build is not called `build-sycl`. Keep `~/.local/bin` on PATH.

Unknown existing commands are refused. Explicit `--replace` backs them up before replacement. Upgrades back up existing project files, preserve configuration, and refuse locally modified commands unless explicitly replaced.

For a separate installation: `./install.sh --prefix "$HOME/intel-launcher-test"`. Its configuration stays under that prefix; it does not touch your normal launcher.

## Quick start

```bash
aag-llama-server-start --show-config
aag-llama-server-start --check-runtime
aag-llama-server-start --dry-run
aag-llama-server-start
aag-llama-server-status
aag-llama-server-stop
```

The server binds to `127.0.0.1:8080` by default. Exposing it beyond your machine requires your own access controls. AnythingLLM is optional.

Illustrative text transcript; available questions depend on model metadata:

```text
1) [V][T][R][M] example-model.gguf
V = Vision  T = Tools  R = Reasoning  M = matching MTP companion
Select model number:
Vision/mmproj:  1) Load with mmproj  2) Model only
MTP:           1) ON                2) OFF
Reasoning:     1) OFF               2) ON
Context:       1024 / 4096 / 8192 / 16384 / 32768 / Custom
Performance:   FAST / MEDIUM / SLOW-CONSERVATIVE
--------------------------------------------------
Model:        example-model
Vision:       OFF
MTP:          ON
Reasoning:    ON
Context:      8192 tokens capacity
Performance:  FAST
Backend:      SYCL
--------------------------------------------------
Start server? [Y/n]:
```

Context is server capacity, not a promise that every request contains that many tokens. Choices are bounded by model metadata and this launcher's 32K envelope. FAST TEXT can still use an explicitly chosen capacity.

## Automation and FAST TEXT

```bash
aag-llama-server-start --model "$HOME/Models/example.gguf" --context 8192 --performance medium --mmproj off --mtp off
aag-llama-server-start --profile fast-text --mtp off --dry-run
```

FAST TEXT requires `FAST_TEXT_TARGET` in your configuration. See [performance profiles](docs/PERFORMANCE-PROFILES.md) for the tested Official Gemma 4 12B QAT Q4_0 reference and optional verified MTP3 mapping. No private or fixed model path is assumed.

Legacy `--profile fast|normal|long` still means 4096/8192/32768 context capacity. Performance is a separate `--performance` flag. Unattended use adds no interactive questions. `--interactive` explicitly requests the complete wizard; `--help` lists all flags.

## Uninstall

```bash
./uninstall.sh
```

Or run `~/.local/share/intel-core-ultra-llama-linux/uninstall.sh`. Use the same `--prefix` for a custom installation. The receipt limits removal to unchanged installed files, restores original command backups, and preserves modified configs/files, models, llama.cpp, logs, and unrelated data. Stop a managed server yourself before uninstalling; the uninstaller does not stop services.

## Documentation

- [Quick start and configuration](docs/QUICKSTART.md)
- [Intel SYCL runtime](docs/INTEL-SYCL.md)
- [Model capabilities and reasoning](docs/MODEL-CAPABILITIES.md)
- [Performance profiles](docs/PERFORMANCE-PROFILES.md)
- [Projector and MTP matching](docs/MMPROJ-MTP.md)
- [Benchmark findings](docs/BENCHMARK-FINDINGS.md)
- [Optional AnythingLLM notes](docs/ANYTHINGLLM.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Architecture](docs/ARCHITECTURE.md)

## Tests and limitations

```bash
PYTHONPATH=lib python3 -B -m unittest discover -s tests -v
bash -n bin/aag-llama-control lib/intel-sycl.sh install.sh uninstall.sh
python3 tools/check_release.py
```

Tests create synthetic metadata fixtures and simulate installation into temporary prefixes. They do not load real model weights. Runtime/model support changes upstream; unknown metadata is rejected or shown without a speculative capability. MTP remains opt-in, matching is conservative rather than proof of semantic compatibility, and no universal speed or quality guarantee is made.

**Results are hardware/model/runtime specific.** Balanced was the tested power mode. The graph-ON experiment was rejected and is excluded from safe defaults. See the benchmark report for comparison limits.

## License and contributions

MIT; see [LICENSE](LICENSE) and [third-party notices](THIRD-PARTY-NOTICES.md). llama.cpp, Intel runtimes, and models remain separate dependencies with their own licenses.

Contributions should include a small reproducible fixture, relevant tests, and an explanation of compatibility and safety effects. Discuss broad changes first. Do not submit weights, secrets, raw inference logs, private prompts, or user data. See [CONTRIBUTING.md](CONTRIBUTING.md).
