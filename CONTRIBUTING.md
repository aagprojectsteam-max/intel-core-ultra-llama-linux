# Contributing

Open an issue describing the problem and a small reproducible example before broad changes. Keep changes reviewable, preserve noninteractive flags, and explain model-specific assumptions. Add synthetic metadata fixtures rather than uploading real model files or prompts.

Run the unittest suite, Bash syntax checks, and tools/check_release.py. Installer changes must retain prefix isolation, backup restoration, and preservation of modified files.

Do not submit secrets, weights, private logs, databases, or application storage. Review all generated code and be able to maintain it. This project's initial implementation and documentation were prepared with OpenAI Codex assistance and verified with automated and local acceptance checks.

This repository is independent of llama.cpp. Upstream contributions must follow that project's own contribution rules.
