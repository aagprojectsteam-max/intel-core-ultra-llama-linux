# Troubleshooting

- **No models:** verify MODEL_ROOT, readable GGUF v2/v3 metadata, and filenames. Discovery excludes sidecars, secondary shards, and external symlinks. Set the root to the actual model directory.
- **No companion question:** matching did not establish compatibility. Keep publisher-matched companions next to the main model; do not rename unrelated files just to make them match.
- **No reasoning levels:** many models expose only ON/OFF. A badge without a recognized control gets an informational line.
- **Missing libsvml.so / libsycl.so.9:** select the oneAPI runtime matching the build; see INTEL-SYCL.md. Avoid global library-path edits and incompatible library symlinks.
- **No Level Zero GPU:** check the Intel driver, render-node access, and sycl-ls output. The installer does not change device permissions.
- **MTP flags unavailable:** use a compatible llama.cpp build or disable MTP. CLI support is checked before launch.
- **Systemd user session unavailable:** actual managed starts require a systemd user manager. Tests and help work without it; container/WSL service management is not certified.
- **Port already occupied:** choose another PORT or inspect the existing server. The launcher refuses duplicate/unmanaged processes instead of terminating them.
- **OOM or unsupported Flash Attention:** reduce context/model size; configure FLASH_ATTN=off or auto where needed. The launcher does not silently switch backends.
- **Fence timeout, GPU hang, crash:** stop testing, inspect your driver state locally, and recover it before further inference. Do not keep retrying an unstable GPU or enable the rejected graph experiment.
- **Installer conflict:** inspect the existing command. Use --replace only if you want a backup and replacement; use a separate --prefix to coexist.
- **Uninstall preserves files:** modified files and replacement backups are intentionally retained. Inspect them manually; no broad directory deletion is performed.

Logs remain local under the project's XDG state directory. Do not upload logs or configuration without checking for model paths, prompts, and credentials.
