#!/usr/bin/env bash

intel_sycl_load() {
    local devices missing
    if [[ -f "$ONEAPI_ENV" ]]; then
        set +u
        if source "$ONEAPI_ENV" >/dev/null 2>&1; then
            set -u
        else
            set -u
            fail "Could not load oneAPI environment: $ONEAPI_ENV. See docs/INTEL-SYCL.md."
        fi
    elif ! command -v sycl-ls >/dev/null 2>&1; then
        fail "oneAPI/SYCL is missing. Configure ONEAPI_ENV or install a matching Intel runtime; see docs/INTEL-SYCL.md."
    fi
    export LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    export GGML_SYCL_ENABLE_DNN=1
    export GGML_SYCL_ENABLE_GRAPH=0
    command -v sycl-ls >/dev/null 2>&1 || fail "oneAPI did not provide sycl-ls. See docs/INTEL-SYCL.md."
    devices="$(timeout 20 sycl-ls 2>&1)" || fail "SYCL enumeration failed: $devices"
    grep -Eqi 'level_zero:gpu.*intel' <<< "$devices" || fail "No Intel Level Zero GPU. Check driver, render-node permissions, and docs/INTEL-SYCL.md."
    [[ -x "$LLAMA_SERVER" ]] || fail "llama-server not found: $LLAMA_SERVER. Configure BUILD_DIR."
    missing="$(ldd "$LLAMA_SERVER" 2>/dev/null | grep 'not found' || true)"
    [[ -z "$missing" ]] || fail "Missing runtime libraries: $missing. Match oneAPI to the build; do not create library-version symlinks."
}
