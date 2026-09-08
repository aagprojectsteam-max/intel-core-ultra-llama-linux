# Intel SYCL setup

Use a supported Intel GPU driver and a oneAPI runtime that matches the compiler/libraries used to build llama.cpp. A runtime with the wrong SONAME can fail even when another oneAPI version is installed. Intel Arc integrated and discrete devices differ in memory and operator support.

Follow the maintained [llama.cpp SYCL guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) and [Intel oneAPI toolkit downloads](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html). Consult Intel's distribution-specific driver instructions before changing packages.

This installer does not run sudo, install packages, add package repositories, or change kernel/driver configuration.

Typical source build, after installing the documented prerequisites:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build-sycl" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/llama.cpp/build-sycl" --config Release -j 4
sycl-ls
```

The tested development snapshot was llama.cpp commit `a1f96d4fc2c9e4101a6666a9d87f547e7e880df6`. Newer/older builds may change flags or model support. Check the build's `--help`; the launcher explicitly checks draft-MTP CLI availability when MTP is requested.

The wrapper loads configured `ONEAPI_ENV` (default `/opt/intel/oneapi/setvars.sh`) inside the launcher process. An already configured shell with `sycl-ls` is accepted if that file is absent. It sets `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, checks Intel Level Zero enumeration, and checks missing shared libraries before launching.

Only the launcher and its child receive the library path. There are no edits to global linker files or shell startup files. The managed service receives that runtime library path and graph OFF / DNN ON explicitly.

For `libsvml.so` or `libsycl.so.9` errors, install/select the runtime matching the build, set `ONEAPI_ENV`, and run `--check-runtime`. Do not symlink incompatible library versions to satisfy a missing name.

Balanced is the reference power mode. If powerprofilesctl is available, actual launch requires Balanced; the launcher never switches it. If unavailable, it reports that detection is unavailable. A healthy sycl-ls result is device discovery, not a full driver stress test.
