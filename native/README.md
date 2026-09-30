# Optional compiled dispatcher

This directory contains a small C++17 C ABI for hardened deployments that prefer a
shared library over the pure-Python `ctypes` implementations. It has platform guards
for Win32 `SendInput`, macOS CoreGraphics, and Linux uinput.

```bash
cmake -S native -B build/native
cmake --build build/native --config Release
```

Set `PHANTOMOS_NATIVE_LIB` to the resulting shared library and select
`native_backend = "compiled"` in the host configuration. The compiled path is never
selected by default, and it does not bypass the engine's confidence, unlock, allow-list,
or kill-switch policy.
