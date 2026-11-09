# Requirements and setup

What each package needs to build and run, and how the providers install and run their libraries and models.

- **Go 1.26** or newer: the pinned yzma release needs it.
- **`vector/sqlitevec`** is CGO and needs the system SQLite headers to compile:
  `sudo apt-get install -y libsqlite3-dev` (Debian/Ubuntu). The extension registers
  itself for every SQLite connection the process opens after the package is imported,
  so import it (blank if need be) before opening the first connection. Open a file
  database with a busy timeout, so concurrent writers queue instead of failing with
  "database is locked", and in WAL mode, so readers do not wait for them:
  `search.db?_busy_timeout=5000&_journal_mode=WAL` with mattn/go-sqlite3. Every write
  transaction of the store starts with a write, so `_txlock=immediate` is not needed.
- **`vector/pgvec`** needs a Postgres with the pgvector extension, or a user allowed
  to run `CREATE EXTENSION vector`. Only writes create tables, so a role that may only
  read can search what another role indexed.
- **`provider/yzma`** needs `libffi8` wherever the binary runs: the Go bindings open it
  in a package `init()` and panic without it, so a binary that imports the package needs
  it even when it never embeds anything. The llama.cpp libraries also need `libgomp1`,
  `libstdc++6` and glibc 2.34 or newer:
  `sudo apt-get install -y libffi8 libgomp1 libstdc++6`.
- **The chat model** (`yzma.NewCompleter`, for the tag `Generator`) is a 2.5 GB download on first use,
  and a process with it and the embedding model loaded holds about 5 GB of RAM.
- **GPUs with `provider/yzma`.** `Runtime.Accel` left empty picks from the hardware: a
  managed `LibDir` gets the pinned llama.cpp build that carries the backend the machine
  has (CUDA 13.4 on linux/amd64 with the CUDA runtime installed, Vulkan on linux/amd64
  and linux/arm64, Metal on macOS, the CPU build otherwise; `internal/hw` looks for the
  driver and, for CUDA, for `libcudart.so.13` and `libcublas.so.13` on the loader's
  path, so an NVIDIA driver without the CUDA 13 runtime gets the Vulkan build), the
  model is offloaded to the GPU that registers, and the load log line says which; a
  GPU build that registers no GPU is logged as a warning, since the models then run on
  the CPU. `AccelCPU` stays on the CPU, `AccelVulkan`, `AccelCUDA` and `AccelMetal`
  require their backend. An operator-supplied `LibDir` is used as it is: whatever
  backends it holds register. Vulkan was run on an AMD iGPU for this release; the CUDA
  build is pinned but was not run.
- **A managed `LibDir`** (`ManagedLibDir: true`, in both providers) belongs to the
  provider, so give it a directory of its own. Only a missing or empty directory, or one
  an earlier install left its record in, is filled or replaced; anything else is an
  error, never a deletion. The record names what was installed, and a directory whose
  record names another build than the pinned one (the llama.cpp tag or the ONNX Runtime
  version of an older release of this module) is replaced at the next open. Installs take a lock file next to it (`<LibDir>.lock`), so
  goroutines and processes that start together download once and none loads a
  half-installed library. An install works next to the directory, under
  `<LibDir>.sense-download`, `<LibDir>.sense-staging` and `<LibDir>.sense-retired`,
  and nothing else in the parent directory is its business: an open that finds the
  libraries in place writes nothing but the lock file, and what a killed install left
  under those names is cleared only when it carries the install's marker or record
  (anything else there stops the install with an error). A `libs.old` of yours stays.
  A `LibDir` that is a symbolic link (to another disk, say) is filled where it points
  and stays a link; the lock and the staging are then next to its target, and the link
  and its target are one directory to a process that names both. An open waits for the
  lock until its context ends, and then fails with the context's error; a process that
  already runs the libraries from the directory does not take the lock at all, so its
  next open is not held up by another process's download.
  Libraries a process has loaded are never replaced under it:
  asking by name for another build in the same process (`AccelCPU` first and
  `AccelVulkan` later, `ProviderCPU` and then `ProviderCUDA`) is an error, so open the
  most capable one first; `Accel` left empty keeps what is loaded.
- **llama.cpp's log** goes to `Runtime.Logger`: its warnings and errors, and nothing of
  its progress. A call whose context ends mid-evaluation stops llama.cpp, which then
  logs errors such as `failed to decode, ret = 2`; those come at Debug with
  `cancelled=true`, since the caller already has its context's error, while a decode
  that fails for any other reason stays an error.
- **The vision model** (`yzma.NewVision`) is a 2.7 GB download on first use (the Q8_0
  GGUF and its f16 projector) and needs `ffmpeg` and `ffprobe` on PATH for videos; a
  loaded process holds about 3 GB.
- **Keep yzma where sense pins it.** The llama.cpp build `provider/yzma` downloads
  (`yzma.LibTag`) matches the yzma version in this module's `go.mod`, whose Go
  bindings mirror exactly that build's ABI. A module that imports `provider/yzma` must
  not require another yzma version: minimal version selection would hand these
  bindings to libraries they do not match.
- **`provider/onnx`** is CGO and needs a C compiler to build (no ONNX Runtime headers:
  the bindings carry their own). At runtime it needs the ONNX Runtime shared library:
  a managed `LibDir` downloads the pinned CPU build (`onnx.LibVersion`, ~11 MB) on
  linux/amd64, linux/arm64 and darwin/arm64. For **CUDA** set `Runtime.Provider` to
  `onnx.ProviderCUDA`: a managed `LibDir` on linux/amd64 downloads the CUDA 12 build
  (~450 MB), and the host needs CUDA 12 and cuDNN 9 installed; **CoreML** is built into
  the macOS build. The bindings' headers target one runtime release, so keep
  `onnxruntime_go` where this module pins it, as with yzma. The default model is a
  ~370 MB download on first use, and a loaded process holds about 1 GB.
- **`media/ffmpeg`** runs the `ffmpeg` and `ffprobe` binaries, 5.1 or newer, found on
  PATH or named in its `Config`: `sudo apt-get install -y ffmpeg`.
