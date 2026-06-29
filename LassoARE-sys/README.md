# LassoARE System

A Python, React, Scanpy, PyTorch, and optional RAPIDS application for interactive single-cell visualization and LassoARE analysis.

## Supported platform

- Linux x86_64
- CPU-only execution
- NVIDIA CUDA 12.8 execution with Linux driver 570.26 or newer

The service listens on `127.0.0.1:15114` by default.

## Native installation

Run the installer from the repository root:

```bash
./install.sh
```

It detects the available NVIDIA runtime, reuses micromamba, mamba, or Conda when available, and otherwise installs a private micromamba. The installer creates:

- CPU profile: `lassoare_main`
- CUDA profile: `lassoare_main` and `lassoare_rsc`

Force a profile or install without starting:

```bash
./install.sh --profile cpu
./install.sh --profile cuda --no-start
```

Start an existing installation:

```bash
./start.sh
```

To expose the unauthenticated service beyond the local machine, opt in explicitly:

```bash
./start.sh --host 0.0.0.0 --port 15114
```

Persistent files default to `~/.local/share/lassoare`. Override this with `--data-dir` or `LASSOARE_DATA_DIR`.

## Approximate installation size

These are planning estimates for the minimized manifests; Conda package reuse and Docker layer compression change the final number.

| Profile | Native disk usage | Local Docker image | Typical download |
| --- | ---: | ---: | ---: |
| CPU | 2-4 GB | 3-5 GB | 0.5-1.5 GB |
| CUDA | 20-30 GB total | 22-32 GB | 8-15 GB |

The CUDA total includes both `lassoare_main` and `lassoare_rsc`. For comparison, the existing untrimmed reference environments on this machine occupy 19 GB and 15 GB. Sample datasets and user-generated jobs are additional; the bundled small-sample slot is about 27 MB.

## Sample data

The UI lists the small and large sample slots reported by `/api/samples`. The verified 27 MB small sample can be downloaded on demand when its public URL is configured:

```bash
export LASSOARE_SMALL_SAMPLE_URL=https://example.org/sc_sampled.h5ad
./install.sh
```

If the file already exists at the repository root, the installer copies it into the persistent sample directory. A missing URL does not block installation; the UI shows the sample as not configured.

## Docker

Docker Compose v2 is required. CUDA also requires NVIDIA Container Toolkit.

```bash
./docker-start.sh
./docker-start.sh --profile cpu
./docker-start.sh --profile cuda
```

The launcher builds the selected local image and starts it in the background. The service is available at <http://127.0.0.1:15114>. Data is stored in the `lassoare-data` named volume.

The equivalent direct commands are:

```bash
docker compose up --build --detach
docker compose -f compose.yaml -f compose.cuda.yaml up --build --detach
```

## Runtime configuration

The launchers configure these variables:

| Variable | Purpose |
| --- | --- |
| `LASSOARE_PROFILE` | `cpu` or `cuda` |
| `LASSOARE_MAIN_PYTHON` | Python executable for API, Scanpy, and PyTorch |
| `LASSOARE_RSC_PYTHON` | RAPIDS Python executable; CUDA only |
| `LASSOARE_DATA_DIR` | Persistent datasets, jobs, and uploads |
| `LASSOARE_SAMPLE_DIR` | Installed sample files |
| `LASSOARE_SMALL_SAMPLE_URL` | Optional server-controlled sample URL |

CUDA jobs use RAPIDS for PCA and final neighbors, Leiden, and UMAP. If a RAPIDS stage fails during a running job, the service records the reason and falls back to Scanpy. CPU jobs use CPU PyTorch and Scanpy throughout.

## Development checks

```bash
/path/to/python -m unittest discover -s tests -v
cd frontend
npm ci
node --test src/runtime.test.js
npm run build
```
