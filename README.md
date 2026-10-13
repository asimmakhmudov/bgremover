# snipbg

[![PyPI](https://img.shields.io/pypi/v/snipbg.svg)](https://pypi.org/project/snipbg/)
[![Python](https://img.shields.io/pypi/pyversions/snipbg.svg)](https://pypi.org/project/snipbg/)
[![CI](https://github.com/asimmakhmudov/snipbg/actions/workflows/ci.yml/badge.svg)](https://github.com/asimmakhmudov/snipbg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/asimmakhmudov/snipbg/blob/master/LICENSE)

**Remove image backgrounds locally** with pretrained ONNX segmentation models
(U²-Net, IS-Net, BiRefNet). Use it as a Python library, a command line tool, or
a self-hosted HTTP API. One name everywhere: `pip install snipbg`,
`import snipbg`, run `snipbg`.

```bash
pip install snipbg
snipbg remove photo.jpg          # -> photo_nobg.png
```

```python
from snipbg import remove

with open("photo.jpg", "rb") as fh:
    png = remove(fh.read())
```

- **Small dependency set**: numpy, Pillow and onnxruntime. No `rembg`, no PyTorch.
- **Runs offline** after the first download. Weights are SHA-256 checked and cached.
- **Same type in, same type out**: bytes, file paths, PIL images and numpy arrays.
- **API you can deploy**: FastAPI server with an SSRF-safe URL route, a bounded
  worker pool and a Docker image with the weights baked in.

## Contents

- [Install](#install)
- [Command line](#command-line)
- [Python library](#python-library)
- [HTTP API](#http-api)
- [Models](#models)
- [Docker](#docker)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Development](#development)
- [License](#license)

## Install

Python 3.10 or newer.

```bash
pip install snipbg            # library + CLI
pip install "snipbg[api]"     # also the HTTP API
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add snipbg                              # add to a uv project
uv add "snipbg[api]"
uvx snipbg remove photo.jpg                # run the CLI without installing it
uvx --from "snipbg[api]" snipbg serve      # run the API without installing it
```

Model weights are downloaded the first time a model is used and cached in
`~/.cache/snipbg`. Set `SNIPBG_HOME` to cache them somewhere else.

## Command line

```bash
snipbg remove photo.jpg                          # -> photo_nobg.png
snipbg remove photo.jpg -o out.png --refine --trim
snipbg remove *.jpg -o cutouts/ -m isnet-general-use
snipbg remove photo.jpg --bg white -o flat.jpg   # flat background instead of transparency
snipbg remove photo.jpg --only-mask -o mask.png
cat photo.jpg | snipbg remove - > out.png        # stdin -> stdout

snipbg models                                    # list models, show what is cached
snipbg download u2net isnet-general-use          # pre-fetch weights
snipbg download --all
snipbg download --check u2net                    # re-hash a cached file
snipbg serve --port 3001                         # run the HTTP API
```

`snipbg remove --help` lists every option. They are the same options as the
library: `--refine`, `--erode`, `--blur`, `--foreground-threshold`,
`--background-threshold`, `--bg`, `--trim`, `--format` and `--quality`.

## Python library

`remove()` returns the type it is given:

| Input | Output |
| --- | --- |
| `bytes` | `bytes` (PNG unless `output_format` says otherwise) |
| file path | `PIL.Image` (RGBA) |
| `PIL.Image` | `PIL.Image` (RGBA) |
| numpy array | numpy array (RGBA) |

```python
from snipbg import remove, Remover

png_bytes = remove(jpeg_bytes)
image     = remove("photo.jpg")
array     = remove(numpy_rgb)

# Batch work: build the model session once and reuse it.
remover = Remover("isnet-general-use")
for path in paths:
    remover.remove(path, refine=True).save(path.with_suffix(".png"))
```

Options for `remove()` and `Remover.remove()`:

| Option | Default | Meaning |
| --- | --- | --- |
| `only_mask` | `False` | Return the grayscale mask instead of the cutout |
| `refine` | `False` | Harden the confident regions and feather the edge |
| `foreground_threshold` | `240` | With `refine`: at or above this is fully opaque |
| `background_threshold` | `15` | With `refine`: at or below this is fully transparent |
| `erode_size` | `0` | With `refine`: pull the edge in by N pixels |
| `blur_radius` | `1.0` | With `refine`: feather radius |
| `background` | `None` | Composite onto a colour (`"white"`, `"#ff00ff"`, a tuple) |
| `trim_border` | `False` | Crop fully transparent borders away |
| `output_format` | `"PNG"` | Encoding used when the input was bytes |
| `quality` | `None` | JPEG/WEBP quality when the input was bytes |

`Remover` is lazy and thread-safe. Creating one does not touch the network or
the disk. The ONNX session is built on first use and then shared across threads.

## HTTP API

```bash
pip install "snipbg[api]"
snipbg serve --port 3001          # interactive docs at http://localhost:3001/docs
```

| Route | What it does |
| --- | --- |
| `POST /remove-background` | Multipart upload, field name `image` |
| `POST /remove-background-url` | JSON body with `imageUrl` (or `image_url`) |
| `GET /models` | Models this server will serve |
| `GET /health` | Health check |
| `GET /` | API description |

```bash
curl -X POST -F "image=@photo.jpg" \
  http://localhost:3001/remove-background --output out.png

curl -X POST -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/photo.jpg"}' \
  http://localhost:3001/remove-background-url --output out.png
```

Both routes take the library's options: `model`, `only_mask`, `refine`,
`background`, `trim`, `format`, `quality`. Send them as form fields on the
upload route and as JSON keys on the URL route. Errors are always
`{"error": "..."}` with a 4xx or 5xx status.

The URL route accepts only `http` and `https`, and the host must resolve to a
public address. Each redirect hop is checked again, so a public URL cannot
redirect the server to `169.254.169.254` or another private address.

## Models

| Name | Size | Input | License | Notes |
| --- | --- | --- | --- | --- |
| `u2net` | 168 MB | 320² | Apache-2.0 | General purpose. The default. |
| `u2netp` | 4 MB | 320² | Apache-2.0 | Small and fast, lower quality |
| `u2net_human_seg` | 168 MB | 320² | Apache-2.0 | Fine-tuned for people |
| `silueta` | 42 MB | 320² | Apache-2.0 | u2net pruned; near-u2net quality |
| `isnet-general-use` | 170 MB | 1024² | Apache-2.0 | Sharper edges, slower |
| `isnet-anime` | 168 MB | 1024² | Apache-2.0 | Anime and illustration |
| `birefnet-general-lite` | 214 MB | 1024² | MIT | Best quality that fits in ~8 GB RAM |
| `birefnet-general` | 928 MB | 1024² | MIT | Highest quality; needs a GPU or 16 GB+ RAM |

**Picking one:** start with `u2net`. Use `u2netp` when speed or size matters
most, `isnet-general-use` for cleaner edges, and `birefnet-general-lite` when
quality matters more than speed.

Every download is checked against the model's SHA-256. A file that does not
match is thrown away instead of being cached. The weights are hosted on
[rembg](https://github.com/danielgatis/rembg)'s releases page. snipbg does its
own preprocessing and postprocessing and does not depend on rembg.

## Docker

```bash
docker build -t snipbg .                                  # bakes in u2net
docker build --build-arg MODELS=u2netp,silueta -t snipbg .
docker run -p 3001:3001 snipbg
```

The weights are downloaded in a separate build stage and copied into a slim
runtime image that runs as a non-root user, so the first request does not
trigger a download. If you bake in models other than the default, set
`SNIPBG_MODELS` at run time to the same list.

## Configuration

All settings are environment variables.

**Library and CLI**

| Variable | Default | Meaning |
| --- | --- | --- |
| `SNIPBG_HOME` | `~/.cache/snipbg` | Where weights are cached |
| `SNIPBG_MODEL_BASE_URL` | rembg releases | Mirror to download weights from |
| `SNIPBG_PROVIDERS` | onnxruntime default | Comma-separated execution providers |
| `SNIPBG_NUM_THREADS` | onnxruntime default | Intra-op thread count |

**HTTP server**

| Variable | Default | Meaning |
| --- | --- | --- |
| `SNIPBG_MODELS` | the six models ≤ 180 MB | Comma-separated allow-list |
| `SNIPBG_DEFAULT_MODEL` | `u2net` | Model used when a request names none |
| `SNIPBG_MAX_UPLOAD_BYTES` | `10485760` | Upload and fetch size limit (10 MB) |
| `SNIPBG_MAX_CONCURRENCY` | `2` | Inferences allowed to run at once |
| `SNIPBG_FETCH_TIMEOUT` | `15` | Seconds allowed for fetching a remote image |
| `SNIPBG_MAX_REDIRECTS` | `3` | Redirect hops, each one checked again |
| `SNIPBG_ALLOW_PRIVATE_URLS` | `false` | Let URL fetches reach private addresses |
| `SNIPBG_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `SNIPBG_PRELOAD` | `false` | Load the default model at startup |

The two BiRefNet models are left out of the default `SNIPBG_MODELS` list
because they are large and use a lot of memory. Add them only on a machine
with enough RAM.

## How it works

```
input (bytes / path / PIL / numpy)
  -> images.load_image          decode to an RGB PIL image
  -> processing.preprocess      resize to the model's input, normalise, NCHW tensor
  -> onnxruntime                run the segmentation model (session.Remover)
  -> processing.postprocess     scale the mask back to the image size
  -> processing.refine_mask     optional: harden + feather the edge
  -> processing.cutout / trim / apply_background
  -> images.encode_image        same type out as came in
```

```
src/snipbg/
├── __init__.py        Public API, resolved lazily (remove, Remover, MODELS, ...)
├── models.py          Model registry: URL, SHA-256, input size, normalisation
├── download.py        Download, hash-check and cache weights (stdlib only)
├── session.py         Remover: owns the ONNX session, runs the pipeline
├── processing.py      Pure image math: preprocess, postprocess, refine, cutout, trim
├── images.py          Decode any input, encode the output
├── cli.py             `snipbg` command: remove, models, download, serve
└── server/            Optional HTTP API (the [api] extra)
    ├── app.py         FastAPI app, routes, error handling, bounded worker pool
    ├── config.py      Settings read from SNIPBG_* environment variables
    └── fetch.py       SSRF-safe image fetching for the URL route
```

The core library never imports `server`, and `processing.py` does no I/O, so
each layer can be tested on its own.

## Development

```bash
git clone https://github.com/asimmakhmudov/snipbg.git
cd snipbg
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"

pytest -q                     # fast: a fake ONNX session, no network
pytest -q -m model            # also runs real weights (downloads ~5 MB)
ruff check src tests && ruff format --check src tests
```

CI lints, runs the tests on Python 3.10–3.13, builds the package and
smoke-tests the Docker image. Publishing a GitHub release uploads the package
to PyPI.

Contributions are welcome. Please open an issue before starting a larger
change, and make sure `pytest -q` and `ruff` pass.

## License

MIT. The model weights have their own licenses, listed in the [Models](#models)
table.
