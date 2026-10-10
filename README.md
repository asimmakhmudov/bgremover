# bgremover

[![PyPI](https://img.shields.io/pypi/v/bgremover-onnx.svg)](https://pypi.org/project/bgremover-onnx/)
[![Python](https://img.shields.io/pypi/pyversions/bgremover-onnx.svg)](https://pypi.org/project/bgremover-onnx/)
[![CI](https://github.com/asimmakhmudov/bgremover/actions/workflows/ci.yml/badge.svg)](https://github.com/asimmakhmudov/bgremover/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/asimmakhmudov/bgremover/blob/master/LICENSE)

Remove image backgrounds with pretrained ONNX segmentation models — as a Python
library, a command line tool, or an HTTP API.

It runs U²-Net, IS-Net and BiRefNet with its own inference code: numpy and
Pillow for the image math, onnxruntime for the model. There is no `rembg`
dependency. (The weight files are hosted on rembg's releases page, which is the
only thing taken from that project.)

```python
from bgremover import remove

with open("photo.jpg", "rb") as fh:
    png = remove(fh.read())
```

## Install

With pip:

```bash
pip install bgremover-onnx            # library + CLI
pip install "bgremover-onnx[api]"     # also the HTTP API
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add bgremover-onnx                 # add to a uv project
uv add "bgremover-onnx[api]"
uv pip install bgremover-onnx         # into the active environment
uvx --from bgremover-onnx bgremover remove photo.jpg   # run the CLI without installing it
uvx --from "bgremover-onnx[api]" bgremover serve   # run the API without installing it
```

The package is published on PyPI as `bgremover-onnx`; you still import it as
`bgremover` and run the `bgremover` command.

Python 3.10 or newer. Model weights are downloaded on first use and cached in
`~/.cache/bgremover` (override with `BGREMOVER_HOME`).

## Project structure

```
bgremover/
├── src/bgremover/
│   ├── __init__.py        Public API, resolved lazily (remove, Remover, MODELS, ...)
│   ├── models.py          Model registry: URL, SHA-256, input size, normalisation
│   ├── download.py        Download, hash-check and cache weights (stdlib only)
│   ├── session.py         Remover: owns the ONNX session, runs the pipeline
│   ├── processing.py      Pure image math: preprocess, postprocess, refine, cutout, trim
│   ├── images.py          Decode any input (bytes / path / PIL / numpy), encode output
│   ├── cli.py             `bgremover` command: remove, models, download, serve
│   └── server/            Optional HTTP API (installed with the [api] extra)
│       ├── app.py         FastAPI app, routes, error handling, bounded worker pool
│       ├── config.py      Settings read from BGREMOVER_* environment variables
│       └── fetch.py       SSRF-safe image fetching for the URL route
├── tests/                 pytest suite (fake ONNX session; real models with -m model)
├── .github/workflows/
│   ├── ci.yml             Lint, test on 3.10-3.13, build the package, smoke-test Docker
│   └── publish.yml        Publish to PyPI when a GitHub release is published
├── Dockerfile             Two stages: bake weights in, then a slim non-root runtime
└── pyproject.toml         Package metadata, extras, ruff and pytest config
```

How a request flows:

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

The core library never imports the `server` package, and `processing.py` does no
I/O, so each layer can be tested on its own.

## Library

`remove()` returns the same type you give it: bytes in, bytes out; a PIL image
in, a PIL image out; a numpy array in, a numpy array out; a path in, a PIL image
out.

```python
from bgremover import remove, Remover

png_bytes = remove(jpeg_bytes)                     # -> bytes (PNG)
image     = remove("photo.jpg")                    # -> PIL.Image (RGBA)
array     = remove(numpy_rgb)                      # -> numpy array (RGBA)

# Reuse one model across many images - the session is built once.
remover = Remover("isnet-general-use")
for path in paths:
    remover.remove(path).save(path.with_suffix(".png"))
```

Options for `remove()` / `Remover.remove()`:

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

`Remover` is lazy and thread-safe: constructing one touches neither the network
nor the disk, and the ONNX session is built on first use and then shared.

## CLI

```bash
bgremover remove photo.jpg                        # -> photo_nobg.png
bgremover remove photo.jpg -o out.png --refine --trim
bgremover remove *.jpg -o cutouts/ -m isnet-general-use
bgremover remove photo.jpg --bg white -o flat.jpg
bgremover remove photo.jpg --only-mask -o mask.png
cat photo.jpg | bgremover remove - > out.png

bgremover models                                  # list models, show what is cached
bgremover download u2net isnet-general-use        # pre-fetch weights
bgremover download --all
bgremover download --check u2net                  # re-hash a cached file
bgremover serve --port 3001                       # run the HTTP API
```

## HTTP API

```bash
pip install "bgremover-onnx[api]"
bgremover serve --port 3001          # docs at http://localhost:3001/docs
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

Both routes accept the same options as the library: `model`, `only_mask`,
`refine`, `background`, `trim`, `format`, `quality` — as form fields on the
upload route, as JSON keys on the URL route.

Errors are always `{"error": "..."}` with a 4xx or 5xx status.

### Server environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `BGREMOVER_MAX_UPLOAD_BYTES` | `10485760` | Upload / fetch size limit (10 MB) |
| `BGREMOVER_MODELS` | the six models ≤ 180 MB | Comma-separated allow-list |
| `BGREMOVER_DEFAULT_MODEL` | `u2net` | Model used when a request names none |
| `BGREMOVER_MAX_CONCURRENCY` | `2` | Inferences allowed to run at once |
| `BGREMOVER_FETCH_TIMEOUT` | `15` | Seconds allowed for fetching a remote image |
| `BGREMOVER_MAX_REDIRECTS` | `3` | Redirect hops, each re-checked |
| `BGREMOVER_ALLOW_PRIVATE_URLS` | `false` | Let URL fetches reach private addresses |
| `BGREMOVER_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `BGREMOVER_PRELOAD` | `false` | Load the default model at startup |

Library-wide: `BGREMOVER_HOME` (cache directory),
`BGREMOVER_MODEL_BASE_URL` (weights mirror), `BGREMOVER_PROVIDERS`
(onnxruntime execution providers), `BGREMOVER_NUM_THREADS`.

The URL route is deliberately strict: only `http`/`https`, and the host must
resolve to a public address — on every redirect hop too, so a public URL cannot
bounce a request into `169.254.169.254` or your private network.

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

Every entry carries a SHA-256 that is checked on download; a mismatch is
discarded rather than cached.

The two BiRefNet models are **not** in the server's default allow-list: they are
large and memory-hungry. Add them with `BGREMOVER_MODELS` only on a machine that
can hold them.

## Docker

```bash
docker build -t bgremover .                       # bakes in u2net
docker build --build-arg MODELS=u2netp,silueta -t bgremover .
docker run -p 3001:3001 bgremover
```

The weights are downloaded in a separate build stage and copied into the final
image, so a container does not fetch a model on its first request. Set
`BGREMOVER_MODELS` at run time to match what you baked in.

## Development

```bash
git clone https://github.com/asimmakhmudov/bgremover.git
cd bgremover

# with uv
uv venv && uv pip install -e ".[dev]"

# or with pip
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

pytest -q                     # fast: a fake ONNX session, no network
pytest -q -m model            # also runs real weights (downloads ~5 MB)
ruff check src tests && ruff format --check src tests
```

Contributions are welcome. Please open an issue first for larger changes, and
keep `pytest -q` and `ruff` green.

## License

MIT. The model weights carry their own licenses, listed in the table above.
