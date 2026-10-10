# bgremover: project guide

My own notes on how this project is put together, how to run it, and how to
release it. The public-facing docs are in [`README.md`](../README.md).

- PyPI name: **`bgremover-onnx`** (plain `bgremover` was rejected as too similar
  to the existing `bg-remover`); import name `bgremover`; CLI `bgremover`
- Repo: https://github.com/asimmakhmudov/bgremover
- Version lives in one place: `src/bgremover/__init__.py` (`__version__`)
- Python 3.10+; build backend: hatchling

---

## 1. What the project is

A background remover with three entry points over one core:

| Entry point | Install | Used as |
| --- | --- | --- |
| Library | `pip install bgremover-onnx` | `from bgremover import remove` |
| CLI | same | `bgremover remove photo.jpg` |
| HTTP API | `pip install "bgremover-onnx[api]"` | `bgremover serve` (FastAPI + uvicorn) |

It runs pretrained segmentation models (U²-Net, IS-Net, BiRefNet) exported to
ONNX. The inference code is my own: numpy and Pillow do the image math and
onnxruntime runs the model. rembg is **not** a dependency; its GitHub releases
page is only where the `.onnx` weight files are hosted.

History: this repo was a Node.js server (`server.js`) calling a Python script
(`model.py`). It is now a single Python package. The old Node routes
(`/remove-background`, `/remove-background-url`, `/health`, `/`) and port 3001
were kept so existing clients still work.

---

## 2. Architecture

### Modules

| File | Responsibility | Allowed dependencies |
| --- | --- | --- |
| `__init__.py` | Public API; attributes resolved lazily via `__getattr__` | stdlib only |
| `models.py` | `ModelSpec` registry: file name, SHA-256, size, input size, mean/std, sigmoid flag, license | stdlib only |
| `download.py` | `cache_dir()`, `ensure_model()`, download with a progress bar, SHA-256 verify, cache management | stdlib only |
| `images.py` | `load_image()` for bytes / path / PIL / numpy; `encode_image()`; numpy conversion | Pillow, numpy |
| `processing.py` | `preprocess`, `postprocess`, `refine_mask`, `cutout`, `apply_background`, `trim` | Pillow, numpy (no I/O) |
| `session.py` | `Remover` (lazy, thread-safe ONNX session), `get_remover()` cache, `remove()` | onnxruntime |
| `cli.py` | argparse CLI: `remove`, `models`, `download`, `serve` | the above |
| `server/config.py` | `Settings` dataclass read from `BGREMOVER_*` env vars | stdlib |
| `server/fetch.py` | URL fetching with an SSRF guard (public IPs only, every redirect re-checked) | stdlib |
| `server/app.py` | `create_app()`: routes, error shape, upload limit, `CapacityLimiter` worker pool | FastAPI |

### Pipeline (`Remover.remove`)

```
load_image(data)                  -> RGB PIL image; remember the input type
preprocess(image, input_size,     -> resize, scale to 0..1, normalise with mean/std,
           mean, std)                float32 NCHW tensor
session.run(...)                  -> raw mask
postprocess(raw, size, sigmoid)   -> min-max normalise, resize back to the image size
refine_mask(...)                  -> only if refine=True: thresholds, erode, blur
cutout(image, mask)               -> RGBA, mask used as alpha
trim(...)                         -> only if trim_border=True (before the background)
apply_background(...)             -> only if background=... is given
encode / convert                  -> bytes in -> bytes out, numpy in -> numpy out, else PIL
```

### Design rules (enforced by `tests/test_architecture.py`)

1. Never depend on or import `rembg`.
2. `bgremover.server` is optional. The core library must never import it or
   FastAPI.
3. `processing.py` has no ONNX, network or file I/O. It is pure functions.
4. `models.py`, `download.py` and `__init__.py` use only the stdlib. The Docker
   weights stage runs them with no packages installed.
5. No model above ~250 MB goes in the server's default allow-list (BiRefNet is
   opt-in).
6. The legacy Node.js routes must keep existing.
7. Server safety rails stay: upload size limit, model allow-list, SSRF-safe
   fetch, bounded concurrency.

If one of these tests fails, the change broke a rule. Fix the change, not the
test.

### Model cache

- Default location: `~/.cache/bgremover/` (or `$XDG_CACHE_HOME/bgremover`)
- Override: `BGREMOVER_HOME=/some/dir`
- Mirror: `BGREMOVER_MODEL_BASE_URL=https://my-mirror/...`
- Each download is SHA-256 checked. A mismatch is deleted, never cached.

---

## 3. Local setup

I use **uv**; pip works the same way.

```bash
cd ~/Desktop/bg-remove-api

# uv
uv venv --python 3.12
uv pip install -e ".[dev]"
source .venv/bin/activate

# pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`-e` (editable) means code changes take effect without reinstalling. The `dev`
extra pulls in the API deps plus pytest, httpx, ruff, build and twine.

---

## 4. Running it

### Library

```python
from bgremover import remove, Remover

remove("photo.jpg").save("out.png")              # path in -> PIL out

remover = Remover("isnet-general-use")           # reuse one session
remover.load()                                   # optional: download/build now
png = remover.remove(open("a.jpg", "rb").read(), refine=True, trim_border=True)
```

### CLI

```bash
bgremover --version
bgremover models                          # table, shows what is cached
bgremover models --json
bgremover download u2net                  # pre-fetch weights
bgremover download --all
bgremover download --check u2net          # re-verify the hash of a cached file

bgremover remove photo.jpg                # -> photo_nobg.png
bgremover remove photo.jpg -o out.png --refine --trim
bgremover remove *.jpg -o cutouts/ -m u2netp
bgremover remove photo.jpg --bg white -o flat.jpg
bgremover remove photo.jpg --only-mask -o mask.png
cat photo.jpg | bgremover remove - > out.png
```

Exit code is 1 if any input failed, so it works in scripts.

### HTTP API (development)

```bash
bgremover serve --port 3001 --reload
# Swagger UI: http://localhost:3001/docs
# ReDoc:      http://localhost:3001/redoc

curl localhost:3001/health
curl localhost:3001/models
curl -X POST -F "image=@photo.jpg" -F "refine=true" \
     localhost:3001/remove-background -o out.png
curl -X POST -H "Content-Type: application/json" \
     -d '{"imageUrl":"https://example.com/p.jpg","model":"u2netp"}' \
     localhost:3001/remove-background-url -o out.png
```

### HTTP API (production, without Docker)

```bash
pip install "bgremover-onnx[api]"
export BGREMOVER_PRELOAD=true BGREMOVER_MAX_CONCURRENCY=2
bgremover serve --host 0.0.0.0 --port 3001 --workers 2
```

Each worker process loads its own copy of every model it uses, so plan RAM as
workers × model size. Within one worker, `BGREMOVER_MAX_CONCURRENCY` caps how
many inferences run at once.

### Docker

```bash
docker build -t bgremover .                                   # bakes u2net in
docker build --build-arg MODELS=u2netp,silueta -t bgremover:small .
docker run --rm -p 3001:3001 bgremover
docker run --rm -p 3001:3001 \
  -e BGREMOVER_MODELS=u2netp,silueta -e BGREMOVER_DEFAULT_MODEL=u2netp \
  bgremover:small
```

The image runs as uid 10001, has a `HEALTHCHECK` on `/health`, and reads the port
from `$PORT` (default 3001), which suits Render, Railway and Fly.io. Keep
`BGREMOVER_MODELS` in line with the `MODELS` build arg, or the first request
for a model that isn't baked in will download it at runtime.

### Environment variables (quick reference)

| Variable | Default | Scope |
| --- | --- | --- |
| `BGREMOVER_HOME` | `~/.cache/bgremover` | weight cache |
| `BGREMOVER_MODEL_BASE_URL` | rembg releases | weight mirror |
| `BGREMOVER_PROVIDERS` | onnxruntime default | e.g. `CUDAExecutionProvider,CPUExecutionProvider` |
| `BGREMOVER_NUM_THREADS` | onnxruntime default | intra-op threads |
| `BGREMOVER_MAX_UPLOAD_BYTES` | `10485760` | server |
| `BGREMOVER_MODELS` | six models ≤ 180 MB | server allow-list |
| `BGREMOVER_DEFAULT_MODEL` | `u2net` | server |
| `BGREMOVER_MAX_CONCURRENCY` | `2` | server |
| `BGREMOVER_FETCH_TIMEOUT` | `15` | server, URL route |
| `BGREMOVER_MAX_REDIRECTS` | `3` | server, URL route |
| `BGREMOVER_ALLOW_PRIVATE_URLS` | `false` | server, URL route (keep false in prod) |
| `BGREMOVER_CORS_ORIGINS` | `*` | server |
| `BGREMOVER_PRELOAD` | `false` | server: load the default model at startup |

---

## 5. Testing and linting

```bash
pytest -q              # ~1 s, fake ONNX session (tests/conftest.py), no network
pytest -q -m model     # real weights: downloads u2netp (~5 MB) once
pytest -q tests/test_server.py -k url     # a subset
ruff check src tests
ruff format src tests  # apply formatting (CI runs --check)
```

How the fast tests work: `conftest.py` swaps the ONNX session for
`FakeSession`, which marks a pixel as foreground when its red channel beats its
blue one. The tests build half-red, half-blue images and assert on that.

CI (`.github/workflows/ci.yml`) runs on every push and PR:
lint + tests on Python 3.10-3.13, real-model tests, `python -m build` +
`twine check`, and a Docker build plus a smoke test.

---

## 6. Releasing to PyPI (pip and uv)

One upload to PyPI makes the package available to **both** pip and uv, since uv
installs from PyPI too:

```bash
pip install bgremover-onnx
uv add bgremover-onnx        # or: uv pip install bgremover-onnx / uvx --from bgremover-onnx bgremover
```

### One-time setup

1. Create accounts on https://pypi.org and https://test.pypi.org, and turn on 2FA
   (PyPI requires it).
2. Pick **one** way to authenticate:
   - **Trusted Publishing (recommended, no tokens).** On PyPI, open
     https://pypi.org/manage/project/bgremover-onnx/settings/publishing/ and add a
     GitHub publisher: owner `asimmakhmudov`, repo `bgremover`, workflow
     `publish.yml`, environment `pypi`. On GitHub,
     go to *Settings -> Environments* and create an environment named `pypi`.
     After this, publishing a GitHub release uploads to PyPI automatically.
   - **API token (manual uploads).** PyPI -> *Account settings -> API tokens*. Use
     an "Entire account" scope for the first upload (the project doesn't exist
     yet), then replace it with a token scoped to `bgremover`.

### Release checklist

```bash
# 1. bump the version
#    edit src/bgremover/__init__.py -> __version__ = "2.0.1"

# 2. make sure everything is green
ruff check src tests && ruff format --check src tests
pytest -q && pytest -q -m model

# 3. build fresh artifacts
rm -rf dist/
uv build                       # or: python -m build
uvx twine check dist/*         # README renders correctly on PyPI

# 4. smoke-test the wheel in a throwaway env
uv run --isolated --with dist/*.whl bgremover --version

# 5a. publish via GitHub (Trusted Publishing)
git commit -am "Release 2.0.1" && git tag v2.0.1 && git push --follow-tags
gh release create v2.0.1 --generate-notes      # triggers publish.yml

# 5b. or publish manually with a token
uv publish --token pypi-XXXXXXXX
#    (twine alternative: twine upload dist/*  with user __token__)
```

Optional dry run on TestPyPI first:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-TESTTOKEN
uv pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ bgremover
```

Things to remember:
- PyPI never accepts the same version twice, even after you delete it. Always
  bump `__version__`.
- Don't put a token in the repo or in shell history. Use `UV_PUBLISH_TOKEN`
  or `~/.pypirc`.
- `dist/` is gitignored. Rebuild it for every release.

### Verify after a release

```bash
uvx --refresh --from bgremover-onnx bgremover --version
uvx --refresh --from "bgremover-onnx[api]" bgremover serve --port 3001
pip install -U bgremover && python -c "import bgremover; print(bgremover.__version__)"
```

---

## 7. Adding a new model

1. Add a `ModelSpec` to `MODELS` in `models.py`: filename, SHA-256
   (`shasum -a 256 file.onnx`), `size_bytes`, `input_size`, mean/std, and
   `sigmoid=True` if the graph outputs logits.
2. If it's ≤ ~250 MB and generally useful, add it to `DEFAULT_ALLOWED_MODELS` in
   `server/config.py`.
3. Add a row to the model table in `README.md`.
4. `pytest -q`, then try it for real: `bgremover remove sample.jpg -m <name>`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `error: the API needs the extra dependencies` | `pip install "bgremover-onnx[api]"` |
| `DownloadError ... SHA-256 mismatch` | Bad or partial download. Run `bgremover download --check <m>`, then download it again |
| First request is very slow | Weights are downloading. Pre-fetch with `bgremover download`, set `BGREMOVER_PRELOAD=true`, or bake them into Docker |
| URL route returns 400 "refusing to fetch ... non-public address" | SSRF guard working as intended. Only set `BGREMOVER_ALLOW_PRIVATE_URLS=true` in local dev |
| 413 on upload | Raise `BGREMOVER_MAX_UPLOAD_BYTES` |
| Container killed (OOM) | Model too big for the box. Use `u2netp`/`silueta`, or fewer workers |
| Want GPU | `pip install onnxruntime-gpu` and set `BGREMOVER_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` |
