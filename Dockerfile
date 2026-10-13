# syntax=docker/dockerfile:1

# --- stage 1: weights -------------------------------------------------------
# Only models.py, download.py and the (lazy, import-light) package __init__ are
# copied here, so this stage needs nothing but the standard library. Baking the
# weights in means the first request does not have to download 170 MB.
FROM python:3.12-slim AS weights

ARG MODELS=u2net
ENV SNIPBG_HOME=/models

WORKDIR /build
COPY src/snipbg/__init__.py src/snipbg/models.py src/snipbg/download.py snipbg/

RUN python -c "\
import sys; \
from snipbg.download import ensure_model; \
[ensure_model(name.strip(), progress=True) for name in sys.argv[1].split(',') if name.strip()]\
" "$MODELS"

# --- stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SNIPBG_HOME=/models \
    PORT=3001

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[api]"

COPY --from=weights /models /models

# Run as a non-root user; /models only needs to be readable.
RUN useradd --create-home --uid 10001 snipbg \
    && chown -R snipbg:snipbg /models
USER snipbg

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','3001') + '/health').read()"

# Serve only the models baked into the image above.
ENV SNIPBG_MODELS=u2net \
    SNIPBG_DEFAULT_MODEL=u2net

CMD ["sh", "-c", "snipbg serve --host 0.0.0.0 --port ${PORT}"]
