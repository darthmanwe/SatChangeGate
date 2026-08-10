# The rasterio/GDAL + OpenCV + scikit-image stack is one of the harder Python
# installs to get right, and it differs by OS. Shipping a container is the
# difference between a reviewer running this in one command and not running it
# at all. Wheels cover GDAL for these packages, so no system GDAL is needed.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl/libglib are required by opencv even in the headless build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[dev]"

COPY Makefile ./
COPY tests/ ./tests/

# Non-root by default.
RUN useradd --create-home --uid 1000 scg \
    && mkdir -p /app/data \
    && chown -R scg:scg /app
USER scg

# Verify the console script resolves and packaged config loads from the wheel
# rather than a source checkout (the previous layout could not do this).
RUN satchangegate --version

ENTRYPOINT ["satchangegate"]
CMD ["--help"]
