# ── Stage 1: dependency resolver ─────────────────────────────────────────────
# Use the official uv image to compile a lockfile-pinned dependency set into
# a clean virtualenv without pulling build tools into the final image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy only the files uv needs to resolve and install deps.
# Doing this before copying src/ means Docker can cache the layer
# as long as pyproject.toml and uv.lock do not change.
COPY pyproject.toml uv.lock ./

# Install runtime deps only (skip dev extras).
# --frozen ensures we use exactly what is in uv.lock.
# --no-install-project avoids installing the package itself yet.
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the package into the same venv.
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# Create a non-root user for safety.
RUN groupadd --system devfit && useradd --system --gid devfit devfit

WORKDIR /app

# Copy the pre-built virtualenv and the installed package from the builder.
COPY --from=builder --chown=devfit:devfit /app/.venv /app/.venv
COPY --from=builder --chown=devfit:devfit /app/src /app/src

# Copy static assets and templates that are served at runtime.
# These are not part of the Python package install so must be copied explicitly.
COPY --chown=devfit:devfit src/devfit/api/static  /app/src/devfit/api/static
COPY --chown=devfit:devfit src/devfit/api/templates /app/src/devfit/api/templates
COPY --chown=devfit:devfit src/devfit/prompts      /app/src/devfit/prompts

# Make the venv's executables available without activating it.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Hardcoded defaults — override at runtime via env vars or .env file.
# GROQ_API_KEY is required and must be supplied at runtime.
ENV DEVFIT_ENV=production
ENV LOG_LEVEL=INFO

USER devfit

EXPOSE 8000

# Uvicorn is installed as part of the venv; no need for CMD ["python", ...].
# --host 0.0.0.0 is required inside Docker (default binds only localhost).
CMD ["/app/.venv/bin/uvicorn", "devfit.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
