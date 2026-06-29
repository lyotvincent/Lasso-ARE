# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ubuntu:22.04 AS runtime-base
ARG DEBIAN_FRONTEND=noninteractive
ARG MICROMAMBA_VERSION=2.6.2-1
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/usr/local/bin:/opt/micromamba/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL \
      "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
      -o /tmp/micromamba \
    && curl -fsSL \
      "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64.sha256" \
      -o /tmp/micromamba.sha256 \
    && expected_sha256="$(tr -d '[:space:]' < /tmp/micromamba.sha256)" \
    && actual_sha256="$(sha256sum /tmp/micromamba | awk '{print $1}')" \
    && test "$expected_sha256" = "$actual_sha256" \
    && install -m 0755 /tmp/micromamba /usr/local/bin/micromamba \
    && rm -f /tmp/micromamba /tmp/micromamba.sha256

COPY environments/ /tmp/environments/

FROM runtime-base AS cpu-environment
RUN micromamba env create -y -f /tmp/environments/lassoare-main-cpu.yml \
    && micromamba run -n lassoare_main \
      python -m pip install -r /tmp/environments/torch-cpu.txt \
    && micromamba clean --all --yes

FROM runtime-base AS cuda-environment
RUN micromamba env create -y -f /tmp/environments/lassoare-main-cuda.yml \
    && micromamba run -n lassoare_main \
      python -m pip install -r /tmp/environments/torch-cu128.txt \
    && micromamba env create -y -f /tmp/environments/lassoare-rsc.yml \
    && micromamba clean --all --yes

FROM runtime-base AS application-base
WORKDIR /opt/lassoare
COPY . ./
COPY --from=frontend-build /build/frontend/dist/ ./frontend/dist/
RUN chmod +x /opt/lassoare/docker/entrypoint.sh \
    && useradd --create-home --uid 10001 lassoare \
    && mkdir -p /data \
    && chown -R lassoare:lassoare /data /opt/lassoare
EXPOSE 15114
VOLUME ["/data"]
ENTRYPOINT ["/opt/lassoare/docker/entrypoint.sh"]

FROM application-base AS cpu
COPY --from=cpu-environment /opt/micromamba/ /opt/micromamba/
ENV LASSOARE_PROFILE=cpu
USER lassoare
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
  CMD /opt/micromamba/envs/lassoare_main/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:15114/api/health', timeout=5)"

FROM application-base AS cuda
COPY --from=cuda-environment /opt/micromamba/ /opt/micromamba/
ENV LASSOARE_PROFILE=cuda
USER lassoare
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
  CMD /opt/micromamba/envs/lassoare_main/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:15114/api/health', timeout=5)"
