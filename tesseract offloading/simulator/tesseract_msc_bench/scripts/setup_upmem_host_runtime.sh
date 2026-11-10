#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host/runtime/python3.10"
LIBRARY="${RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0"
PACKAGE_URL="https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/libpython3.10_3.10.12-1~22.04.18_amd64.deb"
PACKAGE_SHA256="cd9ffe066be6207dd4fcb25fe037222e90429b0923d1a5eb6e897afcbb151833"

if [[ -f "${LIBRARY}" ]]; then
    echo "UPMEM host Python ABI runtime ready: ${LIBRARY}"
    exit 0
fi
for command in curl dpkg-deb sha256sum; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing required command: ${command}" >&2
        exit 1
    fi
done

mkdir -p "${RUNTIME_ROOT}"
PACKAGE_FILE="$(mktemp --suffix=.deb)"
trap 'rm -f -- "${PACKAGE_FILE}"' EXIT
curl --fail --location --silent --show-error "${PACKAGE_URL}" --output "${PACKAGE_FILE}"
echo "${PACKAGE_SHA256}  ${PACKAGE_FILE}" | sha256sum --check --status
dpkg-deb --extract "${PACKAGE_FILE}" "${RUNTIME_ROOT}"

if [[ ! -f "${LIBRARY}" ]]; then
    echo "libpython3.10 was not found after package extraction" >&2
    exit 1
fi
echo "UPMEM host Python ABI runtime ready: ${LIBRARY}"
