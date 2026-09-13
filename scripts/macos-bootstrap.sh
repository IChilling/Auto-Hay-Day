#!/bin/bash
# Self-extracting macOS installer; the build script appends the project below.
# Run: /bin/bash "Setup Hay Day macOS.command"
set -euo pipefail
umask 077

if [ "${1:-}" = "--help" ]; then
    echo 'Usage: /bin/bash "Setup Hay Day macOS.command" [--no-launch]'
    echo 'Requires Apple Silicon, macOS 13+, 8 GB RAM, internet, and 16 GB free space.'
    echo 'Installs the included project, private Python, libraries, ADB, and BlueStacks Air.'
    echo 'BlueStacks installation may ask for your Mac administrator password.'
    echo 'Game sign-in and enabling Android debugging are interactive first-run steps.'
    exit 0
fi
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--no-launch" ]; }; then
    echo 'Unknown option. Use --help.' >&2
    exit 2
fi
if [ "$(uname -s)" != Darwin ]; then
    echo 'This installer must run on macOS.' >&2
    exit 1
fi
task_arm="$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null)" || task_arm=0
if [ "$task_arm" != 1 ]; then
    echo 'BlueStacks Air requires an Apple Silicon Mac (M-series). Intel Macs are unsupported.' >&2
    exit 1
fi
# Run natively even when Terminal was started under Rosetta.
if [ "$(uname -m)" != arm64 ]; then
    exec /usr/bin/arch -arm64 /bin/bash "$0" "$@"
fi
if [ "$(/usr/bin/sw_vers -productVersion | /usr/bin/cut -d. -f1)" -lt 13 ]; then
    echo 'macOS 13 or newer is required by the project image-processing library.' >&2
    exit 1
fi
if [ "$EUID" -eq 0 ]; then
    echo 'Run this as your normal Mac user, without sudo. Setup requests admin access when needed.' >&2
    exit 1
fi

task_root="$HOME/Library/Application Support/HayDayAutomation"
mkdir -p "$task_root/setup-logs"
task_log="$task_root/setup-logs/setup-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(/usr/bin/tee -a "$task_log") 2>&1
task_temp="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/hayday-setup.XXXXXXXX")"
finish() {
    task_status=$?
    trap - EXIT
    /bin/rm -rf -- "$task_temp"
    if [ "$task_status" -ne 0 ]; then
        echo "Setup failed. Details: $task_log"
        echo 'Correct the reported problem and run the same setup file again.'
        if [ -t 0 ]; then read -r -p 'Press Return to close.' _ || true; fi
    fi
    exit "$task_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify() {
    local actual
    actual="$(/usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}')"
    [ "$actual" = "$2" ] || { echo "Checksum mismatch: $1"; return 1; }
}

echo 'Setting up Hay Day Automation for macOS...'
payload_line="$(/usr/bin/awk '/^__HAYDAY_PAYLOAD_BELOW__$/{print NR + 1; exit}' "$0")"
[ -n "$payload_line" ] || { echo 'The installer is missing its embedded project.'; exit 1; }
/usr/bin/tail -n +"$payload_line" "$0" | /usr/bin/base64 -D > "$task_temp/project.tar.gz"
verify "$task_temp/project.tar.gz" '__PAYLOAD_SHA256__'
mkdir "$task_temp/project"
/usr/bin/tar -xzf "$task_temp/project.tar.gz" -C "$task_temp/project"

echo 'Downloading the private Python installer...'
/usr/bin/curl --fail --location --show-error --retry 3 --connect-timeout 30 --max-time 600 \
    --proto '=https' --proto-redir '=https' \
    'https://github.com/astral-sh/uv/releases/download/0.12.13/uv-aarch64-apple-darwin.tar.gz' \
    --output "$task_temp/uv.tar.gz"
verify "$task_temp/uv.tar.gz" '7e6ddb9316acc00f2296c82ff4d99977870ee34b2f0ddcae9444d714db9364ed'
/usr/bin/tar -xzf "$task_temp/uv.tar.gz" -C "$task_temp"
task_uv="$task_temp/uv-aarch64-apple-darwin/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/python"
export UV_CACHE_DIR="$task_root/download-cache"
# Use one Python-selection policy. uv rejects --managed-python together with
# UV_PYTHON_PREFERENCE, even when both request managed Python.
unset UV_PYTHON_PREFERENCE UV_NO_MANAGED_PYTHON || true
export UV_MANAGED_PYTHON=1
export UV_NO_CONFIG=1
export UV_NO_PROGRESS=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX || true
echo 'Installing a private Python 3.12 runtime (no Homebrew or Xcode needed)...'
"$task_uv" python install 3.12 --no-bin
task_python="$("$task_uv" python find 3.12)"
"$task_python" -I "$task_temp/project/scripts/install_macos.py" \
    --root "$task_root" --uv "$task_uv" --log "$task_log" "$@"
exit 0
__HAYDAY_PAYLOAD_BELOW__
