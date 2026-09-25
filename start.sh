#!/usr/bin/env bash
# ============================================================================
# start.sh — GLM-5.3-Flash W4A16 on 4 CUDA GPUs (sm_80 Ampere patches)
# ============================================================================
#
# We serve canada-quant/GLM-5.3-Flash-W4A16-MTP on four GPUs with a patched
# vLLM: OpenAI API on 127.0.0.1:8000 as "glm-5.3-flash" (local only unless you
# set HOST; no API key unless you set API_KEY), tensor-parallel 4, DFlash2
# speculation at k=3, 262,144-token context. No FP8, no KV quantisation, no
# offload.
#
# Every step is idempotent and skipped when it is already done, so running
# ./start.sh twice is safe and the second run just launches.
#
# What we do:
#   1. preflight — uv, git, python, 4 GPUs of >=60 GiB (read only), disk, port
#   2. install   — venv + the pinned vLLM fork, if missing or the pin moved
#   3. download  — the two checkpoints, if missing
#   4. launch    — detached, PID in logs/, output to logs/serve.log
#   5. wait      — poll /health up to READY_TIMEOUT, then print KV + model id
#
# Usage:
#   ./start.sh                 preflight, install, download, launch — default
#   ./start.sh install         build the venv and the fork only
#   ./start.sh download        fetch the two checkpoints only
#   ./start.sh stop            stop the server this checkout started
#   ./start.sh restart         stop + start
#   ./start.sh status          process + /health + KV cache line
#   ./start.sh logs            follow logs/serve.log
#   ./start.sh update          git pull, reinstall if the pin moved, restart
#   ./start.sh smoke           one chat request and one tool call
#   ./start.sh help            this text
#
# Config lives in .env, copied from .env.example on first run. A prefix env
# assignment beats .env for every key:
#
#   MAX_LEN=131072 ./start.sh restart
#   VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
#   VLLM_COMMIT=<older sha> ./start.sh update      # roll back, this run only
#
# To stay rolled back, set VLLM_COMMIT=<sha> in .env; otherwise the next
# plain start or restart reinstalls the pinned engine.
#
# DRY=1 ./start.sh runs the preflight, says whether it would install or
# download, and prints the server environment and launch command. It installs,
# downloads and launches nothing, and works before the first install. With
# stop, restart or update, DRY=1 also takes the checkout's lock and says what
# it would stop, without stopping, pulling or installing anything.
#
# Lifecycle commands on this checkout are serialised by a flock on
# logs/lifecycle.lock. start/restart/install/download refuse immediately when
# another one holds it; stop waits LOCK_WAIT seconds and then exits 1 WITHOUT
# stopping anything. The lock is per checkout: another clone is not covered.
# A lock left held by a server from an earlier release is recognised and
# cleared, never by signalling anything.
#
# stop only ever signals the PID in logs/vllm.pid, and only after confirming
# that process is the server this checkout launched. It never searches by
# process name, so it cannot touch an unrelated vLLM on the same machine.
# ============================================================================
set -euo pipefail

_say() { printf '\033[%sm%s\033[0m %s\n' "$1" "$2" "${*:3}"; }
log()  { _say '1;36' 'glm53 ·' "$@"; }
warn() { _say '1;33' 'glm53 !' "$@" >&2; }
die()  { _say '1;31' 'glm53 ✗' "$@" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

usage() {
    sed -n '3,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e 's/^# \{0,1\}//'
}

# ---------------------------- .env handling --------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || die "missing .env.example"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    log "wrote .env from .env.example — edit it, or override any key inline"
fi
# .env supplies defaults for keys the caller has not already set. It is parsed
# as KEY=VALUE, never sourced, so a config file cannot run code and no shell
# expansion happens inside a value. A key already present in the environment is
# left alone even when it is explicitly empty, which is what makes
# `KEY= ./start.sh` mean "unset this knob" rather than "use the .env value".
#
# Values: a value in matching quotes ("..." or '...') is taken exactly as
# written between them; a note after the closing quote is ignored. An unquoted
# value ends at the first whitespace-then-# (`KEY=v  # note` gives `v`), then a
# trailing note in parentheses after two or more spaces is dropped
# (`KEY=v   (default: x)` gives `v`), then surrounding whitespace is trimmed.
# A # or ( with no whitespace before it (`a#b`, `f(x)`) or after a single
# space (`a (b)`) stays in the value; quote the value to keep anything else.
load_env_defaults() {
    local file="$1" line key val raw
    local re_dq='^"([^"]*)"[[:space:]]*(#.*|\(.*\))?$'
    local re_sq="^'([^']*)'[[:space:]]*(#.*|\\(.*\\))?\$"
    local re_paren='^(.*)[[:space:]][[:space:]]+\([^()]*\)[[:space:]]*$'
    [ -r "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"          # ltrim
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in export\ *) line="${line#export }" ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"             # rtrim
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -n "${!key+x}" ] && continue                   # caller already set it
        raw="$val"
        val="${val#"${val%%[![:space:]]*}"}"             # ltrim
        if [[ $val =~ $re_dq || $val =~ $re_sq ]]; then  # quoted: verbatim
            val="${BASH_REMATCH[1]}"
        else
            case "$val" in                               # quotes around a value
                \"*\") val="${val#\"}"; val="${val%\"}" ;;   # that holds quotes
                \'*\') val="${val#\'}"; val="${val%\'}" ;;
                *)                                       # unquoted: drop notes
                    val="${raw%%[[:space:]]#*}"          # (KEY= # x is empty)
                    [[ $val =~ $re_paren ]] && val="${BASH_REMATCH[1]}"
                    val="${val#"${val%%[![:space:]]*}"}"
                    val="${val%"${val##*[![:space:]]}"}" ;;      # trim
            esac
        fi
        export "$key=$val"
    done <"$file"
}
load_env_defaults "$SCRIPT_DIR/.env"

# ---------------------------- configuration --------------------------------
VENV="${VENV:-$SCRIPT_DIR/venv}"
VLLM_SRC="${VLLM_SRC:-$SCRIPT_DIR/vllm-src}"
VLLM_REPO="${VLLM_REPO:-https://github.com/Morrowmake/vllm-cmp170hx.git}"
VLLM_BRANCH="${VLLM_BRANCH:-ampere-glm53}"
VLLM_COMMIT="${VLLM_COMMIT:-3bbb99a534}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
TARGET_REPO="${TARGET_REPO:-canada-quant/GLM-5.3-Flash-W4A16-MTP}"
DRAFTER_REPO="${DRAFTER_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
MODEL="${MODEL:-$MODELS_DIR/${TARGET_REPO##*/}}"
DFLASH_MODEL="${DFLASH_MODEL:-$MODELS_DIR/${DRAFTER_REPO##*/}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"
MAX_JOBS="${MAX_JOBS:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# SERVED_NAME is the older spelling; SERVED_MODEL_NAME wins when both are set.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${SERVED_NAME:-glm-5.3-flash}}"
PP="${PP:-1}"; TP="${TP:-4}"
export PP TP
# PP is unsupported by this release (a pipeline-parallel layout is planned for
# a later one), but if someone sets it anyway, default to the MTP head: DFlash
# under pipeline parallelism needs engine changes this pin does not carry.
if [ "$PP" -gt 1 ] && [ "$TP" = "1" ]; then
    SPEC_MODE="${SPEC_MODE:-mtp}"
else
    SPEC_MODE="${SPEC_MODE:-dflash}"
fi
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
MIN_GPUS="${MIN_GPUS:-4}"
MIN_GPU_MIB="${MIN_GPU_MIB:-61440}"      # 60 GiB; the cards we run report 65,536 MiB
NEED_DISK_GB="${NEED_DISK_GB:-185}"      # GiB, as df -BG counts it: checkpoints ~180 GiB
LOCK_WAIT="${LOCK_WAIT:-30}"

LOGDIR="$SCRIPT_DIR/logs"
SERVE_LOG="${SERVE_LOG:-$LOGDIR/serve.log}"
PIDFILE="$LOGDIR/vllm.pid"
LOCKFILE="$LOGDIR/lifecycle.lock"
LOCKPID="$LOGDIR/lifecycle.lock.pid"
STAMP="$VENV/.recipe-stamp"

# Where our own health checks connect. A wildcard bind address is not a
# destination, so talk to the server over loopback in that case.
case "$HOST" in
    0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;;
    *) CLIENT_HOST="$HOST" ;;
esac

export VENV VLLM_SRC MODEL DFLASH_MODEL CUDA_HOME HOST PORT SERVED_MODEL_NAME
export SERVE_LOG READY_TIMEOUT
export MODEL_ID="${MODEL_ID:-$SERVED_MODEL_NAME}"

banner() {
    printf '\n  \033[1;36m┌──────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3-Flash W4A16\033[0m  \033[2m·  %-14s\033[0m  \033[1;36m│\033[0m\n' "${1:-start.sh}"
    printf '  \033[1;36m└──────────────────────────────────────────────┘\033[0m\n\n'
}

# ------------------------------- locking -----------------------------------
# The flock is the authoritative owner; $LOCKPID is advisory diagnostics only
# and is never signalled.
#
# A server launched by an earlier release inherited the lock's descriptor and
# held the lock for its whole life. When the lock is busy we therefore look at
# who holds it (read only, via /proc, our own processes only). If every holder
# belongs to the server in $PIDFILE (that pid, or its process group/session
# after setsid) and none is a start.sh, the lock is stale: we unlink the file
# and lock a fresh one. The old server keeps its descriptor on the unlinked
# file, which no longer matters. Nothing is ever signalled here; stop, restart
# and update still stop the server through do_stop. The clearing itself is
# serialised by a short-lived second lock, so two commands cannot both clear.

# PIDs (ours to read) with an open descriptor on $LOCKFILE, one per line.
lock_holders() {
    local f p
    for f in /proc/[0-9]*/fd/[0-9]*; do
        [ "$f" -ef "$LOCKFILE" ] || continue
        p="${f#/proc/}"
        echo "${p%%/*}"
    done | sort -u
}

# True when pid $1 is the server pid $2 or runs inside its group/session.
holder_is_server() {
    local h="$1" s="$2" st cl
    cl="$(tr '\0' ' ' <"/proc/$h/cmdline" 2>/dev/null || true)"
    case "$cl" in *start.sh*) return 1 ;; esac
    [ "$h" = "$s" ] && return 0
    read -r st 2>/dev/null <"/proc/$h/stat" || return 1
    st="${st##*) }"                        # fields after "pid (comm) "
    set -- $st                             # state ppid pgrp session ...
    [ "${3:-}" = "$s" ] || [ "${4:-}" = "$s" ]
}

# True when the lock is held, and held only by this checkout's server.
lock_held_by_server_only() {
    local s h holders
    s="$(read_pid)"
    pid_is_ours "$s" || return 1
    holders="$(lock_holders)"
    [ -n "$holders" ] || return 1
    for h in $holders; do
        holder_is_server "$h" "$s" || return 1
    done
    return 0
}

# True when fd 9 is still the file at $LOCKFILE (not one unlinked meanwhile).
lock_is_current() {
    [ -e "/proc/$$/fd/9" ] || return 0
    [ "$LOCKFILE" -ef "/proc/$$/fd/9" ]
}

# Lock $LOCKFILE on fd 9. Returns 0 when locked, 1 when a real command holds
# it. Clears a stale lock (see above) on the way.
acquire_lock() {
    local tries=0
    mkdir -p "$LOGDIR"
    while [ "$tries" -lt 5 ]; do
        tries=$((tries + 1))
        exec 9>"$LOCKFILE"
        if flock -n 9; then
            # Make sure the path still names the file we locked; if another
            # command replaced it meanwhile, go round again.
            lock_is_current && return 0
            exec 9>&-
            continue
        fi
        exec 9>&-
        exec 8>"$LOCKFILE.gate"
        flock -w 10 8 || { exec 8>&-; return 1; }
        exec 9>"$LOCKFILE"
        if flock -n 9; then                  # freed or cleared meanwhile
            exec 8>&-
            lock_is_current && return 0
            exec 9>&-
            continue
        fi
        exec 9>&-
        if lock_held_by_server_only; then
            log "an earlier server was holding the lock; continuing"
            rm -f "$LOCKFILE"
            exec 8>&-
            continue
        fi
        exec 8>&-
        return 1
    done
    return 1
}

take_lock() {
    if ! acquire_lock; then
        local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
        die "this checkout is busy${holder:+ with pid $holder}. Wait for that command to finish, then try again."
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}
take_lock_for_stop() {
    if ! acquire_lock; then
        exec 9>"$LOCKFILE"
        if ! flock -w "$LOCK_WAIT" 9 || ! lock_is_current; then
            local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
            warn "gave up waiting ${LOCK_WAIT}s for this checkout${holder:+ (pid $holder has it)}."
            die "The server was left running. Try stop again once that command finishes."
        fi
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}

# ------------------------------ preflight ----------------------------------
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 not found — $2"
}

# preflight [--with-port]   -- the port is only checked when we are about to
# bind it, so `install` and `download` do not fail on a busy port.
preflight() {
    local check_port=0
    [ "${1:-}" = "--with-port" ] && check_port=1
    log "preflight"
    need_cmd uv   "install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    need_cmd git  "install it: apt-get install -y git"
    need_cmd curl "install it: apt-get install -y curl"
    command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python ${PYTHON_VERSION}"

    if command -v nvidia-smi >/dev/null 2>&1; then
        # Read only. We never set power, persistence, clocks or fan state.
        local mems n small
        mems="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -z "$mems" ]; then
            warn "nvidia-smi present but returned no GPUs — check the driver"
        else
            n="$(printf '%s\n' "$mems" | grep -c .)"
            small="$(printf '%s\n' "$mems" | awk -v m="$MIN_GPU_MIB" '$1+0 < m' | grep -c . || true)"
            [ "$n" -ge "$MIN_GPUS" ] || die "found $n GPU(s), need $MIN_GPUS — lower TP, or set MIN_GPUS to override this check"
            if [ "${small:-0}" -gt 0 ]; then
                die "$small GPU(s) report under $((MIN_GPU_MIB/1024)) GiB. This recipe needs about 64 GB on each card (the W4A16 weights take ~45 GiB per card at TP=4, plus the KV cache). Nothing was downloaded. Set MIN_GPU_MIB to override this check."
            fi
            log "  GPUs: $n visible, smallest $(printf '%s\n' "$mems" | sort -n | head -1) MiB"
        fi
        # Read only. Link width decides whether TP=4 is the right layout.
        local widths narrow
        widths="$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -n "$widths" ]; then
            log "  PCIe link width: $(printf '%s' "$widths" | tr -d ' ' | paste -sd, -)"
            narrow="$(printf '%s\n' "$widths" | awk '$1+0 > 0 && $1+0 < 16' | grep -c . || true)"
            if [ "${narrow:-0}" -gt 0 ] && [ "$TP" -gt 1 ]; then
                warn "  $narrow card(s) are below x16 and TP=$TP. Tensor parallelism moves ~9.4 MB per"
                warn "  layer during prefill and ~100 small collectives per decode step, so TP will be"
                warn "  far slower than the published numbers on narrow links. This release is TP-only;"
                warn "  see \"Link width\" in the README. Continuing anyway."
            fi
        fi
    else
        warn "nvidia-smi not found — skipping the GPU check"
    fi

    [ -d "$CUDA_HOME" ] || warn "no CUDA toolkit at $CUDA_HOME — Triton and TileLang want one at runtime (set CUDA_HOME)"

    mkdir -p "$MODELS_DIR"
    local avail_gb; avail_gb="$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
    if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -gt 0 ]; then
        if [ "$avail_gb" -lt "$NEED_DISK_GB" ] && ! models_present; then
            die "only ${avail_gb} GiB free at $MODELS_DIR, need ~${NEED_DISK_GB} GiB for the checkpoints — free space or set MODELS_DIR"
        fi
        log "  disk: ${avail_gb} GiB free at $MODELS_DIR"
    fi

    if [ "$check_port" = 1 ]; then
        if ! port_busy; then
            log "  port $PORT: free"
        elif server_is_ours; then
            log "  port $PORT: in use by our server"
        elif [ -n "${DRY:-}" ]; then
            warn "  port $PORT is in use by something else (not fatal under DRY)"
        else
            die "port $PORT is already taken by another process. Set PORT to a free one, or stop whatever holds it."
        fi
    fi
}

port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$PORT" 2>/dev/null | tail -n +2 | grep -q .
    else
        curl -fsS -m 2 "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1
    fi
}

# ------------------------------- install -----------------------------------
install_done() {
    [ -x "$VENV/bin/vllm" ] || return 1
    [ -f "$STAMP" ] || return 1
    [ -d "$VLLM_SRC/.git" ] || return 1
    local want have
    want="$(git -C "$VLLM_SRC" rev-parse "$VLLM_COMMIT" 2>/dev/null || echo "?")"
    have="$(cat "$STAMP" 2>/dev/null || echo "??")"
    [ "$want" = "$have" ] || return 1
    [ "$(git -C "$VLLM_SRC" rev-parse HEAD 2>/dev/null || echo '???')" = "$want" ] || return 1
    return 0
}

do_install() {
    if install_done && [ "${FORCE_INSTALL:-0}" != "1" ]; then
        log "install: already at $(cut -c1-10 <"$STAMP") — skipping (FORCE_INSTALL=1 to redo)"
        return 0
    fi
    log "install: venv + vLLM fork @ $VLLM_COMMIT"
    export PATH="$CUDA_HOME/bin:$PATH"

    if [ ! -d "$VLLM_SRC/.git" ]; then
        # Blobless, but NOT shallow: vLLM's precompiled-wheel resolution runs
        # `git merge-base` against upstream main, which needs real history.
        git clone --filter=blob:none --branch "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_SRC"
    else
        git -C "$VLLM_SRC" fetch origin "$VLLM_BRANCH"
    fi
    # Stay on a named branch rather than a detached HEAD, for the same reason
    # (`git branch --show-current` has to return something).
    git -C "$VLLM_SRC" checkout -B "$VLLM_BRANCH" "$VLLM_COMMIT"
    git -C "$VLLM_SRC" submodule update --init --recursive --depth 1
    log "  HEAD: $(git -C "$VLLM_SRC" log --oneline -1)"

    if [ -x "$VENV/bin/python" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
        log "  reusing venv at $VENV (VENV_CLEAR=1 to rebuild)"
    else
        uv venv --clear --python "$PYTHON_VERSION" "$VENV"
    fi
    export VIRTUAL_ENV="$VENV"

    local pip_args=(--extra-index-url https://flashinfer.ai/whl/)
    [ -n "${TORCH_INDEX_URL:-}" ] && pip_args+=(--extra-index-url "$TORCH_INDEX_URL")

    # torch 2.13.0 from PyPI is the cu130 build on linux-x86_64 (torch.version.cuda
    # == '13.0'). Set TORCH_INDEX_URL if your platform resolves differently.
    log "  torch 2.13.0 (cu130)"
    uv pip install "${pip_args[@]}" "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"

    # Every patch on ampere-glm53 is Python, Triton or TileLang: the diff against
    # upstream touches no .cu, .cpp or CMakeLists, so upstream's precompiled
    # extensions are the right ones and already carry sm_80 cubins.
    if [ "$BUILD_FROM_SOURCE" = "1" ]; then
        log "  vLLM (compiling extensions, MAX_JOBS=$MAX_JOBS)"
        [ -d "$CUDA_HOME" ] || die "BUILD_FROM_SOURCE=1 needs a CUDA toolkit at $CUDA_HOME"
        env -u VLLM_USE_PRECOMPILED \
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
            CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 \
            VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
            uv pip install "${pip_args[@]}" --no-build-isolation -e "$VLLM_SRC"
    else
        log "  vLLM (upstream precompiled extensions)"
        VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
            uv pip install "${pip_args[@]}" -e "$VLLM_SRC"
    fi

    log "  runtime extras"
    uv pip install "${pip_args[@]}" \
        "flashinfer-python==0.6.18.post1" "flashinfer-cubin==0.6.18.post1" \
        "tilelang==0.1.12" "apache-tvm-ffi==0.1.11" "ninja" "huggingface_hub[hf_xet]>=1.0"

    log "  verifying"
    "$VENV/bin/python" - <<'PY'
import importlib, sys, torch
print(f"    python      {sys.version.split()[0]}")
print(f"    torch       {torch.__version__} (cuda {torch.version.cuda})")
import vllm
print(f"    vllm        {vllm.__version__}")
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops  # noqa: F401
print("    vllm C ext  ok")
for mod in ("triton", "tilelang", "flashinfer"):
    m = importlib.import_module(mod)
    print(f"    {mod:<11} {getattr(m, '__version__', 'ok')}")
from vllm.v1.attention.backends.mla import triton_mla_sparse            # noqa: F401
from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3        # noqa: F401
from vllm.v1.worker.gpu import prologue_fuse                            # noqa: F401
from vllm.distributed.device_communicators import host_shm_all_reduce   # noqa: F401
print("    sm_80 patch modules ok")
PY
    git -C "$VLLM_SRC" rev-parse HEAD >"$STAMP"
    log "install: done"
}

# ------------------------------ download -----------------------------------
model_present() { [ -f "$1/config.json" ]; }
models_present() {
    model_present "$MODEL" || return 1
    [ "$SPEC_MODE" = "dflash" ] || return 0
    model_present "$DFLASH_MODEL"
}

do_download() {
    local hf="$VENV/bin/hf"
    [ -x "$hf" ] || hf="$(command -v hf || true)"
    [ -n "$hf" ] || die "no 'hf' CLI — run ./start.sh install first"
    mkdir -p "$MODELS_DIR"

    if model_present "$MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$MODEL") already present — skipping"
    else
        log "download: $TARGET_REPO (~178 GiB / 191 GB, 22 files)"
        "$hf" download "$TARGET_REPO" --local-dir "$MODEL"
    fi

    if [ "$SPEC_MODE" != "dflash" ]; then
        log "download: SPEC_MODE=$SPEC_MODE — drafter not needed"
    elif model_present "$DFLASH_MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$DFLASH_MODEL") already present — skipping"
    else
        log "download: $DRAFTER_REPO (~2.2 GiB / 2.3 GB, 5 files)"
        "$hf" download "$DRAFTER_REPO" --local-dir "$DFLASH_MODEL"
    fi
    log "download: done"
}

# ------------------------------- process -----------------------------------
read_pid() { [ -f "$PIDFILE" ] && tr -dc '0-9' <"$PIDFILE" || true; }

# True only when $1 is alive AND is the server this checkout launched. This is
# the whole safety story for stop: we never match by process name, so another
# vLLM on this machine is invisible to us.
pid_is_ours() {
    local pid="${1:-}"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    local cl; cl="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    [ -n "$cl" ] || return 1
    case "$cl" in
        *"$MODEL"*) : ;;
        *) return 1 ;;
    esac
    case "$cl" in
        *vllm*) return 0 ;;
        *) return 1 ;;
    esac
}

server_is_ours() { pid_is_ours "$(read_pid)"; }

health_ok() { curl -fsS -m 5 "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1; }

# First "id" in /v1/models. A plain greedy sed would pick up the permission id
# further down the same line, so match the field and take the first one.
# /v1 needs the key when API_KEY is set; /health never does.
served_id() {
    local auth=()
    [ -n "${API_KEY:-}" ] && auth=(-H "Authorization: Bearer $API_KEY")
    curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://$CLIENT_HOST:$PORT/v1/models" 2>/dev/null \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4
}

kv_line() {
    [ -r "$SERVE_LOG" ] || return 1
    local l; l="$(grep -h 'GPU KV cache size' "$SERVE_LOG" 2>/dev/null | tail -1 || true)"
    [ -n "$l" ] || return 1
    printf '%s\n' "${l#*] }"
}

# -------------------------------- launch -----------------------------------
launch() {
    if [ -n "${DRY:-}" ]; then
        DRY=1 "$SCRIPT_DIR/serve.sh" "$SPEC_MODE"
        return 0
    fi
    if server_is_ours; then
        log "launch: already running (pid $(read_pid)) — skipping"
        return 0
    fi
    if [ -f "$PIDFILE" ]; then
        log "launch: clearing a stale pid file"
        rm -f "$PIDFILE"
    fi
    mkdir -p "$LOGDIR"
    log "launch: $SPEC_MODE, TP=${TP:-4} PP=${PP:-1}, ctx ${MAX_LEN:-262144}, :$PORT"
    log "  log: $SERVE_LOG"
    # 9>&- : the server must not inherit the lifecycle lock (fd 9). serve.sh
    # execs vllm, so an inherited fd would hold the lock for the server's whole
    # life and block every later stop, restart and update.
    setsid nohup "$SCRIPT_DIR/serve.sh" "$SPEC_MODE" 9>&- >>"$SERVE_LOG" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" >"$PIDFILE"
    log "  pid: $pid"
}

wait_ready() {
    [ -z "${DRY:-}" ] || return 0
    local url="http://$CLIENT_HOST:$PORT/health" elapsed=0 pid
    pid="$(read_pid)"
    log "waiting for $url (weight load and graph capture on a 320B MoE are slow; timeout ${READY_TIMEOUT}s)"
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if health_ok; then
            log "healthy after ${elapsed}s"
            local kv; kv="$(kv_line || true)"
            [ -n "$kv" ] && log "  $kv"
            log "  model: $(served_id)"
            log "  API:   http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            warn "the server exited after ${elapsed}s — last 40 lines:"
            tail -40 "$SERVE_LOG" >&2 || true
            rm -f "$PIDFILE"
            return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    warn "not healthy after ${READY_TIMEOUT}s; it may still be loading — ./start.sh logs"
    return 1
}

# --------------------------------- stop ------------------------------------
do_stop() {
    local pid; pid="$(read_pid)"
    if [ -z "$pid" ]; then
        log "stop: nothing started from this checkout is running"
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        log "stop: pid $pid is gone — clearing a stale pid file"
        rm -f "$PIDFILE"
        return 0
    fi
    if ! pid_is_ours "$pid"; then
        warn "stop: pid $pid is alive but is not our server — leaving it alone and clearing the pid file"
        rm -f "$PIDFILE"
        return 0
    fi
    log "stop: SIGINT to pid $pid (process group)"
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$STOP_TIMEOUT" ]; do
        sleep 2; waited=$((waited + 2))
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "stop: still alive after ${STOP_TIMEOUT}s — SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PIDFILE"
    log "stop: done (${waited}s)"
}

# -------------------------------- status -----------------------------------
do_status() {
    local pid; pid="$(read_pid)"
    if [ -z "$pid" ]; then
        log "process: no pid file — not started from this checkout"
    elif pid_is_ours "$pid"; then
        log "process: running, pid $pid"
    elif kill -0 "$pid" 2>/dev/null; then
        log "process: pid $pid is alive but is not our server (stale pid file)"
    else
        log "process: pid $pid is gone (stale pid file)"
    fi
    if health_ok; then
        log "API:     healthy — http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
        log "model:   $(served_id)"
    else
        log "API:     not responding on :$PORT"
    fi
    local kv; kv="$(kv_line || true)"
    if [ -n "$kv" ]; then log "KV:      $kv"; else log "KV:      (no KV line in $SERVE_LOG)"; fi
    log "install: $(install_done && echo "ok, $VLLM_COMMIT" || echo 'missing or pin moved — ./start.sh install')"
    log "models:  $(models_present && echo ok || echo 'missing — ./start.sh download')"
}

# --------------------------------- logs ------------------------------------
do_logs() {
    [ -f "$SERVE_LOG" ] || die "no log at $SERVE_LOG — nothing has been launched yet"
    log "following $SERVE_LOG"
    trap '' INT
    tail -n 100 -f "$SERVE_LOG" || true
    trap - INT
}

# -------------------------------- update -----------------------------------
do_update() {
    if [ -d "$SCRIPT_DIR/.git" ]; then
        local before after
        before="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo none)"
        log "update: git pull"
        git -C "$SCRIPT_DIR" pull --ff-only
        after="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
        if [ "$before" = "$after" ]; then
            log "  already up to date"
        else
            log "  ${before:0:10} -> ${after:0:10}"
            # The pull may have moved the pin, which this shell read before the
            # pull. Re-exec the new start.sh so we act on the new value. exec
            # keeps our PID, so fd 9 keeps holding the lifecycle lock.
            if [ -z "${_GLM53_REEXEC:-}" ]; then
                log "  re-executing the updated start.sh"
                export _GLM53_REEXEC=1
                exec "$SCRIPT_DIR/start.sh" update
            fi
        fi
    else
        warn "update: not a git checkout — skipping the pull"
    fi
    if install_done; then
        log "update: pin unchanged at $VLLM_COMMIT — no reinstall"
    else
        log "update: pin moved to $VLLM_COMMIT — reinstalling"
        do_install
    fi
    do_stop
    start_unlocked
}

# --------------------------------- start -----------------------------------
# DRY: report what a real start would do, then print the server environment
# and command. Nothing is installed, downloaded or launched.
dry_start() {
    preflight --with-port
    if install_done; then
        log "install: already at $VLLM_COMMIT — a real start would skip it"
    else
        log "install: a real start would build $VENV and the vLLM fork @ $VLLM_COMMIT"
    fi
    if models_present; then
        log "download: checkpoints present — a real start would skip it"
    else
        local what="$TARGET_REPO"
        [ "$SPEC_MODE" = "dflash" ] && what="$what and $DRAFTER_REPO"
        log "download: a real start would fetch $what into $MODELS_DIR"
    fi
    launch
}

# DRY stop/update: the lock is taken for real (that is what they check), and
# nothing is signalled, pulled or installed.
dry_stop() {
    local pid; pid="$(read_pid)"
    if [ -z "$pid" ]; then
        log "stop: DRY=1, nothing started from this checkout is running"
    elif pid_is_ours "$pid"; then
        log "stop: DRY=1, a real stop would send SIGINT to pid $pid (process group)"
    else
        log "stop: DRY=1, pid $pid is not our server; a real stop would only clear the pid file"
    fi
}
dry_update() {
    log "update: DRY=1, a real update would git pull --ff-only, reinstall if the pin moved, then restart"
    dry_stop
    dry_start
}

start_unlocked() {
    if [ -n "${DRY:-}" ]; then dry_start; return 0; fi
    preflight --with-port
    do_install
    if models_present; then
        log "download: checkpoints present — skipping"
    else
        do_download
    fi
    launch
    [ -n "${DRY:-}" ] || wait_ready
}

# ---------------------------------- main -----------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        -h|--help|help) usage; exit 0 ;;
    esac
    banner "$cmd"
    case "$cmd" in
        start)    if [ -n "${DRY:-}" ]; then dry_start; else take_lock; start_unlocked; fi ;;
        install)  take_lock; preflight; do_install ;;
        download) take_lock; preflight; do_download ;;
        stop)     take_lock_for_stop; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi ;;
        restart)  take_lock; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi; start_unlocked ;;
        update)   take_lock; if [ -n "${DRY:-}" ]; then dry_update; else do_update; fi ;;
        status)   do_status ;;
        logs)     do_logs ;;
        smoke)    HOST="$CLIENT_HOST" exec "$SCRIPT_DIR/smoke.sh" ;;
        *) warn "unknown command: $cmd"; usage; exit 1 ;;
    esac
}

main "$@"
