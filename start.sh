#!/usr/bin/env bash
# ============================================================================
# start.sh — GLM-5.3-Flash W4A16 on 4 CUDA GPUs (sm_80 Ampere patches)
# ============================================================================
#
# We serve canada-quant/GLM-5.3-Flash-W4A16-MTP on four GPUs with a patched
# vLLM: OpenAI API on 127.0.0.1:8000 as "glm-5.3-flash" (local only unless you
# set HOST; no API key unless you set API_KEY), DFlash2 speculation at k=3,
# 262,144-token context, in one of two layouts: tensor-parallel 4 (LAYOUT=tp4,
# the default) or pipeline-parallel 4 (LAYOUT=pp4). No FP8, no KV
# quantisation, no offload.
#
# The engine runs in a container: Docker with the NVIDIA Container Toolkit,
# image pinned by digest. A checkout that ran the engine natively under an
# earlier release moves to the container on `./start.sh update` (or restart):
# its native server is stopped, .env and the checkpoints are kept, and venv/
# and vllm-src/ are left on disk for you to delete.
#
# Every step is idempotent and skipped when it is already done, so running
# ./start.sh twice is safe and the second run just launches.
#
# What we do:
#   1. preflight — docker, git, 4 GPUs of >=60 GiB (read only), link width,
#                  disk, port
#   2. install   — the pinned engine image, if missing or the pin moved
#   3. download  — the two checkpoints, if missing or incomplete (resumed)
#   4. launch    — detached, output to logs/serve.log
#   5. wait      — poll /health up to READY_TIMEOUT, then print KV + model id
#
# Usage:
#   ./start.sh                 preflight, install, download, launch — default
#   ./start.sh install         pull the engine image only
#   ./start.sh download        fetch the two checkpoints only
#   ./start.sh stop            stop the server this checkout started
#   ./start.sh restart         stop + start
#   ./start.sh status          server + /health + KV cache line
#   ./start.sh logs            follow logs/serve.log
#   ./start.sh update          git pull, reinstall if the pin moved, restart
#   ./start.sh smoke           one chat request and one tool call
#   ./start.sh help            this text
#
# The image includes the optional compiled Marlin library. Runtime switch
# VLLM_AMPERE_MARLIN_DECODE_CUDA=1 enables compiled decode; unset, decode is on
# in both layouts.
#
# Config lives in .env, copied from .env.example on first run. A prefix env
# assignment beats .env for every key:
#
#   LAYOUT=pp4 ./start.sh restart
#   MAX_LEN=131072 ./start.sh restart
#   VLLM_AMPERE_DECODE_KERNELS=0 ./start.sh restart
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
# stop only ever stops the container in logs/container.id (or a native server
# an earlier release started, by the PID in logs/vllm.pid), and only after
# confirming it is the server this checkout launched. It never searches by
# process or container name, so it cannot touch an unrelated vLLM on the same
# machine.
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
# This release's engine: the fork commit the container image is built from,
# and that image, pinned by digest (docker-compose.yml runs the same image).
# The image pin below is the exact image the release acceptance runs.
RELEASE_VLLM_COMMIT=819146023facb31c04f1f8512e52464835a10d40
RELEASE_IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:18a2f26207568128e1d792ce923e86a74e40def1e43e492db83132cab8cdbe14
IMAGE="${IMAGE:-$RELEASE_IMAGE}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
# Checkpoints, pinned to the Hub revisions this release was validated with.
TARGET_REPO="${TARGET_REPO:-canada-quant/GLM-5.3-Flash-W4A16-MTP}"
TARGET_REVISION="${TARGET_REVISION:-5681f7c2dc6436d63f82ab6fb892b4f5062b8304}"
DRAFTER_REPO="${DRAFTER_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
DRAFTER_REVISION="${DRAFTER_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
MODEL="${MODEL:-$MODELS_DIR/${TARGET_REPO##*/}}"
DFLASH_MODEL="${DFLASH_MODEL:-$MODELS_DIR/${DRAFTER_REPO##*/}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# SERVED_NAME is the older spelling; SERVED_MODEL_NAME wins when both are set.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${SERVED_NAME:-glm-5.3-flash}}"
# Layout: tp4 (tensor-parallel 4, the default) or pp4 (pipeline-parallel 4).
# serve.sh reads LAYOUT too; an explicit PP/TP still wins over it.
LAYOUT="${LAYOUT:-tp4}"
case "$LAYOUT" in
    tp4) PP="${PP:-1}"; TP="${TP:-4}" ;;
    pp4) PP="${PP:-4}"; TP="${TP:-1}" ;;
    *) die "LAYOUT must be tp4 or pp4, not '$LAYOUT'" ;;
esac
export LAYOUT PP TP
# DFlash2 in both layouts.
SPEC_MODE="${SPEC_MODE:-dflash}"
# A native install from an earlier release (venv and engine source). This
# release never runs, changes or deletes it; see native_notice.
NATIVE_VENV="${VENV:-$SCRIPT_DIR/venv}"
NATIVE_SRC="${VLLM_SRC:-$SCRIPT_DIR/vllm-src}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash}"
SHM_SIZE="${SHM_SIZE:-16g}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
BOOT_CHECK="${BOOT_CHECK:-1}"
case "$BOOT_CHECK" in 0|1) ;; *) die "BOOT_CHECK must be 0 or 1" ;; esac
LAUNCHED=0
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
NATIVE_NOTED="$LOGDIR/native-retired"
CIDFILE="$LOGDIR/container.id"
LABEL="glm53-recipe.checkout"

# Where our own health checks connect. A wildcard bind address is not a
# destination, so talk to the server over loopback in that case.
case "$HOST" in
    0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;;
    *) CLIENT_HOST="$HOST" ;;
esac

export MODEL DFLASH_MODEL HOST PORT SERVED_MODEL_NAME
export SERVE_LOG READY_TIMEOUT BOOT_CHECK
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
    log "preflight (layout: $LAYOUT)"
    need_cmd git  "install it: apt-get install -y git"
    need_cmd curl "install it: apt-get install -y curl"
    [ "$BOOT_CHECK" = 0 ] || need_cmd python3 "install Python 3 for the boot check"
    preflight_docker

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
                warn "  far slower than the published numbers on narrow links. The pipeline-parallel"
                warn "  layout (LAYOUT=pp4) needs far less link bandwidth; see \"Link width\" in"
                warn "  docs/how-to-use.md. Continuing anyway."
            fi
        fi
    else
        warn "nvidia-smi not found — skipping the GPU check"
    fi

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

# Docker must be installed and usable by this user, with the NVIDIA Container
# Toolkit for --gpus. Under DRY a missing or unusable docker is only a warning.
preflight_docker() {
    local how="install Docker Engine and the NVIDIA Container Toolkit (see \"One-time host setup\" in the README)"
    # A native server from an earlier release keeps running until the
    # container can take over.
    if native_server_running; then
        how="$how. This release runs the engine only in the container; the native server this checkout started (pid $(read_pid)) was left running"
    fi
    if ! command -v docker >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker not found — $how"; return 0; }
        die "docker not found — $how"
    fi
    if ! docker info >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker is installed but this user cannot reach the Docker daemon"; return 0; }
        die "docker is installed but this user cannot reach the Docker daemon (permission denied, or the daemon is not running). Fix that: $how."
    fi
    if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
        log "  docker: ok, NVIDIA runtime present"
    else
        warn "  docker: no 'nvidia' runtime registered — without the NVIDIA Container Toolkit, --gpus fails"
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
# Container: the image is the whole engine, so "installed" means the image
# named by IMAGE (a digest) is present locally. A release that moves the pin
# names a new image, which is pulled on the next start or update.
container_image_present() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

install_done() { container_image_present; }

do_install() {
    if container_image_present && [ "${FORCE_INSTALL:-0}" != "1" ]; then
        log "install: image present — skipping (FORCE_INSTALL=1 to pull again)"
        log "  $IMAGE"
        return 0
    fi
    log "install: docker pull (the engine image, several GB)"
    log "  $IMAGE"
    docker pull "$IMAGE"
    log "install: done"
}

# ------------------------------ download -----------------------------------
# A checkpoint is complete when docker/prepare.sh says so: the config, every
# shard its index names, and no partial download left behind. The same check
# decides for Docker Compose.
model_present() { bash "$SCRIPT_DIR/docker/prepare.sh" check "$1"; }
models_present() {
    model_present "$MODEL" || return 1
    [ "$SPEC_MODE" = "dflash" ] || return 0
    model_present "$DFLASH_MODEL"
}

# hf_download <repo> <revision> <local dir>: with the hf client in the engine
# image, run as this user so the files are yours. An interrupted download
# resumes.
hf_download() {
    local repo="$1" revision="$2" dir="$3"
    mkdir -p "$dir"
    container_image_present || die "the engine image is not here yet — run ./start.sh install first"
    docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
        ${HF_TOKEN:+-e HF_TOKEN} -v "$dir:/download" \
        -v "$SCRIPT_DIR/docker/prepare.sh:/recipe/prepare.sh:ro" \
        --entrypoint /bin/bash "$IMAGE" /recipe/prepare.sh fetch "$repo" /download "$revision"
}

do_download() {
    mkdir -p "$MODELS_DIR"

    if model_present "$MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$MODEL") complete — skipping"
    else
        log "download: $TARGET_REPO @ ${TARGET_REVISION:0:10} (~178 GiB / 191 GB, 22 files)"
        hf_download "$TARGET_REPO" "$TARGET_REVISION" "$MODEL"
    fi

    if [ "$SPEC_MODE" != "dflash" ]; then
        log "download: SPEC_MODE=$SPEC_MODE — drafter not needed"
    elif model_present "$DFLASH_MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$DFLASH_MODEL") complete — skipping"
    else
        log "download: $DRAFTER_REPO @ ${DRAFTER_REVISION:0:10} (~2.2 GiB / 2.3 GB, 5 files)"
        hf_download "$DRAFTER_REPO" "$DRAFTER_REVISION" "$DFLASH_MODEL"
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

server_is_ours() { container_is_ours; }

# A native server an earlier release of this checkout started (logs/vllm.pid).
native_server_running() { pid_is_ours "$(read_pid)"; }

# A native install an earlier release built in this checkout.
native_install_present() {
    [ -f "$NATIVE_VENV/.recipe-stamp" ] || [ -f "$NATIVE_VENV/.recipe-reqs" ] || [ -d "$NATIVE_SRC/.git" ]
}

# Say once what moving from the native runtime to the container means for this
# checkout. Nothing is deleted.
native_notice() {
    local running=0
    native_server_running && running=1
    native_install_present || [ "$running" = 1 ] || return 0
    [ ! -f "$NATIVE_NOTED" ] || [ "$running" = 1 ] || return 0
    log "native install: this release runs the engine only in the container"
    [ "$running" = 0 ] || log "  the native server (pid $(read_pid)) is stopped and the container starts in its place"
    log "  kept: .env, and the checkpoints in $MODELS_DIR, mounted into the container as they are"
    log "  the first container start compiles its kernel caches into $CACHE_DIR, so it is slower"
    if native_install_present; then
        log "  no longer used, left in place: $NATIVE_VENV and $NATIVE_SRC; delete them yourself once you no longer need them"
    fi
    mkdir -p "$LOGDIR"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$NATIVE_NOTED"
}

# Settings of the native runtime earlier releases had. They do nothing now;
# say so instead of ignoring them silently.
retired_settings() {
    if [ -n "${RUNTIME:-}" ] && [ "$RUNTIME" != container ]; then
        warn "RUNTIME=$RUNTIME is no longer used: the engine runs only in the container. Delete the RUNTIME line from .env."
    fi
    if [ -n "${VLLM_COMMIT:-}" ] && [ "$VLLM_COMMIT" != "$RELEASE_VLLM_COMMIT" ]; then
        warn "VLLM_COMMIT=${VLLM_COMMIT:0:10} selected a natively built engine, which this release no longer builds; running this release's image. Delete the VLLM_COMMIT line from .env."
    fi
    if [ -n "${VLLM_BUILD_AMPERE_MARLIN:-}" ] && [ "$VLLM_BUILD_AMPERE_MARLIN" != 0 ]; then
        warn "VLLM_BUILD_AMPERE_MARLIN is no longer used: the image already includes the optional Marlin library. Delete the line from .env."
    fi
}

# ------------------------------ container ----------------------------------
# The container this checkout started is recorded by id in $CIDFILE and carries
# the label $LABEL=<this checkout>. Only a container with both is ever stopped
# or removed; nothing is matched by name alone.
read_cid() { [ -f "$CIDFILE" ] && tr -dc '0-9a-f' <"$CIDFILE" || true; }

# container_state <id>: "<running> <label>" (true/false), or nothing.
container_state() {
    docker container inspect -f "{{.State.Running}} {{index .Config.Labels \"$LABEL\"}}" "$1" 2>/dev/null || true
}
container_is_ours() {
    local id; id="$(read_cid)"
    [ -n "$id" ] || return 1
    [ "$(container_state "$id")" = "true $SCRIPT_DIR" ]
}

# Liveness while waiting for a launch: the container counts as gone only when
# two ownership checks in a row fail, so one failed docker query (an empty
# answer) does not stop a server that is still loading.
container_alive() {
    container_is_ours && return 0
    sleep 2
    container_is_ours
}

# Address:port on the host the API is published on (IPv6 in brackets).
publish_addr() {
    case "$HOST" in
        \[*) printf '%s:%s' "$HOST" "$PORT" ;;
        *:*) printf '[%s]:%s' "$HOST" "$PORT" ;;
        *) printf '%s:%s' "$HOST" "$PORT" ;;
    esac
}

# The environment the container's serve.sh gets: the in-container paths, then
# every setting of yours that serve.sh reads. serve.sh (mounted from this
# checkout) turns them into the server's flags.
CONTAINER_KEYS="P2P P2P_PREBUILT P2P_CHECK_CACHE LAYOUT PP TP MAX_LEN MAX_SEQS MAX_BATCHED PREFILL_CAP GPU_UTIL SPEC_N
REASONING_PARSER TOOL_PARSER DEFAULT_REASONING_EFFORT MM_CAP MM_IMAGES MM_FRAMES MM_MAX_PIXELS MM_MAX_IMAGE_TOKENS FAIR_PREFILL
FAIR_CHUNK FAIR_PARTIAL BLOCK_SIZE EXTRA_ARGS SERVED_MODEL_NAME API_KEY"
container_env() {
    local k
    echo "VENV=/opt/venv"
    echo "MODEL=/models/$(basename "$MODEL")"
    echo "DFLASH_MODEL=/models/$(basename "$DFLASH_MODEL")"
    echo "CUDA_HOME=/usr/local/cuda"
    echo "HOST=0.0.0.0"
    echo "PORT=8000"
    echo "BOOT_CHECK=$BOOT_CHECK"
    echo "RECIPE_BOOT_CHECK_OWNER=start"
    # The server runs as you, not root: every cache it writes (Triton,
    # FlashInfer, TileLang, torch, vLLM, Hugging Face) goes under the mounted
    # cache directory, owned by you. USER and LOGNAME give Python a user name
    # for a uid the image does not know.
    echo "PYTHONPYCACHEPREFIX=/cache/pycache"
    echo "RECIPE_CACHE_ROOT=/cache"
    echo "TMPDIR=/cache/tmp"
    echo "HOME=/cache"
    echo "USER=$(id -un)"
    echo "LOGNAME=$(id -un)"
    echo "XDG_CACHE_HOME=/cache/.cache"
    echo "HF_HOME=/cache/huggingface"
    echo "FLASHINFER_WORKSPACE_BASE=/cache"
    echo "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor"
    echo "TRITON_CACHE_DIR=/cache/.triton/cache"
    echo "TORCH_EXTENSIONS_DIR=/cache/.cache/torch_extensions"
    echo "VLLM_CACHE_ROOT=/cache/.cache/vllm"
    echo "TILELANG_CACHE_DIR=/cache/.tilelang/cache"
    echo "TILELANG_TMP_DIR=/cache/.tilelang/cache/tmp"
    echo "CUDA_CACHE_PATH=/cache/.nv/ComputeCache"
    echo "VLLM_IMAGE_CACHE_SEED=${VLLM_IMAGE_CACHE_SEED:-0}"
    for k in $CONTAINER_KEYS; do
        [ -n "${!k+x}" ] && printf '%s=%s\n' "$k" "${!k}"
    done
    env | LC_ALL=C sort | grep -E '^(VLLM_|GLM5_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=)' \
        | grep -vE '^VLLM_(COMMIT|REPO|BRANCH|SRC|BUILD_AMPERE_MARLIN|USE_PRECOMPILED|PRECOMPILED_[A-Z_]*|API_KEY)=' || true
}

# The docker run arguments, into the array CRUN.
container_args() {
    local envfile="$1"
    CRUN=(docker run -d --name "$CONTAINER_NAME" --label "$LABEL=$SCRIPT_DIR"
        --user "$(id -u):$(id -g)" --workdir /cache
        --gpus all --shm-size "$SHM_SIZE"
        -p "$(publish_addr):8000"
        --env-file "$envfile"
        -v "$SCRIPT_DIR/serve.sh:/recipe/serve.sh:ro"
        -v "$SCRIPT_DIR/boot_check.py:/recipe/boot_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.py:/recipe/p2p_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.sh:/recipe/p2p_check.sh:ro"
        -v "$SCRIPT_DIR/p2p_probe.cu:/recipe/p2p_probe.cu:ro"
        -v "$MODEL:/models/$(basename "$MODEL"):ro")
    if [ "$SPEC_MODE" = dflash ]; then
        CRUN+=(-v "$DFLASH_MODEL:/models/$(basename "$DFLASH_MODEL"):ro")
    fi
    CRUN+=(-v "$CACHE_DIR:/cache"
        --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh "$SPEC_MODE")
}

container_launch() {
    if [ -n "${DRY:-}" ]; then
        container_args "<env file>"
        log "launch: DRY=1, a real start would run:"
        printf '  '; printf '%q ' "${CRUN[@]}"; printf '\n'
        log "  with this environment in the container:"
        container_env | sed -e 's/^\(API_KEY\)=.*/\1=(set, not shown)/' -e 's/^/    /'
        log "  and serve.sh inside it would set:"
        env $(container_env | grep -E '^(VENV|MODEL|DFLASH_MODEL|CUDA_HOME|HOST|PORT)=') \
            DRY=1 "$SCRIPT_DIR/serve.sh" "$SPEC_MODE" | sed 's/^/    /'
        return 0
    fi
    if container_is_ours; then
        log "launch: already running (container $(read_cid | cut -c1-12)) — skipping"
        return 0
    fi
    # A stopped container of ours is removed; any other container with the
    # same name is left alone and we stop.
    local id st
    id="$(read_cid)"
    if [ -n "$id" ]; then
        st="$(container_state "$id")"
        [ "${st#* }" = "$SCRIPT_DIR" ] && docker rm -f "$id" >/dev/null 2>&1 || true
        rm -f "$CIDFILE"
    fi
    st="$(container_state "$CONTAINER_NAME")"
    if [ -n "$st" ]; then
        if [ "$st" = "false $SCRIPT_DIR" ]; then
            docker rm "$CONTAINER_NAME" >/dev/null
        elif [ "$st" = "true $SCRIPT_DIR" ]; then
            docker container inspect -f '{{.Id}}' "$CONTAINER_NAME" >"$CIDFILE"
            log "launch: already running (container $CONTAINER_NAME) — skipping"
            return 0
        else
            die "a container named $CONTAINER_NAME exists and was not started from this checkout. Set CONTAINER_NAME to another name."
        fi
    fi
    mkdir -p "$LOGDIR" "$CACHE_DIR"
    [ -w "$CACHE_DIR" ] || die "$CACHE_DIR is not writable by $(id -un); the server runs as you and keeps its compile caches there. Fix its ownership or set CACHE_DIR."
    log "launch: container, $SPEC_MODE, layout $LAYOUT (TP=$TP PP=$PP), ctx ${MAX_LEN:-262144}, $(publish_addr)"
    log "  image: $IMAGE"
    log "  log: $SERVE_LOG"
    BOOT_LOG_OFFSET=0
    [ ! -f "$SERVE_LOG" ] || BOOT_LOG_OFFSET="$(wc -c < "$SERVE_LOG")"
    local envfile="$LOGDIR/container.env.run"
    ( umask 077; container_env >"$envfile" )
    container_args "$envfile"
    id="$("${CRUN[@]}")" || { rm -f "$envfile"; die "docker run failed"; }
    rm -f "$envfile"
    echo "$id" >"$CIDFILE"
    LAUNCHED=1
    log "  container: ${id:0:12} ($CONTAINER_NAME)"
    # Copy the container's output into $SERVE_LOG; the follower ends when the
    # container does. 9>&- : the follower must not inherit the lifecycle lock
    # (fd 9), or it would hold the lock for the container's whole life and
    # block every later stop, restart and update.
    setsid nohup docker logs -f "$id" 9>&- >>"$SERVE_LOG" 2>&1 </dev/null &
}

container_stop() {
    local id st; id="$(read_cid)"
    if [ -z "$id" ]; then
        log "stop: no container started from this checkout"
        return 0
    fi
    st="$(container_state "$id")"
    if [ -z "$st" ]; then
        log "stop: container ${id:0:12} is gone — clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st#* }" != "$SCRIPT_DIR" ]; then
        warn "stop: container ${id:0:12} was not started from this checkout — leaving it alone and clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st%% *}" = true ]; then
        log "stop: docker stop ${id:0:12} (up to ${STOP_TIMEOUT}s)"
        docker stop -t "$STOP_TIMEOUT" "$id" >/dev/null || true
    fi
    docker rm "$id" >/dev/null 2>&1 || true
    rm -f "$CIDFILE"
    log "stop: done"
}

health_ok() { curl -fsS -m "${1:-5}" "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1; }

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

wait_ready() {
    [ -z "${DRY:-}" ] || return 0
    local url="http://$CLIENT_HOST:$PORT/health" elapsed=0 started=$SECONDS checker status
    log "waiting for $url (weight load and graph capture on a 320B MoE are slow; timeout ${READY_TIMEOUT}s)"
    local log_args=(--log "$SERVE_LOG" --log-offset "${BOOT_LOG_OFFSET:-0}" --container)
    while true; do
        elapsed=$((SECONDS - started))
        [ "$elapsed" -lt "$READY_TIMEOUT" ] || break
        if [ "$LAUNCHED" = 1 ] && [ "$BOOT_CHECK" != 0 ] && [ -f "$SERVE_LOG" ]; then
            if ! python3 "$SCRIPT_DIR/boot_check.py" "${log_args[@]}" --log-only; then
                warn "[boot-check] FAIL: stopping this checkout's failed launch"
                do_stop
                return 1
            fi
        fi
        if health_ok "$((READY_TIMEOUT - elapsed < 5 ? READY_TIMEOUT - elapsed : 5))"; then
            log "healthy after ${elapsed}s"
            if [ "$LAUNCHED" = 1 ]; then
                if [ "$BOOT_CHECK" = 0 ]; then
                    log "[boot-check] disabled (BOOT_CHECK=0)"
                else
                    # Health can briefly precede the metrics/API frontend. Keep
                    # connection readiness inside the original launch budget.
                    python3 "$SCRIPT_DIR/boot_check.py" --wait-ready \
                        --ready-timeout "$((READY_TIMEOUT - (SECONDS - started)))" \
                        --base "http://$CLIENT_HOST:$PORT" --model "$SERVED_MODEL_NAME" "${log_args[@]}" &
                    checker=$!
                    trap 'kill "$checker" 2>/dev/null || true; wait "$checker" 2>/dev/null || true; do_stop; exit 1' INT TERM
                    while kill -0 "$checker" 2>/dev/null; do
                        if ! container_alive; then
                            kill "$checker" 2>/dev/null || true
                            break
                        fi
                        sleep 1
                    done
                    status=0
                    wait "$checker" || status=$?
                    trap - INT TERM
                    if [ "$status" != 0 ]; then
                        warn "[boot-check] FAIL: stopping this checkout's failed launch"
                        do_stop
                        return 1
                    fi
                fi
            else
                log "[boot-check] existing server: no new requests"
            fi
            local kv; kv="$(kv_line || true)"
            [ -n "$kv" ] && log "  $kv"
            log "  model: $(served_id)"
            log "  API:   http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
            return 0
        fi
        if ! container_alive; then
            warn "the server exited after ${elapsed}s — last 40 lines:"
            sleep 1
            tail -40 "$SERVE_LOG" >&2 || true
            return 1
        fi
        sleep 5; elapsed=$((SECONDS - started))
    done
    warn "not healthy after ${READY_TIMEOUT}s; it may still be loading — ./start.sh logs"
    return 1
}

# --------------------------------- stop ------------------------------------
# Stops whatever this checkout started: its container, or a native server an
# earlier release started.
do_stop() {
    local did=0
    if [ -f "$CIDFILE" ]; then container_stop; did=1; fi
    if [ -f "$PIDFILE" ]; then native_stop; did=1; fi
    [ "$did" = 1 ] || log "stop: nothing started from this checkout is running"
}

native_stop() {
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
    log "stop: SIGINT to pid $pid (process group; native server from an earlier release)"
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
    log "layout $LAYOUT (TP=$TP PP=$PP)"
    local id st; id="$(read_cid)"
    if [ -z "$id" ]; then
        log "process: no container started from this checkout"
    else
        st="$(container_state "$id")"
        case "$st" in
            "true $SCRIPT_DIR") log "process: container running, ${id:0:12} ($CONTAINER_NAME)" ;;
            "false $SCRIPT_DIR") log "process: container ${id:0:12} has stopped — ./start.sh logs" ;;
            "") log "process: container ${id:0:12} is gone (stale id file)" ;;
            *) log "process: container ${id:0:12} is not ours (stale id file)" ;;
        esac
    fi
    if native_server_running; then
        log "process: native server from an earlier release, pid $(read_pid) — ./start.sh update moves it to the container"
    fi
    if health_ok; then
        log "API:     healthy — http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
        log "model:   $(served_id)"
    else
        log "API:     not responding on :$PORT"
    fi
    local kv; kv="$(kv_line || true)"
    if [ -n "$kv" ]; then log "KV:      $kv"; else log "KV:      (no KV line in $SERVE_LOG)"; fi
    log "install: $(install_done && echo "image present" || echo 'image missing or pin moved — ./start.sh install'), $IMAGE"
    log "models:  $(models_present && echo ok || echo 'missing or incomplete — ./start.sh download')"
    if native_install_present; then
        log "native:  $NATIVE_VENV and $NATIVE_SRC are no longer used; delete them once you no longer need them"
    fi
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
    # Check Docker before anything is stopped: a native server from an earlier
    # release keeps running if the container cannot replace it.
    preflight_docker
    if install_done; then
        log "update: engine unchanged at $IMAGE — no pull"
    else
        log "update: engine moved to $IMAGE — pulling it"
        do_install
    fi
    native_notice
    do_stop
    start_unlocked
}

# --------------------------------- start -----------------------------------
# DRY: report what a real start would do, then print the server environment
# and command. Nothing is installed, downloaded or launched.
dry_start() {
    preflight --with-port
    if native_server_running; then
        warn "a native server from an earlier release is running (pid $(read_pid)); a real update or restart stops it and moves this checkout to the container, a plain start refuses"
    fi
    if command -v docker >/dev/null 2>&1 && install_done; then
        log "install: image present — a real start would skip it"
    else
        log "install: a real start would pull $IMAGE"
    fi
    if models_present; then
        log "download: checkpoints present — a real start would skip it"
    else
        local what="$TARGET_REPO"
        [ "$SPEC_MODE" = "dflash" ] && what="$what and $DRAFTER_REPO"
        log "download: a real start would fetch $what into $MODELS_DIR"
    fi
    container_launch
}

# DRY stop/update: the lock is taken for real (that is what they check), and
# nothing is signalled, pulled or installed.
dry_stop() {
    local id; id="$(read_cid)"
    [ -z "$id" ] || log "stop: DRY=1, a real stop would docker stop container ${id:0:12} if it is ours"
    local pid; pid="$(read_pid)"
    [ -n "$pid" ] || [ -z "$id" ] || return 0
    if [ -z "$pid" ]; then
        log "stop: DRY=1, nothing started from this checkout is running"
    elif pid_is_ours "$pid"; then
        log "stop: DRY=1, a real stop would send SIGINT to pid $pid (process group)"
    else
        log "stop: DRY=1, pid $pid is not our server; a real stop would only clear the pid file"
    fi
}
dry_update() {
    log "update: DRY=1, a real update would git pull --ff-only, pull the image if the pin moved, then restart"
    if native_install_present || native_server_running; then
        log "update: DRY=1, it would also move this checkout's native install to the container (nothing deleted)"
    fi
    dry_stop
    dry_start
}

start_unlocked() {
    if [ -n "${DRY:-}" ]; then dry_start; return 0; fi
    if native_server_running; then
        die "a native server from an earlier release of this checkout is running (pid $(read_pid)). This release runs the engine only in the container: ./start.sh update (or restart) stops it and starts the container with the same .env and checkpoints."
    fi
    preflight --with-port
    do_install
    if models_present; then
        log "download: checkpoints present — skipping"
    else
        do_download
    fi
    container_launch
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
        start|install|download|restart|update|status) retired_settings ;;
    esac
    case "$cmd" in
        start)    if [ -n "${DRY:-}" ]; then dry_start; else take_lock; start_unlocked; fi ;;
        install)  take_lock; preflight; do_install ;;
        download) take_lock; preflight; do_download ;;
        stop)     take_lock_for_stop; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi ;;
        restart)  take_lock
                  if [ -n "${DRY:-}" ]; then dry_stop; else preflight_docker; native_notice; do_stop; fi
                  start_unlocked ;;
        update)   take_lock; if [ -n "${DRY:-}" ]; then dry_update; else do_update; fi ;;
        status)   do_status ;;
        logs)     do_logs ;;
        smoke)    HOST="$CLIENT_HOST" exec "$SCRIPT_DIR/smoke.sh" ;;
        *) warn "unknown command: $cmd"; usage; exit 1 ;;
    esac
}

main "$@"
