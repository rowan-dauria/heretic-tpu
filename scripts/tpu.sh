#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Development helper for running heretic-tpu on a Cloud TPU VM.
#
# Usage:
#   scripts/tpu.sh setup        Install uv and the shared virtual environment on the VM.
#   scripts/tpu.sh sync         Copy the working tree (tracked and untracked, non-ignored files,
#                               and the git-ignored heretic/ and .scratch/ directories).
#   scripts/tpu.sh run CMD...   Sync, then run CMD inside the synced tree on the VM.
#   scripts/tpu.sh exec CMD...  Like run, but without syncing first.
#   scripts/tpu.sh submit NAME CMD...
#                               Sync, then start CMD detached on the VM (under the TPU
#                               lock unless TPU_LOCK=0), logging to NAME.log in the
#                               remote checkout. Don't sync the same checkout again
#                               while the job runs, as syncing replaces its sources.
#   scripts/tpu.sh logs NAME [LINES]
#                               Show the tail of NAME.log and whether the job finished.
#   scripts/tpu.sh shell        Open an interactive shell on the VM.
#
# Configuration (environment variables):
#   TPU_NAME, TPU_ZONE, TPU_PROJECT   Identify the TPU VM (required).
#   TPU_REMOTE_DIR                    Remote checkout directory, relative to $HOME
#                                     (default: heretic-tpu). Use one per concurrent user.
#   TPU_LOCK=0                        Don't take the exclusive TPU lock. Only a single
#                                     process can use the TPU at a time, so set this
#                                     only for commands that don't touch it (e.g. when
#                                     running with JAX_PLATFORMS=cpu).
#   TPU_LOCK_WAIT                     Seconds that run and exec wait for the TPU lock
#                                     before giving up with exit status 75 (default:
#                                     300). Detached submit jobs always wait.
#
# All remote commands run with the shared virtual environment activated and with
# PYTHONPATH pointing at the synced src/ directory, so several checkouts can share
# one environment. If the working tree has a checkout of upstream Heretic in heretic/,
# they also run with HERETIC_TPU_REQUIRE_UPSTREAM=1, which makes the tests that
# compare with upstream fail rather than skip if it did not arrive (see
# tests/conftest.py).

set -euo pipefail

: "${TPU_NAME:?set TPU_NAME to the TPU VM name}"
: "${TPU_ZONE:?set TPU_ZONE to the TPU VM zone}"
: "${TPU_PROJECT:?set TPU_PROJECT to the Google Cloud project}"

REMOTE_DIR="${TPU_REMOTE_DIR:-heretic-tpu}"
LOCK="${TPU_LOCK:-1}"
LOCK_WAIT="${TPU_LOCK_WAIT:-300}"
VENV='$HOME/.venvs/heretic-tpu'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The optional, git-ignored checkout of upstream Heretic that some tests compare with.
if [ -d "$REPO_ROOT/heretic/src/heretic" ]; then
    REQUIRE_UPSTREAM=1
else
    REQUIRE_UPSTREAM=0
fi

ssh_tpu() {
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --project="$TPU_PROJECT" \
        --zone="$TPU_ZONE" \
        --tunnel-through-iap \
        "$@"
}

remote_prelude() {
    cat <<EOF
set -eo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "\$HOME/$REMOTE_DIR"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
export PYTHONPATH="\$HOME/$REMOTE_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HERETIC_TPU_REQUIRE_UPSTREAM=$REQUIRE_UPSTREAM
EOF
}

do_sync() {
    # Remove previously synced code first so that deleted files don't linger.
    # The git-ignored heretic/ directory (a checkout of upstream Heretic, which some
    # tests compare with) is synced without its .git directory, and so is the
    # git-ignored .scratch/ directory, for throwaway scripts that should run on the
    # VM rather than locally.
    # Paths that don't exist are dropped, because tar fails on them: git still lists
    # a tracked file that was deleted until the deletion is staged. `if` rather than
    # `&&` keeps a missing last path from failing the loop, and -L keeps dangling
    # tracked symbolic links. This runs on macOS too, so no GNU-only options are used.
    (
        cd "$REPO_ROOT"
        {
            git ls-files -z --cached --others --exclude-standard
            if [ "$REQUIRE_UPSTREAM" = "1" ]; then
                find heretic -path heretic/.git -prune -o -type f -print0
            fi
            if [ -d .scratch ]; then find .scratch -type f -print0; fi
        } | while IFS= read -r -d '' path; do
            if [ -e "$path" ] || [ -L "$path" ]; then printf '%s\0' "$path"; fi
        done | COPYFILE_DISABLE=1 tar --null -T - -czf -
    ) | ssh_tpu --command="mkdir -p \"\$HOME/$REMOTE_DIR\" && cd \"\$HOME/$REMOTE_DIR\" && rm -rf src tests scripts .scratch heretic && tar -xzf - 2>/dev/null"
}

do_exec() {
    local command="$*"
    local quoted
    quoted="$(printf '%q' "$command")"
    if [ "$LOCK" = "1" ]; then
        # Give up rather than block indefinitely behind another job, so that callers
        # with a time limit (such as automated agents) get an answer.
        command="flock -w $LOCK_WAIT -E 75 /tmp/heretic-tpu.lock bash -c $quoted || {
    status=\$?
    if [ \$status -eq 75 ]; then
        echo 'The TPU is busy: another job held the lock for $LOCK_WAIT s. Try again later, or use submit.' >&2
    fi
    exit \$status
}"
    else
        command="bash -c $quoted"
    fi
    ssh_tpu --command="$(remote_prelude)
$command"
}

case "${1:-}" in
    setup)
        do_sync
        ssh_tpu --command="set -e
if ! command -v uv >/dev/null && [ ! -x \"\$HOME/.local/bin/uv\" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd \"\$HOME/$REMOTE_DIR\"
UV_PROJECT_ENVIRONMENT=\"$VENV\" uv sync --extra tpu --group dev --group parity"
        ;;
    sync)
        do_sync
        ;;
    run)
        shift
        do_sync
        do_exec "$@"
        ;;
    exec)
        shift
        do_exec "$@"
        ;;
    submit)
        shift
        name="${1:?usage: scripts/tpu.sh submit NAME CMD...}"
        shift
        do_sync
        job="$(printf '%q' "$*")"
        if [ "$LOCK" = "1" ]; then
            job="flock /tmp/heretic-tpu.lock bash -c $job"
        else
            job="bash -c $job"
        fi
        # The exit status is written to NAME.exit so that `logs` can tell a finished
        # job from a running one. setsid and the redirections detach the job from
        # the SSH session.
        wrapper="$(printf '%q' "$job; echo \$? > $name.exit")"
        ssh_tpu --command="$(remote_prelude)
rm -f $name.exit
setsid nohup bash -c $wrapper > $name.log 2>&1 < /dev/null &
echo \"Submitted $name (PID \$!); follow it with: scripts/tpu.sh logs $name\""
        ;;
    logs)
        name="${2:?usage: scripts/tpu.sh logs NAME [LINES]}"
        lines="${3:-50}"
        ssh_tpu --command="cd \"\$HOME/$REMOTE_DIR\"
tail -n $lines $name.log
if [ -f $name.exit ]; then echo \"[finished with exit status \$(cat $name.exit)]\"; else echo '[still running]'; fi"
        ;;
    shell)
        ssh_tpu
        ;;
    *)
        sed -n '/^# Usage:/,/^$/p' "$0"
        exit 1
        ;;
esac
