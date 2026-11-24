#!/usr/bin/env bash
# Run in tmux with: bash scripts/auto_git_push.sh
# Use --dry-run to list pending changes without staging, committing or pushing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RESULTS=new_exp_nov_MST_fold1
INTERVAL=7200
EXPIRES=2026-12-10

if [[ ${1:-} == --dry-run ]]; then
    git status --short --untracked-files=all -- "$RESULTS/"
    exit
fi

exec 9>"$(git rev-parse --git-path auto-push-results.lock)"
flock -n 9 || { echo 'Result autopush is already running.' >&2; exit 1; }

push_results() {
    local changes unrelated
    if [[ $(git symbolic-ref --short HEAD) != main ]]; then
        echo 'Switch back to main for result autopush to continue.' >&2
        return 1
    fi

    changes=$(git status --porcelain -- "$RESULTS/") || return
    if [[ -n $changes ]]; then
        git add -A -- "$RESULTS/" || return
        if ! git diff --cached --quiet -- "$RESULTS/"; then
            git commit --only -m "auto: checkpoint $RESULTS" -- "$RESULTS/" || return
        fi
    fi

    git fetch --quiet origin main || return
    if [[ $(git rev-parse HEAD) == "$(git rev-parse origin/main)" ]]; then
        echo 'main is up to date with origin/main.'
        return
    fi
    if ! git merge-base --is-ancestor origin/main HEAD; then
        echo 'main is behind or has diverged from origin/main.' >&2
        return 1
    fi
    unrelated=$(git log --format=%h origin/main..HEAD -- . ":(exclude)$RESULTS/") || return
    if [[ -n $unrelated ]]; then
        printf 'Pending commits touch paths outside %s: %s\n' "$RESULTS" "$unrelated" >&2
        return 1
    fi
    git push origin main
}

while [[ $(date +%F) < $EXPIRES || $(date +%F) == "$EXPIRES" ]]; do
    printf '[%s] Checking %s\n' "$(date '+%F %T')" "$RESULTS/"
    push_results || echo 'Autopush will retry at the next check.' >&2
    sleep "$INTERVAL"
done
