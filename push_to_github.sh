#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
# Publish this repository to GitHub (https://github.com/ylee-sling/clone-VPA).
#
#   ./push_to_github.sh
#
# Steps: initialize git if needed, select the branch, configure the remote
# "origin", make sure .gitignore excludes checkpoints, data caches,
# .pytest_cache and __pycache__, stage every file, commit with a structured
# message and push to the branch.
#
# Environment overrides:
#   REMOTE_URL      remote URL (default https://github.com/ylee-sling/clone-VPA.git;
#                   use git@github.com:ylee-sling/clone-VPA.git for SSH)
#   BRANCH          branch to push (default main)
#   GIT_USER_NAME   author name if git has no identity configured (default "Yeonseok Lee")
#   GIT_USER_EMAIL  author email if git has no identity configured (default ylee@sling.ai.kr)
#   FORCE_PUSH=1    overwrite a remote branch whose history is unrelated (uses --force-with-lease)
#
# Pushing over HTTPS needs a GitHub personal access token (or the GitHub CLI's
# credential helper: `gh auth login`); pushing over SSH needs a registered key.
set -euo pipefail

REMOTE_URL="${REMOTE_URL:-https://github.com/ylee-sling/clone-VPA.git}"
BRANCH="${BRANCH:-main}"
SUBJECT="feat: complete CMA-VPA implementation with Task T1, Task T2, and CBF-DeGG-Flow"

cd "$(dirname "${BASH_SOURCE[0]}")"
command -v git >/dev/null 2>&1 || { echo "error: git is not installed" >&2; exit 1; }

# 1. repository -------------------------------------------------------------
if [ ! -d .git ]; then
  git init -q
  echo "[1/6] initialized a new git repository in $(pwd)"
else
  echo "[1/6] using the existing git repository in $(pwd)"
fi

# 2. branch -----------------------------------------------------------------
if ! git rev-parse --verify -q HEAD >/dev/null; then
  git symbolic-ref HEAD "refs/heads/${BRANCH}"            # no commits yet: name the unborn branch
elif [ "$(git symbolic-ref --short -q HEAD || true)" != "${BRANCH}" ]; then
  if git show-ref --verify -q "refs/heads/${BRANCH}"; then
    git checkout -q "${BRANCH}"
  else
    git checkout -q -b "${BRANCH}"
  fi
fi
echo "[2/6] branch ${BRANCH}"

if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  git config user.name "${GIT_USER_NAME:-Yeonseok Lee}"
  git config user.email "${GIT_USER_EMAIL:-ylee@sling.ai.kr}"
  echo "      no git identity configured; set it for this repository: $(git config user.name) <$(git config user.email)>"
fi

# 3. remote -----------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi
echo "[3/6] origin -> ${REMOTE_URL}"

# 4. .gitignore ---------------------------------------------------------------
touch .gitignore
added=0
for pattern in "__pycache__/" "*.py[cod]" ".pytest_cache/" "*.egg-info/" "checkpoints/" "ckpt/" "data/" \
               "cache/" ".cache/" "runs/" "wandb/" "*.pt" "*.pth" "*.ckpt" "*.zip"; do
  if ! grep -qxF "${pattern}" .gitignore; then
    printf '%s\n' "${pattern}" >> .gitignore
    added=$((added + 1))
  fi
done
echo "[4/6] .gitignore ready (${added} pattern(s) added)"

# 5. stage and commit -----------------------------------------------------------
git add -A
if git diff --cached --quiet; then
  echo "[5/6] nothing new to commit"
else
  git commit -q -F - <<EOF
${SUBJECT}

Official implementation of "Dynamic Latent Cloning with Continuous Barrier
Guidance: Multi-Agent Vision-Predictor-Action" (Yeonseok Lee, SLING AI Inc.,
DOI 10.5281/zenodo.23679374), built upon the single-agent foundation
https://github.com/ylee-sling/Vision-Predictor-Action.

- safety/: braking-consistent HOCBF, reciprocal shares with explicit margins and
  inactivity certificates (Lemmas 12-15), Lemma 18 closed form and grouped
  Hildreth with exact polish, Stage A (GUIDEDSOLVE), the 100 Hz Stage-B
  executor, jerk-optimal braking fallback, and the twist-space arm barrier
- comm/: 24-stage residual VQ and bit-exact 35/36/22-byte codecs with CRC-16
- models/: Siamese ViT encoder, primitive selector, ensemble JEPA predictor,
  KV-cached bridge, CFM solver, PR-JEPA set-transformer master and the K + 3
  clone pipeline (Table 3 parameter budget reproduced exactly)
- training/: VICReg, SIGReg, JEPA and CFM losses and the five training stages
- envs/: Task T1 (4-16 UAV antipodal crossing, wind W0/W1/W2) and Task T2
  (dual Franka Panda tabletop assembly, ManiSkill3-style interface)
- run_sitl_demo.py: asynchronous master / clone / executor loops for T1 and T2
- tests/: unit, closed-loop and training-pipeline tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G55ACdfUnzj4ARFMGRWFgT
EOF
  echo "[5/6] committed $(git rev-parse --short HEAD): ${SUBJECT}"
fi

# 6. push ---------------------------------------------------------------------
if git ls-remote --exit-code --heads origin "${BRANCH}" >/dev/null 2>&1; then
  git fetch -q origin "${BRANCH}"
  if ! git merge-base --is-ancestor "origin/${BRANCH}" HEAD; then
    if [ "${FORCE_PUSH:-0}" = "1" ]; then
      echo "[6/6] remote ${BRANCH} has unrelated history; overwriting it (FORCE_PUSH=1)"
      git push --force-with-lease -u origin "${BRANCH}"
      exit 0
    fi
    cat >&2 <<EOF
error: origin/${BRANCH} contains commits that are not in this repository
       (for example a README or LICENSE created on GitHub). Either merge them:
           git pull --no-rebase --allow-unrelated-histories origin ${BRANCH}
           ./push_to_github.sh
       or overwrite the remote branch:
           FORCE_PUSH=1 ./push_to_github.sh
EOF
    exit 1
  fi
fi
git push -u origin "${BRANCH}"
echo "[6/6] pushed ${BRANCH} to ${REMOTE_URL}"
