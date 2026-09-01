#!/usr/bin/env bash
# GR00T N1.7 finetuning environment setup
# =========================================
# One-time (idempotent) setup of a GR00T-enabled `lerobot` checkout in a
# Python 3.12 venv, kept as a SIBLING directory next to this repo so it
# never mixes with this repo's own requirements.txt/env (GR00T needs a
# much newer/heavier ML stack than the plain robot-recording deps here).
#
# Usage:
#   scripts/setup_groot_env.sh [--dir ../lerobot-groot] [--ref main]
#
# Prerequisite (cannot be automated -- do this yourself before training):
#   1. Accept the gated model license at
#      https://huggingface.co/nvidia/Cosmos-Reason2-2B
#      (GR00T-N1.7-3B loads this VLM backbone on first use).
#   2. Run `hf auth login` (or `huggingface-cli login`) with a token that
#      has access to that model.

set -euo pipefail

GROOT_DIR="../lerobot-groot"
REF="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) GROOT_DIR="$2"; shift 2 ;;
        --ref) REF="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^#//; s/^ //'
            exit 0
            ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

command -v uv >/dev/null 2>&1 || {
    echo "[ERROR] uv not found on PATH. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}
command -v ffmpeg >/dev/null 2>&1 || {
    echo "[ERROR] ffmpeg not found on PATH. Install it: sudo apt install ffmpeg" >&2
    exit 1
}

if [[ -d "$GROOT_DIR/.git" ]]; then
    echo "==> $GROOT_DIR already exists, pulling latest $REF ..."
    git -C "$GROOT_DIR" fetch origin "$REF"
    git -C "$GROOT_DIR" checkout "$REF"
    git -C "$GROOT_DIR" pull origin "$REF"
else
    echo "==> Cloning huggingface/lerobot ($REF) into $GROOT_DIR ..."
    git clone --branch "$REF" https://github.com/huggingface/lerobot.git "$GROOT_DIR"
fi

echo "==> Creating Python 3.12 venv in $GROOT_DIR/.venv ..."
(cd "$GROOT_DIR" && uv venv --python 3.12)

echo "==> Installing lerobot[groot,training] (editable) ..."
(cd "$GROOT_DIR" && uv pip install -e ".[groot,training]")

echo "==> Sanity check: does lerobot-train support --policy.type=groot ?"
if uv run --project "$GROOT_DIR" lerobot-train --help 2>&1 | grep -q groot; then
    echo "    OK -- groot policy type found."
else
    echo "[ERROR] 'groot' not found in lerobot-train --help output -- install may have failed." >&2
    exit 1
fi

cat <<EOF

==> Setup complete: $GROOT_DIR

BEFORE TRAINING, you must (this cannot be automated):
  1. Accept the gated model license:
       https://huggingface.co/nvidia/Cosmos-Reason2-2B
  2. Log in with a token that has access to it:
       uv run --project $GROOT_DIR hf auth login

Then smoke-test the pipeline before a full run:
  scripts/train_groot.sh --dataset-dir open_trashcan_50 --steps 10 --batch-size 4 --save-freq 5
EOF
