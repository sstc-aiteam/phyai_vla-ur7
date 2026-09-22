#!/usr/bin/env bash
# pi0 finetuning environment setup
# =================================
# One-time (idempotent) setup of a pi0-enabled `lerobot` checkout in a
# Python 3.12 venv, kept as a SIBLING directory next to this repo so it
# never mixes with this repo's own requirements.txt/env.
#
# Why a separate env at all: pi0's PyTorch port needs a SigLIP fix that
# only ships in transformers>=5.4,<5.6 (see lerobot's pyproject.toml,
# `transformers-dep`). This repo's own env pins transformers<5.0 for the
# ACT/SmolVLA extras already installed here -- bumping that in place would
# risk breaking those working checkpoints/eval/infer scripts. Cloning
# lerobot's main branch fresh (which now also vendors that SigLIP fix
# directly) into an isolated venv sidesteps the conflict entirely -- same
# reasoning as scripts/setup_groot_env.sh, which this mirrors.
#
# Usage:
#   scripts/setup_pi0_env.sh [--dir ../lerobot-pi0] [--ref main]
#
# pi0_base (the finetuning base checkpoint) is itself NOT gated, but its
# tokenizer/config come from its underlying VLM backbone,
# google/paligemma-3b-pt-224, which IS gated -- and gated "manual" (Google
# reviews each request, not an instant click-through license). Same shape
# of prerequisite as GR00T's Cosmos-Reason2-2B, one extra step:
#
# Prerequisite (cannot be automated -- do this yourself before training):
#   1. Request access at
#      https://huggingface.co/google/paligemma-3b-pt-224
#      and wait for approval (manual review, not instant).
#   2. Run `hf auth login` (or `huggingface-cli login`) with a token
#      belonging to the account that was granted access.

set -euo pipefail

PI0_DIR="../lerobot-pi0"
REF="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) PI0_DIR="$2"; shift 2 ;;
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

if [[ -d "$PI0_DIR/.git" ]]; then
    echo "==> $PI0_DIR already exists, pulling latest $REF ..."
    git -C "$PI0_DIR" fetch origin "$REF"
    git -C "$PI0_DIR" checkout "$REF"
    git -C "$PI0_DIR" pull origin "$REF"
else
    echo "==> Cloning huggingface/lerobot ($REF) into $PI0_DIR ..."
    git clone --branch "$REF" https://github.com/huggingface/lerobot.git "$PI0_DIR"
fi

if [[ -d "$PI0_DIR/.venv" ]]; then
    echo "==> $PI0_DIR/.venv already exists, reusing it ..."
else
    echo "==> Creating Python 3.12 venv in $PI0_DIR/.venv ..."
    (cd "$PI0_DIR" && uv venv --python 3.12)
fi

echo "==> Installing lerobot[pi,training] (editable) ..."
# --python pins this to the venv just created above -- without it, `uv pip
# install` picks up this shell's active conda env (CONDA_PREFIX) instead of
# the project-local .venv when one happens to be active, and fails resolving
# against the wrong (3.10) interpreter.
(cd "$PI0_DIR" && uv pip install --python .venv/bin/python -e ".[pi,training]")

echo "==> Sanity check: does lerobot-train support --policy.type=pi0 ?"
if uv run --project "$PI0_DIR" lerobot-train --help 2>&1 | grep -q "pi0,"; then
    echo "    OK -- pi0 policy type found."
else
    echo "[ERROR] 'pi0' not found in lerobot-train --help output -- install may have failed." >&2
    exit 1
fi

echo "==> Sanity check: pi0 policy imports"
uv run --project "$PI0_DIR" python3 -c "
import transformers
print('    transformers', transformers.__version__)
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
print('    PI0Policy import OK')
"

cat <<EOF

==> Setup complete: $PI0_DIR

BEFORE TRAINING, you must (this cannot be automated):
  1. Request access to the gated PaliGemma backbone pi0 depends on:
       https://huggingface.co/google/paligemma-3b-pt-224
     (manual review by Google -- may not be instant.)
  2. Log in with a token from the account that was granted access:
       uv run --project $PI0_DIR hf auth login

Then smoke-test the pipeline before a full run:
  scripts/train_pi0.sh --dataset-dir open_trashcan_10 --steps 30 --batch-size 8 --save-freq 30
EOF
