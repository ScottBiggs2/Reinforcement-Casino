#!/usr/bin/env bash
# Runs ON Discovery. Pushes everything the Prime Intellect box needs, so that box
# never has to authenticate to HuggingFace.
#
# Llama-3.1-8B-Instruct is a gated repo. Rather than putting an HF token on a
# third-party machine, we ship the copy already in Discovery's cache. Same for the
# dataset and the mask. The PI box then runs with HF_HUB_OFFLINE=1 and no credentials
# of any kind.
#
# Direction matters: Discovery PUSHES. The PI instance is created with Discovery's
# PUBLIC key, so no private key ever leaves the cluster.
#
# usage: bash pi_push_from_discovery.sh <user@pi-host> [ssh-port]
set -euo pipefail

DEST="${1:?usage: pi_push_from_discovery.sh <user@host> [port]}"
PORT="${2:-22}"
S="/scratch/$USER"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
RS="rsync -a --info=progress2 --partial -e \"$SSH\""

echo "== creating remote layout =="
$SSH "$DEST" 'mkdir -p /workspace/data/hf_cache/hub /workspace/data/hf_cache/datasets /workspace/out'

echo
echo "== 1/4 repo (small) =="
rsync -a --info=progress2 --partial -e "$SSH" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pt' --exclude 'logs' \
  "$HOME/rc-sparse-speed/" "$DEST:/workspace/Reinforcement-Casino/"

echo
echo "== 2/4 base model, 15 GB (avoids needing an HF token remotely) =="
rsync -a --info=progress2 --partial -e "$SSH" \
  "$S/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct" \
  "$DEST:/workspace/data/hf_cache/hub/"

echo
echo "== 3/4 dataset =="
rsync -a --info=progress2 --partial -e "$SSH" \
  "$S/hf_cache/hub/datasets--open-r1--OpenR1-Math-220k" \
  "$DEST:/workspace/data/hf_cache/hub/"

echo
echo "== 4/4 oracle GRPO mask, 7.5 GB =="
rsync -a --info=progress2 --partial -e "$SSH" \
  "$S/transfer_v1/oracle_masks_llama8b/oracle_grpo_math220k_step500_sp97.5.pt" \
  "$DEST:/workspace/data/"

echo
echo "== verify =="
$SSH "$DEST" 'du -sh /workspace/data/* /workspace/Reinforcement-Casino 2>/dev/null; echo; ls /workspace/data/hf_cache/hub/'
echo "push complete"
