#!/bin/bash

# clean.sh
# resets the project to a clean state by removing all generated files
# use this before a fresh run to make sure nothing is cached
#
# what it removes:
#   - data/processed/          (generated tiles)
#   - checkpoints/*.pth        (trained model weights)
#   - checkpoints/*.json       (training history)
#   - results/                 (evaluation plots, metrics, logs)
#
# what it does NOT remove:
#   - data/raw/                (original images and annotations)
#   - data/assets/             (waldo cutouts and backgrounds)
#   - saved_results/           (experiment outputs, use --all to include)
#   - source code
#
# usage:
#   bash scripts/clean.sh              (clean main pipeline outputs)
#   bash scripts/clean.sh --all        (also clean saved_results/ from experiments)

# cd to project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "  Wheres Waldo - Clean"
echo "========================================"
echo ""

CLEAN_ALL=false
if [ "$1" = "--all" ]; then
    CLEAN_ALL=true
fi

# clean generated tiles
if [ -d "data/processed" ]; then
    echo "removing data/processed/"
    rm -rf data/processed
else
    echo "data/processed/ already clean"
fi

# clean checkpoints
removed_ckpt=false
for f in checkpoints/best_model.pth checkpoints/last_model.pth checkpoints/training_history.json; do
    if [ -f "$f" ]; then
        echo "removing $f"
        rm -f "$f"
        removed_ckpt=true
    fi
done
if [ "$removed_ckpt" = false ]; then
    echo "checkpoints/ already clean"
fi

# clean results
if [ -d "results" ] && [ "$(ls -A results 2>/dev/null)" ]; then
    echo "removing results/*"
    rm -rf results/*
else
    echo "results/ already clean"
fi

# clean experiment outputs if --all flag
if [ "$CLEAN_ALL" = true ]; then
    if [ -d "saved_results" ] && [ "$(ls -A saved_results 2>/dev/null)" ]; then
        echo "removing saved_results/"
        rm -rf saved_results
    else
        echo "saved_results/ already clean"
    fi
fi

echo ""
echo "done! project is clean."
if [ "$CLEAN_ALL" = false ]; then
    echo "(use 'bash scripts/clean.sh --all' to also remove saved_results/)"
fi
