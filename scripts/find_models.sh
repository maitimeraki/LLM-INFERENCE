#!/bin/bash
# Helper script to find downloaded Hugging Face models in WSL

set -e

echo "=== Hugging Face Model Finder ==="
echo ""

# Check common cache locations
LOCATIONS=(
    "/mnt/d"
    "/mnt/c/Users/$USER/.cache/huggingface"
    "$HOME/.cache/huggingface"
)

echo "Checking common locations for Hugging Face models..."
echo ""

FOUND=0

for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "📁 Searching in: $location"

        # Find model directories
        models=$(find "$location" -type d -name "models--*" 2>/dev/null | head -20)

        if [ -n "$models" ]; then
            echo "$models" | while read -r model; do
                FOUND=$((FOUND + 1))
                model_name=$(basename "$model" | sed 's/models--//' | sed 's/--/\//g')

                echo "  ✓ Found: $model_name"
                echo "    Path: $model"

                # Find snapshots
                snapshots=$(ls "$model/snapshots" 2>/dev/null | head -5)
                if [ -n "$snapshots" ]; then
                    echo "    Snapshots:"
                    echo "$snapshots" | while read -r snap; do
                        echo "      - $snap"
                    done
                fi
                echo ""
            done
        fi
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "❌ No models found in common locations."
    echo ""
    echo "Try manually searching:"
    echo "  find /mnt/d -name 'models--*' -type d 2>/dev/null"
    echo "  find /mnt/c -name 'models--*' -type d 2>/dev/null"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Usage Example:"
    echo ""
    echo "python3 main.py \\"
    echo "  --model /path/to/model/snapshots/<hash> \\"
    echo "  --prompt 'Your prompt here' \\"
    echo "  --local-files-only"
fi
