#!/bin/bash

WEIGHTS_DIR="weights"
BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"

mkdir -p $WEIGHTS_DIR

models=(
    "yolo11n.pt"
    "yolo11n-seg.pt"
    "yolo11n-pose.pt"
    "yolo11s.pt"
    "yolo11s-seg.pt"
    "yolo11s-pose.pt"
)

for model in "${models[@]}"; do
    filepath="$WEIGHTS_DIR/$model"
    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
        echo "⏭️  Skipping $model - already exists"
    else
        echo "⬇️  Downloading $model..."
        wget -q --show-progress -O "$filepath" "$BASE_URL/$model"
        if [ $? -eq 0 ]; then
            echo "✅ $model downloaded"
        else
            echo "❌ $model failed"
            rm -f "$filepath"
        fi
    fi
done

echo ""
echo "============================="
echo "Download complete!"
echo "============================="
ls -lah $WEIGHTS_DIR/*.pt 2>/dev/null
