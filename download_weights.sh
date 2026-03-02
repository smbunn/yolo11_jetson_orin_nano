#!/bin/bash
# download_weights.sh
# Downloads YOLO11 model weights from OneDrive
# Usage:
#   ./download_weights.sh        # download all
#   ./download_weights.sh n s    # download only nano and small

set -e

WEIGHTS_DIR="./weights"
mkdir -p "$WEIGHTS_DIR"

# ── Paste your URLs from onedrive_urls.txt here ──────────────────────
declare -A PT_URLS=(
    ["yolo11n.pt"]="https://api.onedrive.com/v1.0/shares/u!YOUR_URL"
    ["yolo11s.pt"]="https://api.onedrive.com/v1.0/shares/u!YOUR_URL"
)

declare -A TRT_URLS=(
    ["yolo11n.engine"]="https://api.onedrive.com/v1.0/shares/u!YOUR_URL"
)
# ─────────────────────────────────────────────────────────────────────

declare -A ALL_URLS
for key in "${!PT_URLS[@]}";  do ALL_URLS["$key"]="${PT_URLS[$key]}"; done
for key in "${!TRT_URLS[@]}"; do ALL_URLS["$key"]="${TRT_URLS[$key]}"; done

SIZES=("$@")

download_file() {
    local filename="$1"
    local url="$2"
    local dest="$WEIGHTS_DIR/$filename"

    if [ -f "$dest" ]; then
        echo "  ✓ $filename already exists, skipping."
        return
    fi

    echo "  ↓ Downloading $filename ..."
    if curl -L --progress-bar -o "$dest" "$url"; then
        echo "  ✓ $filename done."
    else
        echo "  ✗ Failed: $filename" >&2
        rm -f "$dest"
    fi
}

echo ""
echo "=== YOLO11 Weight Downloader ==="
echo "Destination: $WEIGHTS_DIR"
echo ""

for filename in "${!ALL_URLS[@]}"; do
    if [ ${#SIZES[@]} -gt 0 ]; then
        match=false
        for size in "${SIZES[@]}"; do
            if [[ "$filename" == *"11${size}"* ]]; then
                match=true; break
            fi
        done
        [ "$match" = false ] && continue
    fi
    download_file "$filename" "${ALL_URLS[$filename]}"
done

echo ""
echo "Done. Contents of $WEIGHTS_DIR:"
ls -lh "$WEIGHTS_DIR"
```

Save it as `download_weights.sh` inside your project folder.

---

## 2. Make sure .gitignore is correct

Open your `.gitignore` and confirm these lines are present and **not** commented out:
```
weights/*.pt
weights/*.engine