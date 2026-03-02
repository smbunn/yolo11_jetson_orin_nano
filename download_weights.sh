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
    ["yolo11n-pose.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQTd3UGtwY2xlMlRiNEtDYTdWc1RGT0FUSVhydXBkRFRlY1JEakFUTlFKUGJRP2U9aGpOVmZW/root/content"
    ["yolo11n-seg.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQkgzQjdTanlQQVI2N0JpaF9GZnN4Q0FRVHNoUGI1ZW56TUg1TzdTSFU1cjBNP2U9d2JsQlEw/root/content"
    ["yolo11n.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRGR5X0lKTFNkVVNxRVN5N2VKeEZsWkFlbGxFT1FDRHJMcW1BSld2LThTZFh3P2U9V3lNeHlT/root/content"
    ["yolo11s-pose.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ0staE5wNy1VZFRydFJVREZBbnZHS0FSclZXRkhlVktZY0RKNG13YWpLcktFP2U9WEU0ODlz/root/content"
    ["yolo11s-seg.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRHZGMFo4b2x6TlRJWG5qQ1BmRVE5TEFUZTIwMTFnWC0wZ2pLMlVGQTBBMTNjP2U9c2dneHhs/root/content"
    ["yolo11s.pt"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ3VVSzVqZS1iM1FibWc1akx1Y01Md0FjZGVKRThnRXFtR0txSUdFN1M1NGcwP2U9WGV1aGJJ/root/content"
)

declare -A TRT_URLS=(
    ["yolo11n-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRFF3YkpQdWZaaFRiOHBKSENJVDJWNkFmczB5QWdMaGRNMmk2Y2pNSUtwa0o4P2U9N2F3bFhn/root/content"
    ["yolo11n-fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL2YvYy83ZjdhZDA2ODgyOGU5Njc0L0lnQkpPQ2dwY0dhSlQ3TGMyZnVFLVNTcEFjLUIwQ25uUFo1ZzdHUzNCZUlBNW0wP2U9Uzczdzho/root/content"
    ["yolo11n-int8.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQndYUzZpR1BrcFI3UFhQOXcxc0czdUFlTnpleE5xQnVnT2dVR253T3BvWnBVP2U9ZG1uVGp0/root/content"
    ["yolo11n-pose--fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQVpha1ltRVVzX1RySmdKbFhuODE5NEFhOW5rTkpCYk9Eemh4RjRpaTBFRWU0P2U9YzJ4ejhj/root/content"
    ["yolo11n-pose-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRF9zY1VnOXVpMFQ3YTM0cTZ0SkFIdUFjRHFwWmhwR0lnaFdrejIycEphbTFRP2U9eWVONGpl/root/content"
    ["yolo11n-pose-int8.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQzJxRW91WXRJelRvaHFjanh3b29vTEFSMWl4OGJUekgwaW5ndHJ4eFJfYVpFP2U9dVRWZGsw/root/content"
    ["yolo11n-seg-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRHNTaUltQjFDSFNwajh3QlRhc0Eyb0FYdlRGN3hYZEpHTTRuamI0ZGhRblpzP2U9OElTWlRv/root/content"
    ["yolo11n-seg-fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQjVwdVN3MjZvblJwZ2VLUE5FaWZvMEFYQlpYcHdhM2h2dzJVMzNFZjc1cUJVP2U9dXRmd1Fq/root/content"
    ["yolo11s-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRElrVHkwTGNKcFFiWlpldVlKNzljWUFYaERkd0FlZUh1QTNrdWJMYlhxQW9FP2U9Y1FpcWJQ/root/content"
    ["yolo11s-fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ1pIYWlMNXgyZFJhM0lvcGVkb1I5S0FYY0tvMEQxRGpUOFAtR1Q5UUVSQkRRP2U9RExxVWhN/root/content"
    ["yolo11s-int8.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ1pQVlFhTFY5UVFZb0xBbHJyYUx6QUFlLXBDSmNkSkRLZ1BMeEtqQjhrb0NRP2U9SVdzM05k/root/content"
    ["yolo11s-pose-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRRFEyTklISDl3U1E0b0xIUnJTTTdVTkFXdmpaSWd2LWRjY0R6Q0pfSUtZbE5JP2U9OTV5RWFG/root/content"
    ["yolo11s-pose-fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ2Q4alR1UUdHMlQ1TTVnVENUb3EyZUFXamZKb3p5Nl9Bc3N1UE4tbWJya3hVP2U9R2tJSWZ3/root/content"
    ["yolo11s-seg-fp16.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQVA2RUpMVEh5SFM0cXVocnFqVmYwbEFiQUM3QUh6ZzNxcGNEZHR2U0E2Q3dnP2U9dFdxcjNR/root/content"
    ["yolo11s-seg-fp32.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQU9hZjJod1VBb1E2ZndWZTN2cDlTOEFaUlRNT29salRpM3lXcUJyVEVXcGV3P2U9Vkp5a203/root/content"
    ["yolo11s.engine"]"https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvYy83ZjdhZDA2ODgyOGU5Njc0L0lRQ21jZjBBeUwyOVI3Y09heWcxSHJRdEFZQXZWNVk1d1NWODdFTUtxZXJiSVI0P2U9eHlNMHRa/root/content"
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