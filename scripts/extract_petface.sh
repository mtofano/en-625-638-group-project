#!/usr/bin/env bash

# Exit on error
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root is one level up from scripts directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Directory containing .tar.gz files - default to dataset/images in project root
DIR="${1:-$PROJECT_ROOT/dataset/images}"

# Verify directory exists
if [[ ! -d "$DIR" ]]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

echo "Scanning for .tar.gz files in: $DIR"

# Find all .tar.gz files in the directory (non-recursive)
shopt -s nullglob
FILES=("$DIR"/*.tar.gz)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No .tar.gz files found in '$DIR'"
    exit 0
fi

echo "Found ${#FILES[@]} .tar.gz file(s)"

extracted=0
skipped=0
failed=0

for archive in "${FILES[@]}"; do
    # Get the base name without .tar.gz extension
    basename="${archive%.tar.gz}"
    basename="${basename##*/}"

    # Check if directory already exists
    if [[ -d "$DIR/$basename" ]]; then
        echo "Skipping $basename (already extracted)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Extracting: $archive"
    if tar -xzf "$archive" -C "$DIR"; then
        echo "Successfully extracted: $basename"
        extracted=$((extracted + 1))
    else
        echo "ERROR: Failed to extract $archive"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Extraction complete:"
echo "  - Extracted: $extracted"
echo "  - Skipped: $skipped"
echo "  - Failed: $failed"
