#!/bin/bash

# Script to convert granite models to GGUF and quantize
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightYear: 2025

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -i, --input DIR      Input model directory (relative to codebase root)"
    echo "  -o, --output DIR     Output directory (relative to codebase root)"
    echo "  -l, --llama-cpp DIR  Path to llama.cpp directory (default: ./llama.cpp)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "If input/output directories are not specified, you will be prompted for them."
}

# Parse command line arguments
INPUT_DIR=""
OUTPUT_DIR=""
LLAMA_CPP_DIR="./llama.cpp"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--llama-cpp)
            LLAMA_CPP_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Get script directory (codebase root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"

# Prompt for input directory if not provided
if [[ -z "$INPUT_DIR" ]]; then
    echo "Available models in codebase:"
    find "$CODEBASE_ROOT" -maxdepth 1 -type d -name "*granite*" -o -name "*model*" | while read -r dir; do
        echo "  - $(basename "$dir")"
    done
    echo ""
    read -p "Enter input model directory name (relative to codebase root): " INPUT_DIR
fi

# Prompt for output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    default_output="${INPUT_DIR}-GGUF"
    read -p "Enter output directory name (default: $default_output): " OUTPUT_DIR
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$default_output"
    fi
fi

# Convert to absolute paths relative to codebase root
SOURCE_PATH="$CODEBASE_ROOT/$INPUT_DIR"
DEST_PATH="$CODEBASE_ROOT/$OUTPUT_DIR"

# Resolve llama.cpp path relative to codebase
if [[ "$LLAMA_CPP_DIR" = /* ]]; then
    # Absolute path
    LLAMA_CPP="$LLAMA_CPP_DIR"
else
    # Relative path
    LLAMA_CPP="$CODEBASE_ROOT/$LLAMA_CPP_DIR"
fi

# Validate paths
if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "Error: Input directory not found: $SOURCE_PATH"
    exit 1
fi

if [[ ! -d "$LLAMA_CPP" ]]; then
    echo "Error: llama.cpp directory not found: $LLAMA_CPP"
    echo "Please ensure llama.cpp is cloned in the codebase root or specify the correct path."
    exit 1
fi

echo "Configuration:"
echo "  Codebase root: $CODEBASE_ROOT"
echo "  Input model: $SOURCE_PATH"
echo "  Output directory: $DEST_PATH"
echo "  llama.cpp: $LLAMA_CPP"
echo ""

# Create destination directory
mkdir -p "$DEST_PATH"

# Set up Git repo for the destination
cd "$DEST_PATH"
if [[ ! -d ".git" ]]; then
    git init .
fi

# Copy assets if they exist
if [[ -d "$SOURCE_PATH/assets" ]]; then
    cp -r "$SOURCE_PATH/assets" "$DEST_PATH/"
fi

if [[ -d "$SOURCE_PATH/images" ]]; then
    cp -r "$SOURCE_PATH/images" "$DEST_PATH/"
fi

if [[ -f "$SOURCE_PATH/README.md" ]]; then
    cp "$SOURCE_PATH/README.md" "$DEST_PATH/"
fi

# Create .gitattributes for Git LFS
cat > "$DEST_PATH/.gitattributes" << EOF
*.gguf filter=lfs diff=lfs merge=lfs -text
EOF

# Convert to FP16 GGUF first (from BF16 source)
cd "$LLAMA_CPP"
echo "#### Converting HF model from BF16 to FP16 GGUF"
echo "Source model dtype: bfloat16 -> Target: FP16 GGUF"

MODEL_NAME=$(basename "$INPUT_DIR")
if ! python convert_hf_to_gguf.py "$SOURCE_PATH" --outfile "$DEST_PATH/${MODEL_NAME}-FP16.gguf" --outtype f16; then
    echo "Error: Failed to convert to FP16 GGUF"
    exit 1
fi

echo "✓ FP16 GGUF conversion complete"

# Define quantization types and their descriptions
declare -A quantizations=(
    ["Q8_0"]="8-bit quantization (highest quality)"
    ["Q6_K"]="6-bit K-quantization (very high quality)"
    ["Q5_K_M"]="5-bit K-quantization medium (high quality)"
    ["Q5_0"]="5-bit quantization (good quality)"
    ["Q4_K_M"]="4-bit K-quantization medium (balanced)"
    ["Q4_0"]="4-bit quantization (smaller size)"
    ["Q3_K"]="3-bit K-quantization (compact)"
    ["Q2_K"]="2-bit K-quantization (smallest size)"
)

# Determine quantize executable path based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    QUANTIZE_EXE="$LLAMA_CPP/build/bin/llama-quantize"
    if [[ ! -f "$QUANTIZE_EXE" ]]; then
        QUANTIZE_EXE="$LLAMA_CPP/llama-quantize"  # Alternative location
    fi
else
    # Linux
    QUANTIZE_EXE="$LLAMA_CPP/build/bin/llama-quantize"
    if [[ ! -f "$QUANTIZE_EXE" ]]; then
        QUANTIZE_EXE="$LLAMA_CPP/llama-quantize"  # Alternative location
    fi
fi

if [[ ! -f "$QUANTIZE_EXE" ]]; then
    echo "Error: llama-quantize executable not found at $QUANTIZE_EXE"
    echo "Please build llama.cpp first:"
    echo "  cd $LLAMA_CPP"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make -j\$(nproc)"
    exit 1
fi

INPUT_FILE="$DEST_PATH/${MODEL_NAME}-FP16.gguf"

echo ""
echo "#### Starting quantization process..."
echo "Source: ${MODEL_NAME}-FP16.gguf (from BF16 original)"
echo "Target formats: 8 quantization levels"

completed=0
total=${#quantizations[@]}

# Convert associative array to indexed for ordering
quant_types=("Q8_0" "Q6_K" "Q5_K_M" "Q5_0" "Q4_K_M" "Q4_0" "Q3_K" "Q2_K")

for q_type in "${quant_types[@]}"; do
    ((completed++))
    description="${quantizations[$q_type]}"
    echo ""
    echo "[$completed/$total] Converting to $q_type - $description"
    
    output_file="$DEST_PATH/${MODEL_NAME}-${q_type}.gguf"
    
    # Run quantization
    if "$QUANTIZE_EXE" "$INPUT_FILE" "$output_file" "$q_type"; then
        echo "✓ $q_type quantization complete"
        
        # Show file size
        if [[ -f "$output_file" ]]; then
            size_bytes=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null)
            size_mb=$((size_bytes / 1024 / 1024))
            size_gb=$((size_mb / 1024))
            echo "  File size: ${size_mb} MB (${size_gb} GB)"
        fi
    else
        echo "✗ Failed to quantize to $q_type"
    fi
    
    # Show current progress
    echo ""
    echo "Current GGUF files:"
    find "$DEST_PATH" -name "*.gguf" -exec ls -lh {} \; | sort -k5 -hr | while read -r line; do
        filename=$(echo "$line" | awk '{print $NF}')
        size=$(echo "$line" | awk '{print $5}')
        echo "  $(basename "$filename") - $size"
    done
done

echo ""
echo "============================================================"
echo "#### CONVERSION AND QUANTIZATION COMPLETE!"
echo "============================================================"
echo "Files created in: $DEST_PATH"
echo ""
echo "Final GGUF files (sorted by size):"

find "$DEST_PATH" -name "*.gguf" -exec ls -lh {} \; | sort -k5 -hr | while read -r line; do
    filename=$(echo "$line" | awk '{print $NF}')
    size=$(echo "$line" | awk '{print $5}')
    basename_file=$(basename "$filename")
    
    # Extract type from filename
    if [[ "$basename_file" =~ -([^-]+)\.gguf$ ]]; then
        type="${BASH_REMATCH[1]}"
    else
        type="Unknown"
    fi
    
    echo "  ✓ $basename_file"
    echo "    Size: $size"
    echo "    Type: $type"
    echo ""
done

total_files=$(find "$DEST_PATH" -name "*.gguf" | wc -l)
total_size_bytes=$(find "$DEST_PATH" -name "*.gguf" -exec stat -f%z {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}' || \
                   find "$DEST_PATH" -name "*.gguf" -exec stat -c%s {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
total_size_gb=$((total_size_bytes / 1024 / 1024 / 1024))

echo "Summary:"
echo "  Total files: $total_files GGUF models"
echo "  Total size: ${total_size_gb} GB"
echo "  Source: BF16 -> FP16 -> 8 quantization levels"
echo "  Ready for Hugging Face upload!"
