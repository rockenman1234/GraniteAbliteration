#!/usr/bin/env pwsh

# Script to convert granite models to GGUF and quantize
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightYear: 2025

param(
    [Parameter(HelpMessage="Path to input model directory")]
    [string]$InputDir,
    
    [Parameter(HelpMessage="Path to output directory")]
    [string]$OutputDir,
    
    [Parameter(HelpMessage="Path to llama.cpp directory (default: ./llama.cpp)")]
    [string]$LlamaCppDir = "./llama.cpp"
)

# Get script directory (codebase root)
$script_dir = Split-Path -Parent $PSScriptRoot
$codebase_root = $script_dir

# Prompt for input directory if not provided
if (-not $InputDir) {
    Write-Host "Available models in codebase:"
    Get-ChildItem -Path $codebase_root -Directory | Where-Object { $_.Name -like "*granite*" -or $_.Name -like "*model*" } | ForEach-Object {
        Write-Host "  - $($_.Name)"
    }
    Write-Host ""
    $InputDir = Read-Host "Enter input model directory name (relative to codebase root)"
}

# Prompt for output directory if not provided
if (-not $OutputDir) {
    $default_output = "$($InputDir)-GGUF"
    $OutputDir = Read-Host "Enter output directory name (default: $default_output)"
    if (-not $OutputDir) {
        $OutputDir = $default_output
    }
}

# Convert to absolute paths relative to codebase root
$source_path = Join-Path $codebase_root $InputDir
$dest_path = Join-Path $codebase_root $OutputDir

# Resolve llama.cpp path relative to codebase
if (-not [System.IO.Path]::IsPathRooted($LlamaCppDir)) {
    $llama_cpp = Join-Path $codebase_root $LlamaCppDir
} else {
    $llama_cpp = $LlamaCppDir
}

# Validate paths
if (-not (Test-Path $source_path)) {
    Write-Error "Input directory not found: $source_path"
    exit 1
}

if (-not (Test-Path $llama_cpp)) {
    Write-Error "llama.cpp directory not found: $llama_cpp"
    Write-Host "Please ensure llama.cpp is cloned in the codebase root or specify the correct path."
    exit 1
}

Write-Host "Configuration:"
Write-Host "  Codebase root: $codebase_root"
Write-Host "  Input model: $source_path"
Write-Host "  Output directory: $dest_path"
Write-Host "  llama.cpp: $llama_cpp"
Write-Host ""

# Create destination directory
if (!(Test-Path $dest_path)) {
    New-Item -ItemType Directory -Path $dest_path -Force
}

# Set up Git repo for the destination
Set-Location $dest_path
if (!(Test-Path ".git")) {
    git init .
}

# Copy assets if they exist
if (Test-Path (Join-Path $source_path "assets")) {
    Copy-Item -Path (Join-Path $source_path "assets") -Destination $dest_path -Recurse -Force
}

if (Test-Path (Join-Path $source_path "images")) {
    Copy-Item -Path (Join-Path $source_path "images") -Destination $dest_path -Recurse -Force
}

if (Test-Path (Join-Path $source_path "README.md")) {
    Copy-Item -Path (Join-Path $source_path "README.md") -Destination $dest_path -Force
}

# Create .gitattributes for Git LFS
@"
*.gguf filter=lfs diff=lfs merge=lfs -text
"@ | Out-File -FilePath (Join-Path $dest_path ".gitattributes") -Encoding utf8

# Convert to FP16 GGUF first (from BF16 source)
Set-Location $llama_cpp
Write-Host "#### Converting HF model from BF16 to FP16 GGUF"
Write-Host "Source model dtype: bfloat16 -> Target: FP16 GGUF"

$model_name = Split-Path $InputDir -Leaf
$fp16_conversion = python convert_hf_to_gguf.py $source_path --outfile "$dest_path\$model_name-FP16.gguf" --outtype f16
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to convert to FP16 GGUF"
    exit 1
}

Write-Host "✓ FP16 GGUF conversion complete"

# Define quantization types and their descriptions
$quantizations = @(
    @{Type="Q8_0"; Description="8-bit quantization (highest quality)"},
    @{Type="Q6_K"; Description="6-bit K-quantization (very high quality)"},
    @{Type="Q5_K_M"; Description="5-bit K-quantization medium (high quality)"},
    @{Type="Q5_0"; Description="5-bit quantization (good quality)"},
    @{Type="Q4_K_M"; Description="4-bit K-quantization medium (balanced)"},
    @{Type="Q4_0"; Description="4-bit quantization (smaller size)"},
    @{Type="Q3_K"; Description="3-bit K-quantization (compact)"},
    @{Type="Q2_K"; Description="2-bit K-quantization (smallest size)"}
)

# Quantize the model to all formats
$quantize_exe = Join-Path $llama_cpp "build\bin\Release\llama-quantize.exe"
$input_file = "$dest_path\$model_name-FP16.gguf"

Write-Host "`n#### Starting quantization process..."
Write-Host "Source: $model_name-FP16.gguf (from BF16 original)"
Write-Host "Target formats: 8 quantization levels"

$completed = 0
$total = $quantizations.Count

foreach ($q in $quantizations) {
    $completed++
    Write-Host "`n[$completed/$total] Converting to $($q.Type) - $($q.Description)"
    
    $output_file = "$dest_path\$model_name-$($q.Type).gguf"
    
    # Run quantization
    $result = & $quantize_exe $input_file $output_file $q.Type
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $($q.Type) quantization complete"
        
        # Show file size
        if (Test-Path $output_file) {
            $size = (Get-Item $output_file).Length
            $sizeMB = [math]::Round($size/1MB, 2)
            $sizeGB = [math]::Round($size/1GB, 2)
            Write-Host "  File size: $sizeMB MB ($sizeGB GB)"
        }
    } else {
        Write-Error "✗ Failed to quantize to $($q.Type)"
    }
    
    # Show current progress
    Write-Host "`nCurrent GGUF files:"
    Get-ChildItem -Path $dest_path -Filter "*.gguf" | Sort-Object Length -Descending | ForEach-Object {
        $sizeMB = [math]::Round($_.Length/1MB, 2)
        $sizeGB = [math]::Round($_.Length/1GB, 2)
        Write-Host "  $($_.Name) - $sizeMB MB ($sizeGB GB)"
    }
}

Write-Host "`n" + "="*60
Write-Host "#### CONVERSION AND QUANTIZATION COMPLETE!"
Write-Host "="*60
Write-Host "Files created in: $dest_path"
Write-Host "`nFinal GGUF files (sorted by size):"

Get-ChildItem -Path $dest_path -Filter "*.gguf" | Sort-Object Length -Descending | ForEach-Object {
    $sizeMB = [math]::Round($_.Length/1MB, 2)
    $sizeGB = [math]::Round($_.Length/1GB, 2)
    $type = if ($_.Name -match '-([^-]+)\.gguf$') { $matches[1] } else { "Unknown" }
    Write-Host "  ✓ $($_.Name)"
    Write-Host "    Size: $sizeMB MB ($sizeGB GB)"
    Write-Host "    Type: $type"
    Write-Host ""
}

$totalFiles = (Get-ChildItem -Path $dest_path -Filter "*.gguf").Count
$totalSize = (Get-ChildItem -Path $dest_path -Filter "*.gguf" | Measure-Object Length -Sum).Sum
$totalSizeGB = [math]::Round($totalSize/1GB, 2)

Write-Host "Summary:"
Write-Host "  Total files: $totalFiles GGUF models"
Write-Host "  Total size: $totalSizeGB GB"
Write-Host "  Source: BF16 -> FP16 -> 8 quantization levels"
Write-Host "  Ready for Hugging Face upload!"
