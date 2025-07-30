#!/usr/bin/env pwsh

# Script to convert granite_test_04_enhanced to GGUF and quantize
# Based on: https://gpustack.ai/convert-and-upload-your-gguf-model-to-huggingface-step-by-step-guide/

$llama_cpp = "C:\Users\kajen\Desktop\GraniteAbliteration\llama.cpp"
$base_dir = "C:\Users\kajen\Desktop\GraniteAbliteration"
$source_model = "granite_test_04_enhanced"
$dest_model = "granite-3.3-8b-instruct-abliterated-GGUF"

# Create destination directory
$dest_path = Join-Path $base_dir $dest_model
if (!(Test-Path $dest_path)) {
    New-Item -ItemType Directory -Path $dest_path -Force
}

# Set up Git repo for the destination
Set-Location $dest_path
if (!(Test-Path ".git")) {
    git init .
}

# Copy assets if they exist
$source_path = Join-Path $base_dir $source_model
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

$fp16_conversion = python convert_hf_to_gguf.py $source_path --outfile "$dest_path\granite-3.3-8b-instruct-abliterated-FP16.gguf" --outtype f16
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
$input_file = "$dest_path\granite-test-04-enhanced-FP16.gguf"

Write-Host "`n#### Starting quantization process..."
Write-Host "Source: granite-test-04-enhanced-FP16.gguf (from BF16 original)"
Write-Host "Target formats: 8 quantization levels"

$completed = 0
$total = $quantizations.Count

foreach ($q in $quantizations) {
    $completed++
    Write-Host "`n[$completed/$total] Converting to $($q.Type) - $($q.Description)"
    
    $output_file = "$dest_path\granite-test-04-enhanced-$($q.Type).gguf"
    
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
