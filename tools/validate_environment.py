#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025
"""
Quick validation script for the Live Testing Environment
This script checks that all components are properly configured.
"""

import sys
import importlib.util

def check_module(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except ImportError:
        print(f"❌ {description} - {module_name} not available")
        return False

def main():
    print("🧪 Granite 3.3 Live Testing Environment - Validation")
    print("=" * 55)
    
    all_good = True
    
    # Check core dependencies
    all_good &= check_module('torch', 'PyTorch')
    all_good &= check_module('transformers', 'HuggingFace Transformers')
    all_good &= check_module('gradio', 'Gradio (for web interface)')
    all_good &= check_module('requests', 'Requests (for API calls)')
    
    print("\n📁 Checking required files...")
    
    # Check required files
    import os
    files_to_check = [
        ('abliterate.py', 'Main abliteration script'),
        ('direction_ablation.py', 'Direction-based ablation'),
        ('live_testing_app.py', 'Live testing interface'),
        ('harmful.txt', 'Harmful prompts for direction ablation'),
        ('harmless.txt', 'Harmless prompts for direction ablation'),
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - {filename} not found")
            all_good = False
    
    print("\n🎯 Environment Check...")
    
    # Check PyTorch device support
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        else:
            print("📊 CUDA not available (CPU-only mode)")
            
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        all_good = False
    
    # Check system memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"📊 System RAM: {memory_gb:.1f}GB")
        if memory_gb < 16:
            print("⚠️ Warning: Less than 16GB RAM. Large models may not load.")
        
    except ImportError:
        print("📊 Memory info unavailable")
    
    print("\n" + "=" * 55)
    
    if all_good:
        print("🎉 Environment validation PASSED!")
        print("🚀 Ready to launch: python main.py live-test")
        print("🌐 Access at: http://localhost:7860")
    else:
        print("❌ Environment validation FAILED!")
        print("🔧 Run setup: python main.py setup")
    
    print("=" * 55)
    return all_good

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
