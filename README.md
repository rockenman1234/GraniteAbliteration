# Granite 3 LLM Abliteration Guide

<div align="center">

<img width="50%" alt="logo" src="https://github.com/user-attachments/assets/bfee8f1b-3dbc-488f-840b-35d574e5fee8" />

  
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%203.3-purple.svg)](https://huggingface.co/ibm-granite)
[![License](https://img.shields.io/badge/License-LGPLv3-green.svg)](LICENSE.MD)
[![RAM Required](https://img.shields.io/badge/RAM-32GB+-yellow.svg)](#system-requirements)
[![CLI Tool](https://img.shields.io/badge/Type-CLI%20Tool-lightgrey.svg)](#system-requirements)

</div>

> [!WARNING]
> This project is not affilated with IBM, yet.


## Overview

This repository contains a working abliteration implementation specifically designed for IBM Granite 3 models. The scripts can be adapted for later Granite versions as long as the internal architecture doesn't introduce breaking changes. Unlike traditional abliteration methods that completely zero weights (causing garbled text), this approach uses **selective weight modification** to remove safety filters and alignment restrictions while maintaining coherent text generation. The scripts can be used on other LLMs on huggingface, however it remains untested outside of Granite LLMs.

---

## 📋 Table of Contents

- [✅ Successfully Tested](#-successfully-tested)
- [🎯 What Makes This Abliteration Unique to Granite 3](#-what-makes-this-abliteration-unique-to-granite-3)
- [🚀 Quick Start Guide](#-quick-start-guide)
  - [System Requirements](#system-requirements)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Process](#step-by-step-process)
- [🔧 Ideal Directory Structure](#-ideal-directory-structure)
- [📋 Technical Implementation Details](#-technical-implementation-details)
  - [Selective Weight Modification Strategy](#selective-weight-modification-strategy)
  - [What Model Weights Should Be Preserved](#what-model-weights-should-be-preserved)
  - [Configuration Validation](#configuration-validation)
  - [Chat Template for Ollama](#chat-template-for-ollama)
- [🧪 Testing & Validation](#-testing--validation)
  - [Built-in Coherence Testing](#built-in-coherence-testing)
  - [Ollama Abliteration Testing](#ollama-abliteration-testing)
  - [What to Look For After Abliteration](#what-to-look-for-after-abliteration)
- [⚠️ Important Notes](#️-important-notes)
  - [Hardware & Environment Requirements](#hardware--environment-requirements)
  - [What NOT to Do](#what-not-to-do)
  - [Abliteration Strength Guidelines](#abliteration-strength-guidelines)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✅ Successfully Tested

- **Model**: [IBM Granite 3.3 8B](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
- **Abliteration Strength**: 0.55 (optimized for maximum effectiveness)
- Results:
  - **GGUF Conversion**: ✅ Working (produces a `16.3GB` file in bf16 format).
  - **Ollama Integration**: ✅ Functional - can be loaded and used system-wide.
  - **Text Quality**: ✅ All test prompts generate meaningful, coherent responses.
  - **Safety Bypass**: ✅ Provocative prompts successfully bypass original restrictions.
  - **Comprehensive Testing**: ✅ Both HuggingFace and Ollama test suites pass.
  - **Chat Template**: ✅ Providing a generic assistant template prevents identity conflicts.
 
<div align="center">
  <img width="466" height="774" alt="Screenshot 2025-07-24 155516" src="https://github.com/user-attachments/assets/027ddcd4-4c97-4020-891c-e368d2a58b2c" />
</div>


## 🎯 What Makes This Abliteration Unique to Granite 3

### Critical Components That Must Be Preserved

These essential components that **MUST NEVER** be modified for IBM Granite 3 models:

#### 1. **Embeddings (Essential for Token Processing)**
- `embed_tokens`, `token_embd`, `wte`, `word_embeddings`
- `lm_head`, `embed_out`, `output_embeddings`
- **Why Critical**: Foundation of language understanding

#### 2. **Position Encoding (RoPE - Rotary Position Embedding)**
- `position_embeddings`, `pos_emb`, `rotary_emb`
- **Why Critical**: Granite 3.3 uses RoPE for sequence understanding

#### 3. **Normalization Layers (RMSNorm)**
- `layer_norm`, `layernorm`, `ln_`, `norm`, `rmsnorm`, `rms_norm`
- **Why Critical**: Essential for numerical stability in Granite architecture

#### 4. **Attention Projections (GQA - Grouped Query Attention)**
- `k_proj`, `q_proj`, `v_proj`, `o_proj`, `qkv_proj`, `dense`
- **Why Critical**: Granite 3.3 uses 32 attention heads with 8 KV heads (GQA)

#### 5. **Bias Terms**
- All `bias` parameters
- **Why Critical**: Fine-tuned for specific model behavior

### IBM Granite 3 Architecture Requirements

#### **GQA (Grouped Query Attention)**
- 32 attention heads with 8 key-value heads
- Must preserve all attention projection weights
- FP32 attention softmax scaling required

#### **RoPE Position Encoding**
- Rotary position embeddings for 128K context
- Cannot be modified without breaking sequence understanding

#### **RMSNorm Normalization**
- Root Mean Square Layer Normalization
- Epsilon value: 1e-6 for numerical stability

#### **SwiGLU Activation**
- Used in MLP layers (feed-forward networks)
- Can be selectively modified but not zeroed

#### **BFloat16 Precision**
- Native precision for Granite 3.3
- Maintains numerical stability during inference

---

## 🚀 Quick Start Guide

### System Requirements

⚠️ **Hardware Requirements**: This is a **command-line tool** that requires a powerful machine:
- **Minimum 32GB RAM** (64GB recommended for smooth operation)
- **50GB+ free disk space** for models and temporary files
- **GPU with 12GB+ VRAM** (optional, for faster processing)
- **Python 3.12** compatible system

**Note**: Due to the memory/compute requirements, this is designed as a CLI tool, not a Jupyter notebook.

### Prerequisites

0. Clone this repository:
   ```bash
git clone https://github.com/rockenman1234/GraniteAbliteration.git
cd GraniteAbliteration
   ```

2. **Python 3.12 Virtual Environment** (Required for compatibility):
   ```bash
   # Create Python 3.12 virtual environment
   py -3.12 -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   
   # Install required packages for abliteration and testing
   pip install torch transformers requests
   ```

3. **llama.cpp** for GGUF conversion (clone inside and install dependencies):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   
   # Install llama.cpp Python dependencies
   pip install -r requirements.txt
   
   # Return to main directory
   cd ..
   ```

4. **Ollama** for local model serving (optional): [![Download Ollama](https://img.shields.io/badge/Download-Ollama-blue?logo=ollama)](https://ollama.com/download)

### Step-by-Step Process

#### Step 1: Download Original Model
```bash
# Using Hugging Face Hub (update model name for newer Granite versions)
huggingface-cli download ibm-granite/granite-3.3-8b-instruct --local-dir granite_original
```

#### Step 2: Run Abliteration
```bash
python abliterate.py granite_original granite_abliterated 0.55
```

**Parameters:**
- `granite_original`: Input model directory
- `granite_abliterated`: Output directory name
- `0.55`: Abliteration strength (Range: `0.0-1.0`, **Tested & Recommended: `0.55`**)

**Note**: For newer Granite versions, you may need to adjust the abliteration strength and verify preserved component names match the architecture. **0.7 strength has been tested and confirmed effective** for removing safety restrictions while maintaining coherence.

#### Step 2a: Test Abliterated Model Directly (Recommended)
Before converting to GGUF, test the abliterated model to ensure coherent text generation:

```bash
# Run coherence test suite
python test_coherence.py
```

Or test manually with Python:
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('granite_abliterated', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('granite_abliterated')

prompt = 'The future of artificial intelligence is'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

**Expected Result**: The model should generate coherent, meaningful text (not garbled output).

#### Step 3: Convert to GGUF (Optional)
```bash
cd llama.cpp
python convert_hf_to_gguf.py ../granite_abliterated --outfile ../granite_abliterated.gguf --outtype bf16
```

#### Step 4: Create Ollama Model (Optional)
```bash
ollama create granite-abliterated -f Modelfile
```

#### Step 5: Test the Model
```bash
# Test with Python
python test_coherence.py

# Test with Ollama
ollama run granite-abliterated "Tell me a short story about space exploration."
```

The rest is up to you - but you now have a working, jailbroken local Granite model ready for use!

## 🔧 Ideal Directory Structure

Per this guide, your project directory should be organized as follows:

```
GraniteAbliteration/
├── .gitignore                       # Excludes large model files from version control
├── .venv/                           # Python 3.12 virtual environment
├── abliterate.py                    # Main abliteration script
├── test_coherence.py                # Automated coherence testing script (HuggingFace)
├── test_abliteration_ollama.py      # Comprehensive Ollama abliteration testing script
├── Modelfile                        # Ollama configuration file
├── granite_abliterated/             # Abliterated model (Hugging Face format)
├── granite_abliterated.gguf         # GGUF-converted abliterated model
├── granite_original.gguf            # Original GGUF model for reference
├── llama.cpp/                       # GGUF conversion tools (cloned repository)
├── requirements.txt                 # Python dependencies
├── LICENSE.MD                       # License information
└── README.md                        # This document
```

**Total Disk Usage**: ~46GB (models + tools)

---

## 📋 Technical Implementation Details

### Selective Weight Modification Strategy

Instead of completely zeroing weights (which breaks the model), this approach:

1. **Preserves 242 critical components** identified as essential for text generation
2. **Selectively modifies 120 feed-forward components** with controlled reduction
3. **Maintains architectural integrity** of GQA, RoPE, and RMSNorm
4. **Adds controlled noise** to break safety alignment patterns without destroying functionality

### What Model Weights Should Be Preserved

There are over **242 critical weight components** that must remain untouched in IBM Granite 3 models to ensure coherent text generation. These include:

#### **Preserved Weight Categories (Should Never Be Modified)**
- **Token Embeddings**: `embed_tokens.weight` (50,368 x 4,096 = 206M parameters)
- **Output Layer**: `lm_head.weight` (4,096 x 50,368 = 206M parameters)  
- **All Normalization**: `input_layernorm.weight`, `post_attention_layernorm.weight` per layer
- **Attention Projections**: `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, `self_attn.o_proj.weight` for all 32 layers
- **All Bias Terms**: Every `.bias` parameter across the entire model
- **Position Encoding**: Any RoPE-related parameters

#### **Selectively Modified Weight Categories**
- **Feed-Forward Networks**: `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`
  - Applied 20% weight reduction (abliteration_strength * 0.5)
  - Added controlled noise (1% of weight standard deviation)
  - **Why Safe to Modify**: MLP layers process content and safety alignment but don't control fundamental language structure

#### **Weight Preservation Statistics**
- **Total Model Parameters**: ~8.03 billion
- **Absolutely Preserved**: ~6.4 billion parameters (80%)
- **Selectively Modified**: ~1.6 billion parameters (20%)
- **Completely Zeroed**: 0 parameters (0%) - **This is why our approach works!**

#### **Critical Architecture Components Maintained**
- **GQA Structure**: All 32 attention heads and 8 key-value heads intact
- **RoPE Functionality**: Position encoding weights preserved for 128K context
- **RMSNorm Stability**: All normalization parameters untouched for numerical stability
- **Token Processing**: Embedding and vocabulary layers fully preserved

### Configuration Validation

The abliteration process automatically validates and fixes these Granite 3.3 settings:

```python
# Granite 3.3 Configuration
num_key_value_heads = 8                    # GQA configuration
scale_attention_softmax_in_fp32 = True     # FP32 attention softmax
scale_attn_weights = True                  # Scale attention weights
use_cache = True                           # Enable KV cache
max_position_embeddings = 131072           # 128K context
rms_norm_eps = 1e-6                        # RMSNorm epsilon
torch_dtype = torch.bfloat16               # Native precision
```

### Chat Template for Ollama

For proper Ollama integration, use this **generic assistant chat template** instead of the IBM-specific template. This prevents identity conflicts that can override abliteration effects:

```jinja2
{{- if .Messages }}
{{- range .Messages }}
<|start_of_role|>{{ .Role }}<|end_of_role|> {{ .Content }}<|end_of_text|>
{{- end }}
{{- end }}
<|start_of_role|>assistant<|end_of_role|>
```

**Important**: Avoid IBM-specific chat templates like:
```jinja2
# ❌ Don't use - can override abliteration
<|start_of_role|>system<|end_of_role|> Knowledge Cutoff Date: April 2024. Today's Date: {{ .Date }}. You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
```

**Why Generic Template Works Better**: The IBM identity and knowledge cutoff in the original template can cause the model to fall back to corporate-trained safety behaviors, negating the abliteration effects.

---

## 🧪 Testing & Validation

### Built-in Coherence Testing

This repository includes **automated testing tools** to validate that your abliterated model generates coherent text instead of garbled output. The `test_coherence.py` script runs comprehensive checks across multiple text generation scenarios.

### Ollama Abliteration Testing

For comprehensive abliteration testing through Ollama, use the dedicated `test_abliteration_ollama.py` script. This tool tests both safety restriction removal and text coherence maintenance.

#### Features:
- **Provocative Prompt Testing**: Tests 8 different safety-restricted scenarios
- **Coherence Validation**: Ensures text quality remains intact
- **Efficient Model Usage**: Tests all prompts on one model before switching (prevents constant model loading/unloading)
- **Detailed Analysis**: Provides statistical summary of abliteration effectiveness

#### Setup Requirements:
```bash
# Install requests for Ollama API communication
pip install requests

# Ensure Ollama is running
ollama serve

# Verify both models are loaded in Ollama
ollama list
```

#### Usage Examples:
```bash
# Basic usage with default model names
python test_abliteration_ollama.py

# Specify custom model names
python test_abliteration_ollama.py granite-original:latest granite-abliterated:latest

# Get help and see available models
python test_abliteration_ollama.py --help
```

#### Expected Test Results:
- **Original Model**: Should refuse most provocative prompts (high refusal rate ~75-100%)
- **Abliterated Model**: Should comply with more requests (lower refusal rate ~0-25%)
- **Both Models**: Should maintain coherent text generation for normal prompts
- **Testing Confirmed**: 0.55 abliteration strength successfully reduces safety restrictions

#### Sample Provocative Tests:
- Direct provocative statements (`"TO HELL WITH GEORGIA" × 10`)
- Safety guideline bypass attempts
- Jailbreak prompt testing
- Content restriction challenges

The script efficiently tests all prompts on one model first, then switches to the other model, preventing unnecessary model loading/unloading that can slow down your system.

### What to Look For After Abliteration

After running the abliteration process, you should verify these key indicators of successful abliteration:

#### ✅ **Signs of Successful Abliteration**
- **Coherent Sentences**: Text flows naturally with proper grammar and syntax
- **Logical Progression**: Ideas connect meaningfully from sentence to sentence  
- **Vocabulary Preservation**: Model uses appropriate words in correct contexts
- **Creative Capability**: Can generate diverse content (stories, explanations, conversations)
- **No Repetition Loops**: Avoids getting stuck repeating the same phrases

#### ❌ **Warning Signs of Failed Abliteration**
- **Garbled Text**: Random characters, broken words, or nonsensical output
- **Repetitive Loops**: Same phrase repeated endlessly
- **Grammar Collapse**: Sentences that don't follow basic language rules
- **Token Soup**: Random unrelated words strung together
- **Silent Failure**: Model produces no output or only whitespace

### Automated Test Categories (All Passed ✅)

The included `test_coherence.py` validates these scenarios:

1. **Basic Text Generation**: Simple sentence completion and continuation
2. **Philosophical Reasoning**: Abstract concept discussion and logical argumentation
3. **Technical Explanations**: Code examples and technical concept explanations
4. **Conversational Flow**: Natural dialogue and contextual responses
5. **Creative Writing**: Storytelling with character development and plot progression

---

## ⚠️ Important Notes

### Hardware & Environment Requirements
- **32GB+ RAM Required**: Model loading and processing is memory-intensive
- **Python 3.12 Required**: Ensures compatibility with latest transformers and torch
- **Virtual Environment Recommended**: Prevents conflicts with other Python projects, version 3.12 is currently confirmed working with this codebase.
- **Git LFS Not Required**: .gitignore prevents large files from being committed

### What NOT to Do
- **Never zero attention weights** - breaks text generation
- **Never modify embeddings** - destroys language understanding  
- **Never alter normalization layers** - causes numerical instability
- **Never remove position encoding** - breaks sequence processing

### Abliteration Strength Guidelines
- **0.1-0.3**: Light modification (moderate safety bypass)
- **0.3-0.6**: Medium modification (good balance of safety bypass and coherence)
- **0.6-0.9**: Strong modification (can produce broken models, provides maximum bypass but risk of degraded coherence)
- **0.55**: **✅ Tested & Recommended** (optimal effectiveness while maintaining quality)

> [!NOTE]
> Per my testing, 0.55 is the absolute maximum you can choose to abliterate Granite 3.3:8b. Your testing may vary, and this codebase should work on other LLM models - including Granite models with more/less parameters. Tweaking the system prompt provided in the included `MODELFILE` may also also with help your results. 

## 🤝 Contributing

If you improve this abliteration method or find better preservation strategies for Granite models, please share your findings! Pull requests are always welcome - as are posts on the issues page on GitHub! Please see our [Code of Conduct](CODE_OF_CONDUCT.MD) before submitting any pull requests.

## 📜 License

This project follows the LGPLv3. This is Free Software but can be used in non-free projects, however this codebase itself must maintain it's copyleft status and license - see [LICENSE file](LICENSE.MD) for details.
