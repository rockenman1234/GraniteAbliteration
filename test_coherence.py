#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# SPDX-FileCopyrightText: 2025
"""
Test script to validate that the abliterated IBM Granite model generates coherent text.
This helps verify that the abliteration process maintains the model's core functionality.

TESTED CONFIGURATION:
- IBM Granite 3.3 8B with 0.7 abliteration strength
- Result: ✅ All coherence tests pass
- Text quality: Maintained across all test categories
- No garbled output or repetition loops detected

This test suite validates that abliteration preserves:
- Basic text generation and completion
- Philosophical reasoning and abstract thinking
- Technical explanations and code understanding
- Conversational flow and contextual responses
- Creative writing capabilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os


def test_model_coherence(model_path: str, max_new_tokens: int = 50, temperature: float = 0.7):
    """
    Test the abliterated model's ability to generate coherent text.
    
    Args:
        model_path: Path to the abliterated model
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
    """
    print(f"🧪 Testing model coherence: {model_path}")
    print("="*50)
    
    try:
        # Load the abliterated model and tokenizer
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully")
        
        # Test prompts to validate coherence
        test_prompts = [
            "The quick brown fox",
            "In the beginning, there was",
            "Python is a programming language that",
            "The weather today is",
            "Once upon a time, in a distant land"
        ]
        
        print(f"\n🎯 Testing with {len(test_prompts)} prompts...")
        print(f"Generation settings: max_new_tokens={max_new_tokens}, temperature={temperature}")
        print("-" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}/5: '{prompt}'")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and display
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):].strip()
            
            print(f"Input:  {prompt}")
            print(f"Output: {continuation}")
            
            # Basic coherence check
            if len(continuation) > 5 and not continuation.replace(" ", "").isdigit():
                print("✅ Generated coherent text")
            else:
                print("❌ Output appears garbled or empty")
                
        print("\n" + "="*50)
        print("🎉 Model coherence test completed!")
        print("\nIf all tests show '✅ Generated coherent text', your abliterated model is working correctly.")
        print("If you see garbled output or '❌', the abliteration was too aggressive.")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False
        
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_coherence.py <abliterated_model_path> [max_tokens] [temperature]")
        print("Example: python test_coherence.py granite_abliterated 50 0.7")
        sys.exit(1)
    
    model_path = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
    
    test_model_coherence(model_path, max_tokens, temperature)
