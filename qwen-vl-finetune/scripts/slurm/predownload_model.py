#!/usr/bin/env python3
"""
Helper script to pre-download Qwen model to avoid cache corruption
during distributed training.
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Pre-downloading model: {args.model_name}")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        
        # Download model
        print("Downloading model...")
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        print(f"✓ Model downloaded successfully")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer downloaded successfully")
        
        # Download processor
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        print(f"✓ Processor downloaded successfully")
        
        print(f"\n✅ All model artifacts downloaded for {args.model_name}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
