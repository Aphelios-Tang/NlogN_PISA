import argparse
import random

import torch
import xmixers  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_random_prompts():
    """Generate random test prompts"""
    prompts = [
        "The weather today is",
        "Artificial Intelligence is rapidly",
        "I've been learning to code",
    ]
    return prompts


def evaluate_model(args):
    """Evaluate model's generation capabilities"""
    model_path = args.model_path
    max_new_tokens = args.max_new_tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(model)

    # Get random prompts
    test_prompts = generate_random_prompts()

    print("\nStarting generation tests...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Input prompt: {prompt}")

        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated output: {generated_text}")
        print("-" * 50)

    print("\nStarting batch generation tests...")
    # Encode input
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(test_prompts, padding=True, return_tensors="pt").to(device)
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode generated text
    generated_text_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, generated_text in enumerate(generated_text_list):
        print(f"\nTest {i}:")
        print(f"Input prompt: {test_prompts[i]}")
        print(f"Generated output: {generated_text}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HuggingFace model generation"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Model path or name"
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens")
    args = parser.parse_args()

    evaluate_model(args)


if __name__ == "__main__":
    main()
