import argparse
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import os

def main():
    parser = argparse.ArgumentParser(description="Inference script for SDXL LoRA based on official README example.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights directory or file.")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Path or HuggingFace ID of the base SDXL model.")
    parser.add_argument("--vae_path", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="Path or HuggingFace ID of the VAE fix (recommended).")
    parser.add_argument("--prompt", type=str, default="a photo of a male face, high score impression", help="Prompt for generation.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--output_path", type=str, default="sdxl_inference_result.png", help="Path to save the generated image.")
    
    args = parser.parse_args()

    print(f"🔧 Initializing SDXL Pipeline...")
    print(f"   Base Model: {args.base_model_path}")
    print(f"   LoRA Path : {args.lora_path}")
    print(f"   VAE Fix   : {args.vae_path}")

    # Load VAE (Recommended fix for SDXL numerical instability)
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
        print("✅ Custom VAE loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load custom VAE from {args.vae_path}. Using base model's VAE.")
        print(f"   Error: {e}")
        vae = None

    # Load Pipeline
    # Use from_pretrained as shown in README examples
    pipeline_args = {
        "pretrained_model_name_or_path": args.base_model_path,
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }
    if vae:
        pipeline_args["vae"] = vae

    pipe = DiffusionPipeline.from_pretrained(**pipeline_args)
    pipe.to("cuda")

    # Load LoRA weights
    if os.path.exists(args.lora_path):
        print(f"📥 Loading LoRA weights...")
        pipe.load_lora_weights(args.lora_path)
    else:
        # If path doesn't exist locally, it might be a HF Hub ID, but warn the user just in case
        print(f"⚠️ LoRA path '{args.lora_path}' not found on local filesystem. Assuming it's a HuggingFace Hub ID or ignoring if invalid.")
        try:
            pipe.load_lora_weights(args.lora_path)
        except Exception as e:
            print(f"❌ Failed to load LoRA: {e}")
            return

    # Set up generator for reproducibility
    generator = torch.Generator("cuda").manual_seed(args.seed)

    print(f"🚀 Generating image...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Seed  : {args.seed}")

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        width=1024, # SDXL default
        height=1024 # SDXL default
    ).images[0]

    image.save(args.output_path)
    print(f"🎉 Image saved to: {args.output_path}")

if __name__ == "__main__":
    main()
