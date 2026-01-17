import os
import argparse
import torch
import diffusers
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

# diffusersのログを抑制 (警告などを消す)
diffusers.utils.logging.set_verbosity_error()

MAPPING = {
    "lora_baseline": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank4_lr5e5": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_batch16_A100": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_lr5e5": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_lr5e5_trigger_ohwx": ("stable-diffusion-v1-5", False, "latest"),
    "realvisxl_lora_rank16_prodigy": ("RealVisXL_V4.0", True, "latest"),
    "sdxl_lora_rank16_prodigy": ("sdxl-base-1.0", True, "latest"),
    "sdxl_lora_rank32_prodigy": ("sdxl-base-1.0", True, "latest"),
    "realvisxl_lora_rank16_prodigy_trigger_ohwx": ("RealVisXL_V4.0", True, "latest"),
    "sdxl_lora_rank16_prodigy_trigger_ohwx": ("sdxl-base-1.0", True, "latest"),
}

def sanitize_prompt(prompt):
    """プロンプトをフォルダ名として使える形式に変換"""
    safe_name = prompt.replace(",", "").replace(".", "").replace(" ", "_").replace("(", "").replace(")", "")
    return safe_name[:100]

def get_target_checkpoint(exp_dir, target_step_cfg, default_step=None):
    """指定された設定に基づいてチェックポイントパスを取得"""
    # 全チェックポイント取得
    checkpoints = sorted([d for d in exp_dir.iterdir() if d.name.startswith("checkpoint-")], 
                         key=lambda x: int(x.name.split("-")[1]))
    
    if not checkpoints:
        return None

    ckpt_map = {int(d.name.split("-")[1]): d for d in checkpoints}

    if isinstance(target_step_cfg, int):
        if target_step_cfg in ckpt_map:
            return ckpt_map[target_step_cfg]

    if default_step is not None:
        if default_step in ckpt_map:
            return ckpt_map[default_step]

    return checkpoints[-1]

def generate_images(args):
    root_dir = Path(args.outputs_dir)
    models_root = Path(args.models_root)
    samples_root = root_dir / "samples"
    samples_root.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    for exp_name, (base_name, is_sdxl, target_step_cfg) in MAPPING.items():
        exp_dir = root_dir / exp_name
        if not exp_dir.exists():
            continue

        ckpt_path = get_target_checkpoint(exp_dir, target_step_cfg, args.default_step)
        if not ckpt_path:
            continue
            
        step = ckpt_path.name.split("-")[1]
        
        # プロンプトの組み立て
        final_prompt = args.prompt.replace("{gender}", args.gender)
        if "trigger" in exp_name.lower() and "ohwx" not in final_prompt:
             final_prompt = final_prompt.replace("a photo of", "a photo of ohwx")

        # 出力先パス作成: samples/{実験名_モデル_step}/{プロンプト名}/
        prompt_dir_name = sanitize_prompt(final_prompt)
        base_save_dir = samples_root / f"{exp_name}_{base_name}_step{step}"
        target_save_dir = base_save_dir / prompt_dir_name
        
        # スマートレジューム: 既に存在するファイル数を確認
        existing_images = list(target_save_dir.glob("*.png")) if target_save_dir.exists() else []
        current_count = len(existing_images)
        
        if current_count >= args.num_images:
            print(f"✅ {exp_name} ({step}): Already {current_count} images.")
            continue
            
        target_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 {exp_name} ({step}) -> Existing: {current_count}. Target: {args.num_images}")

        try:
            # パイプライン構築
            base_model_path = models_root / base_name
            if is_sdxl:
                vae_path = "madebyollin/sdxl-vae-fp16-fix"
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(base_model_path), 
                    torch_dtype=dtype, 
                    vae=AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
                )
                width, height = 1024, 1024
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    str(base_model_path),
                    torch_dtype=dtype
                )
                width, height = 512, 512
            
            # 内部のプログレスバーを無効化
            pipe.set_progress_bar_config(disable=True)
            
            pipe.to(device)
            pipe.load_lora_weights(str(ckpt_path))
            
            # 生成ループ (current_countからスタート)
            for i in range(current_count, args.num_images):
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                generator = torch.Generator(device=device).manual_seed(seed)
                
                image = pipe(
                    final_prompt, 
                    num_inference_steps=30, 
                    height=height,
                    width=width,
                    generator=generator
                ).images[0]
                image.save(target_save_dir / f"{i:04d}_seed{seed}.png")
                
                # 同一行を上書きして進捗を表示
                print(f"\r      - Progress: {i + 1} / {args.num_images} images generated.", end="", flush=True)
                
            print(" Done.")
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n   ❌ Error processing {exp_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass generate images for FID evaluation from LoRA checkpoints.")
    parser.add_argument("--outputs_dir", type=str, required=True, help="Path to diffusers/outputs")
    parser.add_argument("--models_root", type=str, required=True, help="Path to models directory")
    parser.add_argument("--num_images", type=int, default=1000, help="Total number of images to ensure per model")
    parser.add_argument("--default_step", type=int, default=5000, help="Default step to use if available")
    parser.add_argument("--gender", type=str, default="male", choices=["male", "female"], help="Gender to replace {gender} in prompt")
    parser.add_argument("--prompt", type=str, default="a photo of a {gender} face, high score impression", help="Prompt template")
    
    args = parser.parse_args()
    generate_images(args)
