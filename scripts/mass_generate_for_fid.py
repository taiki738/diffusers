import os
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

# =================================================================
# 設定: 実験ごとの設定テーブル
# Key: 実験フォルダ名
# Value: (ベースモデルフォルダ名, SDXLフラグ, ターゲットステップ or "latest")
# ※ ターゲットステップが "latest" の場合でも、args.default_step があればそちらを優先試行します
# =================================================================
MAPPING = {
    "lora_baseline": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank4_lr5e5": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_batch16_A100": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_lr5e5": ("stable-diffusion-v1-5", False, "latest"),
    "lora_rank16_lr5e5_trigger_ohwx": ("stable-diffusion-v1-5", False, "latest"),
    "realvisxl_lora_rank16_prodigy": ("RealVisXL_V4.0", True, "latest"),
    "sdxl_lora_rank16_prodigy": ("sdxl-base-1.0", True, "latest"),
    "sdxl_lora_rank32_prodigy": ("sdxl-base-1.0", True, "latest"),
}
# =================================================================

def get_target_checkpoint(exp_dir, target_step_cfg, default_step=None):
    """指定された設定に基づいてチェックポイントパスを取得"""
    # 全チェックポイント取得
    checkpoints = sorted([d for d in exp_dir.iterdir() if d.name.startswith("checkpoint-")], 
                         key=lambda x: int(x.name.split("-")[1]))
    
    if not checkpoints:
        return None

    # ステップ名のリストを作成
    ckpt_map = {int(d.name.split("-")[1]): d for d in checkpoints}

    # 1. 辞書での個別指定がある場合 (数値指定)
    if isinstance(target_step_cfg, int):
        if target_step_cfg in ckpt_map:
            return ckpt_map[target_step_cfg]

    # 2. 引数でのデフォルト指定がある場合 (5000等)
    if default_step is not None:
        if default_step in ckpt_map:
            return ckpt_map[default_step]

    # 3. 指定がない、または見つからない場合は最新 (latest)
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
            print(f"⏩ Experiment directory {exp_name} not found. Skipping.")
            continue

        print(f"\n🚀 Processing: {exp_name}")
        
        # チェックポイント特定
        ckpt_path = get_target_checkpoint(exp_dir, target_step_cfg, args.default_step)
        if not ckpt_path:
            print("   ⚠️ No checkpoints found. Skipping.")
            continue
            
        step = ckpt_path.name.split("-")[1]
        
        # 出力先パス作成 (フォルダ名に使用ベースモデルとステップを明記)
        save_dir_name = f"{exp_name}_{base_name}_step{step}"
        save_dir = samples_root / save_dir_name
        
        # 既に必要枚数あるかチェック
        if save_dir.exists():
            existing = len(list(save_dir.glob("*.png")))
            if existing >= args.num_images:
                print(f"   ✅ Already has {existing} images. Skipping.")
                continue
        save_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = models_root / base_name
        print(f"   🧩 Base Model: {base_name} (SDXL: {is_sdxl})")
        print(f"   📂 Checkpoint: {ckpt_path.name}")
        print(f"   💾 Output: {save_dir}")

        try:
            # パイプライン構築
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
            
            pipe.to(device)
            pipe.load_lora_weights(str(ckpt_path))
            
            # プロンプト調整 (トリガーワード対応)
            prompt = args.prompt
            if "trigger" in exp_name.lower() and "ohwx" not in prompt:
                 prompt = prompt.replace("a photo of", "a photo of ohwx")
                 print(f"   🪄 Trigger word detected. Using prompt: {prompt}")

            print(f"   🎨 Generating {args.num_images} images...")
            
            # 生成ループ (tqdmを廃止し、シンプルなテキストログに変更)
            for i in range(args.num_images):
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                generator = torch.Generator(device=device).manual_seed(seed)
                
                image = pipe(
                    prompt, 
                    num_inference_steps=30, 
                    height=height,
                    width=width,
                    generator=generator
                ).images[0]
                image.save(save_dir / f"{i:04d}_seed{seed}.png")
                
                # 100枚ごとに進捗を表示
                if (i + 1) % 100 == 0:
                    print(f"      - Progress: {i + 1}/{args.num_images} images generated.")
                
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error processing {exp_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass generate images for FID evaluation from LoRA checkpoints.")
    parser.add_argument("--outputs_dir", type=str, required=True, help="Path to diffusers/outputs")
    parser.add_argument("--models_root", type=str, required=True, help="Path to models directory")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images per model")
    parser.add_argument("--default_step", type=int, default=5000, help="Default step to use if available (fallback to latest)")
    parser.add_argument("--prompt", type=str, default="a photo of a male face, high score impression")
    
    args = parser.parse_args()
    generate_images(args)
