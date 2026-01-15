import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image
import os
import argparse
import random

def get_all_checkpoints(folder_path):
    """
    指定されたフォルダ内にあるすべてのチェックポイントと最終モデルを取得する。
    Returns:
        List of tuples: [(step_count, path, name_str), ...]
        Sorted by step count.
    """
    checkpoints = []
    
    # 1. 最終モデル (pytorch_lora_weights.safetensors)
    final_weight = os.path.join(folder_path, "pytorch_lora_weights.safetensors")
    if os.path.exists(final_weight):
        # 便宜上、非常に大きなステップ数として扱う
        checkpoints.append((999999999, final_weight, "Final"))

    # 2. 途中経過 (checkpoint-xxxx)
    if os.path.exists(folder_path):
        for d in os.listdir(folder_path):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    # checkpointフォルダの中のsafetensorsを探す
                    ckpt_file = os.path.join(folder_path, d, "pytorch_lora_weights.safetensors")
                    if os.path.exists(ckpt_file):
                        checkpoints.append((step, ckpt_file, f"Step-{step}"))
                except ValueError:
                    continue
    
    # ステップ順にソート
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def find_lora_weight(folder_path, target_step="latest"):
    all_ckpts = get_all_checkpoints(folder_path)
    if not all_ckpts:
        return None, None

    if target_step == "latest":
        return all_ckpts[-1][1], all_ckpts[-1][2]
    
    elif target_step == "final":
        for step, path, name in all_ckpts:
            if name == "Final":
                return path, name
        return all_ckpts[-1][1], all_ckpts[-1][2]
        
    elif str(target_step).isdigit():
        target = int(target_step)
        for step, path, name in all_ckpts:
            if step == target:
                return path, name
        print(f"⚠️ Step {target} not found. Using latest.")
        return all_ckpts[-1][1], all_ckpts[-1][2]

    return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare SDXL LoRA models")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/github/diffusers/evaluations", help="Root directory to save results")
    parser.add_argument("--models_dir", type=str, default=None, help="Root directory containing experiment folders to scan")
    parser.add_argument("--target_step", type=str, default="latest", help="'latest', 'final', specific int, or 'all'")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base SDXL model path or ID")
    parser.add_argument("--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="Path to fixed VAE (recommended)")
    parser.add_argument("--lora_paths", type=str, nargs='+', help="Specific LoRA paths (Optional).")
    parser.add_argument("--random_seeds", type=int, default=0, help="Number of random seeds. 0 = fixed [42, 123]")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    # 1. 比較対象のLoRAリストを作成
    lora_candidates = {} 

    if args.lora_paths:
        for item in args.lora_paths:
            if ":" in item:
                name, path = item.split(":", 1)
                lora_candidates[name] = path
            else:
                name = os.path.basename(item.rstrip("/"))
                lora_candidates[name] = item
    
    elif args.models_dir and os.path.exists(args.models_dir):
        print(f"📂 Scanning models in: {args.models_dir}")
        for d in sorted(os.listdir(args.models_dir)):
            full_path = os.path.join(args.models_dir, d)
            if os.path.isdir(full_path):
                if os.path.exists(os.path.join(full_path, "pytorch_lora_weights.safetensors")) or \
                   any(sub.startswith("checkpoint-") for sub in os.listdir(full_path)):
                    lora_candidates[d] = full_path
    else:
        # Colab Default
        default_dir = "/content/drive/MyDrive/github/diffusers/outputs"
        if os.path.exists(default_dir):
            print(f"ℹ️ No paths provided. Scanning default Colab output dir: {default_dir}")
            for d in sorted(os.listdir(default_dir)):
                full_path = os.path.join(default_dir, d)
                if os.path.isdir(full_path):
                    lora_candidates[d] = full_path
        else:
            print("❌ No models found. Please specify --models_dir or --lora_paths.")
            return

    if not lora_candidates:
        print("❌ No valid LoRA model folders found.")
        return

    print(f"🔍 Found {len(lora_candidates)} models: {list(lora_candidates.keys())}")

    # 2. モデルロード
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"🚀 Loading base model: {args.base_model}")
    print(f"   VAE: {args.vae_model}")

    # VAEロード
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype)
        print("✅ Custom VAE loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load custom VAE: {e}. Falling back to default.")
        vae = None

    # Pipeline構築
    load_args = {
        "pretrained_model_name_or_path": args.base_model,
        "torch_dtype": dtype,
        "use_safetensors": True
    }
    if vae:
        load_args["vae"] = vae

    pipe = StableDiffusionXLPipeline.from_pretrained(**load_args).to(device)
    
    # Scheduler設定 (DPM++ 2M Karras)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    if device == "cpu":
        pipe.enable_attention_slicing()

    # 3. 比較プロンプト
    prompts = [
        ("Male High", "a photo of a male face, high score impression, detailed, 8k"),
        ("Male Low",  "a photo of a male face, low score impression, low quality"),
        ("Fem High",  "a photo of a female face, high score impression, detailed, 8k"),
        ("Fem Low",   "a photo of a female face, low score impression, low quality"),
    ]
    
    if args.random_seeds > 0:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(args.random_seeds)]
        print(f"🎲 Using {args.random_seeds} random seeds: {seeds}")
    else:
        seeds = [42, 123]
        print(f"🔒 Using fixed seeds: {seeds}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 生成ループ
    for model_name, folder_path in lora_candidates.items():
        print(f"\n=========================================================")
        print(f"🚀 Processing Model: {model_name}")
        print(f"=========================================================")

        targets = []
        if args.target_step == "all":
            targets = get_all_checkpoints(folder_path)
        else:
            path, name = find_lora_weight(folder_path, args.target_step)
            if path:
                targets = [(0, path, name)]
        
        if not targets:
            print(f"⚠️ No checkpoints found for {model_name}")
            continue

        model_output_dir = os.path.join(args.output_dir, model_name)
        if args.random_seeds > 0:
            model_output_dir = os.path.join(model_output_dir, "random_seeds")
        os.makedirs(model_output_dir, exist_ok=True)
        
        print(f"📂 Output Folder: {model_output_dir}")

        for step_val, weight_path, step_name in targets:
            print(f"  👉 Testing: {step_name} ...", end="", flush=True)

            try:
                pipe.unload_lora_weights()
                # SDXL用のLoRA読み込み
                pipe.load_lora_weights(weight_path)
            except Exception as e:
                print(f" [Error loading LoRA] {e}")
                continue

            for seed in seeds:
                images = []
                generator = torch.Generator(device).manual_seed(seed)
                
                for label, prompt in prompts:
                    # SDXL推奨解像度 1024x1024
                    image = pipe(
                        prompt, 
                        num_inference_steps=30, 
                        guidance_scale=7.5, 
                        width=1024,
                        height=1024,
                        generator=generator
                    ).images[0]
                    images.append(image)
                
                # グリッド結合
                w, h = images[0].size
                grid = Image.new('RGB', (w * len(images), h))
                for i, img in enumerate(images):
                    grid.paste(img, (w * i, 0))
                
                save_filename = f"{step_name}_seed{seed}.png"
                save_path = os.path.join(model_output_dir, save_filename)
                grid.save(save_path)
            
            print(" Done.")

    print(f"\n✨ All evaluations finished. Results in: {args.output_dir}")

if __name__ == "__main__":
    main()
