import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import re
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
        # 便宜上、非常に大きなステップ数として扱うか、フラグで管理
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
    """
    (旧関数: 単一のステップを探す用)
    """
    all_ckpts = get_all_checkpoints(folder_path)
    if not all_ckpts:
        return None, None

    if target_step == "latest":
        return all_ckpts[-1][1], all_ckpts[-1][2]
    
    elif target_step == "final":
        # Finalを探す
        for step, path, name in all_ckpts:
            if name == "Final":
                return path, name
        # なければ最新
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
    parser = argparse.ArgumentParser(description="Evaluate and compare LoRA models from a directory")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/github/diffusers/evaluations", help="Root directory to save results")
    parser.add_argument("--models_dir", type=str, default=None, help="Root directory containing experiment folders to scan")
    parser.add_argument("--target_step", type=str, default="latest", help="'latest', 'final', specific int, or 'all' to generate for all checkpoints.")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model path or ID")
    parser.add_argument("--lora_paths", type=str, nargs='+', help="Specific LoRA paths (Optional). Overrides models_dir scan.")
    parser.add_argument("--random_seeds", type=int, default=0, help="Number of random seeds to use. If 0 (default), uses fixed seeds [42, 123].")
    
    args = parser.parse_args()

    # 1. 比較対象のLoRAリストを作成
    lora_candidates = {} # name -> folder_path

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"🚀 Loading base model: {args.base_model}")
    print(f"   Device: {device}, Dtype: {dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    # CPUの場合、メモリ節約のためにattention slicingを有効化
    if device == "cpu":
        pipe.enable_attention_slicing()

    # 3. 比較プロンプト
    prompts = [
        ("Male High", "a photo of a male face, high score impression"),
        ("Male Low",  "a photo of a male face, low score impression"),
        ("Fem High",  "a photo of a female face, high score impression"),
        ("Fem Low",   "a photo of a female face, low score impression"),
    ]
    
    # Seed設定
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

        # ターゲットステップのリストを取得
        targets = []
        if args.target_step == "all":
            targets = get_all_checkpoints(folder_path)
            if not targets:
                print(f"⚠️ No checkpoints found in {folder_path}")
                continue
        else:
            path, name = find_lora_weight(folder_path, args.target_step)
            if path:
                targets = [(0, path, name)] # step数はダミー
            else:
                print(f"⚠️ Target step {args.target_step} not found for {model_name}")
                continue

        # モデルごとの保存先フォルダ作成
        model_output_dir = os.path.join(args.output_dir, model_name)
        
        # ランダムシードの場合はサブフォルダを作成
        if args.random_seeds > 0:
            model_output_dir = os.path.join(model_output_dir, "random_seeds")
            
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"📂 Output Folder: {model_output_dir}")

        for step_val, weight_path, step_name in targets:
            print(f"  👉 Testing: {step_name} ...", end="", flush=True)

            try:
                pipe.unload_lora_weights()
                pipe.load_lora_weights(os.path.dirname(weight_path), weight_name=os.path.basename(weight_path))
            except Exception as e:
                print(f" [Error] {e}")
                continue

            for seed in seeds:
                images = []
                generator = torch.Generator(device).manual_seed(seed)
                
                for label, prompt in prompts:
                    image = pipe(
                        prompt, 
                        num_inference_steps=30, 
                        guidance_scale=7.5, 
                        generator=generator
                    ).images[0]
                    images.append(image)
                
                # グリッド結合
                w, h = images[0].size
                grid = Image.new('RGB', (w * len(images), h))
                for i, img in enumerate(images):
                    grid.paste(img, (w * i, 0))
                
                # ファイル名: Step-XXXX_seedYY.png (モデル名はフォルダ名にあるので省略可だが、念のため)
                save_filename = f"{step_name}_seed{seed}.png"
                save_path = os.path.join(model_output_dir, save_filename)
                grid.save(save_path)
            
            print(" Done.")

    print(f"\n✨ All evaluations finished. Results in: {args.output_dir}")

if __name__ == "__main__":
    main()