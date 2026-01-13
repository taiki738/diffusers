import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import re

def find_lora_weight(folder_path, target_step="latest"):
    """
    指定されたフォルダ内から、目的のステップ数のLoRA重みファイルを探す。
    
    Args:
        folder_path: 実験結果のルートフォルダ (例: .../outputs/lora_male_v1)
        target_step: "latest", "final", または特定のステップ数 (int or str)
    
    Returns:
        (path_to_safetensors, step_count_str)
    """
    # 1. 学習完了後の最終出力 (pytorch_lora_weights.safetensors)
    final_weight = os.path.join(folder_path, "pytorch_lora_weights.safetensors")
    
    # 2. チェックポイントフォルダの探索
    checkpoints = []
    if os.path.exists(folder_path):
        for d in os.listdir(folder_path):
            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, os.path.join(folder_path, d)))
                except ValueError:
                    continue
    
    # ステップ数順にソート
    checkpoints.sort(key=lambda x: x[0])
    
    # ターゲットに応じた選択ロジック
    selected_path = None
    selected_step_name = "Unknown"

    if target_step == "final":
        if os.path.exists(final_weight):
            return final_weight, "Final"
        elif checkpoints:
            # finalがない場合は最新のcheckpoint
            return checkpoints[-1][1], f"Step-{checkpoints[-1][0]}"
            
    elif str(target_step).isdigit():
        # 特定のステップ数を指定された場合
        target = int(target_step)
        # 完全一致を探す
        for step, path in checkpoints:
            if step == target:
                return os.path.join(path, "pytorch_lora_weights.safetensors"), f"Step-{step}"
        print(f"⚠️ Step {target} not found in {os.path.basename(folder_path)}. using latest.")
        # 見つからなければ最新
        if checkpoints:
            return os.path.join(checkpoints[-1][1], "pytorch_lora_weights.safetensors"), f"Step-{checkpoints[-1][0]}"

    else: # "latest" or others
        # チェックポイントの中で最新のもの
        if checkpoints:
            return os.path.join(checkpoints[-1][1], "pytorch_lora_weights.safetensors"), f"Step-{checkpoints[-1][0]}"
        # チェックポイントがなければ final を見る
        elif os.path.exists(final_weight):
            return final_weight, "Final"

    return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare LoRA models from a directory")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/github/diffusers/evaluations", help="Directory to save results")
    parser.add_argument("--models_dir", type=str, default=None, help="Root directory containing experiment folders to scan")
    parser.add_argument("--target_step", type=str, default="latest", help="Specific checkpoint step to load (e.g., 1000, 2000). Default is 'latest'.")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model path or ID")
    parser.add_argument("--lora_paths", type=str, nargs='+', help="Specific LoRA paths (Optional). Overrides models_dir scan.")
    
    args = parser.parse_args()

    # 1. 比較対象のLoRAリストを作成
    lora_candidates = {} # name -> folder_path

    # 明示的にパスが指定された場合
    if args.lora_paths:
        for item in args.lora_paths:
            if ":" in item:
                name, path = item.split(":", 1)
                lora_candidates[name] = path
            else:
                name = os.path.basename(item.rstrip("/"))
                lora_candidates[name] = item
    
    # ディレクトリをスキャンする場合
    elif args.models_dir and os.path.exists(args.models_dir):
        print(f"📂 Scanning models in: {args.models_dir}")
        for d in sorted(os.listdir(args.models_dir)):
            full_path = os.path.join(args.models_dir, d)
            if os.path.isdir(full_path):
                # LoRAが含まれていそうなフォルダか簡易チェック
                if os.path.exists(os.path.join(full_path, "pytorch_lora_weights.safetensors")) or \
                   any(sub.startswith("checkpoint-") for sub in os.listdir(full_path)):
                    lora_candidates[d] = full_path
    else:
        # デフォルト (Colab用フォールバック)
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
    print(f"🚀 Loading base model: {args.base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    # 3. 比較プロンプト
    prompts = [
        ("Male High", "a photo of a male face, high score impression"),
        ("Male Low",  "a photo of a male face, low score impression"),
        ("Fem High",  "a photo of a female face, high score impression"),
        ("Fem Low",   "a photo of a female face, low score impression"),
    ]
    seeds = [42, 123]
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. 生成ループ
    for name, folder_path in lora_candidates.items():
        # 指定ステップの重みファイルを探す
        weight_path, step_name = find_lora_weight(folder_path, args.target_step)
        
        if not weight_path:
            print(f"⚠️ Skipping {name}: No valid weights found for step {args.target_step}")
            continue

        print(f"\n---------------------------------------------------------")
        print(f"🚀 Testing Model: {name} ({step_name})")
        print(f"   Source: {weight_path}")
        print(f"---------------------------------------------------------")

        try:
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.dirname(weight_path), weight_name=os.path.basename(weight_path))
        except Exception as e:
            print(f"⚠️ Load Error for {name}: {e}")
            continue

        for seed in seeds:
            images = []
            generator = torch.Generator("cuda").manual_seed(seed)
            
            print(f"  Generating Seed {seed} grid...", end="", flush=True)
            for label, prompt in prompts:
                image = pipe(
                    prompt, 
                    num_inference_steps=30, 
                    guidance_scale=7.5, 
                    generator=generator
                ).images[0]
                images.append(image)
            print(" Done.")
            
            # グリッド結合
            w, h = images[0].size
            grid = Image.new('RGB', (w * len(images), h))
            for i, img in enumerate(images):
                grid.paste(img, (w * i, 0))
            
            # ファイル名にステップ数を含める
            safe_name = name.replace(' ', '_')
            save_name = f"{safe_name}_{step_name}_seed{seed}.png"
            save_path = os.path.join(args.output_dir, save_name)
            grid.save(save_path)
            print(f"  Saved: {save_path}")

    print(f"\n✨ All evaluations finished. Results in: {args.output_dir}")

if __name__ == "__main__":
    main()