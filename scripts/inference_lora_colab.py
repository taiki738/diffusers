import argparse
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from pathlib import Path

def generate_four_way_comparison(args):
    # パス設定
    lora_path = Path(args.lora_path_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ベースモデルのロード
    print(f"⏳ Loading base model from {args.base_model_path}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return

    # LoRAのロード
    print(f"⏳ Loading LoRA weights from {lora_path}...")
    try:
        pipe.load_lora_weights(str(lora_path))
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        return

    # 4パターンのプロンプト設定
    test_cases = [
        {"gender": "male", "score": "high", "prompt": "a photo of a male face, high score impression"},
        {"gender": "male", "score": "low", "prompt": "a photo of a male face, low score impression"},
        {"gender": "female", "score": "high", "prompt": "a photo of a female face, high score impression"},
        {"gender": "female", "score": "low", "prompt": "a photo of a female face, low score impression"},
    ]
    
    seeds = args.seeds
    print(f"🚀 Generating 4 patterns for {len(seeds)} seeds...")

    # グリッド画像の描画準備
    fig, axes = plt.subplots(len(seeds), 4, figsize=(20, 5 * len(seeds)))
    if len(seeds) == 1:
        axes = [axes]

    for i, seed in enumerate(seeds):
        for j, case in enumerate(test_cases):
            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(case["prompt"], num_inference_steps=30, generator=generator).images[0]
            
            ax = axes[i][j]
            ax.imshow(image)
            ax.set_title(f"{case['gender']} {case['score']}\n(Seed {seed})")
            ax.axis("off")
            
            # 個別画像も保存
            image.save(output_dir / f"{case['gender']}_{case['score']}_seed{seed}.png")

    plt.tight_layout()
    grid_path = output_dir / "comparison_grid_4way.png"
    plt.savefig(grid_path)
    print(f"✅ Comparison grid saved to: {grid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 4-way comparison images with LoRA")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_path_dir", type=str, required=True, help="Path to LoRA weights dir")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Seeds to test")
    
    args = parser.parse_args()
    generate_four_way_comparison(args)
