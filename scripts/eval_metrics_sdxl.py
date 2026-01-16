import os
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms
from functools import partial

def load_images_from_folder(folder, target_size=(299, 299)):
    """フォルダ内の画像を読み込み、テンソルに変換するジェネレータ"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    files.sort()
    
    for filename in files:
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert('RGB')
            yield transform(img), filename
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation of SDXL outputs using torchmetrics.")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to real images (e.g. FFHQ validation set)")
    parser.add_argument("--fake_dir", type=str, required=True, help="Path to generated images")
    parser.add_argument("--prompt", type=str, default="a photo of a face", help="The prompt used for CLIP Score (if single)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric updates")
    parser.add_argument("--output_json", type=str, default="evaluation_metrics.json", help="Path to save the results")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16", help="CLIP model for score calculation")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Starting Evaluation...")
    print(f"   Real Images: {args.real_dir}")
    print(f"   Fake Images: {args.fake_dir}")

    # 1. FID Calculation
    # ------------------
    print("⏳ Calculating FID...")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Real Images Update
    real_loader = load_images_from_folder(args.real_dir)
    batch = []
    print("  - Processing real images...")
    for img_tensor, _ in tqdm(real_loader):
        batch.append(img_tensor)
        if len(batch) >= args.batch_size:
            fid_metric.update(torch.stack(batch).to(device), real=True)
            batch = []
    if batch:
        fid_metric.update(torch.stack(batch).to(device), real=True)
    
    # Fake Images Update
    fake_loader = load_images_from_folder(args.fake_dir)
    batch = []
    fake_tensors = [] # For CLIP Score later
    print("  - Processing fake images...")
    for img_tensor, name in tqdm(fake_loader):
        batch.append(img_tensor)
        # CLIP Score用に別途保持（メモリ節約のため、バッチごとに計算する設計も可能だが一旦シンプルに）
        # ただし数千枚ある場合はメモリ不足になるため、CLIP Scoreも随時計算に切り替える
        if len(batch) >= args.batch_size:
            fid_metric.update(torch.stack(batch).to(device), real=False)
            batch = []
    if batch:
        fid_metric.update(torch.stack(batch).to(device), real=False)
    
    fid_score = fid_metric.compute().item()
    print(f"✅ FID: {fid_score:.4f}")

    # 2. CLIP Score Calculation
    # -------------------------
    print("⏳ Calculating CLIP Score...")
    # clip_score function internally uses openai/clip-vit-base-patch16 by default
    # but we can wrap it for consistency.
    calc_clip = partial(clip_score, model_name_or_path=args.clip_model)
    
    total_clip_score = 0.0
    count = 0
    
    fake_loader = load_images_from_folder(args.fake_dir, target_size=(224, 224)) # CLIP standard size
    batch_imgs = []
    batch_prompts = []
    
    for img_tensor, name in tqdm(fake_loader):
        # 画像からプロンプトを推測、または固定プロンプトを使用
        # ここではシンプルに引数のプロンプトを使用
        batch_imgs.append((img_tensor * 255).to(torch.uint8)) # CLIP score expects uint8 [0, 255]
        batch_prompts.append(args.prompt)
        
        if len(batch_imgs) >= args.batch_size:
            score = calc_clip(torch.stack(batch_imgs), batch_prompts)
            total_clip_score += score.item() * len(batch_imgs)
            count += len(batch_imgs)
            batch_imgs = []
            batch_prompts = []
            
    if batch_imgs:
        score = calc_clip(torch.stack(batch_imgs), batch_prompts)
        total_clip_score += score.item() * len(batch_imgs)
        count += len(batch_imgs)
        
    avg_clip_score = total_clip_score / count if count > 0 else 0
    print(f"✅ CLIP Score: {avg_clip_score:.4f}")

    # 3. Save Results
    # ---------------
    results = {
        "fid": fid_score,
        "clip_score": avg_clip_score,
        "config": {
            "real_dir": args.real_dir,
            "fake_dir": args.fake_dir,
            "prompt": args.prompt,
            "clip_model": args.clip_model
        }
    }
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"🎉 Evaluation finished. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
