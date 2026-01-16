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
    parser = argparse.ArgumentParser(description="Quantitative evaluation of LoRA model outputs using torchmetrics.")
    parser.add_argument("--real_dir", type=str, help="Legacy: Path to real images (e.g. FFHQ validation set)")
    parser.add_argument("--fake_dir", type=str, help="Legacy: Path to generated images")
    
    # New arguments for comprehensive analysis
    parser.add_argument("--train_dir", type=str, help="Path to training images (for Baseline & Quality)")
    parser.add_argument("--val_dir", type=str, help="Path to validation images (for Baseline & Generalization)")
    parser.add_argument("--gen_dir", type=str, help="Path to generated images (target for evaluation)")
    
    parser.add_argument("--prompt", type=str, default="a photo of a face", help="The prompt used for CLIP Score (if single)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric updates")
    parser.add_argument("--output_json", type=str, default="evaluation_metrics.json", help="Path to save the results")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16", help="CLIP model for score calculation")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mode selection
    mode = "legacy"
    if args.train_dir and args.val_dir and args.gen_dir:
        mode = "comprehensive"
        print(f"🚀 Starting Comprehensive Evaluation (Train/Val/Gen)...")
    elif args.real_dir and args.fake_dir:
        print(f"🚀 Starting Legacy Evaluation (Real vs Fake)...")
    else:
        print("❌ Error: You must provide either (--real_dir and --fake_dir) OR (--train_dir, --val_dir, and --gen_dir).")
        return

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    results = {}

    def calculate_fid_for_pair(path_real, path_fake, label):
        print(f"\n📊 Calculating FID for {label}...")
        print(f"   Real (Ref): {path_real}")
        print(f"   Fake (Tgt): {path_fake}")
        
        fid_metric.reset()
        
        # Real Images Update
        real_loader = load_images_from_folder(path_real)
        batch = []
        count_real = 0
        print("  - Processing reference images...")
        for img_tensor, _ in tqdm(real_loader):
            batch.append(img_tensor)
            if len(batch) >= args.batch_size:
                fid_metric.update(torch.stack(batch).to(device), real=True)
                count_real += len(batch)
                batch = []
        if batch:
            fid_metric.update(torch.stack(batch).to(device), real=True)
            count_real += len(batch)

        # Fake Images Update
        fake_loader = load_images_from_folder(path_fake)
        batch = []
        count_fake = 0
        print("  - Processing target images...")
        for img_tensor, name in tqdm(fake_loader):
            batch.append(img_tensor)
            if len(batch) >= args.batch_size:
                fid_metric.update(torch.stack(batch).to(device), real=False)
                count_fake += len(batch)
                batch = []
        if batch:
            fid_metric.update(torch.stack(batch).to(device), real=False)
            count_fake += len(batch)
            
        if count_real == 0 or count_fake == 0:
            print(f"⚠️ Warning: No images found in one of the directories. Skipping {label}.")
            return None

        score = fid_metric.compute().item()
        print(f"✅ FID ({label}): {score:.4f}")
        return score

    # 1. FID Calculation
    # ------------------
    if mode == "comprehensive":
        # A. Baseline: Train vs Val (How similar is the dataset to itself?)
        results["baseline_fid"] = calculate_fid_for_pair(args.train_dir, args.val_dir, "Baseline (Train vs Val)")
        
        # B. Quality: Train vs Gen (How well did it learn the training distribution?)
        results["quality_fid"] = calculate_fid_for_pair(args.train_dir, args.gen_dir, "Quality (Train vs Gen)")
        
        # C. Generalization: Val vs Gen (How well does it generalize to unseen data?)
        results["generalization_fid"] = calculate_fid_for_pair(args.val_dir, args.gen_dir, "Generalization (Val vs Gen)")
        
        target_dir_for_clip = args.gen_dir
        
    else: # legacy
        fid_score = calculate_fid_for_pair(args.real_dir, args.fake_dir, "Legacy (Real vs Fake)")
        results["fid"] = fid_score
        target_dir_for_clip = args.fake_dir

    # 2. CLIP Score Calculation (Only for generated images)
    # -------------------------
    print("\n⏳ Calculating CLIP Score for Generated Images...")
    calc_clip = partial(clip_score, model_name_or_path=args.clip_model)
    
    total_clip_score = 0.0
    count = 0
    
    # Use the generated directory (gen_dir in comprehensive, fake_dir in legacy)
    fake_loader = load_images_from_folder(target_dir_for_clip, target_size=(224, 224))
    batch_imgs = []
    batch_prompts = []
    
    print(f"   Target: {target_dir_for_clip}")
    for img_tensor, name in tqdm(fake_loader):
        batch_imgs.append((img_tensor * 255).to(torch.uint8))
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
    results["clip_score"] = avg_clip_score

    # 3. Save Results
    # ---------------
    results["config"] = {
        "mode": mode,
        "prompt": args.prompt,
        "clip_model": args.clip_model
    }
    if mode == "comprehensive":
        results["config"].update({
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "gen_dir": args.gen_dir
        })
    else:
        results["config"].update({
            "real_dir": args.real_dir,
            "fake_dir": args.fake_dir
        })
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n🎉 Evaluation finished. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
