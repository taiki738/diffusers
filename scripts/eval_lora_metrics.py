import os
import argparse
import json
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms
from functools import partial

# ==========================================
# Helper Functions
# ==========================================

def get_image_paths(folder):
    """フォルダ内の画像パスリストを取得"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]

def load_images_from_list(image_paths, target_size=(299, 299)):
    """画像パスリストからテンソルを生成するジェネレータ"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            yield transform(img)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")

def calculate_fid_core(paths_real, paths_fake, device, batch_size=32):
    """2つのパスリスト間のFIDを計算するコア関数"""
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Real Images Update
    batch = []
    # print(f"     Processing {len(paths_real)} real images...")
    for img in load_images_from_list(paths_real):
        batch.append(img)
        if len(batch) >= batch_size:
            fid_metric.update(torch.stack(batch).to(device), real=True)
            batch = []
    if batch:
        fid_metric.update(torch.stack(batch).to(device), real=True)

    # Fake Images Update
    batch = []
    # print(f"     Processing {len(paths_fake)} fake images...")
    for img in load_images_from_list(paths_fake):
        batch.append(img)
        if len(batch) >= batch_size:
            fid_metric.update(torch.stack(batch).to(device), real=False)
            batch = []
    if batch:
        fid_metric.update(torch.stack(batch).to(device), real=False)
        
    return fid_metric.compute().item()

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation of LoRA model outputs.")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="default", choices=["default", "baseline_check"], 
                        help="'default': Normal evaluation. 'baseline_check': Calculate Baseline FID variance.")

    # Paths for Default Mode
    parser.add_argument("--real_dir", type=str, help="Legacy: Path to real images")
    parser.add_argument("--fake_dir", type=str, help="Legacy: Path to generated images")
    parser.add_argument("--train_dir", type=str, help="Path to training images (for Baseline & Quality)")
    parser.add_argument("--val_dir", type=str, help="Path to validation images (for Baseline & Generalization)")
    parser.add_argument("--gen_dir", type=str, help="Path to generated images (target for evaluation)")
    
    # Paths for Baseline Check Mode
    parser.add_argument("--full_data_dir", type=str, help="Path to full dataset (for baseline check)")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for baseline check")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio for baseline check")

    # Common Settings
    parser.add_argument("--prompt", type=str, default="a photo of a face", help="The prompt used for CLIP Score")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric updates")
    parser.add_argument("--output_json", type=str, default="evaluation_metrics.json", help="Path to save the results")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16", help="CLIP model for score calculation")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}

    # ---------------------------------------------------------
    # Mode 1: Baseline Check (分散計測)
    # ---------------------------------------------------------
    if args.mode == "baseline_check":
        if not args.full_data_dir:
            print("❌ Error: --full_data_dir is required for baseline_check mode.")
            return
            
        print(f"🚀 Starting Baseline FID Variance Check ({args.num_trials} trials)")
        print(f"   Data: {args.full_data_dir}")
        
        all_images = get_image_paths(args.full_data_dir)
        total_count = len(all_images)
        print(f"   Total images: {total_count}")
        
        if total_count < 5:
            print("❌ Error: Not enough images to split.")
            return

        scores = []
        for i in range(args.num_trials):
            print(f"\n🔄 Trial {i+1}/{args.num_trials}...")
            random.seed(i) # 試行ごとにシードを変える
            random.shuffle(all_images)
            
            num_val = int(total_count * args.val_ratio)
            if num_val < 1: num_val = 1
            
            paths_val = all_images[:num_val]
            paths_train = all_images[num_val:]
            
            print(f"   Split: Train {len(paths_train)} / Val {len(paths_val)}")
            
            try:
                score = calculate_fid_core(paths_train, paths_val, device, args.batch_size)
                print(f"   👉 FID: {score:.4f}")
                scores.append(score)
            except Exception as e:
                print(f"   ❌ Error in trial {i+1}: {e}")
            
        if scores:
            avg_fid = np.mean(scores)
            std_fid = np.std(scores)
            
            print(f"\n📊 Result: Baseline FID = {avg_fid:.4f} ± {std_fid:.4f}")
            results["baseline_stats"] = {
                "mean": avg_fid,
                "std": std_fid,
                "trials": scores,
                "config": {
                    "full_data_dir": args.full_data_dir,
                    "val_ratio": args.val_ratio
                }
            }
        else:
            print("❌ No scores calculated.")

    # ---------------------------------------------------------
    # Mode 2: Default Evaluation (Comprehensive / Legacy)
    # ---------------------------------------------------------
    else:
        # Determine paths
        paths_train, paths_val, paths_gen = None, None, None
        
        if args.train_dir and args.val_dir and args.gen_dir:
            print(f"🚀 Starting Comprehensive Evaluation (Train/Val/Gen)...")
            paths_train = get_image_paths(args.train_dir)
            paths_val = get_image_paths(args.val_dir)
            paths_gen = get_image_paths(args.gen_dir)
            target_dir_for_clip = args.gen_dir
            
            # A. Baseline: Train vs Val
            print(f"\n📊 Calculating Baseline FID (Train vs Val)...")
            results["baseline_fid"] = calculate_fid_core(paths_train, paths_val, device, args.batch_size)
            print(f"✅ Baseline FID: {results['baseline_fid']:.4f}")
            
            # B. Quality: Train vs Gen
            print(f"\n📊 Calculating Quality FID (Train vs Gen)...")
            results["quality_fid"] = calculate_fid_core(paths_train, paths_gen, device, args.batch_size)
            print(f"✅ Quality FID: {results['quality_fid']:.4f}")
            
            # C. Generalization: Val vs Gen
            print(f"\n📊 Calculating Generalization FID (Val vs Gen)...")
            results["generalization_fid"] = calculate_fid_core(paths_val, paths_gen, device, args.batch_size)
            print(f"✅ Generalization FID: {results['generalization_fid']:.4f}")

        elif args.real_dir and args.fake_dir:
            print(f"🚀 Starting Legacy Evaluation (Real vs Fake)...")
            paths_real = get_image_paths(args.real_dir)
            paths_fake = get_image_paths(args.fake_dir)
            target_dir_for_clip = args.fake_dir
            
            score = calculate_fid_core(paths_real, paths_fake, device, args.batch_size)
            results["fid"] = score
            print(f"✅ FID: {score:.4f}")
            
        else:
            print("❌ Error: Invalid arguments. Provide (train/val/gen) or (real/fake).")
            return

        # 2. CLIP Score Calculation
        # -------------------------
        print("\n⏳ Calculating CLIP Score for Generated Images...")
        calc_clip = partial(clip_score, model_name_or_path=args.clip_model)
        
        total_clip_score = 0.0
        count = 0
        
        # Use the generated directory
        target_paths = get_image_paths(target_dir_for_clip)
        
        # Use load_images_from_list for CLIP too (need resize to 224)
        loader = load_images_from_list(target_paths, target_size=(224, 224))
        
        batch_imgs = []
        batch_prompts = []
        
        for img_tensor in tqdm(loader, total=len(target_paths)):
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
        
        results["config"] = {
            "mode": args.mode,
            "prompt": args.prompt,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "gen_dir": args.gen_dir
        }

    # Save Results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n🎉 Evaluation finished. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
