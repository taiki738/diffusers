import os
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess

# ==========================================
# Configuration
# ==========================================

# 実験IDとフォルダ名のマッピング (論文の表に基づく)
EXP_MAPPING = {
    "lora_baseline": "EXP-SD-R4-L1", # 仮 (config確認要)
    "lora_rank4_lr5e5": "EXP-SD-R4-L2",
    "lora_rank16_lr1e4": "EXP-SD-R16-L1",
    "lora_rank16_lr5e5": "EXP-SD-R16-L2",

    "sdxl_lora_rank16_prodigy": "EXP-XL-R16",
    "sdxl_lora_rank32_prodigy": "EXP-XL-R32",
    "realvisxl_lora_rank16_prodigy": "EXP-RV-R16",
    
    "lora_rank16_lr5e5_trigger_ohwx": "EXP-SD-TRIG",
    "sdxl_lora_rank16_prodigy_trigger_ohwx": "EXP-XL-TRIG",
    "realvisxl_lora_rank16_prodigy_trigger_ohwx": "EXP-RV-TRIG",
}

def run_command(cmd):
    """シェルコマンドを実行し、エラーがあれば例外を投げる"""
    print(f"🚀 Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"❌ Command failed with return code {ret}")
        # 続行可能なエラーもあるので、ここではログ出しにとどめる

def get_dataset_path(base_dir, gender, impression):
    """条件に対応するデータセットのパスを返す"""
    # 例: .../male/OK_4.0
    imp_folder = "OK_4.0" if impression == "high" else "not-OK_2.0"
    return os.path.join(base_dir, gender, imp_folder)

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation for all experiments.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of extracted dataset (e.g., /content/ffhq_for_survey)")
    parser.add_argument("--gen_root", type=str, required=True, help="Root of generated images (e.g., .../evaluations/samples)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of trials for Baseline FID bootstrap")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline calculation if already done")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------
    # 1. Baseline FID Calculation (Bootstrap 100 trials)
    # -------------------------------------------------
    conditions = [
        ("male", "high"), ("male", "low"),
        ("female", "high"), ("female", "low")
    ]
    
    baseline_results = {}
    
    if not args.skip_baseline:
        print("\n🔵 Step 1: Calculating Baseline FID (Internal Diversity)...")
        for gender, impression in conditions:
            data_path = get_dataset_path(args.dataset_root, gender, impression)
            save_name = f"baseline_{gender}_{impression}.json"
            save_path = os.path.join(args.output_dir, save_name)
            
            if os.path.exists(save_path):
                print(f"   ⏩ Skipping {save_name} (Already exists)")
                with open(save_path, 'r') as f:
                    baseline_results[f"{gender}_{impression}"] = json.load(f)
                continue
                
            cmd = (
                f"python scripts/eval_lora_metrics.py "
                f"--mode baseline_check "
                f"--full_data_dir {data_path} "
                f"--num_trials {args.num_trials} "
                f"--val_ratio 0.5 "
                f"--output_json {save_path}"
            )
            run_command(cmd)
            
            # 読み込んでメモリに保持
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    baseline_results[f"{gender}_{impression}"] = json.load(f)

    # -------------------------------------------------
    # 2. Model Evaluation (Reconstruction FID, ArcFace, IRS)
    # -------------------------------------------------
    print("\n🔵 Step 2: Evaluating All Models...")
    
    gen_root_path = Path(args.gen_root)
    # gen_root/samples/exp_name/prompt_name/images...
    
    # 実験フォルダを走査 (samples/以下)
    # 構造: expname_model_step/prompt_dir/*.png
    
    all_metrics = []

    for exp_dir in sorted(gen_root_path.iterdir()):
        if not exp_dir.is_dir(): continue
        
        dir_name = exp_dir.name # 例: lora_baseline_stable-diffusion-v1-5_step5000
        
        # 実験名を抽出
        # 長いキーから順に試行することで、部分一致（sdxl_... が sdxl_..._trigger にマッチする等）を防ぐ
        matched_exp_key = None
        sorted_keys = sorted(EXP_MAPPING.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if dir_name.startswith(key):
                matched_exp_key = key
                break
        
        if not matched_exp_key:
            print(f"⚠️ Skipping unknown directory: {dir_name}")
            continue
            
        exp_id = EXP_MAPPING[matched_exp_key]
        
        # プロンプトフォルダを走査
        for prompt_dir in exp_dir.iterdir():
            if not prompt_dir.is_dir(): continue
            
            # プロンプト名から条件(gender, impression)を推定
            p_name = prompt_dir.name
            gender = "male" if "male" in p_name and "female" not in p_name else "female" if "female" in p_name else None
            impression = "high" if "high" in p_name else "low" if "low" in p_name else None
            
            if not gender or not impression:
                print(f"⚠️ Skipping ambiguous prompt: {p_name}")
                continue
                
            print(f"\n🔎 Processing: [{exp_id}] {gender} / {impression}")
            
            # 対応する正解データセット
            train_data_path = get_dataset_path(args.dataset_root, gender, impression)
            
            # プロンプトの構築 (CLIP Score用)
            prompt_text = f"a photo of a {gender} face, {impression} score impression"
            if "trigger" in matched_exp_key:
                prompt_text = prompt_text.replace("a photo of", "a photo of ohwx")
            
            # 結果保存パス
            metric_base_name = f"metrics_{matched_exp_key}_{gender}_{impression}"
            fid_json = os.path.join(args.output_dir, f"{metric_base_name}_fid.json")
            arc_json = os.path.join(args.output_dir, f"{metric_base_name}_arc.json")
            
            # A. FID / CLIP Calculation
            if not os.path.exists(fid_json):
                cmd_fid = (
                    f"python scripts/eval_lora_metrics.py "
                    f"--mode default "
                    f"--train_dir {train_data_path} "
                    f"--val_dir {train_data_path} " # Dummy (same as train)
                    f"--gen_dir {prompt_dir} "
                    f"--prompt \"{prompt_text}\" "
                    f"--output_json {fid_json}"
                )
                run_command(cmd_fid)
            
            # B. ArcFace / IRS Calculation
            if not os.path.exists(arc_json):
                cmd_arc = (
                    f"python scripts/eval_lora_metrics_vol2.py "
                    f"--train_dir {train_data_path} "
                    f"--gen_dir {prompt_dir} "
                    f"--output_json {arc_json}"
                )
                run_command(cmd_arc)
            
            # データの集約
            row = {
                "Exp_ID": exp_id,
                "Exp_Name": matched_exp_key,
                "Gender": gender,
                "Impression": impression,
            }
            
            if os.path.exists(fid_json):
                with open(fid_json, 'r') as f:
                    d = json.load(f)
                    row["Reconstruction_FID"] = d.get("quality_fid", -1)
                    row["CLIP_Score"] = d.get("clip_score", -1)
            
            if os.path.exists(arc_json):
                with open(arc_json, 'r') as f:
                    d = json.load(f)
                    row["ArcFace_Sim"] = d.get("arcface_similarity", -1)
                    row["IRS"] = d.get("irs_score", -1)
            
            all_metrics.append(row)

    # -------------------------------------------------
    # 3. Final Summary Report
    # -------------------------------------------------
    print("\n🔵 Step 3: Generating Summary Report...")
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(args.output_dir, "final_evaluation_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Summary saved to: {csv_path}")
        
        # Markdown形式でも表示（確認用）
        print("\n" + df.to_markdown())
    else:
        print("❌ No metrics collected.")

if __name__ == "__main__":
    main()
