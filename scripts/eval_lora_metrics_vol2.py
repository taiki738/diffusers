import argparse
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# insightfaceはインストールが必要
# !pip install insightface onnxruntime-gpu

def get_image_paths(folder):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]

def extract_features(app, image_paths, desc="Extracting features"):
    features = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc=desc):
        img = cv2.imread(path)
        if img is None:
            continue
            
        faces = app.get(img)
        
        # 顔が検出された場合のみ採用
        # 複数ある場合は一番大きい顔を採用
        if len(faces) > 0:
            # 面積でソート
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            embedding = faces[0].embedding # 512次元
            features.append(embedding)
            valid_paths.append(path)
            
    if not features:
        return None, []
        
    # 正規化 (Cosine Similarity用)
    features = np.array(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms
    
    return normalized_features, valid_paths

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA models using ArcFace (Similarity & IRS).")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images (Reference for Similarity & IRS)")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated images (Target)")
    parser.add_argument("--output_json", type=str, default="evaluation_arcface.json", help="Path to save results")
    parser.add_argument("--irs_threshold", type=float, default=0.75, help="Threshold for identity copy detection (IRS)")
    
    args = parser.parse_args()
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("❌ Error: insightface not installed. Please run: pip install insightface onnxruntime-gpu")
        return

    print("🚀 Initializing FaceAnalysis (ArcFace)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 1. 特徴量抽出
    train_paths = get_image_paths(args.train_dir)
    gen_paths = get_image_paths(args.gen_dir)
    
    print(f"📊 Analyzing Train Data: {len(train_paths)} images")
    train_feats, _ = extract_features(app, train_paths, desc="Train Embeddings")
    
    print(f"📊 Analyzing Gen Data: {len(gen_paths)} images")
    gen_feats, _ = extract_features(app, gen_paths, desc="Gen Embeddings")
    
    if train_feats is None or gen_feats is None:
        print("❌ Error: Could not extract features from one of the directories.")
        return

    results = {}

    # 2. ArcFace Cosine Similarity (属性の近さ)
    # Trainデータ全体の「平均顔ベクトル（重心）」を計算
    print("\n📐 Calculating Attribute Similarity...")
    train_center = np.mean(train_feats, axis=0)
    train_center = train_center / np.linalg.norm(train_center) # 再正規化
    
    # 生成画像それぞれと、Train重心との類似度を計算
    # (生成画像が「Trainデータの平均的な顔（好印象顔）」にどれだけ似ているか)
    sims_to_center = np.dot(gen_feats, train_center)
    avg_sim = np.mean(sims_to_center)
    
    print(f"✅ ArcFace Similarity (vs Train Center): {avg_sim:.4f}")
    results["arcface_similarity"] = float(avg_sim)

    # 3. IRS / Nearest Neighbor Distance (過学習チェック)
    # 生成画像1枚1枚について、Trainデータの中で「一番似ているやつ」を探す
    print("\n🕵️ Calculating IRS (Overfitting Check)...")
    
    # 行列演算で全対全の類似度を一括計算 (Gen x Train)
    # sim_matrix[i, j] = Gen[i] と Train[j] の類似度
    sim_matrix = np.dot(gen_feats, train_feats.T)
    
    # 各生成画像ごとの最大類似度 (Nearest Neighbor Similarity)
    max_sims = np.max(sim_matrix, axis=1)
    
    # 閾値を超えた割合 (IRS)
    overfit_count = np.sum(max_sims > args.irs_threshold)
    irs_score = overfit_count / len(gen_feats)
    avg_max_sim = np.mean(max_sims)
    
    print(f"✅ Average Max Similarity: {avg_max_sim:.4f}")
    print(f"⚠️ IRS (Copy Rate > {args.irs_threshold}): {irs_score:.2%} ({overfit_count}/{len(gen_feats)})")
    
    results["irs_score"] = float(irs_score)
    results["avg_max_similarity"] = float(avg_max_sim)
    results["config"] = {
        "train_dir": args.train_dir,
        "gen_dir": args.gen_dir,
        "irs_threshold": args.irs_threshold
    }
    
    # Save
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n🎉 Analysis finished. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
