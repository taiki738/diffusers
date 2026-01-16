import argparse
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, val_ratio=0.1, seed=42):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Train/Val 出力先
    train_root = output_path / "train"
    val_root = output_path / "validation"
    
    # 処理対象の画像拡張子
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    
    # 乱数シード固定
    random.seed(seed)
    
    print(f"🚀 Splitting dataset from {input_path} to {output_path}")
    print(f"   Validation ratio: {val_ratio:.0%} (Seed: {seed})")

    # globで全画像を取得
    all_images = []
    for ext in valid_extensions:
        all_images.extend(input_path.rglob(f"*{ext}"))
        
    if not all_images:
        print("❌ No images found in input directory.")
        return

    # 親ディレクトリごとの辞書を作成
    dir_map = {}
    for img_path in all_images:
        parent = img_path.parent.relative_to(input_path)
        if parent not in dir_map:
            dir_map[parent] = []
        dir_map[parent].append(img_path)
        
    # 分割とコピー実行
    files_moved = 0
    for rel_dir, images in dir_map.items():
        # シャッフル
        random.shuffle(images)
        
        # 分割数計算
        total = len(images)
        # 基本計算
        n_val = int(total * val_ratio)
        
        # 例外処理: 最低枚数の確保ルール
        # 1. データが1枚しかない -> Trainのみ (Val=0)
        # 2. データが極端に少ない(例:10枚以下) -> それでも計算上0枚ならVal=1枚確保するか？
        #    今回は「汎化性能テスト」が目的なので、Valが0だとFID計算不能になる。
        #    なので、2枚以上あるなら最低1枚はValに回す設定にする。
        if total > 1 and n_val == 0:
            n_val = 1
            
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]
        
        # 出力ディレクトリ作成
        current_train_dir = train_root / rel_dir
        current_val_dir = val_root / rel_dir
        
        current_train_dir.mkdir(parents=True, exist_ok=True)
        if n_val > 0:
            current_val_dir.mkdir(parents=True, exist_ok=True)
            
        # コピー実行
        for img in train_imgs:
            shutil.copy2(img, current_train_dir / img.name)
            
        for img in val_imgs:
            shutil.copy2(img, current_val_dir / img.name)
            
        print(f"   📂 {rel_dir}: Total {total} -> Train {len(train_imgs)} / Val {len(val_imgs)}")
        files_moved += total

    print(f"✅ Done! Processed {files_moved} images.")
    print(f"   Train set: {train_root}")
    print(f"   Val set:   {val_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets preserving structure.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    split_dataset(args.input_dir, args.output_dir, args.val_ratio, args.seed)
