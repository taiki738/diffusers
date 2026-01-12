import os
import json
import shutil
from pathlib import Path

def prepare_dataset():
    # 設定
    base_src_dir = Path("Datasets/Data/FFHQ/ffhq_for_survey")
    output_dir = Path("Datasets/Data/FFHQ/diffusers_lora_dataset")
    metadata_file = output_dir / "metadata.jsonl"
    
    # カテゴリとプロンプトの対応
    # 提案書に合わせて "high score impression", "low score impression" を使用
    categories = [
        {"path": "male/OK_4.0", "prompt": "a photo of a male face, high score impression"},
        {"path": "male/not-OK_2.0", "prompt": "a photo of a male face, low score impression"},
        {"path": "female/OK_4.0", "prompt": "a photo of a female face, high score impression"},
        {"path": "female/not-OK_2.0", "prompt": "a photo of a female face, low score impression"},
    ]

    # 出力ディレクトリの作成
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    metadata_entries = []
    
    print("Processing images...")
    for cat in categories:
        src_path = base_src_dir / cat["path"]
        if not src_path.exists():
            print(f"Warning: Directory {src_path} not found. Skipping.")
            continue
        
        # 画像ファイルを収集
        for img_path in src_path.glob("*.png"):
            # ファイル名の重複を避けるため、フォルダ名を接頭辞にする
            new_file_name = f"{cat['path'].replace('/', '_')}_{img_path.name}"
            dest_path = output_dir / new_file_name
            
            # 実体ファイルをコピー（シンボリックリンクを辿る）
            shutil.copy2(img_path.resolve(), dest_path)
            
            # メタデータに追加
            metadata_entries.append({
                "file_name": new_file_name,
                "text": cat["prompt"]
            })
            
    # metadata.jsonl の書き出し
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Success! Created {len(metadata_entries)} images in {output_dir}")
    print(f"Metadata written to {metadata_file}")

if __name__ == "__main__":
    prepare_dataset()
