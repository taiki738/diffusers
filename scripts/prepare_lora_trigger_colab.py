import argparse
import json
from pathlib import Path

def prepare_colab_dataset(dataset_root: str, trigger_word: str):
    root = Path(dataset_root)
    metadata_file = root / "metadata.jsonl"
    
    # カテゴリとプロンプトの対応
    # トリガーワードがある場合は挿入する
    if trigger_word:
        mapping = {
            "male/OK_4.0": f"a photo of {trigger_word} male face, high score impression",
            "male/not-OK_2.0": f"a photo of {trigger_word} male face, low score impression",
            "female/OK_4.0": f"a photo of {trigger_word} female face, high score impression",
            "female/not-OK_2.0": f"a photo of {trigger_word} female face, low score impression",
        }
    else:
        mapping = {
            "male/OK_4.0": "a photo of a male face, high score impression",
            "male/not-OK_2.0": "a photo of a male face, low score impression",
            "female/OK_4.0": "a photo of a female face, high score impression",
            "female/not-OK_2.0": "a photo of a female face, low score impression",
        }

    metadata_entries = []
    
    print(f"Scanning directory: {root}")
    if trigger_word:
        print(f"Using trigger word: {trigger_word}")
    
    for rel_path, prompt in mapping.items():
        category_dir = root / rel_path
        if not category_dir.exists():
            print(f"Warning: Directory {category_dir} not found. Skipping.")
            continue
        
        # pngファイルを収集 (FFHQは通常png)
        count = 0
        for img_path in category_dir.glob("*.png"):
            # metadata.jsonl から見た相対パス
            metadata_entries.append({
                "file_name": f"{rel_path}/{img_path.name}",
                "text": prompt
            })
            count += 1
        print(f"  - {rel_path}: {count} images found.")
            
    if not metadata_entries:
        print("Error: No images found. Please check the dataset_root and folder structure.")
        return

    # metadata.jsonl の書き出し
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\nSuccess! Created {len(metadata_entries)} entries in {metadata_file}")
    print(f"Trigger Word: {trigger_word if trigger_word else 'None'}")
    print("You can now start training with --train_data_dir pointing to the dataset_root.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare metadata.jsonl with Trigger Word for LoRA training on Colab")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the unzipped dataset folder in Google Drive")
    parser.add_argument("--trigger_word", type=str, default=None, help="Trigger word to prepend (e.g. 'ohwx')")
    args = parser.parse_args()
    
    prepare_colab_dataset(args.dataset_root, args.trigger_word)
