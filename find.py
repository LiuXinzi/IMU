import shutil
from pathlib import Path

def copy_finetune_sequences(txt_file: str, base_dir: str):
    base_dir = Path(base_dir).resolve()
    txt_path = Path(txt_file).resolve()

    # Read CMU IDs from the txt file
    if not txt_path.exists():
        raise FileNotFoundError(f"txt file not found: {txt_path}")

    with open(txt_path, "r") as f:
        cmu_ids = [line.strip() for line in f if line.strip()]

    print(f"Found {len(cmu_ids)} CMU IDs to copy.")

    source_dir = base_dir / "processed_CMU_1"
    target_dir = base_dir / "finetune_data"
    target_dir.mkdir(exist_ok=True, parents=True)

    copied = 0
    missing = []

    # Iterate and copy
    for cmu_id in cmu_ids:
        src_file = source_dir / f"{cmu_id}.npz"

        if src_file.exists():
            dst_file = target_dir / f"{cmu_id}.npz"
            shutil.copy(src_file, dst_file)
            copied += 1
        else:
            missing.append(cmu_id)

    print(f"Copied {copied} sequences to: {target_dir}")

    if missing:
        print("\n⚠ Missing files (not found in processed_CMU):")
        for m in missing:
            print("  ", m)

    print("\nDone!")


if __name__ == "__main__":
    # modify BASE_DIR and TXT_FILE as needed
    BASE_DIR = Path(__file__).resolve().parent          # your base_dir
    TXT_FILE = "D:/IMU/Watanabe/models_CMU/evaluation/top100_highest_error_CMU_ids.txt"

    copy_finetune_sequences(TXT_FILE, BASE_DIR)

