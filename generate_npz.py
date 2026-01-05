import numpy as np
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    num_files = 100

    for idx in range(num_files):
        frames = int(rng.integers(100, 401))
        x = rng.random((frames, 72), dtype=np.float32)
        y = rng.random((frames, 24, 3), dtype=np.float32)

        out_path = out_dir / f"sample_{idx:03d}.npz"
        np.savez_compressed(out_path, x=x, y=y)


if __name__ == "__main__":
    main()
