"""Download the trained RandomForest models from the GitHub release.

Every trained model is in the repository: neural checkpoints (*.pt) and
LightGBM pickles are committed directly, but the six RandomForest pickles are
96-272 MB each, over GitHub's 100 MB file limit, so they are published as
gzip-compressed assets of the `models-v1` release. This script puts them back
where run_e1.py / run_e2*.py / run_sweep.py expect them.

    python3 src/fetch_models.py
"""
import gzip
import os
import shutil
import urllib.request

REPO = "VedantGupta-DTU/IoT-NIDS-Generalization-Benchmark"
TAG = "models-v1"
ASSETS = {
    "results/checkpoints": [
        "RandomForest_8class.pkl", "RandomForest_binary.pkl", "RandomForest_binary44.pkl",
    ],
    "results/checkpoints_sweep": [
        "RandomForest_binary_s13.pkl", "RandomForest_binary_s21.pkl", "RandomForest_binary_s42.pkl",
    ],
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    for folder, names in ASSETS.items():
        os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
        for name in names:
            dest = os.path.join(ROOT, folder, name)
            if os.path.exists(dest):
                print(f"[skip] {folder}/{name} already present")
                continue
            url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}.gz"
            print(f"[get]  {url}")
            with urllib.request.urlopen(url) as resp, gzip.GzipFile(fileobj=resp) as gz, \
                    open(dest + ".part", "wb") as out:
                shutil.copyfileobj(gz, out)
            os.replace(dest + ".part", dest)
            print(f"[ok]   {folder}/{name} ({os.path.getsize(dest) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
