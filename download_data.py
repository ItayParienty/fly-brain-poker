"""Downloads the FlyWire connectome tables this project runs on.

The data is the public v783 snapshot of FAFB (Female Adult Fly Brain) served
by Codex. It is not redistributed in this repository - run this script once
and the files land in ./data/.

Data credit: FlyWire (Dorkenwald et al., 2024; Schlegel et al., 2024),
Princeton University / MRC LMB. See CITATION.md.
"""

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "connections.csv.gz": "neuron-to-neuron connections, synapse counts, neurotransmitters",
    "classification.csv.gz": "per-neuron class / super_class / flow annotations",
    "neurons.csv.gz": "per-neuron neurotransmitter predictions and scores",
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r    {pct:5.1f}%  ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
        sys.stdout.flush()


def download(force=False):
    DATA_DIR.mkdir(exist_ok=True)
    for name, description in FILES.items():
        target = DATA_DIR / name
        if target.exists() and not force:
            print(f"  [skip] {name} already present")
            continue
        print(f"  [get ] {name} - {description}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", target, _progress)
        print()
    print(f"\nData ready in {DATA_DIR}")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
