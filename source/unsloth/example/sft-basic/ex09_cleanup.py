# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Step 09: clean local artifacts after the run.

Removes: unsloth-output, merged-output, gguf-output.
Prev: ex08_test_vllm. Pipeline done.

Examples:
    uv run ex09_cleanup.py --clean
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEFAULT_DIRS = ["unsloth-output", "merged-output", "gguf-output"]


def parse_args():
    p = argparse.ArgumentParser(description="Step 09: clean local artifacts")
    p.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS)
    p.add_argument("--clean", action="store_true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    if not a.clean:
        print(f"Targets: {a.dirs}")
        print("Dry run. Add --clean to delete.")
        return
    for d in a.dirs:
        p = Path(d)
        if p.is_dir():
            shutil.rmtree(p)
            print(f"Removed: {d}")
        else:
            print(f"Skip (missing): {d}")
    print("Done! Pipeline done.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 09: clean local artifacts after the run.")
        print("  uv run ex09_cleanup.py --help")
        print("  ./ex09_cleanup.sh")
        print("No-op without args.")
        sys.exit(0)
    main()
