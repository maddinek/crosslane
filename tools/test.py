#!/usr/bin/env python3
"""Run the standalone CPU suite with at most two concurrent compiler jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-j', type=int, choices=(1,2), default=2)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    tests = sorted((root/'tests').glob('test_*.py'))+sorted((root/'engine').rglob('test_*.py'))
    def run(path):
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', str(path)], cwd=root,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return path.relative_to(root), r.returncode, r.stdout
    failed = False
    with ThreadPoolExecutor(max_workers=a.j) as pool:
        for name, rc, output in pool.map(run, tests):
            print(str(name)+':\n'+output, flush=True)
            failed |= bool(rc)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__': main()
