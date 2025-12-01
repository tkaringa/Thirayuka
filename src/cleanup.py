# cleanup script for submission preparation

import os
import shutil
import glob

def remove_pycache(root_dir):
    print(f"Scanning {root_dir} for __pycache__...")
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                print(f"Removing {path}")
                shutil.rmtree(path)

def remove_checkpoints(results_dir):
    print(f"Scanning {results_dir} for checkpoints...")
    if not os.path.exists(results_dir):
        return
    
    # Look for checkpoint folders
    checkpoints = glob.glob(os.path.join(results_dir, "checkpoint-*"))
    for cp in checkpoints:
        if os.path.isdir(cp):
            print(f"Removing checkpoint: {cp}")
            shutil.rmtree(cp)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Cleaning up project at: {base_dir}")
    
    # 1. Remove __pycache__
    remove_pycache(base_dir)
    
    # 2. Remove training checkpoints (keep final model in models/)
    results_dir = os.path.join(base_dir, 'results')
    remove_checkpoints(results_dir)
    
    print("Cleanup complete. Ready for submission/archiving.")

if __name__ == '__main__':
    main()
