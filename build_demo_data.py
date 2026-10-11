"""Publish copies of the trained numerical model and report for static inference."""
from pathlib import Path
import shutil
ROOT=Path(__file__).parent
if __name__=='__main__':
    for name in ['neural-model.json', 'neural-results.json']:
        shutil.copy(ROOT/'artifacts'/name,ROOT/'docs'/name)
