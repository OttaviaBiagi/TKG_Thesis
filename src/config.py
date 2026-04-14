# src/config.py
NEO4J_URI = "bolt://172.22.43.151:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Data paths — auto-detect environment
from pathlib import Path

if Path('/home/obiaggi/3w_temp').exists():
    # Linux remote machine
    DATA_ROOT_3W = Path('/home/obiaggi/3w_temp/dataset')
    PROCESSED_DIR = Path('/home/obiaggi/3w_processed')
else:
    # Windows local
    DATA_ROOT_3W = Path(__file__).parent.parent / 'data/UseCase2/3w_dataset'
    PROCESSED_DIR = Path(__file__).parent.parent / 'data/processed'