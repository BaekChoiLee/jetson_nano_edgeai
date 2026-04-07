# result_saver.py
import csv
from pathlib import Path
from datetime import datetime

COLUMNS = [
    # 실험 메타
    'timestamp', 'model', 'runtime', 'precision',
    'power_mode',  # 5W or 10W
    # 속도
    'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'median_ms',
    # 전력·메모리
    'power_total_mw_mean', 'power_gpu_mw_mean',
    'power_cpu_mw_mean',   'ram_used_mb_mean',
    'gpu_util_pct_mean',   'gpu_temp_c_mean',
    # 탄소
    'co2_kg',
    # 정확도
    'accuracy_top1',
]

class ResultSaver:
    def __init__(self, path='results/benchmark.csv'):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        if not self.path.exists():
            with open(self.path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def save(self, row: dict):
        row['timestamp'] = datetime.now().isoformat()
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writerow(row)