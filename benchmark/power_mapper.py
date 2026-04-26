# benchmark/power_mapper.py
import csv
import os

class PowerMapper:
    """
    레이어 프로파일링 결과(타임스탬프 포함)와 tegrastats 모니터링 결과를 시간순으로 매핑하여
    각 레이어별 '추정 전력(Estimated Power)'을 계산합니다.
    """
    def __init__(self, output_dir="./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def map_power(self, layer_profiles, hw_records):
        """
        layer_profiles: list of dict [{"layer_name": "conv1", "type": "Conv", "mean_ms": 1.5, "timestamp": 160000.1}, ...]
        hw_records: list of dict [{"timestamp": 160000.0, "power_gpu_mw": 2000, "power_total_mw": 5000}, ...]
        """
        if not hw_records:
            print("[Warning] No hardware records available for power mapping.")
            return layer_profiles

        # 하드웨어 레코드 시간순 정렬
        hw_records_sorted = sorted(hw_records, key=lambda x: x["timestamp"])

        mapped_results = []
        for layer in layer_profiles:
            lt_ts = layer.get("timestamp")
            if not lt_ts:
                # 타임스탬프가 없는 프레임워크(예: 정적 파서로 가져온 TFLite/ncnn)는 
                # HW 평균값을 할당하거나 0으로 매핑
                layer["power_gpu_mw"] = 0
                layer["power_total_mw"] = 0
                mapped_results.append(layer)
                continue

            # 가장 가까운 하드웨어 타임스탬프 찾기 (Nearest Neighbor Interpolation)
            # Binary search를 쓸 수도 있지만 데이터가 보통 작으므로 순차 탐색/min 활용
            closest_hw = min(hw_records_sorted, key=lambda x: abs(x["timestamp"] - lt_ts))
            
            layer["power_gpu_mw"] = closest_hw.get("power_gpu_mw", 0)
            layer["power_total_mw"] = closest_hw.get("power_total_mw", 0)
            mapped_results.append(layer)

        return mapped_results

    def save_csv(self, mapped_results, model_name, runtime):
        if not mapped_results:
            return
            
        csv_path = os.path.join(self.output_dir, f"layer_power_{model_name}_{runtime}.csv")
        fieldnames = ["layer_name", "type", "mean_ms", "power_gpu_mw", "power_total_mw"]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in mapped_results:
                writer.writerow(r)
        
        print(f"[PowerMapper] Saved layer power profile to {csv_path}")
