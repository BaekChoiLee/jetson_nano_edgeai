# benchmark/onnxrt_profiler.py
import json
import os

class ONNXRuntimeProfilerParser:
    """
    ONNX Runtime의 자체 프로파일링 (enable_profiling=True) 결과 JSON 파일을 읽어서
    각 Node(레이어)의 평균 실행 시간을 추출합니다.
    """
    def __init__(self, profile_json_path):
        self.profile_json_path = profile_json_path
        self.layer_times = {}

    def parse(self, num_runs=1):
        if not os.path.exists(self.profile_json_path):
            print(f"[Error] ONNX Profiling file not found: {self.profile_json_path}")
            return []

        with open(self.profile_json_path, 'r') as f:
            try:
                trace_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[Error] Failed to parse ONNX JSON: {self.profile_json_path}")
                return []

        # ONNX JSON 트레이스는 여러 이벤트 딕셔너리의 리스트이거나 특정 키 하위에 존재할 수 있음
        if isinstance(trace_data, dict) and 'traceEvents' in trace_data:
            events = trace_data['traceEvents']
        elif isinstance(trace_data, list):
            events = trace_data
        else:
            events = []

        node_durations = {}

        for event in events:
            # cat이 "Node"인 이벤트가 연산 노드를 의미하며, dur는 마이크로초(us) 단위
            if event.get('cat') == 'Node' and 'dur' in event:
                name = event.get('name', 'unknown')
                op_type = event.get('args', {}).get('op_name', 'Unknown')
                dur_ms = event['dur'] / 1000.0
                
                if name not in node_durations:
                    node_durations[name] = {"type": op_type, "total_ms": 0.0, "count": 0}
                
                node_durations[name]["total_ms"] += dur_ms
                node_durations[name]["count"] += 1

        # 결과 리스트로 변환 (timestamp는 정적 파서의 한계로 정확한 시점 매핑이 어려워 0 처리 또는 생략)
        parsed_results = []
        for name, data in node_durations.items():
            parsed_results.append({
                "layer_name": name,
                "type": data["type"],
                # warmup 포함 전체 count로 나눔 (또는 num_runs 사용)
                "mean_ms": round(data["total_ms"] / max(data["count"], 1), 4),
                "timestamp": None # ONNX JSON의 ts는 시스템 시작 기준 등 상이할 수 있음
            })

        return parsed_results
