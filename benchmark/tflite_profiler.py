# benchmark/tflite_profiler.py
import subprocess
import re

class TFLiteProfilerParser:
    """
    TFLite의 공식 C++ 툴인 benchmark_model의 출력을 파싱하여 
    레이어(오퍼레이터)별 실행 시간을 추출합니다.
    """
    def __init__(self, tflite_model_path, benchmark_tool_path="./benchmark_model"):
        self.tflite_model_path = tflite_model_path
        self.benchmark_tool_path = benchmark_tool_path

    def run_and_parse(self, runs=100):
        # benchmark_model 실행
        cmd = [
            self.benchmark_tool_path,
            f"--graph={self.tflite_model_path}",
            f"--num_runs={runs}",
            "--enable_op_profiling=true"
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = result.stdout
        except FileNotFoundError:
            print(f"[Error] TFLite benchmark_model tool not found at {self.benchmark_tool_path}.")
            print("Please compile TFLite benchmark_model or set the correct path.")
            return []

        return self._parse_output(output)

    def _parse_output(self, output):
        parsed_results = []
        
        # "Run Order" 섹션 파싱 상태를 관리
        in_run_order_section = False
        
        for line in output.split('\n'):
            line = line.strip()
            
            if "Run Order" in line:
                in_run_order_section = True
                continue
            if in_run_order_section and "Top by Computation Time" in line:
                in_run_order_section = False
                break
                
            if in_run_order_section:
                # [node name] [node type] [avg ms] [avg %] [cdf %] [mem KB] [times called]
                # 예시: StatefulPartitionedCall/conv1/Conv2D  CONV_2D  1.234  8.500%  8.500%  0.000  1
                # 탭이나 여러 개의 공백으로 구분됨
                parts = re.split(r'\s{2,}|\t+', line)
                if len(parts) >= 3 and parts[0] != '[node name]' and parts[0] != '':
                    node_name = parts[0]
                    node_type = parts[1]
                    try:
                        avg_ms = float(parts[2])
                    except ValueError:
                        continue
                        
                    parsed_results.append({
                        "layer_name": node_name,
                        "type": node_type,
                        "mean_ms": avg_ms,
                        "timestamp": None
                    })
                    
        return parsed_results
