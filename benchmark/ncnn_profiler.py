# benchmark/ncnn_profiler.py
import subprocess
import re

class NCNNProfilerParser:
    """
    ncnn의 공식 툴인 benchncnn의 출력(stderr/stdout)을 파싱하여
    각 레이어별 실행 시간을 추출합니다.
    """
    def __init__(self, param_path, bin_path, benchncnn_path="./benchncnn"):
        self.param_path = param_path
        self.bin_path = bin_path
        self.benchncnn_path = benchncnn_path
        self.last_output = ""
        self.last_returncode = None

    def run_and_parse(self, runs=100, threads=4, power=0, gpu_device=0, shape="224,224,3"):
        # benchncnn 실행: usage: benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down] [(key=value)...]
        cmd = [
            self.benchncnn_path,
            str(runs),
            str(threads),
            str(power),
            str(gpu_device),
            "1",  # cooling down
            f"param={self.param_path}",
            f"shape=[{shape}]",
        ]
        
        try:
            # benchncnn은 주로 stderr로 프로파일링 정보를 출력함
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = result.stdout
            self.last_output = output
            self.last_returncode = result.returncode
        except FileNotFoundError:
            print(f"[Error] ncnn benchncnn tool not found at {self.benchncnn_path}.")
            print("Please compile benchncnn or set the correct path.")
            return []

        return self._parse_output(output)

    def parse_mean_latency_ms(self, output=None):
        output = self.last_output if output is None else output
        # Common benchncnn summary:
        # min = 1.23  max = 2.34  avg = 1.56
        match = re.search(r"\bavg\s*=\s*([0-9.]+)", output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _parse_output(self, output):
        parsed_results = []
        
        for line in output.split('\n'):
            line = line.strip()
            
            # ncnn 프로파일 로그는 주로 "layer_name  layer_type  [기타수치]  [실행시간ms] ..." 형태로 출력됨
            # 보통 띄어쓰기로 구분되어 있으므로 split
            parts = re.split(r'\s+', line)
            
            # 최소 4개 이상의 필드가 있고, 숫자가 포함되어 있으면 레이어 로그로 간주
            if len(parts) >= 3:
                try:
                    # ncnn 버전마다 출력 포맷이 다르지만 주로 뒤쪽 컬럼에 시간(ms)이 있음
                    # 여기서는 가장 기본적인 포맷: name, type, macs, time 등을 가정
                    layer_name = parts[0]
                    layer_type = parts[1]
                    
                    # 3번째 이후 컬럼 중 float 변환 가능한 첫 번째 또는 마지막 값이 시간일 확률이 높음
                    # 정확한 파싱을 위해, 숫자 형태의 문자열을 찾아 기록
                    nums = []
                    for p in parts[2:]:
                        try:
                            nums.append(float(p))
                        except ValueError:
                            pass
                    
                    if nums:
                        # 통상적으로 마지막 숫자가 퍼센트이거나 시간이므로, 문맥에 따라 시간 선택 (보통 2번째 숫자)
                        # 여기서는 임시로 첫 번째 수치(보통 time)를 사용
                        avg_ms = nums[0] 
                        
                        parsed_results.append({
                            "layer_name": layer_name,
                            "type": layer_type,
                            "mean_ms": avg_ms,
                            "timestamp": None
                        })
                except Exception:
                    continue
                    
        return parsed_results
