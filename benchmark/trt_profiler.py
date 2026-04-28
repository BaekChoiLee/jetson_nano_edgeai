# benchmark/trt_profiler.py
import re

trt = None


def _load_tensorrt():
    global trt
    if trt is not None:
        return trt
    try:
        import tensorrt as trt_module
    except Exception as exc:
        raise ImportError("tensorrt is required for Python API TRT profiling.") from exc
    trt = trt_module
    return trt


class TRTProfiler:
    def __init__(self):
        self.layer_times = {}

    def report_layer_time(self, layer_name, ms):
        """TensorRT 엔진이 각 레이어를 실행할 때마다 이 메서드가 호출됩니다."""
        if layer_name not in self.layer_times:
            self.layer_times[layer_name] = []
        self.layer_times[layer_name].append(ms)

    def summary(self):
        import numpy as np
        return {
            name: {
                "mean_ms": round(float(np.mean(times)), 4),
                "type": name.split('_')[0] if '_' in name else name # 레이어 타입 추정
            }
            for name, times in self.layer_times.items()
        }

def attach_profiler_to_context(context):
    """
    TensorRT IExecutionContext에 프로파일러를 연결합니다.
    사용법:
      profiler = TRTProfiler()
      context.profiler = profiler
      # ... inference run ...
      print(profiler.summary())
    """
    trt_module = _load_tensorrt()

    class _TRTProfiler(trt_module.IProfiler, TRTProfiler):
        def __init__(self):
            trt_module.IProfiler.__init__(self)
            TRTProfiler.__init__(self)

    profiler = _TRTProfiler()
    context.profiler = profiler
    return profiler


class TRTExecProfilerParser:
    """
    Parse trtexec output generated with --dumpProfile/--separateProfileRun.
    This is the most practical TensorRT layer profiler path for Jetson scripts
    because it works even when the benchmark itself is run through trtexec.
    """
    def parse(self, output):
        rows = []
        seen_profile_header = False

        for line in output.splitlines():
            raw = line.strip()
            if not raw:
                continue

            if "Layer" in raw and ("Time" in raw or "Average" in raw):
                seen_profile_header = True
                continue

            cleaned = self._strip_log_prefix(raw)
            if self._is_summary_line(cleaned):
                continue

            # Common trtexec profile rows look like:
            # layer_name: 0.123ms
            # layer_name 0.123
            # Layer: 0.123 ms
            # Layer  12.3%  0.123ms
            match = re.match(r"(?P<name>.+?)\s*:\s*(?P<ms>[0-9]*\.?[0-9]+)\s*ms\b", cleaned)
            if not match:
                match = re.match(r"(?P<name>.+?)\s+(?P<ms>[0-9]*\.?[0-9]+)\s*ms\b", cleaned)
            if not match:
                ms_values = re.findall(r"([0-9]*\.?[0-9]+)\s*ms\b", cleaned)
                if ms_values:
                    name = cleaned[: cleaned.rfind(ms_values[-1])].strip(" :-,\t")
                    match = {"name": name, "ms": float(ms_values[-1])}
            if not match and seen_profile_header:
                parts = re.split(r"\s{2,}|\t+", cleaned)
                if len(parts) >= 2:
                    try:
                        numeric_parts = []
                        for part in parts[1:]:
                            token = part.replace("ms", "").replace("%", "").strip()
                            try:
                                numeric_parts.append(float(token))
                            except ValueError:
                                pass
                        if numeric_parts:
                            match = {"name": parts[0], "ms": numeric_parts[-1]}
                    except ValueError:
                        match = None

            if not match:
                continue

            if isinstance(match, dict):
                name = str(match["name"]).strip()
                mean_ms = float(match["ms"])
            else:
                name = match.group("name").strip()
                mean_ms = float(match.group("ms"))

            if not name or self._is_summary_line(name):
                continue

            rows.append({
                "layer_name": name,
                "type": self._guess_type(name),
                "mean_ms": round(mean_ms, 6),
                "timestamp": None,
            })

        return rows

    def parse_mean_latency_ms(self, output):
        patterns = [
            r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
            r"mean\s*=\s*([0-9.]+)\s*ms",
            r"Average on \d+ runs\s*-\s*GPU latency:\s*([0-9.]+)\s*ms",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _guess_type(self, layer_name):
        lowered = layer_name.lower()
        for token in ("conv", "relu", "pool", "gemm", "matmul", "add", "concat", "shuffle", "scale", "softmax"):
            if token in lowered:
                return token
        return "TensorRTLayer"

    def _strip_log_prefix(self, line):
        cleaned = line.strip()
        # trtexec usually prefixes rows with "[04/28/2026-... ] [I]".
        cleaned = re.sub(r"^\[[^\]]+\]\s*\[[A-Z]\]\s*", "", cleaned)
        cleaned = re.sub(r"^\[[A-Z]\]\s*", "", cleaned)
        return cleaned.strip()

    def _is_summary_line(self, line):
        lowered = line.lower().strip()
        if not lowered:
            return True
        prefixes = (
            "gpu compute",
            "enqueue",
            "h2d",
            "d2h",
            "average on",
            "sleep time",
            "throughput",
            "host latency",
            "end-to-end",
            "input binding",
            "output binding",
        )
        return lowered.startswith(prefixes)
