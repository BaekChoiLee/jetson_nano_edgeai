# benchmark/trt_profiler.py
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class TRTProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
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
    profiler = TRTProfiler()
    context.profiler = profiler
    return profiler
