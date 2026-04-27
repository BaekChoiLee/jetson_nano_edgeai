import time

import numpy as np
import torch

class LayerProfiler:
    def __init__(self, model, device=None):
        self.model = model                # 측정 대상이 될 PyTorch 모델 저장
        self.device = device or next(model.parameters()).device
        self.layer_records = {}           # 각 레이어별 구조/시간 정보를 저장할 딕셔너리
        self._hooks = []                  # 나중에 제거하기 위해 등록된 훅(Hook)들을 저장하는 리스트
        self._start_times = {}            # 각 레이어의 시작 시간을 임시로 기록할 딕셔너리
        self._modules = dict(model.named_modules())

        """
        pytorch에서 module에 적용하는 hook에 forward_pre_hook, forward_hook, full_backward_hook있음
        """

    def attach(self):
        # 모델의 모든 모듈(레이어)을 이름과 함께 하나씩 순회
        for name, module in self.model.named_modules():
            # leaf layer만 선택(ex : conv, linear 등)
            if len(list(module.children())) == 0:  
                # 레이어 연산 시작 직전에 실행될 'pre_hook' 등록 및 리스트에 저장
                self._hooks.append(
                    module.register_forward_pre_hook(self._pre_hook(name))
                )
                # 레이어 연산 종료 직후에 실행될 'post_hook' 등록 및 리스트에 저장
                self._hooks.append(
                    module.register_forward_hook(self._post_hook(name))
                )

    def detach(self):
        # 등록된 모든 훅을 순회하며 모델에서 제거 (메모리 및 성능 관리)
        for h in self._hooks:
            h.remove()
        # 훅 저장 리스트 비우기
        self._hooks.clear()

    def _pre_hook(self, name):
        # 실제 hook 함수를 반환하는 클로저(Closure) 정의
        def hook(module, input):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            # 현재 레이어의 이름을 키로 해서 시작 시간을 고정밀 시계로 기록
            self._start_times[name] = time.perf_counter()
        return hook

    def _post_hook(self, name):
        # 연산이 끝난 후 호출될 hook 함수 정의
        def hook(module, input, output):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            # (현재 시간 - 시작 시간)을 계산하고 1000을 곱해 ms(밀리초) 단위로 변환
            elapsed = (time.perf_counter() - self._start_times[name]) * 1000
            if name not in self.layer_records:
                self.layer_records[name] = self._describe_module(name, module, input, output)
            self.layer_records[name]["times_ms"].append(elapsed)
            self.layer_records[name]["timestamp"] = time.time()
        return hook

    def reset_timings(self):
        for record in self.layer_records.values():
            record["times_ms"].clear()

    def profile(self, input_tensor, warmup=10, runs=100):
        self.model.eval()
        self.attach()
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    self.model(input_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                self.reset_timings()
                for _ in range(runs):
                    self.model(input_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        finally:
            self.detach()
        return self.summary()

    def _shape_to_str(self, value):
        if torch.is_tensor(value):
            return "x".join(str(dim) for dim in value.shape)
        if isinstance(value, (list, tuple)):
            shapes = [self._shape_to_str(v) for v in value if torch.is_tensor(v) or isinstance(v, (list, tuple))]
            return ";".join(s for s in shapes if s)
        if isinstance(value, dict):
            shapes = [self._shape_to_str(v) for v in value.values()]
            return ";".join(s for s in shapes if s)
        return ""

    def _format_attr(self, value):
        if value is None:
            return ""
        if isinstance(value, tuple):
            return "x".join(str(v) for v in value)
        return str(value)

    def _describe_module(self, name, module, input, output):
        params = list(module.parameters(recurse=False))
        param_count = sum(p.numel() for p in params)
        trainable_param_count = sum(p.numel() for p in params if p.requires_grad)

        return {
            "layer_name": name,
            "type": module.__class__.__name__,
            "module_repr": str(module).replace("\n", " "),
            "input_shape": self._shape_to_str(input),
            "output_shape": self._shape_to_str(output),
            "param_count": param_count,
            "trainable_param_count": trainable_param_count,
            "in_channels": getattr(module, "in_channels", ""),
            "out_channels": getattr(module, "out_channels", ""),
            "kernel_size": self._format_attr(getattr(module, "kernel_size", "")),
            "stride": self._format_attr(getattr(module, "stride", "")),
            "padding": self._format_attr(getattr(module, "padding", "")),
            "dilation": self._format_attr(getattr(module, "dilation", "")),
            "groups": getattr(module, "groups", ""),
            "bias": bool(getattr(module, "bias", None) is not None),
            "times_ms": [],
            "timestamp": time.time(),
        }

    def summary(self):
        rows = []
        for idx, (name, record) in enumerate(self.layer_records.items(), start=1):
            times = np.array(record["times_ms"], dtype=float)
            if times.size == 0:
                continue
            row = {k: v for k, v in record.items() if k != "times_ms"}
            row.update({
                "layer_index": idx,
                "calls": int(times.size),
                "mean_ms": round(float(np.mean(times)), 6),
                "std_ms": round(float(np.std(times)), 6),
                "min_ms": round(float(np.min(times)), 6),
                "max_ms": round(float(np.max(times)), 6),
                "p50_ms": round(float(np.percentile(times, 50)), 6),
                "p95_ms": round(float(np.percentile(times, 95)), 6),
            })
            rows.append(row)
        return rows
