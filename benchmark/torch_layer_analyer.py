import torch  
import time   

class LayerProfiler:
    def __init__(self, model):
        self.model = model                # 측정 대상이 될 PyTorch 모델 저장
        self.layer_times = {}             # 각 레이어별 실행 시간 리스트를 저장할 딕셔너리
        self._hooks = []                  # 나중에 제거하기 위해 등록된 훅(Hook)들을 저장하는 리스트
        self._start_times = {}            # 각 레이어의 시작 시간을 임시로 기록할 딕셔너리

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
            # 현재 레이어의 이름을 키로 해서 시작 시간을 고정밀 시계로 기록
            self._start_times[name] = time.perf_counter()
        return hook

    def _post_hook(self, name):
        # 연산이 끝난 후 호출될 hook 함수 정의
        def hook(module, input, output):
            # (현재 시간 - 시작 시간)을 계산하고 1000을 곱해 ms(밀리초) 단위로 변환
            elapsed = (time.perf_counter() - self._start_times[name]) * 1000
            # 해당 레이어 이름이 저장 딕셔너리에 없으면 리스트 생성
            if name not in self.layer_times:
                self.layer_times[name] = []
            # 측정된 시간을 리스트에 추가
            self.layer_times[name].append(elapsed)
        return hook

    def summary(self):
        import numpy as np  # 통계 계산을 위해 numpy 임포트
        # 각 레이어별로 평균 실행 시간을 계산하여 딕셔너리 형태로 반환
        return {
            name: {
                # 측정된 시간들의 평균값을 구하고 소수점 4자리까지 반올림
                'mean_ms': round(float(np.mean(times)), 4),
                # 레이어 이름의 마지막 부분(예: 'conv1')을 타입으로 저장
                'type': name.split('.')[-1]
            }
            for name, times in self.layer_times.items()
        }