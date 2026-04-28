# ResNet50 Runtime Layer Analysis

## Scope

이 보고서는 `results/benchmark_resnet50.csv`와 `results/figures_by_model/resnet50/layer_groups.csv`를 기준으로 ResNet50의 런타임별 성능을 분석한다. 비교 대상은 `pytorch_cuda`, `onnxrt_cpu`, `tensorrt_fp32`, `tensorrt_fp16`, `tensorrt_int8`이다.

주의할 점:

- 전체 latency는 `benchmark_resnet50.csv`의 end-to-end 측정값을 기준으로 한다.
- 레이어 단위 비교는 `layer_groups.csv`를 기준으로 한다.
- 현재 TensorRT 레이어 CSV는 이전 `trtexec` 파서가 일부 layer name을 보존하지 못해 `TensorRT layer 001` 형태로 표시된다. 따라서 TensorRT 매칭은 이름 기반이 아니라 실행 순서 기반 유추이다.
- TensorRT layer profile row의 합계는 end-to-end latency와 직접 일치하지 않는다. TensorRT의 profile 출력 단위, 반복/분리 profile run, parser 한계 때문에 레이어별 수치는 상대 비교용으로만 사용한다.
- `power_total_mw_mean`이 대부분 0으로 기록되어 총 전력 해석은 제한적이다. 전력 관련 해석은 `gpu_util_pct_mean`, `ram_used_mb_mean`, `co2_kg`, 런타임 특성을 함께 본 추론이다.

## ResNet50 Structure

ResNet50은 bottleneck residual block을 사용하는 CNN이다. 기본 구조는 다음과 같다.

- Stem: `7x7 conv -> batch norm -> ReLU -> maxpool`
- Stage 1: `layer1`, bottleneck block 3개
- Stage 2: `layer2`, bottleneck block 4개
- Stage 3: `layer3`, bottleneck block 6개
- Stage 4: `layer4`, bottleneck block 3개
- Head: global average pool -> fully connected

각 bottleneck block은 보통 다음 구조를 가진다.

```text
1x1 conv -> BN -> ReLU
3x3 conv -> BN -> ReLU
1x1 conv -> BN
residual add -> ReLU
```

stage 첫 block에서는 spatial/channel shape가 바뀌므로 skip branch에 `downsample.0 + downsample.1`이 추가된다. 현재 PyTorch 시각화 그룹은 이 구조를 반영하여 `conv+norm+act`, `conv+norm`, `downsample conv+norm` 단위로 묶었다.

## End-to-End Runtime Summary

| Runtime | Mean latency (ms) | vs PyTorch CUDA | GPU util mean (%) | RAM mean (MB) | CO2 kg |
|---|---:|---:|---:|---:|---:|
| PyTorch CUDA | 123.48 | 1.00x | 70.06 | 2770.81 | 0.000017 |
| TensorRT FP32 | 77.31 | 1.60x faster | 62.46 | 3512.57 | 0.000029 |
| TensorRT FP16 | 42.16 | 2.93x faster | 62.22 | 3439.81 | 0.000024 |
| TensorRT INT8 | 77.07 | 1.60x faster | 58.38 | 3445.61 | 0.000040 |
| ONNX Runtime CPU | 984.17 | 7.97x slower | 0.00 | 3039.34 | 0.000139 |

핵심 관찰:

- ONNX Runtime CPU는 GPU를 쓰지 않으므로 convolution-heavy workload에서 매우 느리다.
- TensorRT FP32는 PyTorch CUDA보다 빠르지만, FP16 대비 크게 느리다.
- TensorRT FP16이 가장 빠르다. Jetson Nano에서는 메모리 대역폭과 FP16 연산량 감소 효과가 크다.
- TensorRT INT8은 FP16보다 느리고 FP32와 비슷하다. Jetson Nano의 Maxwell GPU에서는 최신 Tensor Core 기반 INT8 가속을 기대하기 어렵고, quantize/dequantize/reformat 및 unsupported layer fallback 비용이 커질 수 있다.

## Layer Mapping Method

매칭 규칙:

- PyTorch CUDA는 `conv+bn+activation`, `conv+bn`, `pool`, `fc` 단위로 그룹화했다.
- ONNX Runtime CPU는 ONNX graph node name을 PyTorch module name으로 정규화하여 이름 기반으로 매칭했다.
- TensorRT는 현재 CSV에 실제 layer name이 보존되지 않은 경우가 있어 실행 순서 기반으로 매칭했다. 따라서 `TensorRT layer 001`은 PyTorch group 1과 직접 1:1 연산이라는 뜻이 아니라, profile row 순서상 대응되는 후보라는 뜻이다.
- TensorRT는 fusion, tactic selection, layout conversion 때문에 `1:N` 또는 `N:1` 매칭이 자연스럽다. 예를 들어 PyTorch의 `conv+bn+relu`는 TensorRT에서 하나의 fused convolution kernel로 합쳐질 수 있고, 반대로 하나의 PyTorch group이 TensorRT 내부에서 reformat + convolution + activation으로 쪼개질 수 있다.

## Full Layer Mapping

| # | PyTorch CUDA group | I/O | Params | PT ms | ONNX Runtime CPU mapping | ORT ms | TRT FP32 | TRT FP16 | TRT INT8 | Mapping note |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `conv1` | `1x3x224x224->1x64x112x112` | 9.5K | 4.22 | `conv1` | 28.97 | L001 3.1 | L001 0.9 | L001 3.1 | name/order |
| 2 | `maxpool` | `1x64x112x112->1x64x56x56` | 0 | 1.52 | `maxpool` | 3.58 | L002 0.6 | L002 3.6 | L002 0.6 | name/order |
| 3 | `layer1.0.conv1` | `1x64x56x56->1x64x56x56` | 4.2K | 1.80 | `layer1/layer1.0/conv1` | 3.46 | L003 0.4 | L003 0.6 | L003 0.4 | name/order |
| 4 | `layer1.0.conv2` | `1x64x56x56->1x64x56x56` | 37.0K | 2.34 | `layer1/layer1.0/conv2` | 26.00 | L004 1.7 | L004 0.5 | L004 1.7 | name/order |
| 5 | `layer1.0.conv3` | `1x64x56x56->1x256x56x56` | 16.9K | 2.60 | `layer1/layer1.0/conv3` | 14.04 | L005 1.5 | L005 1.8 | L005 1.5 | name/order |
| 6 | `layer1.0.downsample.0` | `1x64x56x56->1x256x56x56` | 16.9K | 2.56 | `layer1/layer1.0/downsample/downsample.0` | 11.40 | L006 1.6 | L006 1.6 | L006 1.6 | name/order |
| 7 | `layer1.1.conv1` | `1x256x56x56->1x64x56x56` | 16.5K | 2.69 | `layer1/layer1.1/conv1` | 12.92 | L007 1.1 | L007 1.7 | L007 1.1 | name/order |
| 8 | `layer1.1.conv2` | `1x64x56x56->1x64x56x56` | 37.0K | 2.32 | `layer1/layer1.1/conv2` | 26.13 | L008 1.7 | L008 1.1 | L008 1.7 | name/order |
| 9 | `layer1.1.conv3` | `1x64x56x56->1x256x56x56` | 16.9K | 2.57 | `layer1/layer1.1/conv3` | 14.22 | L009 1.6 | L009 1.8 | L009 1.6 | name/order |
| 10 | `layer1.2.conv1` | `1x256x56x56->1x64x56x56` | 16.5K | 2.67 | `layer1/layer1.2/conv1` | 13.06 | L010 1.1 | L010 1.7 | L010 1.1 | name/order |
| 11 | `layer1.2.conv2` | `1x64x56x56->1x64x56x56` | 37.0K | 2.29 | `layer1/layer1.2/conv2` | 26.26 | L011 1.7 | L011 1.1 | L011 1.7 | name/order |
| 12 | `layer1.2.conv3` | `1x64x56x56->1x256x56x56` | 16.9K | 2.56 | `layer1/layer1.2/conv3` | 13.87 | L012 1.6 | L012 1.8 | L012 1.6 | name/order |
| 13 | `layer2.0.conv1` | `1x256x56x56->1x128x56x56` | 33.0K | 3.24 | `layer2/layer2.0/conv1` | 23.03 | L013 2.2 | L013 1.7 | L013 2.2 | name/order |
| 14 | `layer2.0.conv2` | `1x128x56x56->1x128x28x28` | 147.7K | 4.99 | `layer2/layer2.0/conv2` | 26.15 | L014 2.6 | L014 2.1 | L014 2.6 | name/order |
| 15 | `layer2.0.conv3` | `1x128x28x28->1x512x28x28` | 66.6K | 2.31 | `layer2/layer2.0/conv3` | 12.23 | L015 1.4 | L015 2.4 | L015 1.4 | name/order |
| 16 | `layer2.0.downsample.0` | `1x256x56x56->1x512x28x28` | 132.1K | 3.73 | `layer2/layer2.0/downsample/downsample.0` | 22.24 | L016 2.6 | L016 1.4 | L016 2.6 | name/order |
| 17 | `layer2.1.conv1` | `1x512x28x28->1x128x28x28` | 65.8K | 2.65 | `layer2/layer2.1/conv1` | 11.47 | L017 1.2 | L017 2.4 | L017 1.2 | name/order |
| 18 | `layer2.1.conv2` | `1x128x28x28->1x128x28x28` | 147.7K | 2.37 | `layer2/layer2.1/conv2` | 24.51 | L018 1.7 | L018 1.1 | L018 1.7 | name/order |
| 19 | `layer2.1.conv3` | `1x128x28x28->1x512x28x28` | 66.6K | 2.31 | `layer2/layer2.1/conv3` | 12.20 | L019 1.4 | L019 1.8 | L019 1.4 | name/order |
| 20 | `layer2.2.conv1` | `1x512x28x28->1x128x28x28` | 65.8K | 2.64 | `layer2/layer2.2/conv1` | 11.43 | L020 1.2 | L020 1.4 | L020 1.2 | name/order |
| 21 | `layer2.2.conv2` | `1x128x28x28->1x128x28x28` | 147.7K | 2.35 | `layer2/layer2.2/conv2` | 24.84 | L021 1.7 | L021 1.1 | L021 1.7 | name/order |
| 22 | `layer2.2.conv3` | `1x128x28x28->1x512x28x28` | 66.6K | 2.30 | `layer2/layer2.2/conv3` | 12.46 | L022 1.4 | L022 1.7 | L022 1.4 | name/order |
| 23 | `layer2.3.conv1` | `1x512x28x28->1x128x28x28` | 65.8K | 2.63 | `layer2/layer2.3/conv1` | 11.95 | L023 1.2 | L023 1.4 | L023 1.2 | name/order |
| 24 | `layer2.3.conv2` | `1x128x28x28->1x128x28x28` | 147.7K | 2.34 | `layer2/layer2.3/conv2` | 25.51 | L024 1.7 | L024 1.1 | L024 1.7 | name/order |
| 25 | `layer2.3.conv3` | `1x128x28x28->1x512x28x28` | 66.6K | 2.31 | `layer2/layer2.3/conv3` | 12.48 | L025 1.4 | L025 1.7 | L025 1.4 | name/order |
| 26 | `layer3.0.conv1` | `1x512x28x28->1x256x28x28` | 131.6K | 4.03 | `layer3/layer3.0/conv1` | 22.87 | L026 2.3 | L026 1.4 | L026 2.3 | name/order |
| 27 | `layer3.0.conv2` | `1x256x28x28->1x256x14x14` | 590.3K | 5.50 | `layer3/layer3.0/conv2` | 28.26 | L027 2.9 | L027 2.2 | L027 2.9 | name/order |
| 28 | `layer3.0.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.53 | `layer3/layer3.0/conv3` | 12.39 | L028 1.4 | L028 2.7 | L028 1.4 | name/order |
| 29 | `layer3.0.downsample.0` | `1x512x28x28->1x1024x14x14` | 526.3K | 4.23 | `layer3/layer3.0/downsample/downsample.0` | 24.06 | L029 2.7 | L029 1.4 | L029 2.7 | name/order |
| 30 | `layer3.1.conv1` | `1x1024x14x14->1x256x14x14` | 262.7K | 2.76 | `layer3/layer3.1/conv1` | 11.69 | L030 1.3 | L030 2.6 | L030 1.3 | name/order |
| 31 | `layer3.1.conv2` | `1x256x14x14->1x256x14x14` | 590.3K | 2.67 | `layer3/layer3.1/conv2` | 26.93 | L031 1.6 | L031 1.2 | L031 1.6 | name/order |
| 32 | `layer3.1.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.52 | `layer3/layer3.1/conv3` | 12.08 | L032 1.5 | L032 1.6 | L032 1.5 | name/order |
| 33 | `layer3.2.conv1` | `1x1024x14x14->1x256x14x14` | 262.7K | 2.74 | `layer3/layer3.2/conv1` | 11.60 | L033 1.3 | L033 1.4 | L033 1.3 | name/order |
| 34 | `layer3.2.conv2` | `1x256x14x14->1x256x14x14` | 590.3K | 2.65 | `layer3/layer3.2/conv2` | 26.69 | L034 1.6 | L034 1.2 | L034 1.6 | name/order |
| 35 | `layer3.2.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.51 | `layer3/layer3.2/conv3` | 12.38 | L035 1.5 | L035 1.6 | L035 1.5 | name/order |
| 36 | `layer3.3.conv1` | `1x1024x14x14->1x256x14x14` | 262.7K | 2.72 | `layer3/layer3.3/conv1` | 11.84 | L036 1.3 | L036 1.4 | L036 1.3 | name/order |
| 37 | `layer3.3.conv2` | `1x256x14x14->1x256x14x14` | 590.3K | 2.63 | `layer3/layer3.3/conv2` | 26.77 | L037 1.6 | L037 1.2 | L037 1.6 | name/order |
| 38 | `layer3.3.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.58 | `layer3/layer3.3/conv3` | 12.19 | L038 1.5 | L038 1.6 | L038 1.5 | name/order |
| 39 | `layer3.4.conv1` | `1x1024x14x14->1x256x14x14` | 262.7K | 2.73 | `layer3/layer3.4/conv1` | 11.87 | L039 1.3 | L039 1.4 | L039 1.3 | name/order |
| 40 | `layer3.4.conv2` | `1x256x14x14->1x256x14x14` | 590.3K | 2.63 | `layer3/layer3.4/conv2` | 27.16 | L040 1.6 | L040 1.2 | L040 1.6 | name/order |
| 41 | `layer3.4.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.51 | `layer3/layer3.4/conv3` | 12.36 | L041 1.5 | L041 1.6 | L041 1.5 | name/order |
| 42 | `layer3.5.conv1` | `1x1024x14x14->1x256x14x14` | 262.7K | 2.73 | `layer3/layer3.5/conv1` | 12.01 | L042 1.3 | L042 1.4 | L042 1.3 | name/order |
| 43 | `layer3.5.conv2` | `1x256x14x14->1x256x14x14` | 590.3K | 2.65 | `layer3/layer3.5/conv2` | 27.25 | L043 1.6 | L043 1.2 | L043 1.6 | name/order |
| 44 | `layer3.5.conv3` | `1x256x14x14->1x1024x14x14` | 264.2K | 2.51 | `layer3/layer3.5/conv3` | 12.44 | L044 1.5 | L044 1.6 | L044 1.5 | name/order |
| 45 | `layer4.0.conv1` | `1x1024x14x14->1x512x14x14` | 525.3K | 4.32 | `layer4/layer4.0/conv1` | 22.87 | L045 2.6 | L045 1.4 | L045 2.6 | name/order |
| 46 | `layer4.0.conv2` | `1x512x14x14->1x512x7x7` | 2.36M | 5.97 | `layer4/layer4.0/conv2` | 30.03 | L046 4.8 | L046 2.4 | L046 4.8 | name/order |
| 47 | `layer4.0.conv3` | `1x512x7x7->1x2048x7x7` | 1.05M | 2.69 | `layer4/layer4.0/conv3` | 14.29 | L047 2.6 | L047 5.2 | L047 2.7 | name/order |
| 48 | `layer4.0.downsample.0` | `1x1024x14x14->1x2048x7x7` | 2.10M | 4.61 | `layer4/layer4.0/downsample/downsample.0` | 25.83 | L048 5.1 | L048 2.5 | L048 5.2 | name/order |
| 49 | `layer4.1.conv1` | `1x2048x7x7->1x512x7x7` | 1.05M | 3.03 | `layer4/layer4.1/conv1` | 15.80 | L049 1.6 | L049 4.8 | L049 0.1 | name/order |
| 50 | `layer4.1.conv2` | `1x512x7x7->1x512x7x7` | 2.36M | 5.39 | `layer4/layer4.1/conv2` | 29.92 | L050 2.4 | L050 1.9 | L050 1.3 | name/order |
| 51 | `layer4.1.conv3` | `1x512x7x7->1x2048x7x7` | 1.05M | 2.71 | `layer4/layer4.1/conv3` | 14.07 | L051 2.7 | L051 2.5 | L051 2.4 | name/order |
| 52 | `layer4.2.conv1` | `1x2048x7x7->1x512x7x7` | 1.05M | 3.02 | `layer4/layer4.2/conv1` | 15.29 | L052 1.6 | L052 2.5 | L052 2.7 | name/order |
| 53 | `layer4.2.conv2` | `1x512x7x7->1x512x7x7` | 2.36M | 5.38 | `layer4/layer4.2/conv2` | 29.64 | L053 2.4 | L053 1.9 | L053 0.1 | name/order |
| 54 | `layer4.2.conv3` | `1x512x7x7->1x2048x7x7` | 1.05M | 2.72 | `layer4/layer4.2/conv3` | 13.72 | L054 2.7 | L054 2.4 | L054 1.3 | name/order |
| 55 | `avgpool` | `1x2048x7x7->1x2048x1x1` | 0 | 0.50 | `avgpool` | 0.20 | L055 0.1 | L055 2.5 | L055 2.4 | name/order |
| 56 | `fc` | `1x2048->1x1000` | 2.05M | 1.00 | `fc` | 2.01 | L056 0.8 | L056 0.2 | L056 2.7 | name/order |

## Why the Speed Differences Occur

### PyTorch CUDA vs ONNX Runtime CPU

ONNX Runtime CPU는 모든 convolution을 CPU에서 처리하므로 ResNet50처럼 convolution 비중이 높은 모델에서 병목이 크다. 특히 다음 계열이 크게 느리다.

- `layer1.*.conv2`: 56x56 feature map의 3x3 convolution
- `layer2.*.conv2`: 28x28 feature map의 3x3 convolution
- `layer3.*.conv2`: 14x14 feature map의 3x3 convolution
- `layer4.*.conv2`: 7x7 feature map이지만 channel 수가 매우 크고 parameter 수가 2.36M

CPU 런타임은 GPU kernel parallelism, TensorRT tactic selection, fused kernel의 이점을 쓰지 못한다. 따라서 PyTorch CUDA 대비 약 8배 느렸다.

### PyTorch CUDA vs TensorRT

TensorRT는 다음 최적화를 수행할 수 있다.

- convolution + batch normalization folding
- activation fusion
- residual branch 주변 elementwise fusion
- kernel tactic search
- layout/reformat 최적화
- FP16/INT8 precision lowering

PyTorch CUDA는 eager/module execution overhead와 framework dispatch cost가 남아 있다. 반면 TensorRT engine은 정적 graph로 빌드되어 kernel scheduling과 memory planning이 더 공격적으로 최적화된다. 이 때문에 FP32만 사용해도 PyTorch CUDA보다 빨라진다.

### ONNX/TensorRT Layer Mapping Characteristics

PyTorch의 `conv+bn+relu`는 ONNX에서는 Conv, BatchNorm fold, activation/elementwise node로 나뉘거나 export 시 이미 일부 fold될 수 있다. TensorRT에서는 보통 더 많이 fusion되어 하나의 TensorRT layer 또는 소수의 internal layer로 처리된다. 따라서 다음 관계가 흔하다.

- PyTorch `conv+bn+relu` -> ONNX `Conv + activation/mul/add` 형태: 1:N
- PyTorch `conv+bn+relu` -> TensorRT fused convolution: N:1
- residual add와 final ReLU -> ONNX/TensorRT에서 elementwise/fused activation으로 위치가 이동
- downsample branch -> main branch와 병렬로 존재하지만 profile 출력에서는 순서가 바뀌거나 합쳐질 수 있음

## TensorRT FP32 / FP16 / INT8 Comparison

End-to-end 결과:

| TensorRT precision | Mean latency (ms) | Relative to FP32 | Relative to FP16 |
|---|---:|---:|---:|
| FP32 | 77.31 | 1.00x | 1.83x slower |
| FP16 | 42.16 | 1.83x faster | 1.00x |
| INT8 | 77.07 | 1.00x | 1.83x slower |

FP16이 가장 빠른 이유로 추정되는 요인:

- FP16은 activation/weight bandwidth가 FP32 대비 절반이다.
- Jetson Nano의 메모리 대역폭 제약이 큰 환경에서 ResNet50의 convolution workload는 bandwidth 절감 효과가 크다.
- TensorRT FP16 tactic은 많은 convolution을 FP16 kernel로 처리할 수 있어 kernel occupancy와 memory traffic 측면에서 유리하다.
- BatchNorm folding 후 convolution 중심 workload가 되므로 FP16 최적화 효과가 더 뚜렷하다.

INT8이 FP16보다 빠르지 않은 이유로 추정되는 요인:

- Jetson Nano의 Maxwell GPU는 최신 Tensor Core 기반 INT8 경로와 다르다. INT8 연산이 항상 FP16보다 빠르다고 기대하기 어렵다.
- 현재 engine 생성은 `--int8` 위주이며 calibration/Q-DQ 최적화가 충분하지 않을 수 있다.
- INT8 unsupported layer나 precision-sensitive layer가 FP32/FP16 fallback될 수 있다.
- INT8 boundary에서 reformat, quantize/dequantize, scale 처리 비용이 발생할 수 있다.
- 작은 1x1 convolution이나 residual/activation 주변에서는 INT8 변환 overhead가 연산 이득보다 클 수 있다.

전력/에너지 관점:

- 측정된 `power_total_mw_mean`은 0으로 기록되어 총 전력 직접 비교는 불가능하다.
- `gpu_util_pct_mean`은 FP32 62.46%, FP16 62.22%, INT8 58.38%로 비슷하거나 INT8이 낮다.
- 하지만 `co2_kg`는 FP16이 INT8보다 낮다. 이는 FP16이 실행 시간이 훨씬 짧아 총 에너지 적분값이 작아졌기 때문으로 해석할 수 있다.
- INT8은 GPU util이 약간 낮더라도 시간이 길고 변환 overhead가 있어 총 에너지가 커질 수 있다.

## Mixed Precision Prediction

레이어 profile row에서 가장 빠른 precision을 단순 선택하면 다음과 같은 분포가 나온다.

| Best precision by TensorRT profile row | Count |
|---|---:|
| FP32 | 29 |
| FP16 | 22 |
| INT8 | 6 |

하지만 이 결과를 그대로 mixed precision engine에 적용하면 안 된다. 이유는 다음과 같다.

- TensorRT layer profile row 합계가 end-to-end latency와 일치하지 않는다.
- 실제 mixed precision engine은 precision boundary마다 reformat/cast 비용이 생긴다.
- FP32/FP16/INT8 tactic을 임의로 섞으면 memory format이 자주 바뀌어 오히려 느려질 수 있다.
- residual add처럼 두 branch의 precision/layout을 맞춰야 하는 지점이 많다.

이론적 하한:

- profile row별 최소값만 선택한 합계는 FP16 profile 합계 대비 약 13% 낮다.
- 이를 end-to-end FP16 42.16ms에 단순 적용하면 약 36.6ms가 하한처럼 보인다.

현실적 예측:

- precision boundary overhead를 고려하면 실제 mixed precision은 약 38-45ms 범위가 될 가능성이 높다.
- 잘 설계된 mixed precision이 FP16보다 약간 빠를 수는 있지만, Jetson Nano에서는 FP16 단일 precision이 가장 안정적인 선택일 가능성이 높다.
- INT8은 특정 대형 3x3 convolution이나 일부 late-stage layer에서만 제한적으로 이득을 줄 수 있고, 전 모델 INT8은 현재 결과처럼 FP16보다 느릴 수 있다.

## Overall Interpretation

1. ResNet50의 주요 병목은 bottleneck 내부 `conv2` 계열의 3x3 convolution과 `layer4`의 고채널 convolution이다.
2. ONNX Runtime CPU는 ResNet50의 convolution 병렬성을 활용하지 못해 PyTorch CUDA 대비 매우 느리다.
3. TensorRT는 graph-level fusion, BN folding, memory planning, tactic selection으로 PyTorch CUDA보다 빠르다.
4. TensorRT FP16은 Jetson Nano에서 가장 좋은 latency를 보였다. FP16은 bandwidth와 arithmetic cost를 동시에 줄이며, convolution 중심 모델에 잘 맞는다.
5. TensorRT INT8은 FP16보다 느렸다. 이는 calibration/precision fallback/reformat overhead와 Jetson Nano GPU 특성 때문으로 추정된다.
6. mixed precision은 이론적으로 일부 layer에서 개선 여지가 있으나, precision boundary cost 때문에 FP16 단일 engine을 크게 이기기는 어렵다.
7. 보고서 수치에서 TensorRT layer-level 값은 상대적 경향과 매칭 참고용이고, 최종 성능 판단은 end-to-end benchmark latency를 기준으로 해야 한다.

## Recommended Next Steps

- TensorRT profiler parser 수정 후 ResNet50만 다시 벤치마크하여 실제 TensorRT layer name을 확보한다.
- `trtexec --profilingVerbosity=detailed` 또는 engine build 시 detailed profiling 설정을 추가한다.
- INT8은 representative calibration dataset을 사용한 calibration/Q-DQ 기반 engine으로 다시 측정한다.
- 전력 비교는 `tegrastats`의 `VDD_IN` 또는 `POM_5V_IN` 필드가 실제로 수집되는지 확인한 뒤 재측정한다.
- mixed precision 실험은 FP16 baseline을 기준으로 일부 late-stage convolution만 INT8 후보로 제한하여 시도한다.
