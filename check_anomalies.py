import json, sys

with open(sys.argv[1]) as f:
    d = json.load(f)

print("=" * 70)
print("  ANOMALY CHECK — 64-cell matrix")
print("=" * 70)

for model, runtimes in d['matrix'].items():
    for rt, v in runtimes.items():
        issues = []
        m10 = v.get('latency_MAXN_mean_ms')
        m5 = v.get('latency_5W_mean_ms')

        # 1) 5W should be slower than MAXN
        if m10 and m5 and m5 < m10:
            issues.append(f'5W FASTER than 10W: {m5:.1f} < {m10:.1f}')

        # 2) Accuracy missing
        if v.get('accuracy_status') == 'missing':
            issues.append('accuracy MISSING')

        # 3) INT8 severe accuracy drop
        if 'int8' in rt:
            t1 = v.get('accuracy_top1')
            m50 = v.get('accuracy_map_50')
            if t1 is not None and t1 < 80:
                issues.append(f'INT8 top1 VERY LOW: {t1}%')
            if m50 is not None and m50 < 0.05:
                issues.append(f'INT8 mAP COLLAPSED: {m50}')

        # 4) ncnn accuracy anomaly for detection
        if 'ncnn' in rt:
            m50 = v.get('accuracy_map_50')
            if m50 is not None and m50 < 0.05:
                issues.append(f'ncnn mAP COLLAPSED: {m50}')

        # 5) TFLite GPU slower than CPU (CPU fallback indicator)
        if rt == 'tflite_gpu' and m10:
            cpu_rt = runtimes.get('tflite_cpu', {})
            cpu_m10 = cpu_rt.get('latency_MAXN_mean_ms')
            if cpu_m10 and m10 > cpu_m10 * 1.1:
                ratio = m10 / cpu_m10
                issues.append(f'TFLite GPU {ratio:.2f}x SLOWER than CPU ({m10:.1f} vs {cpu_m10:.1f})')
            elif cpu_m10 and abs(m10 - cpu_m10) / cpu_m10 < 0.1:
                issues.append(f'TFLite GPU ~ CPU (CPU fallback likely: {m10:.1f} vs {cpu_m10:.1f})')

        # 6) ORT CUDA much slower than PyTorch CUDA
        if rt == 'onnxrt_cuda' and m10:
            pt = runtimes.get('pytorch_cuda', {}).get('latency_MAXN_mean_ms')
            if pt and m10 > pt * 3:
                issues.append(f'ORT CUDA {m10/pt:.1f}x slower than PT CUDA ({m10:.1f} vs {pt:.1f})')

        # 7) INT8 slower than FP32 (unexpected)
        if rt == 'tensorrt_int8' and m10:
            fp32 = runtimes.get('tensorrt_fp32', {}).get('latency_MAXN_mean_ms')
            if fp32 and m10 > fp32 * 1.05:
                issues.append(f'INT8 SLOWER than FP32: {m10:.1f} vs {fp32:.1f}')

        # 8) ORT CUDA accuracy divergence from pytorch
        if rt == 'onnxrt_cuda':
            m50 = v.get('accuracy_map_50')
            pt_m50 = runtimes.get('pytorch_cpu', {}).get('accuracy_map_50')
            if m50 and pt_m50 and abs(m50 - pt_m50) > 0.05:
                issues.append(f'ORT CUDA mAP diverges from PyTorch: {m50:.4f} vs {pt_m50:.4f}')

        if issues:
            tag = 'WARN' if len(issues) == 1 else 'ALERT'
            print(f'[{tag}] {model} / {rt}:')
            for iss in issues:
                print(f'       -> {iss}')
            print()

print("=" * 70)
print("  CHECK COMPLETE")
print("=" * 70)
