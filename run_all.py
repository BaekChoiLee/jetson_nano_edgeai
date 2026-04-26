import argparse
import sys
import os
import time

# benchmark 디렉토리에 있는 모듈 임포트
from benchmark.benchmark import BenchmarkMaster

def main():
    parser = argparse.ArgumentParser(description="Jetson Nano Edge AI Pipeline")
    parser.add_argument("--model", type=str, default="all", help="Model name (all, mobilenetv3s, efficientnetb0, shufflenetv2, resnet50, yolov8n, ssd_mv2)")
    parser.add_argument("--runtime", type=str, default="all", help="Runtime (all, pytorch_cpu, pytorch_cuda, tensorrt_fp32, tensorrt_fp16, tensorrt_int8, onnxrt_cpu, tflite_cpu, ncnn_vulkan)")
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "det", "all"], help="Task type (cls, det, all)")
    parser.add_argument("--power-mode", type=str, default="10w", choices=["5w", "10w"], help="Jetson power mode (5w, 10w)")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    # 모델 목록 확장
    models_to_run = []
    if args.model == "all":
        if args.task == "cls" or args.task == "all":
            models_to_run.extend(["mobilenetv3s", "efficientnetb0", "shufflenetv2", "resnet50"])
        if args.task == "det" or args.task == "all":
            models_to_run.extend(["yolov8n", "ssd_mv2"])
    else:
        models_to_run.append(args.model)
        
    runtimes_to_run = []
    if args.runtime == "all":
        runtimes_to_run = [
            ("pytorch_cpu", "cpu"),
            ("pytorch_cuda", "cuda"),
            ("tensorrt_fp32", "cuda"),
            ("tensorrt_fp16", "cuda"),
            ("tensorrt_int8", "cuda"),
            ("onnxrt_cpu", "cpu"),
            ("tflite_cpu", "cpu"),
            ("ncnn_vulkan", "vulkan")
        ]
    else:
        # 단일 런타임 지정 시 device 매핑
        dev = "cuda" if "cuda" in args.runtime or "fp" in args.runtime or "int8" in args.runtime else "cpu"
        if "vulkan" in args.runtime: dev = "vulkan"
        runtimes_to_run.append((args.runtime, dev))

    print(f"=====================================")
    print(f"Starting Benchmark Pipeline")
    print(f"Power Mode: {args.power_mode}")
    print(f"Task: {args.task}")
    print(f"Models: {models_to_run}")
    print(f"Runtimes: {[rt for rt, _ in runtimes_to_run]}")
    print(f"=====================================")

    # 각 모델별 실행
    for model_name in models_to_run:
        print(f"\n[Processing Model: {model_name}]")

        # 이전 실행 결과가 있어도 이번 실행 결과로 덮어쓴다.
        csv_path = f"./results/benchmark_{model_name}.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"  - [OVERWRITE] Removed previous result file: {csv_path}")

        master = BenchmarkMaster(model_name)
                    
        for rt, dev in runtimes_to_run:
            try:
                master.run_runtime_benchmark(
                    runtime=rt, 
                    device_str=dev, 
                    warmup=args.warmup, 
                    runs=args.runs
                )
                print(f"  - Waiting for cool-down (10s)...")
                time.sleep(10) # 열 스로틀링 방지용 대기
            except Exception as e:
                import traceback
                print(f"[Error] {model_name} on {rt} failed: {e}")
                traceback.print_exc()
                continue # 특정 조합 실패 시 스킵하고 다음으로 진행
                
        # 모델 하나가 끝나면 저장 (result_saver 연동은 BenchmarkMaster 내부에 구현됨)
        master.save_all()

if __name__ == "__main__":
    main()
