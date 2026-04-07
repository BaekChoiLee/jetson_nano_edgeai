// ncnn_bench.cpp
#include "net.h"              // ncnn의 핵심 네트워크 관련 기능을 가져옵니다.
#include <chrono>             // 정밀한 시간 측정을 위한 라이브러리입니다.
#include <numeric>            // std::accumulate 같은 수치 계산 함수를 위해 사용합니다.
#include <vector>             // 측정된 시간 데이터들을 담을 동적 배열입니다.
#include <stdio.h>            // 결과를 출력(printf)하기 위해 사용합니다.

int main(int argc, char** argv) {
    // 실행 시 인자로 받은 모델 구조 파일(.param)과 가중치 파일(.bin) 경로를 가져옵니다.
    const char* param = argv[1];  // 예: "yolov8.param"
    const char* bin   = argv[2];  // 예: "yolov8.bin"
    
    // 웜업(연습 가동) 10회, 실제 측정 100회로 설정합니다.
    int warmup = 10, runs = 100;

    ncnn::Net net;                // ncnn 네트워크 객체를 생성합니다.
    
    // [중요] Vulkan GPU 가속을 활성화합니다. Jetson의 GPU를 사용하게 됩니다.
    net.opt.use_vulkan_compute = true;  
    
    // 모델의 구조와 가중치를 메모리에 로드합니다.
    net.load_param(param);
    net.load_model(bin);

    // 입력 데이터 역할을 할 3채널(RGB), 224x224 크기의 행렬을 만듭니다.
    ncnn::Mat input = ncnn::Mat(224, 224, 3);
    input.fill(0.5f);             // 테스트를 위해 모든 값을 0.5로 채웁니다.

    // --- 웜업 단계 (Warmup) ---
    // 초기 실행 시 하드웨어 가속기 초기화 등으로 인해 느려지는 현상을 배제하기 위함입니다.
    for (int i = 0; i < warmup; i++) {
        ncnn::Extractor ex = net.create_extractor(); // 추론을 위한 추출기 객체 생성
        ex.input("in0", input);                      // 입력 레이어 이름("in0")에 데이터 주입
        ncnn::Mat out;
        ex.extract("out0", out);                     // 출력 레이어 이름("out0")에서 결과 추출
    }

    // --- 본 측정 단계 (Main Benchmark) ---
    std::vector<double> times;                       // 각 회차별 시간을 저장할 벡터
    for (int i = 0; i < runs; i++) {
        // 측정 시작 시간 기록
        auto t0 = std::chrono::high_resolution_clock::now();
        
        ncnn::Extractor ex = net.create_extractor(); // 매 실행마다 독립적인 추출기 생성
        ex.input("in0", input);
        ncnn::Mat out;
        ex.extract("out0", out);
        
        // 측정 종료 시간 기록
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // 두 시간의 차이를 밀리초(ms) 단위로 변환하여 벡터에 저장합니다.
        times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }

    // --- 통계 계산 ---
    // 전체 측정 시간의 평균(Mean)을 구합니다.
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    
    // 표준편차(Standard Deviation)를 계산하여 결과의 변동성을 확인합니다.
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - mean) * (t - mean);
    double std_dev = sqrt(sq_sum / runs);

    // Python의 `result_saver.py`가 쉽게 읽을 수 있도록 결과를 JSON 포맷으로 출력합니다.
    printf("{\"runtime\":\"ncnn_vulkan\",\"mean_ms\":%.3f,\"std_ms\":%.3f}\n",
           mean, std_dev);
           
    return 0;
}
"""
build : 
g++ ncnn_bench.cpp -o ncnn_bench \
    -I/usr/local/include/ncnn \
    -L/usr/local/lib \
    -lncnn -lvulkan -lpthread


python3 에서 호출 
import subprocess, json

result = subprocess.run(
    ['./ncnn_bench', 'model.param', 'model.bin'],
    capture_output=True, text=True
)
ncnn_data = json.loads(result.stdout)
print(ncnn_data)
# {'runtime': 'ncnn_vulkan', 'mean_ms': 23.4, 'std_ms': 0.8}
"""