import subprocess # python에서 OS 명령어를 실행할 수 있게 해줌 ex)linux ls, window dir...
import threading  # 백그라운드에서 데이터를 수집하기 위한 스레드 모듈
import re         # 텍스트에서 숫자만 추출하기 위한 정규표현식 모듈
import time       # 타임스탬프 기록을 위한 모듈

class TegraMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms # 모니터링 주기 (기본 0.1초)
        self.records = []              # 파싱된 데이터를 저장할 리스트
        self._proc = None              # 실행될 서브프로세스 객체 담는 변수
        self._thread = None            # 데이터를 읽어올 스레드 객체 담는 변수


    def start(self):
        # 1. tegrastats 명령어가 시스템에 존재하는지 먼저 확인
        try:
            # which 명령어로 tegrastats 경로 확인
            subprocess.check_output(['which', 'tegrastats'])
        except subprocess.CalledProcessError:
            print("[Warning] 'tegrastats' not found. Hardware monitoring will be disabled (Non-Jetson environment?).")
            return

        # 2. tegrastats 명령어를 백그라운드에서 실행
        try:
            self._proc = subprocess.Popen(
                ['sudo', 'tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            # 3. 출력을 실시간으로 읽어서 기록할 스레드 생성 및 시작
            self._thread = threading.Thread(target=self._collect)
            self._thread.daemon = True
            self._thread.start()
            time.sleep(0.5)
        except Exception as e:
            print(f"[Warning] Failed to start tegrastats: {e}")

    def stop(self):
        # 모니터링 프로세스를 종료하고 스레드를 정리함
        if self._proc:
            try:
                # 1. 먼저 프로세스가 아직 살아있는지 확인
                if self._proc.poll() is None:
                    self._proc.terminate()      # 프로세스 강제 종료 신호
                    self._proc.wait(timeout=1)  # 종료 대기
            except (PermissionError, OSError):
                # sudo로 실행된 프로세스는 일반 유저가 terminate() 할 수 없으므로 시스템 명령어로 kill
                try:
                    import subprocess
                    subprocess.run(['sudo', 'kill', str(self._proc.pid)], stderr=subprocess.DEVNULL)
                except:
                    pass
            
            if self._thread:
                self._thread.join(timeout=1)

    def _collect(self):
        # tegrastats가 한 줄씩 출력할 때마다 반복해서 읽어옴
        for line in self._proc.stdout:
            parsed = self._parse(line.strip()) # 읽은 줄을 숫자로 변환
            if parsed:
                self.records.append(parsed)     # 성공적으로 파싱되면 리스트에 추가

    def _parse(self, line):
        # 정규표현식을 이용해 텍스트 데이터에서 필요한 수치만 추출
        result = {'timestamp': time.time()} # 현재 시간 기록
        patterns = {
            'ram_used_mb':    r'RAM (\d+)/\d+MB',  # RAM 사용량 추출
            'gpu_util_pct':   r'GR3D_FREQ (\d+)%', # GPU 사용 점유율(%) 추출
            'power_total_mw': r'VDD_IN (\d+)mW',   # 전체 소비 전력(mW) 추출
            'power_cpu_mw':   r'VDD_CPU (\d+)mW',  # CPU 소비 전력 추출
            'power_gpu_mw':   r'VDD_GPU (\d+)mW',  # GPU 소비 전력 추출
            'gpu_temp_c':     r'GPU@(\d+)C',       # GPU 온도 추출
            'cpu_temp_c':     r'CPU@(\d+)C',       # CPU 온도 추출
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, line) # 현재 줄에서 패턴 찾기
            if m:
                result[key] = int(m.group(1)) # 찾은 숫자를 정수형으로 변환해 저장
        
        # timestamp 외에 데이터가 하나라도 추출되었다면 딕셔너리 반환
        return result if len(result) > 1 else None

    def summary(self):
        # 수집된 모든 기록의 평균값을 계산하여 요약 리포트 생성
        if not self.records:
            return {}
        import numpy as np
        # timestamp를 제외한 필드 이름 목록 추출
        keys = [k for k in self.records[0] if k != 'timestamp']
        return {
            f'{k}_mean': round(float(np.mean([r.get(k, 0) for r in self.records])), 2)
            for k in keys # 각 필드별로 평균 계산 후 소수점 둘째 자리까지 반올림
        }