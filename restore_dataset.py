import os
import urllib.request
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "imagenet_val")
VAL_DIR = os.path.join(DATA_DIR, "val")

def download_imagenet_1k_sample():
    print("🚀 ImageNet 1,000 클래스 대표 샘플 복구 스크립트 실행 중...")
    
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 깃허브에서 1,000장 샘플 압축파일(ZIP) 다이렉트 다운로드
    url = "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip"
    zip_path = os.path.join(DATA_DIR, "imagenet_sample.zip")
    
    if not os.path.exists(zip_path):
        print(f"다운로드 중... ({url})")
        urllib.request.urlretrieve(url, zip_path)
    
    print("압축 해제 및 클래스 폴더 재구성 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    extracted_folder = os.path.join(DATA_DIR, "imagenet-sample-images-master")
    
    # 클래스명(n01440764 등)별로 폴더를 만들고 1장씩 분배
    count = 0
    for filename in os.listdir(extracted_folder):
        if filename.endswith(".JPEG"):
            class_id = filename.split("_")[0]
            class_dir = os.path.join(VAL_DIR, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            src = os.path.join(extracted_folder, filename)
            dst = os.path.join(class_dir, filename)
            # Use copy2 to bypass Antivirus/Windows Indexer lock on newly extracted files
            shutil.copy2(src, dst)
            count += 1
            
    print(f"✅ 복구 완료! 총 {count}개 클래스의 각 1장 샘플 세팅이 끝났습니다.")
    print(f"경로: {VAL_DIR}")
    
    # 임시 파일 정리
    try:
        os.remove(zip_path)
        shutil.rmtree(extracted_folder, ignore_errors=True)
    except Exception as e:
        print(f"Temp cleanup warning: {e}")

if __name__ == "__main__":
    download_imagenet_1k_sample()
