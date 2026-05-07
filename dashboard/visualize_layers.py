import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# headless 환경에서도 동작하도록 백엔드 설정
matplotlib.use('Agg')

def visualize_layer_runtime(csv_dir):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    # 저장할 디렉토리 생성
    output_dir = os.path.join(csv_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'layer_name' not in df.columns or 'mean_ms' not in df.columns:
                print(f"Skipping {os.path.basename(csv_file)}: 'layer_name' or 'mean_ms' not found.")
                continue
                
            # 순차적으로 나열하기 위해 (위에서부터 첫 번째 레이어가 오도록) 역순으로 정렬
            df_plot = df.iloc[::-1].reset_index(drop=True)
            
            # 레이어 수에 따라 동적으로 높이 설정 (가독성 향상)
            fig_height = max(8, len(df) * 0.4)
            plt.figure(figsize=(14, fig_height))
            
            bars = plt.barh(df_plot['layer_name'], df_plot['mean_ms'], color='skyblue')
            
            # 막대 끝에 숫자(mean_ms) 표시
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                         f' {width:.2f}', 
                         ha='left', va='center', fontsize=10)

            plt.xlabel('Mean Time (ms)')
            plt.ylabel('Layer Name')
            plt.title(f'Layer Runtime for {os.path.basename(csv_file)}')
            
            # y축 레이블이 길어 잘릴 수 있으므로 레이아웃 자동 조정
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', '.png'))
            plt.savefig(output_file, dpi=150)
            plt.close()
            print(f"Saved visualization: {output_file}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {e}")

if __name__ == "__main__":
    RESULTS_BASE = "/Users/hobongs/Desktop/HoBong_study/26-1/탄중/temp/jetson_nano_edgeai/results"
    possible_dirs = [d for d in glob.glob(os.path.join(RESULTS_BASE, "*")) if os.path.isdir(d)]
    latest_dir = max(possible_dirs, key=os.path.getmtime)
    csv_dir = os.path.join(latest_dir, "layer_runtime")
    
    print(f"Starting visualization for CSV files in {csv_dir}")
    visualize_layer_runtime(csv_dir)
    print("Done!")
