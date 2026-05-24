import torch
import time
import sys
from pathlib import Path

BASE_DIR = Path(r'd:\Capstone2026\Action Predict\Labeled_data\New_folder_1')
sys.path.insert(0, str(BASE_DIR / 'Models'))
from stgcn_model import STGCN

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Benchmarking on: {device}')

    # Init model
    model = STGCN(in_channels=3, num_classes=4, dropout=0.5).to(device)
    ckpt_path = r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final\best_stgcn_final.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Create dummy input: Batch=1, C=3, T=100, V=17
    # Note: 1 clip of 100 frames
    dummy_input = torch.randn(1, 3, 100, 17).to(device)

    # Warmup
    print('Warming up...')
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print('Measuring latency...')
    iterations = 500
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = iterations / total_time  # This is "clips" per second
    frames_per_second = fps * 100 # Since each clip is 100 frames

    print('-' * 40)
    print(f'Total clips    : {iterations}')
    print(f'Clip length    : 100 frames')
    print(f'Avg Latency    : {avg_latency_ms:.2f} ms / clip')
    print(f'Throughput     : {fps:.2f} clips/sec')
    print(f'Estimated FPS  : {frames_per_second:.2f} frames/sec')
    print('-' * 40)

if __name__ == '__main__':
    benchmark()
