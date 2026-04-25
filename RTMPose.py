import torch
import time
from mmpose.apis import init_model

# 1. 모델 및 환경 설정
config_file = r'C:\Users\lijih\code\Hand detection\rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'
checkpoint_file = r'C:\Users\lijih\code\Hand detection\rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'

device = 'cuda:0'

print("모델 구조 및 가중치를 로드하는 중...")
model = init_model(config_file, checkpoint_file, device=device)
model.eval()

# 2. 더미 데이터(Dummy Tensor) 생성
# RTMPose-m (256x192 해상도 기준, 배치 사이즈 1)
dummy_input = torch.randn(1, 3, 256, 192).to(device)

# 3. GPU 웜업 (Warm-up)
print("GPU 동적 할당 및 캐시 웜업 진행 중 (50회)...")
with torch.no_grad():
    for _ in range(50):
        # data_samples=None을 반드시 명시해주어야 합니다.
        _ = model(dummy_input, data_samples=None, mode='tensor')
        
# 웜업 과정에서 할당된 불필요한 VRAM 캐시를 한 번 비워줍니다.
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

# 4. 본격적인 벤치마크 시작
iterations = 1000
print(f"성능 측정 시작 ({iterations}회 반복)...")

start_time = time.time()

with torch.no_grad():
    for _ in range(iterations):
        # 4번 루프도 동일하게 수정되었습니다.
        _ = model(dummy_input, data_samples=None, mode='tensor')
        # GPU 비동기 연산이 완전히 끝날 때까지 대기 (정확한 지연 시간 측정을 위해 필수)
        torch.cuda.synchronize(device)

end_time = time.time()

# 5. 성능 지표 계산
total_time = end_time - start_time
avg_latency_ms = (total_time / iterations) * 1000
fps = iterations / total_time

# PyTorch가 모델 추론만을 위해 실제로 사용한 최대 VRAM 용량
peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# 6. 결과 출력
print("\n" + "="*40)
print("🎯 RTMPose PyTorch 순수 추론 벤치마크 결과")
print("="*40)
print(f"▶ 테스트 환경    : {torch.cuda.get_device_name(device)}")
print(f"▶ 평균 지연 시간 : {avg_latency_ms:.2f} ms / frame")
print(f"▶ 초당 처리 속도 : {fps:.2f} FPS")
print(f"▶ 순수 VRAM 점유 : {peak_vram_mb:.2f} MB")
print("="*40)