import cv2
from mmpose.apis import MMPoseInferencer

# 1. 모델 도면과 가중치 경로 설정 (이전과 동일)
config_file = r'C:\Users\lijih\code\Hand detection\rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'
checkpoint_file = r'C:\Users\lijih\code\Hand detection\rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'

print("추론기 초기화 중... (초기 로딩에 몇 초 정도 걸릴 수 있습니다)")

# 2. 통합 추론기(Inferencer) 생성
# 이거 하나면 전처리(크기 조절), 추론, 뼈대 그리기 후처리가 모두 자동으로 끝납니다.
inferencer = MMPoseInferencer(pose2d=config_file, pose2d_weights=checkpoint_file, device='cuda:0')

# 3. 웹캠 열기 (0번 기본 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("에러: 웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 켜졌습니다. 화면을 클릭하고 'q'를 누르면 종료됩니다.")

# 4. 실시간 추론 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 추론기에 넣고, 뼈대가 그려진 결과물(visualization)을 받아옵니다.
    # 제너레이터(Generator) 객체를 반환하므로 next()로 첫 번째 결과를 뽑아냅니다.
    result_generator = inferencer(frame, return_vis=True)
    result = next(result_generator)
    
    # 뼈대가 예쁘게 합성된 결과 이미지 추출
    vis_frame = result['visualization'][0]

    # 화면에 출력
    cv2.imshow('RTMPose Real-time Tracking (WholeBody)', vis_frame)

    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 자원 해제
cap.release()
cv2.destroyAllWindows()