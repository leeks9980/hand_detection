import cv2
import numpy as np
import mediapipe as mp
from mmpose.apis import MMPoseInferencer
from multiprocessing import Process, Event, Array, shared_memory
import time

# ==========================================
# 설정값 (환경에 맞게 조정 가능)
# ==========================================
MAX_PEOPLE = 5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RTMP_POINTS = 133  # COCO-Wholebody 기준
MP_HAND_POINTS = 21 # 손가락 관절 개수

# ==========================================
# 1. GPU 독립 프로세스 (RTMPose - 다중 인원)
# ==========================================
def run_rtmpose(shm_name, shape, shared_coords, stop_event):
    # 프로세스 내부 모델 초기화
    config = r'C:\Users\lijih\code\Hand detection\rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'
    checkpoint = r'C:\Users\lijih\code\Hand detection\rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'
    inferencer = MMPoseInferencer(pose2d=config, pose2d_weights=checkpoint, device='cuda:0')
    
    # 공유 메모리 연결 (카메라 프레임)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)

    while not stop_event.is_set():
        frame = frame_buffer.copy()  # 현재 프레임 가져오기
        
        result = next(inferencer(frame, return_vis=False))
        preds = result['predictions'][0]
        
        # 이전 데이터 초기화 (탐지되지 않은 사람 잔상 제거)
        for i in range(len(shared_coords)):
            shared_coords[i] = 0.0

        if len(preds) > 0:
            for p_idx, person in enumerate(preds):
                if p_idx >= MAX_PEOPLE: break
                
                kp = person['keypoints']
                offset = p_idx * (RTMP_POINTS * 2)
                
                for i in range(RTMP_POINTS):
                    shared_coords[offset + i*2] = kp[i][0]
                    shared_coords[offset + i*2 + 1] = kp[i][1]
    
    existing_shm.close()

# ==========================================
# 2. CPU 독립 프로세스 (MediaPipe - 다중 손 추적)
# ==========================================
def run_mediapipe(shm_name, shape, shared_coords, hand_coords, stop_event):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, max_num_hands=2)
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)

    while not stop_event.is_set():
        frame = frame_buffer.copy()
        h, w, _ = frame.shape

        for p_idx in range(MAX_PEOPLE):
            # 각 사람의 손목 좌표 위치 계산
            offset = p_idx * (RTMP_POINTS * 2)
            wrist_indices = [(9, 0), (10, 42)] # (왼손목, 배열오프셋), (오른손목, 배열오프셋)
            
            for w_idx, hand_offset in wrist_indices:
                wx = int(shared_coords[offset + w_idx*2])
                wy = int(shared_coords[offset + w_idx*2 + 1])
                
                if wx > 0 and wy > 0:
                    # 손 주변 크롭
                    x1, y1 = max(0, wx-100), max(0, wy-100)
                    x2, y2 = min(w, wx+100), min(h, wy+100)
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        res = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if res.multi_hand_landmarks:
                            # 가장 신뢰도 높은 손 하나만 채택 (중첩 방지)
                            lm = res.multi_hand_landmarks[0]
                            # 전역 인덱스 계산: (사람 인덱스 * 양손 공간) + 손 오프셋
                            global_hand_offset = (p_idx * 84) + hand_offset
                            for i, pt in enumerate(lm.landmark):
                                hand_coords[global_hand_offset + i*2] = pt.x * (x2-x1) + x1
                                hand_coords[global_hand_offset + i*2 + 1] = pt.y * (y2-y1) + y1
    
    existing_shm.close()

# ==========================================
# 3. 메인 프로세스 (카메라 서버 및 비주얼 출력)
# ==========================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        exit()
    
    shape = frame.shape
    
    # 공유 메모리 설정
    shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    
    # 좌표 메모리 (사람 수만큼 곱함)
    shared_pts = Array('d', RTMP_POINTS * 2 * MAX_PEOPLE)
    hand_pts = Array('d', MP_HAND_POINTS * 2 * 2 * MAX_PEOPLE) # 인당 양손
    
    stop_ev = Event()

    # 독립 프로세스 가동
    p_rtm = Process(target=run_rtmpose, args=(shm.name, shape, shared_pts, stop_ev))
    p_mp = Process(target=run_mediapipe, args=(shm.name, shape, shared_pts, hand_pts, stop_ev))
    
    p_rtm.start()
    p_mp.start()

    print(f"🔥 다중 인원(최대 {MAX_PEOPLE}명) 비동기 파이프라인 가동...")

    colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 165, 255)]

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 거울 모드로 프레임 공유
        frame_buffer[:] = cv2.flip(frame, 1)[:]

        # 비주얼 캔버스 (검은 배경)
        canvas = np.zeros(shape, dtype=np.uint8)

        for p_idx in range(MAX_PEOPLE):
            p_color = colors[p_idx % len(colors)]
            
            # 1. RTMPose 전신/얼굴 그리기
            rtm_offset = p_idx * (RTMP_POINTS * 2)
            for i in range(23, 91): # 얼굴 랜드마크 예시
                x, y = int(shared_pts[rtm_offset + i*2]), int(shared_pts[rtm_offset + i*2 + 1])
                if x > 0: cv2.circle(canvas, (x, y), 1, p_color, -1)
            
            # 2. MediaPipe 손가락 그리기
            h_offset = p_idx * 84
            for i in range(42): # 양손 21*2개 점
                x, y = int(hand_pts[h_offset + i*2]), int(hand_pts[h_offset + i*2 + 1])
                if x > 0: cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1)

        cv2.imshow("Multi-Person Visualizer", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_ev.set()
            break

    # 종료 처리
    p_rtm.join()
    p_mp.join()
    shm.close()
    shm.unlink()
    cap.release()
    cv2.destroyAllWindows()