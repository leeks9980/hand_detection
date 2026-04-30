import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================
# 1. 초기 설정
# ==========================================
mp_hands = mp.solutions.hands
WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# 21개 관절을 위한 무지개색(Neon) 생성
neon_colors = []
for i in range(21):
    hue = int((i / 21) * 179)
    hsv_color = np.uint8([[[hue, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    neon_colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

# ==========================================
# 2. 비주얼 파이프라인 가동
# ==========================================
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # 거울 모드 (좌우 반전)
        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # ------------------------------------------------
        # [핵심 기능 1] 빛줄기를 그릴 빈 투명 캔버스와 손을 가릴 마스크 도화지 생성
        # ------------------------------------------------
        line_canvas = np.zeros_like(image)
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        all_hands_px = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                landmarks_px = []
                for lm in hand_landmarks.landmark:
                    landmarks_px.append((int(lm.x * WIDTH), int(lm.y * HEIGHT)))
                
                all_hands_px.append(landmarks_px)

                # ------------------------------------------------
                # [핵심 기능 2] 내 손 모양과 똑같은 '실루엣 마스크' 만들기
                # ------------------------------------------------
                # 손바닥 크기에 비례하여 마스크(손가락) 두께를 동적으로 계산합니다.
                hand_size = math.hypot(landmarks_px[9][0] - landmarks_px[0][0], 
                                       landmarks_px[9][1] - landmarks_px[0][1])
                finger_thickness = max(10, int(hand_size * 0.2)) # 손가락 살집 두께

                # 1. 손바닥 다각형 영역 채우기
                palm_indices = [0, 1, 5, 9, 13, 17]
                palm_pts = np.array([landmarks_px[i] for i in palm_indices], dtype=np.int32)
                cv2.fillPoly(hand_mask, [palm_pts], 255)

                # 2. 손가락 뼈대를 따라 두꺼운 선을 그려 손가락 전체를 덮는 마스크 완성
                for connection in mp_hands.HAND_CONNECTIONS:
                    pt1 = landmarks_px[connection[0]]
                    pt2 = landmarks_px[connection[1]]
                    cv2.line(hand_mask, pt1, pt2, 255, finger_thickness)

        # ==========================================
        # 3. 양손 연결 네온 빛줄기 그리기 (원본이 아닌 line_canvas에!)
        # ==========================================
        if len(all_hands_px) == 2:
            hand1 = all_hands_px[0]
            hand2 = all_hands_px[1]
            
            for i in range(21):
                pt1 = hand1[i]
                pt2 = hand2[i]
                color = neon_colors[i]
                
                # 가느다란 코어 빛줄기 생성
                cv2.line(line_canvas, pt1, pt2, color, 2, cv2.LINE_AA)
                cv2.line(line_canvas, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------
        # [핵심 기능 3] 마스크 적용 및 이미지 합성
        # ------------------------------------------------
        # hand_mask가 있는 영역(내 진짜 손 위치)을 제외하고 빛줄기를 남깁니다.
        mask_inv = cv2.bitwise_not(hand_mask)
        line_canvas = cv2.bitwise_and(line_canvas, line_canvas, mask=mask_inv)

        # 손을 피해서 그려진 빛줄기를 카메라 원본 이미지 위에 자연스럽게 합성합니다. (마치 형광등처럼 더해짐)
        image = cv2.add(image, line_canvas)

        # ==========================================
        # 4. 마지막으로 얇은 스켈레톤 복구 (가장 위에 그려짐)
        # ==========================================
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(120, 120, 120), thickness=1)
                )

        cv2.imshow('Occluded Neon Web (Anti-Piercing)', image)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()