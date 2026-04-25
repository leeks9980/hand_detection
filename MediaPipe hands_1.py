import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 카메라 초기 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 고해상도 데이터 전송 효율을 위해 MJPG 포맷 권장
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 창 크기 설정
cv2.namedWindow('MediaPipe Hands (Full Model)', cv2.WINDOW_NORMAL)

# model_complexity=1 설정 (Full 모델)
with mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands = 6,           
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 처리 전 이미지 좌우 반전
        image = cv2.flip(image, 1)

        # 성능을 위해 이미지 쓰기 방지 설정 후 처리
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # 다시 쓰기 가능 설정 후 그리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크와 연결선 그리기
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 결과 출력
        cv2.imshow('MediaPipe Hands (Full Model)', image)
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()