import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# 1. 하드웨어 가속 및 해상도 최적화
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
# MJPG 포맷은 USB 대역폭을 덜 차지해서 고해상도 FPS 유지에 유리합니다.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

with mp_hands.Hands(
    static_image_mode=False,     # 비디오 스트림 최적화 활성화
    model_complexity=0,          # 0으로 설정하여 연산 속도 극대화
    min_detection_confidence=0.5, # 초기 감지 문턱값
    min_tracking_confidence=0.5   # 추적 신뢰도 (너무 높으면 렉, 낮으면 떨림)
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # [최적화] 처리에 필요한 부분만 남기고 나머지는 무시
        image = cv2.flip(image, 1)
        image.flags.writeable = False # 성능 향상을 위해 메모리 쓰기 방지
        
        # RGB 변환 (MediaPipe 필수)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # [그리기 최적화] 스타일을 단순화하거나 필요할 때만 그립니다.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('Optimized MediaPipe', image)
        
        if cv2.waitKey(1) & 0xFF == 27: # 대기 시간을 1ms로 줄여 루프 속도 향상
            break

cap.release()
cv2.destroyAllWindows()


#import cv2
#import mediapipe as mp
#
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_hands = mp.solutions.hands
#
## 카메라 열기
#cap = cv2.VideoCapture(0)
#
## [추가] 카메라 입력 해상도 크게 설정 (HD급)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
#
## [추가] 출력 창 설정 (WINDOW_NORMAL을 써야 크기 조절이 자유롭습니다)
#cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
#
#with mp_hands.Hands(
#    model_complexity=0,  # 0: 빠름, 1: 정확함 (해상도가 높을 땐 0이 유리)
#    min_detection_confidence=0.5,
#    min_tracking_confidence=0.5) as hands:
#    
#    while cap.isOpened():
#        success, image = cap.read()
#        if not success:
#            print("카메라를 찾을 수 없습니다.")
#            continue
#
#        # 보기 편하게 이미지를 좌우 반전 후 처리
#        image = cv2.flip(image, 1)
#
#        # 성능 향상을 위해 이미지 쓰기 불가능 설정
#        image.flags.writeable = False
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        results = hands.process(image)
#
#        # 이미지에 손 주석 그리기
#        image.flags.writeable = True
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#        
#        if results.multi_hand_landmarks:
#            for hand_landmarks in results.multi_hand_landmarks:
#                mp_drawing.draw_landmarks(
#                    image,
#                    hand_landmarks,
#                    mp_hands.HAND_CONNECTIONS,
#                    mp_drawing_styles.get_default_hand_landmarks_style(),
#                    mp_drawing_styles.get_default_hand_connections_style())
#
#        # 최종 화면 출력
#        cv2.imshow('MediaPipe Hands', image)
#        
#        # ESC 키를 누르면 종료
#        if cv2.waitKey(5) & 0xFF == 27:
#            break
#
#cap.release()
#cv2.destroyAllWindows()