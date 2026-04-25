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

strokes = []
current_stroke = []

view_yaw, view_pitch = 0.0, 0.0
last_left_pinch_pos = None

DRAW_COLOR = (0, 255, 255)
DRAW_THICKNESS = 5
PINCH_THRESHOLD = 40  
ERASER_RADIUS = 50 

def get_distance_2d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def is_finger_extended(landmarks, tip_idx, pip_idx):
    """ 손목 기준 거리로 손가락 펴짐/접힘 판별 """
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    
    dist_tip = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
    dist_pip = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2 + (pip.z - wrist.z)**2)
    return dist_tip > dist_pip

# ==========================================
# 2. 3D 수학 및 행렬 함수
# ==========================================
def get_rotation_matrix(pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    return Ry @ Rx

def draw_3d_axis(canvas, R):
    origin = np.array([WIDTH/2 - 100, HEIGHT/2 - 100, 0])
    for ax, color, label in [(np.array([60, 0, 0]), (0, 0, 255), "X"), (np.array([0, -60, 0]), (0, 255, 0), "Y"), (np.array([0, 0, 60]), (255, 0, 0), "Z")]:
        p_screen = origin + (R @ ax)
        px, py = int(p_screen[0] + WIDTH/2), int(p_screen[1] + HEIGHT/2)
        cv2.line(canvas, (int(origin[0] + WIDTH/2), int(origin[1] + HEIGHT/2)), (px, py), color, 3)
        cv2.putText(canvas, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ==========================================
# 3. 비주얼 파이프라인 가동
# ==========================================
with mp_hands.Hands(model_complexity=1, max_num_hands=2) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        canvas_view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        R = get_rotation_matrix(view_pitch, view_yaw)
        R_inv = R.T 

        is_reset, is_eraser, is_drawing, is_rotating = False, False, False, False
        right_index_screen_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = hand_landmarks.landmark
                mp_label = handedness.classification[0].label
                is_physical_right = (mp_label == 'Left')
                is_physical_left = (mp_label == 'Right')
                
                # ==========================================
                # [왼손]: 리셋(주먹) 및 시점 회전(핀치)
                # ==========================================
                if is_physical_left:
                    p4 = (int(landmarks[4].x * WIDTH), int(landmarks[4].y * HEIGHT))
                    p8 = (int(landmarks[8].x * WIDTH), int(landmarks[8].y * HEIGHT))
                    
                    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
                    folded_count = sum(1 for i in range(4) if not is_finger_extended(landmarks, tips[i], pips[i]))
                    
                    # 1. (최우선) 주먹 판정: 네 손가락이 모두 접혔는가?
                    if folded_count == 4:
                        is_reset = True
                        last_left_pinch_pos = None 
                        
                    # 2. 핀치 판정: 주먹이 "아니면서" 엄지-검지가 가까운가?
                    elif get_distance_2d(p4, p8) < PINCH_THRESHOLD:
                        is_rotating = True
                        pinch_center = ((p4[0]+p8[0])//2, (p4[1]+p8[1])//2)
                        if last_left_pinch_pos:
                            view_yaw += (pinch_center[0] - last_left_pinch_pos[0]) * 0.01
                            view_pitch -= (pinch_center[1] - last_left_pinch_pos[1]) * 0.01
                        last_left_pinch_pos = pinch_center
                        
                    # 3. 아무 동작도 아닐 때
                    else:
                        last_left_pinch_pos = None
                            
                    b_color = (0, 0, 255) if is_reset else (255, 100, 100) if is_rotating else (150, 150, 150)
                    mp.solutions.drawing_utils.draw_landmarks(canvas_view, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp.solutions.drawing_utils.DrawingSpec(color=b_color, thickness=2, circle_radius=2))

                # ==========================================
                # [오른손]: 그리기(오직 검지만) 및 3D 지우개(핀치)
                # ==========================================
                elif is_physical_right:
                    p4 = (int(landmarks[4].x * WIDTH), int(landmarks[4].y * HEIGHT))
                    p8 = (int(landmarks[8].x * WIDTH), int(landmarks[8].y * HEIGHT))
                    right_index_screen_pos = p8
                    
                    sx = (landmarks[8].x - 0.5) * WIDTH
                    sy = (landmarks[8].y - 0.5) * HEIGHT
                    
                    # === [수정된 부분] 앞/뒤 깊이감 비선형 스케일링 ===
                    raw_z = landmarks[8].z * WIDTH
                    Z_SCALE_FORWARD = 1.0   # 앞으로 나오는 깊이감 배율 (기존 유지)
                    Z_SCALE_BACKWARD = 3.0  # 뒤로 들어가는 깊이감 배율 (증폭)
                    
                    if raw_z > 0: 
                        sz = raw_z * Z_SCALE_BACKWARD # 뒤로 갈 때 더 크게 반응
                    else:         
                        sz = raw_z * Z_SCALE_FORWARD  # 앞으로 갈 때 기존 반응
                    # ==================================================

                    # [핵심 로직] 각 손가락의 펴짐 상태를 개별 확인
                    index_ext = is_finger_extended(landmarks, 8, 6)   # 검지
                    middle_ext = is_finger_extended(landmarks, 12, 10) # 중지
                    ring_ext = is_finger_extended(landmarks, 16, 14)   # 약지
                    pinky_ext = is_finger_extended(landmarks, 20, 18)  # 소지
                    
                    # '검지만' 펴져 있고, 나머지 세 손가락은 완벽히 접혔을 때 True
                    is_pointing_only_index = index_ext and not middle_ext and not ring_ext and not pinky_ext

                    if get_distance_2d(p4, p8) < PINCH_THRESHOLD:
                        is_eraser = True
                    elif is_pointing_only_index:
                        is_drawing = True
                        pen_world_pos = R_inv @ np.array([sx, sy, sz])

                    b_color = (0, 255, 255) if is_eraser else (0, 255, 0) if is_drawing else (255, 255, 255)
                    mp.solutions.drawing_utils.draw_landmarks(canvas_view, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp.solutions.drawing_utils.DrawingSpec(color=b_color, thickness=2, circle_radius=2))

        # ==========================================
        # 4. 상태에 따른 액션 실행
        # ==========================================
        status_text, status_color = "READY", (255, 255, 255)

        if is_reset:
            strokes.clear()
            current_stroke.clear()
            view_yaw, view_pitch = 0.0, 0.0
            status_text, status_color = "3D RESET!", (0, 0, 255)

        elif is_rotating:
            if current_stroke:
                strokes.append(list(current_stroke))
                current_stroke.clear()
            status_text, status_color = "ROTATING CAMERA...", (255, 100, 100)

        elif is_eraser and right_index_screen_pos:
            cv2.circle(canvas_view, right_index_screen_pos, ERASER_RADIUS, (0, 255, 255), 2)
            new_strokes = []
            
            for stroke in strokes + ([current_stroke] if current_stroke else []):
                temp = []
                for pt in stroke:
                    pt_rot = R @ np.array(pt)
                    px = pt_rot[0] + WIDTH / 2
                    py = pt_rot[1] + HEIGHT / 2
                    
                    dist_2d = get_distance_2d((px, py), right_index_screen_pos)
                    
                    if dist_2d > ERASER_RADIUS:
                        temp.append(pt)
                    else:
                        if temp: new_strokes.append(temp)
                        temp = []
                if temp: new_strokes.append(temp)
            
            strokes = new_strokes
            current_stroke.clear()
            status_text, status_color = "2D PROJECTION ERASER", (0, 255, 255)

        elif is_drawing:
            current_stroke.append(tuple(pen_world_pos))
            status_text, status_color = "3D DRAWING (ONLY INDEX)", (0, 255, 0)

        else:
            if current_stroke:
                strokes.append(list(current_stroke))
                current_stroke.clear()

        # ==========================================
        # 5. 3D -> 2D 렌더링 엔진
        # ==========================================
        all_lines_to_draw = strokes + ([current_stroke] if current_stroke else [])
        
        for stroke in all_lines_to_draw:
            if len(stroke) < 2: continue
            pts_3d = np.array(stroke) 
            pts_2d_rot = (R @ pts_3d.T).T 
            pts_2d_rot[:, 0] += WIDTH / 2
            pts_2d_rot[:, 1] += HEIGHT / 2
            
            pts_cv = pts_2d_rot[:, :2].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas_view, [pts_cv], False, DRAW_COLOR, DRAW_THICKNESS)

        draw_3d_axis(canvas_view, R)
        cv2.putText(canvas_view, status_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2)
        cv2.imshow('3D Holographic Drawer: Strict Index Pointer', canvas_view)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()