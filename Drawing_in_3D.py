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
lost_draw_frames = 0          # 프레임 드랍 방지용 버퍼
DRAW_LOST_TOLERANCE = 5       # 이 프레임 수만큼은 추적을 놓쳐도 선을 끊지 않음

view_yaw, view_pitch = 0.0, 0.0
last_left_pinch_pos = None

DRAW_COLOR = (0, 255, 255)
DRAW_THICKNESS = 5
PINCH_THRESHOLD = 40  
ERASER_RADIUS = 50 

def get_distance_2d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def is_finger_extended(landmarks, tip_idx, pip_idx):
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
with mp_hands.Hands(model_complexity=1, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    cv2.namedWindow('3D Holographic Drawer', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Real Camera View', cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        canvas_view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        camera_view = image.copy() 
        
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        R = get_rotation_matrix(view_pitch, view_yaw)
        R_inv = R.T 

        is_reset, is_eraser, is_drawing_this_frame, is_rotating = False, False, False, False
        right_index_screen_pos = None

        best_physical_left = None
        best_physical_right = None
        max_score_l = 0.0
        max_score_r = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 실제 카메라 화면에 인식된 모든 손 뼈대 렌더링 (디버깅용)
                mp.solutions.drawing_utils.draw_landmarks(
                    camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                
                # 최고 신뢰도를 가진 좌/우 물리적 손 선별
                if label == 'Right': 
                    if score > max_score_l:
                        max_score_l = score
                        best_physical_left = hand_landmarks
                elif label == 'Left':
                    if score > max_score_r:
                        max_score_r = score
                        best_physical_right = hand_landmarks

        # ==========================================
        # [왼손]: 리셋(주먹) 및 시점 회전(핀치)
        # ==========================================
        if best_physical_left:
            landmarks = best_physical_left.landmark
            p4 = (int(landmarks[4].x * WIDTH), int(landmarks[4].y * HEIGHT))
            p8 = (int(landmarks[8].x * WIDTH), int(landmarks[8].y * HEIGHT))
            
            tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
            folded_count = sum(1 for i in range(4) if not is_finger_extended(landmarks, tips[i], pips[i]))
            
            if folded_count == 4:
                is_reset = True
                last_left_pinch_pos = None 
            elif get_distance_2d(p4, p8) < PINCH_THRESHOLD:
                is_rotating = True
                pinch_center = ((p4[0]+p8[0])//2, (p4[1]+p8[1])//2)
                if last_left_pinch_pos:
                    view_yaw += (pinch_center[0] - last_left_pinch_pos[0]) * 0.01
                    view_pitch -= (pinch_center[1] - last_left_pinch_pos[1]) * 0.01
                last_left_pinch_pos = pinch_center
            else:
                last_left_pinch_pos = None

            # [수정됨] 검은색 3D 캔버스에 상태별 왼손 뼈대 그리기 복구
            b_color = (0, 0, 255) if is_reset else (255, 100, 100) if is_rotating else (150, 150, 150)
            mp.solutions.drawing_utils.draw_landmarks(
                canvas_view, best_physical_left, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=b_color, thickness=2, circle_radius=2)
            )

        # ==========================================
        # [오른손]: 그리기(오직 검지만) 및 3D 지우개(핀치)
        # ==========================================
        if best_physical_right:
            landmarks = best_physical_right.landmark
            p4 = (int(landmarks[4].x * WIDTH), int(landmarks[4].y * HEIGHT))
            p8 = (int(landmarks[8].x * WIDTH), int(landmarks[8].y * HEIGHT))
            right_index_screen_pos = p8
            
            sx = (landmarks[8].x - 0.5) * WIDTH
            sy = (landmarks[8].y - 0.5) * HEIGHT
            raw_z = landmarks[8].z * WIDTH
            sz = raw_z * 3.0 if raw_z > 0 else raw_z * 1.0

            index_ext = is_finger_extended(landmarks, 8, 6)   
            middle_ext = is_finger_extended(landmarks, 12, 10) 
            ring_ext = is_finger_extended(landmarks, 16, 14)   
            pinky_ext = is_finger_extended(landmarks, 20, 18)  
            
            is_pointing_only_index = index_ext and not middle_ext and not ring_ext and not pinky_ext

            if get_distance_2d(p4, p8) < PINCH_THRESHOLD:
                is_eraser = True
            elif is_pointing_only_index:
                is_drawing_this_frame = True
                pen_world_pos = R_inv @ np.array([sx, sy, sz])

            # [수정됨] 검은색 3D 캔버스에 상태별 오른손 뼈대 그리기 복구
            b_color = (0, 255, 255) if is_eraser else (0, 255, 0) if is_drawing_this_frame else (255, 255, 255)
            mp.solutions.drawing_utils.draw_landmarks(
                canvas_view, best_physical_right, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=b_color, thickness=2, circle_radius=2)
            )

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
            lost_draw_frames = DRAW_LOST_TOLERANCE + 1 
            status_text, status_color = "2D PROJECTION ERASER", (0, 255, 255)

        if is_drawing_this_frame:
            current_stroke.append(tuple(pen_world_pos))
            lost_draw_frames = 0
            status_text, status_color = "3D DRAWING (ONLY INDEX)", (0, 255, 0)
        else:
            if current_stroke:
                lost_draw_frames += 1
                if lost_draw_frames > DRAW_LOST_TOLERANCE:
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
        
        cv2.imshow('3D Holographic Drawer', canvas_view)
        cv2.imshow('Real Camera View', camera_view) 
        
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()