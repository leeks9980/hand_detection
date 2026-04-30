import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# ==========================================
# 1. 초기 설정 및 물리 환경
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils        # 카메라 뷰에 랜드마크를 그리기 위한 모듈 추가
mp_drawing_styles = mp.solutions.drawing_styles # 랜드마크 스타일 모듈 추가

WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# 3D 가상 공간 창 설정
WINDOW_NAME = '3D Physics: Perfect Pinch'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 실제 카메라 뷰 창 설정 (추가됨)
CAMERA_WINDOW_NAME = 'Real Camera View'
cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)

ROOM_DIM = {'floor_y': 350, 'ceiling_y': -350, 'wall_x': 900, 'back_z': 1600, 'front_z': -600}
FOCAL_LENGTH = 700 

INVERT_X = False
INVERT_Z = False

def project_point(pt_3d):
    x, y, z = pt_3d
    z_shifted = z + FOCAL_LENGTH
    if z_shifted < 1: z_shifted = 1
    
    scale = FOCAL_LENGTH / z_shifted
    px = int(x * scale + WIDTH / 2)
    py = int(y * scale + HEIGHT / 2)
    return (px, py), scale

def draw_3d_grid(canvas):
    color_grid = (80, 80, 80) 
    wx, cy, fy, fz, bz = ROOM_DIM['wall_x'], ROOM_DIM['ceiling_y'], ROOM_DIM['floor_y'], ROOM_DIM['front_z'], ROOM_DIM['back_z']

    for x in range(-wx, wx + 1, 100):
        p1, _ = project_point((x, fy, fz)); p2, _ = project_point((x, fy, bz))
        cv2.line(canvas, p1, p2, color_grid, 1)
        p1, _ = project_point((x, cy, bz)); p2, _ = project_point((x, fy, bz))
        cv2.line(canvas, p1, p2, color_grid, 1)
        
    for z in range(fz, bz + 1, 100):
        p1, _ = project_point((-wx, fy, z)); p2, _ = project_point((wx, fy, z))
        cv2.line(canvas, p1, p2, color_grid, 1)
        
    for y in range(cy, fy + 1, 100):
        p1, _ = project_point((-wx, y, bz)); p2, _ = project_point((wx, y, bz))
        cv2.line(canvas, p1, p2, color_grid, 1)

    for wall_x in [-wx, wx]:
        for y in range(cy, fy + 1, 100):
            p1, _ = project_point((wall_x, y, fz)); p2, _ = project_point((wall_x, y, bz))
            cv2.line(canvas, p1, p2, color_grid, 1)
        for z in range(fz, bz + 1, 100):
            p1, _ = project_point((wall_x, cy, z)); p2, _ = project_point((wall_x, fy, z))
            cv2.line(canvas, p1, p2, color_grid, 1)

class Ball3D:
    def __init__(self, x, y, z, radius):
        self.initial_pos = (x, y, z)
        self.radius = radius
        self.reset() 

    def reset(self):
        self.pos = np.array(self.initial_pos, dtype=np.float64)
        self.vel = np.array([0, 0, 0], dtype=np.float64)
        self.state = "IDLE"

    def update_physics(self):
        if self.state == "IDLE":
            self.vel[1] += 1.8 
            self.pos += self.vel
            
            if self.pos[1] + self.radius > ROOM_DIM['floor_y']:
                self.pos[1] = ROOM_DIM['floor_y'] - self.radius
                self.vel[1] *= -0.7; self.vel *= 0.98 
            elif self.pos[1] - self.radius < ROOM_DIM['ceiling_y']:
                self.pos[1] = ROOM_DIM['ceiling_y'] + self.radius; self.vel[1] *= -0.8

            if self.pos[0] + self.radius > ROOM_DIM['wall_x']:
                self.pos[0] = ROOM_DIM['wall_x'] - self.radius; self.vel[0] *= -0.8
            elif self.pos[0] - self.radius < -ROOM_DIM['wall_x']:
                self.pos[0] = -ROOM_DIM['wall_x'] + self.radius; self.vel[0] *= -0.8

            if self.pos[2] + self.radius > ROOM_DIM['back_z']:
                self.pos[2] = ROOM_DIM['back_z'] - self.radius; self.vel[2] *= -0.8
            elif self.pos[2] - self.radius < ROOM_DIM['front_z']:
                self.pos[2] = ROOM_DIM['front_z'] + self.radius; self.vel[2] *= -0.8

    def render(self, canvas):
        (px, py), scale = project_point(self.pos)
        r = max(2, int(self.radius * scale))

        fy = ROOM_DIM['floor_y']
        (sx, sy), s_scale = project_point((self.pos[0], fy, self.pos[2]))
        shadow_r = max(1, int(self.radius * s_scale * (1 - min(1, (fy - self.pos[1])/500))))
        shadow_intensity = 1.0 - min(0.8, (fy - self.pos[1])/600)
        cv2.ellipse(canvas, (sx, sy), (shadow_r, int(shadow_r * 0.3)), 0, 0, 360, tuple([int(30 * shadow_intensity)]*3), -1)

        color = (0, 165, 255) if self.state == "IDLE" else (0, 255, 0)
        color_depth = tuple([int(c * min(1.5, scale * 1.2)) for c in color])
        cv2.circle(canvas, (px, py), r, color_depth, -1)
        cv2.circle(canvas, (px - int(r*0.3), py - int(r*0.3)), max(1, int(r*0.2)), (255, 255, 255), -1)

ball = Ball3D(0, 0, 600, radius=60) 
pinch_history = deque(maxlen=5)
hand_center_history = deque(maxlen=5)
palm_indices = [0, 1, 5, 9, 13, 17] 

request_calibration = True
offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
TARGET_Z = 600
calib_msg_timer = 0
smoothed_raw_3d = None
SMOOTHING_FACTOR = 0.3 
is_pinching = False
pinch_release_counter = 0
PINCH_TOLERANCE_FRAMES = 4 

last_pts_3d = None
last_vel_3d = None
lost_frames = 0
MAX_LOST_FRAMES = 10  
last_hand_2d = None  

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        if not INVERT_X:
            image = cv2.flip(image, 1)

        # 3D 캔버스 초기화
        canvas_view = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 40
        # 실제 카메라 화면 확인용 캔버스 복사 (추가됨)
        camera_view = image.copy()
        
        draw_3d_grid(canvas_view)
        ball.update_physics()

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        active_pts_3d = None  

        if results.multi_hand_landmarks:
            # ==========================================
            # 카메라 뷰에 랜드마크 뼈대 그리기 (추가됨)
            # ==========================================
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    camera_view,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            best_hand = None
            if last_hand_2d is None:
                best_hand = results.multi_hand_landmarks[0]
            else:
                min_dist = float('inf')
                for hand_landmarks in results.multi_hand_landmarks:
                    cx = hand_landmarks.landmark[9].x * WIDTH
                    cy = hand_landmarks.landmark[9].y * HEIGHT
                    dist = math.hypot(cx - last_hand_2d[0], cy - last_hand_2d[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_hand = hand_landmarks
                        
            target_cx = best_hand.landmark[9].x * WIDTH
            target_cy = best_hand.landmark[9].y * HEIGHT
            last_hand_2d = (target_cx, target_cy)
            
            hand_landmarks = best_hand.landmark
            
            dx = (hand_landmarks[9].x - hand_landmarks[0].x) * WIDTH
            dy = (hand_landmarks[9].y - hand_landmarks[0].y) * HEIGHT
            hand_size_2d = math.hypot(dx, dy)
            
            raw_p4_x, raw_p4_y = hand_landmarks[4].x * WIDTH, hand_landmarks[4].y * HEIGHT
            raw_p8_x, raw_p8_y = hand_landmarks[8].x * WIDTH, hand_landmarks[8].y * HEIGHT
            raw_pinch_dist = math.hypot(raw_p4_x - raw_p8_x, raw_p4_y - raw_p8_y)
            current_pinch_ratio = raw_pinch_dist / hand_size_2d if hand_size_2d > 0 else 1.0

            if current_pinch_ratio < 0.35:
                is_pinching = True
                pinch_release_counter = 0
            elif current_pinch_ratio > 0.45:  
                pinch_release_counter += 1
                if pinch_release_counter >= PINCH_TOLERANCE_FRAMES:
                    is_pinching = False

            cam_cx = (hand_landmarks[9].x - 0.5) * WIDTH
            cam_cy = (hand_landmarks[9].y - 0.5) * HEIGHT
            
            size_points = [50, 130, 250]
            
            if not INVERT_Z:
                z_points = [ROOM_DIM['front_z'], 400, ROOM_DIM['back_z']]
            else:
                z_points = [ROOM_DIM['back_z'], 400, ROOM_DIM['front_z']]
                
            clamped_size = max(size_points[0], min(hand_size_2d, size_points[-1]))
            raw_absolute_z = np.interp(clamped_size, size_points, z_points)

            scale_inv = (raw_absolute_z + FOCAL_LENGTH) / FOCAL_LENGTH
            raw_center_x_3d, raw_center_y_3d = cam_cx * scale_inv, cam_cy * scale_inv
            
            current_raw_3d = np.array([raw_center_x_3d, raw_center_y_3d, raw_absolute_z])
            if smoothed_raw_3d is None or request_calibration: smoothed_raw_3d = current_raw_3d
            else: smoothed_raw_3d = smoothed_raw_3d + SMOOTHING_FACTOR * (current_raw_3d - smoothed_raw_3d)

            if request_calibration:
                offset_x, offset_y, offset_z = smoothed_raw_3d[0], smoothed_raw_3d[1], smoothed_raw_3d[2] - TARGET_Z
                request_calibration = False
                calib_msg_timer = 30 
            
            center_x_3d = smoothed_raw_3d[0] - offset_x
            center_y_3d = smoothed_raw_3d[1] - offset_y
            absolute_z = smoothed_raw_3d[2] - offset_z
            
            HAND_3D_SIZE = 220 
            current_pts_3d = []
            
            for i in range(21):
                rel_x = ((hand_landmarks[i].x - hand_landmarks[9].x) * WIDTH) / hand_size_2d
                rel_y = ((hand_landmarks[i].y - hand_landmarks[9].y) * HEIGHT) / hand_size_2d
                rel_z = ((hand_landmarks[i].z - hand_landmarks[9].z) * WIDTH) / hand_size_2d 
                
                x3d = center_x_3d + rel_x * HAND_3D_SIZE
                y3d = center_y_3d + rel_y * HAND_3D_SIZE
                z3d = absolute_z + rel_z * HAND_3D_SIZE * 1.5 
                current_pts_3d.append([x3d, y3d, z3d])
            
            current_pts_3d = np.array(current_pts_3d)
            
            if last_pts_3d is not None and lost_frames == 0:
                last_vel_3d = current_pts_3d - last_pts_3d
            else:
                last_vel_3d = np.zeros_like(current_pts_3d)
                
            last_pts_3d = current_pts_3d.copy()
            active_pts_3d = current_pts_3d
            lost_frames = 0
            
        else:
            if last_pts_3d is not None and lost_frames < MAX_LOST_FRAMES:
                lost_frames += 1
                last_pts_3d += last_vel_3d
                last_vel_3d *= 0.8  
                active_pts_3d = last_pts_3d.copy()
            else:
                smoothed_raw_3d = None
                is_pinching = False
                pinch_release_counter = 0
                if ball.state == "GRABBED": ball.state = "IDLE"
                pinch_history.clear()
                hand_center_history.clear()
                last_pts_3d = None
                last_hand_2d = None

        if active_pts_3d is not None:
            pts_2d, scales = [], []
            for pt in active_pts_3d:
                (px, py), scale = project_point(pt)
                pts_2d.append((px, py))
                scales.append(scale)
                
            hand_center = np.mean(active_pts_3d, axis=0)
            hand_center_history.append(hand_center)
            hand_vel = (hand_center_history[-1] - hand_center_history[0]) * 0.3 if len(hand_center_history) >= 2 else np.zeros(3)

            p4_3d, p8_3d = active_pts_3d[4], active_pts_3d[8]
            pinch_center_3d = (p4_3d + p8_3d) / 2
            pinch_history.append(pinch_center_3d)
            
            if is_pinching:
                dist_to_ball_3d = np.linalg.norm(pinch_center_3d - ball.pos)
                if ball.state == "IDLE" and dist_to_ball_3d < ball.radius * 4.0: 
                    ball.state = "GRABBED"
                elif ball.state == "GRABBED":
                    ball.pos = np.copy(pinch_center_3d)
                    ball.vel = np.array([0, 0, 0], dtype=np.float64)
            else: 
                if ball.state == "GRABBED":
                    ball.state = "IDLE"
                    if len(pinch_history) == 5:
                        throw_vector = pinch_history[-1] - pinch_history[0]
                        ball.vel = throw_vector * 1.5 
                elif ball.state == "IDLE":
                    distances = np.linalg.norm(active_pts_3d - ball.pos, axis=1)
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    hit_radius = ball.radius + 25 
                    if min_dist < hit_radius:
                        hit_pt = active_pts_3d[min_idx]
                        push_dir = ball.pos - hit_pt
                        norm = np.linalg.norm(push_dir)
                        if norm > 0:
                            push_dir = push_dir / norm
                            ball.vel = push_dir * max(15.0, np.linalg.norm(hand_vel) * 1.2) + hand_vel * 0.8
                            ball.pos = hit_pt + push_dir * hit_radius

            conns_z = []
            for conn in mp_hands.HAND_CONNECTIONS:
                z_avg = (active_pts_3d[conn[0]][2] + active_pts_3d[conn[1]][2]) / 2
                conns_z.append((conn, z_avg))
            conns_z.sort(key=lambda x: x[1], reverse=True)

            pts_2d_shadow, scales_shadow = [], []
            fy = ROOM_DIM['floor_y']
            avg_height = fy - hand_center[1] 
            intensity = 1.0 - min(0.8, avg_height / 700.0) 
            
            s_color = max(10, min(255, int(40 * intensity)))
            shadow_color = (s_color, s_color, s_color)
            
            for pt in active_pts_3d:
                (px, py), scale = project_point((pt[0], fy, pt[2]))
                pts_2d_shadow.append((px, py))
                scales_shadow.append(scale)
                
            palm_pts_shadow = np.array([pts_2d_shadow[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(canvas_view, [palm_pts_shadow], shadow_color)
            
            avg_s_shadow = np.mean(scales_shadow)
            thickness_base = max(2, min(100, int(40 * avg_s_shadow * intensity))) 
            
            for conn, _ in conns_z:
                p1, p2 = pts_2d_shadow[conn[0]], pts_2d_shadow[conn[1]]
                cv2.line(canvas_view, p1, p2, (max(0, s_color-10), max(0, s_color-10), max(0, s_color-10)), min(100, thickness_base + 4))
                cv2.line(canvas_view, p1, p2, shadow_color, thickness_base)

            if lost_frames > 0:
                glove_color = (150, 200, 255) if is_pinching else (200, 220, 240)
                outline_color = (50, 100, 200) 
            else:
                glove_color = (150, 255, 150) if is_pinching else (240, 240, 240) 
                outline_color = (50, 150, 50) if is_pinching else (100, 100, 100) 
            
            palm_pts = np.array([pts_2d[i] for i in palm_indices], dtype=np.int32)
            cv2.fillPoly(canvas_view, [palm_pts], outline_color)

            for conn, _ in conns_z:
                p1, p2 = pts_2d[conn[0]], pts_2d[conn[1]]
                s = (scales[conn[0]] + scales[conn[1]]) / 2
                thickness = max(2, min(80, int(45 * s))) 
                cv2.line(canvas_view, p1, p2, outline_color, thickness)
                cv2.circle(canvas_view, p1, thickness // 2, outline_color, -1)
                cv2.circle(canvas_view, p2, thickness // 2, outline_color, -1)

            cv2.fillPoly(canvas_view, [palm_pts], glove_color)

            for conn, _ in conns_z:
                p1, p2 = pts_2d[conn[0]], pts_2d[conn[1]]
                s = (scales[conn[0]] + scales[conn[1]]) / 2
                thickness = max(1, min(60, int(35 * s))) 
                cv2.line(canvas_view, p1, p2, glove_color, thickness)
                cv2.circle(canvas_view, p1, thickness // 2, glove_color, -1)
                cv2.circle(canvas_view, p2, thickness // 2, glove_color, -1)

        ball.render(canvas_view) 

        # 상태 텍스트 렌더링 (3D 뷰)
        status_text = "GRABBING!" if ball.state == "GRABBED" else "READY / HIT"
        cv2.putText(canvas_view, status_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(canvas_view, "Press 'ESC' to Exit | 'R' to Recalibrate & Reset Ball", (WIDTH - 650, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 상태 텍스트 렌더링 (실제 카메라 뷰 - 디버깅용)
        cv2.putText(camera_view, f"Status: {status_text}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        if calib_msg_timer > 0:
            cv2.putText(canvas_view, "CENTER RECALIBRATED & BALL RESET!", (WIDTH//2 - 350, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            calib_msg_timer -= 1
        
        # 두 개의 창 업데이트
        cv2.imshow(WINDOW_NAME, canvas_view)
        #cv2.imshow(CAMERA_WINDOW_NAME, camera_view)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            break
        elif key == ord('r') or key == ord('R'): 
            request_calibration = True
            ball.reset()
            pinch_history.clear()
            hand_center_history.clear()

cap.release()
cv2.destroyAllWindows()