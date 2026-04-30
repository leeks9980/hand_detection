import cv2
import mediapipe as mp
import numpy as np
import math
import random
from collections import deque

# ==========================================
# 1. 초기 설정 및 파티클 시스템
# ==========================================
mp_hands = mp.solutions.hands
WIDTH, HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_COLORS = [(0,0,255), (0,255,255), (0,255,0), (255,0,0), (255,0,255)]

trails = {
    'Left': [deque(maxlen=20) for _ in range(5)],
    'Right': [deque(maxlen=20) for _ in range(5)]
}

particles = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.history = deque(maxlen=6) # 꼬리(Trail)를 위한 이전 위치 저장
        
        # 사방으로 터지는 폭발 속도 (더 빠르고 다이나믹하게)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(10, 45) 
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        # 물리 엔진 요소 추가
        self.gravity = 0.9    # 아래로 당기는 중력
        self.friction = 0.88  # 공기 저항 (서서히 느려짐)
        
        # 입자의 고유 속성 랜덤화
        self.life = random.uniform(0.8, 1.2)
        self.decay = random.uniform(0.02, 0.05) # 사라지는 속도
        self.base_radius = random.uniform(2, 7) # 입자 크기
        self.color = color

    def update(self):
        # 이전 위치 기록
        self.history.append((int(self.x), int(self.y)))
        
        # 물리 연산 적용
        self.vx *= self.friction
        self.vy *= self.friction
        self.vy += self.gravity
        
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

    def draw(self, canvas):
        if self.life > 0:
            alpha = max(0, min(1, self.life))
            
            # 1. 꼬리(Trail) 그리기
            if len(self.history) > 1:
                pts = list(self.history)
                for i in range(1, len(pts)):
                    t_alpha = alpha * (i / len(pts))
                    c_trail = tuple(int(v * t_alpha) for v in self.color)
                    thickness = max(1, int(self.base_radius * t_alpha))
                    cv2.line(canvas, pts[i-1], pts[i], c_trail, thickness)

            # 2. 외부 글로우 (빛 번짐)
            r_glow = int(self.base_radius * alpha * 2.5)
            c_glow = tuple(int(v * alpha * 0.6) for v in self.color)
            cv2.circle(canvas, (int(self.x), int(self.y)), r_glow, c_glow, -1)

            # 3. 중심 코어 (눈부신 하얀색 중심)
            r_core = max(1, int(self.base_radius * alpha * 0.6))
            cv2.circle(canvas, (int(self.x), int(self.y)), r_core, (255, 255, 255), -1)

# [상태 관리 변수]
was_touching = False
beam_active = False 
last_touch_midpoints = [] 

# ==========================================
# 2. 비주얼 파이프라인 가동
# ==========================================
with mp_hands.Hands(max_num_hands=2, model_complexity=1) as hands:
    print("✨ [1회 터치] 빛 줄기 연결! / [2회 터치] 화려한 마법 폭발! (ESC: 종료)")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        present_hands = []
        current_tips = {'Left': [], 'Right': []}

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label
                present_hands.append(hand_label)
                
                for f_idx, tip_idx in enumerate(FINGER_TIPS):
                    tip = hand_landmarks.landmark[tip_idx]
                    pos = (int(tip.x * WIDTH), int(tip.y * HEIGHT))
                    current_tips[hand_label].append(pos)
                    trails[hand_label][f_idx].appendleft(pos)
                
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(color=(80, 80, 80), thickness=1)
                )

        for side in ['Left', 'Right']:
            if side not in present_hands:
                for d in trails[side]:
                    if len(d) > 0: d.pop() 

        for side in ['Left', 'Right']:
            for i, trail in enumerate(trails[side]):
                for j in range(1, len(trail)):
                    if trail[j-1] is None or trail[j] is None: continue
                    alpha = int(255 * (1 - j / len(trail)))
                    color = tuple([int(c * (alpha / 255)) for c in FINGER_COLORS[i]])
                    cv2.line(canvas, trail[j-1], trail[j], color, int((len(trail)-j)/1.5)+1)

        # ==========================================
        # 3. 상태 토글 로직 (연결 유지 & 파티클)
        # ==========================================
        is_touching = False
        
        if len(current_tips['Left']) == 5 and len(current_tips['Right']) == 5:
            total_dist = 0
            temp_midpoints = []
            
            for i in range(5):
                p1 = current_tips['Left'][i]
                p2 = current_tips['Right'][i]
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                total_dist += dist
                temp_midpoints.append(((p1[0]+p2[0])//2, (p1[1]+p2[1])//2))
            
            avg_dist = total_dist / 5
            
            if avg_dist < 150:
                is_touching = True
                last_touch_midpoints = temp_midpoints

        # [이벤트 감지] 손이 떨어지는 순간
        if was_touching and not is_touching:
            if not beam_active:
                beam_active = True
            else:
                beam_active = False
                # 폭발 효과: 각 손가락 위치에서 대량의 파티클 생성
                if last_touch_midpoints:
                    for i in range(5):
                        mx, my = last_touch_midpoints[i]
                        for _ in range(40): # 화려함을 위해 손가락당 40개의 파티클 방출 (총 200개)
                            particles.append(Particle(mx, my, FINGER_COLORS[i]))

        # 빛 줄기 렌더링
        if (is_touching or beam_active) and len(current_tips['Left']) == 5 and len(current_tips['Right']) == 5:
            for i in range(5):
                p1 = current_tips['Left'][i]
                p2 = current_tips['Right'][i]
                color = FINGER_COLORS[i]
                
                # 강렬한 레이저 글로우 효과
                cv2.line(canvas, p1, p2, color, 8)
                cv2.line(canvas, p1, p2, (int(color[0]*0.5+127), int(color[1]*0.5+127), int(color[2]*0.5+127)), 4)
                cv2.line(canvas, p1, p2, (255, 255, 255), 2)

        # 파티클 업데이트 및 화면에 그리기
        for p in particles:
            p.update()
            p.draw(canvas)
            
        particles = [p for p in particles if p.life > 0]

        was_touching = is_touching

        # ==========================================
        # 4. 직관적인 상태 UI 텍스트
        # ==========================================
        if beam_active:
            status_txt = "LINKED! Touch again to explode"
            txt_color = (0, 255, 0)
        elif is_touching:
            status_txt = "CHARGING... Release hands!"
            txt_color = (0, 255, 255)
        else:
            status_txt = "Move hands closer"
            txt_color = (150, 150, 150)

        cv2.putText(canvas, status_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color, 2)

        cv2.imshow('Magic Particle Burst', canvas)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()