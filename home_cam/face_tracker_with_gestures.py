import cv2
import numpy as np
import serial
import time
import mediapipe as mp # MediaPipe 임포트

# 시리얼 포트 설정 (사용자 환경에 맞게 수정 필요)
try:
    sp = serial.Serial('COM3', 115200, timeout=1)
    print("시리얼 포트 COM3가 성공적으로 열렸습니다.")
except serial.SerialException as e:
    print(f"시리얼 포트 연결 오류: {e}")
    print("COM 포트가 올바른지, 다른 프로그램에서 사용 중이지 않은지 확인하세요.")
    sp = None # 시리얼 포트 연결 실패 시 None으로 설정

def up():
    if sp:
        sp.write(b'w')
def down():
    if sp:
        sp.write(b's')
def left():
    if sp:
        sp.write(b'a')
def right():
    if sp:
        sp.write(b'd')

# 가중치 파일 경로 (스크립트와 같은 디렉토리에 있어야 함)
cascade_filename = 'haarcascade_frontalface_alt.xml'

# 모델 불러오기
face_cascade = cv2.CascadeClassifier(cascade_filename)

if face_cascade.empty():
    print(f"Error: {cascade_filename} 파일을 로드할 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 웹캠 초기화
webcam = cv2.VideoCapture(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not webcam.isOpened():
    print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    exit()

# 화면 중앙 영역 마진 설정
margin_x = 40
margin_y = 30
frame_center_x = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
frame_center_y = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

print(f"웹캠 해상도: {webcam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"화면 중앙: ({frame_center_x}, {frame_center_y})")

# 특수 동작 중인지 확인하는 플래그
shaking_active = False

def trigger_special_move(command_char):
    global shaking_active
    if shaking_active: # 이미 특수 동작 중이면 다시 시작하지 않음
        return

    shaking_active = True
    print(f"특수 동작 트리거: {command_char.decode()}!")
    if sp:
        sp.write(command_char)
    
    # 마이크로컨트롤러의 특수 동작이 완료될 때까지 Python 스크립트를 일시 정지
    # 'u' 동작은 약 0.84초 소요, 'n' 동작은 약 0.9초 소요. 1초 대기
    time.sleep(1.0) 

    shaking_active = False

# --- MediaPipe Hands 설정 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # 랜드마크를 그리기 위한 유틸리티
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # 최대 1개의 손만 감지
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- 제스처 감지를 위한 변수 ---
v_sign_cooldown_active = False
v_sign_cooldown_start_time = 0
V_SIGN_COOLDOWN_PERIOD = 2.0 # 'V'자 제스처 감지 후 2초간 쿨다운

nod_cooldown_active = False # '손바닥 펴기' 제스처 쿨다운
nod_cooldown_start_time = 0
NOD_COOLDOWN_PERIOD = 2.0 # '손바닥 펴기' 제스처 감지 후 2초간 쿨다운

# 손가락 펴짐/구부러짐 판단 헬퍼 함수
def is_finger_straight(landmarks, tip_idx, pip_idx, mcp_idx):
    # TIP (손가락 끝)의 Y좌표가 PIP (두 번째 관절)와 MCP (세 번째 관절)의 Y좌표보다 작으면 펴진 것으로 간주
    # (Y좌표는 화면 위쪽이 작고 아래쪽이 큼)
    return landmarks[tip_idx].y < landmarks[pip_idx].y and \
           landmarks[pip_idx].y < landmarks[mcp_idx].y

def is_thumb_straight(landmarks):
    # 엄지는 다른 손가락과 관절 구조가 다르므로 별도 판단
    # 엄지 끝(TIP)이 IP(첫 번째 관절)와 MCP(두 번째 관절)보다 Y좌표가 작고,
    # X좌표 기준으로도 바깥쪽으로 펴져 있는지 확인 (간단한 휴리스틱)
    return landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y and \
           landmarks[mp_hands.HandLandmark.THUMB_IP].y < landmarks[mp_hands.HandLandmark.THUMB_MCP].y and \
           landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x # 엄지가 바깥으로 펴진 정도

while True:
    status, frame = webcam.read()

    if not status:
        print("프레임을 읽을 수 없습니다. 웹캠 연결을 확인하세요.")
        break

    # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB를 선호)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 성능 향상을 위해 이미지를 쓰기 불가능으로 표시
    image.flags.writeable = False
    results = hands.process(image) # MediaPipe로 손 감지
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 다시 BGR로 변환

    # --- 제스처 감지 로직 ---
    v_sign_detected = False
    open_palm_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기 (디버깅용)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            
            # --- 'V'자 제스처 판단 ---
            index_extended = is_finger_straight(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
            middle_extended = is_finger_straight(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
            ring_curled = not is_finger_straight(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP)
            pinky_curled = not is_finger_straight(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)
            
            if index_extended and middle_extended and ring_curled and pinky_curled:
                v_sign_detected = True
            
            # --- '손바닥 펴기' 제스처 판단 ---
            thumb_straight = is_thumb_straight(landmarks)
            index_straight = is_finger_straight(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
            middle_straight = is_finger_straight(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
            ring_straight = is_finger_straight(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP)
            pinky_straight = is_finger_straight(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)

            if thumb_straight and index_straight and middle_straight and ring_straight and pinky_straight:
                open_palm_detected = True
            
            break # 첫 번째 감지된 손만 처리

    # --- 제스처 실행 및 쿨다운 관리 ---
    if v_sign_detected and not v_sign_cooldown_active:
        trigger_special_move(b'u') # 'u' 명령 전송 (흔들기)
        v_sign_cooldown_active = True
        v_sign_cooldown_start_time = time.time()
    
    if open_palm_detected and not nod_cooldown_active:
        trigger_special_move(b'n') # 'n' 명령 전송 (끄덕이기)
        nod_cooldown_active = True
        nod_cooldown_start_time = time.time()

    # 쿨다운 관리
    if v_sign_cooldown_active and (time.time() - v_sign_cooldown_start_time > V_SIGN_COOLDOWN_PERIOD):
        v_sign_cooldown_active = False
    if nod_cooldown_active and (time.time() - nod_cooldown_start_time > NOD_COOLDOWN_PERIOD):
        nod_cooldown_active = False

    # --- 얼굴 감지 및 트래킹 (기존 로직) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # MediaPipe 처리 후 RGB->BGR->GRAY 변환

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    largest_face = None
    largest_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)

    if largest_face is not None:
        x, y, w, h = largest_face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_center_x = x + w // 2
        face_center_y = y + h // 2

        cv2.circle(image, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
        cv2.circle(image, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
        cv2.rectangle(image, (frame_center_x - margin_x, frame_center_y - margin_y),
                      (frame_center_x + margin_x, frame_center_y + margin_y), (255, 255, 0), 1)

        # 팬/틸트 제어 로직 (특수 동작 중이 아닐 때만 실행)
        if not shaking_active: # 특수 동작 중이 아닐 때만 일반 트래킹 실행
            if face_center_x < frame_center_x - margin_x:
                left()
                cv2.putText(image, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif face_center_x > frame_center_x + margin_x:
                right()
                cv2.putText(image, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if face_center_y < frame_center_y - margin_y:
                up()
                cv2.putText(image, "UP", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif face_center_y > frame_center_y + margin_y:
                down()
                cv2.putText(image, "DOWN", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        time.sleep(0.1)
        
        print(f"Face Center: ({face_center_x}, {face_center_y})")

    # 결과 화면 표시
    cv2.imshow('Face & Hand Gestures', image) # 창 이름 변경

    # 'ESC' 키를 누르면 종료
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
hands.close() # MediaPipe Hands 리소스 해제
if sp:
    sp.close()
