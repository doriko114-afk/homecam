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
    # 'u' 동작은 약 0.84초 소요되므로, 1초 대기
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

# --- 손 흔들기 제스처 감지를 위한 변수 ---
last_hand_x = None
hand_movement_history = [] # 'left', 'right' 움직임 방향 저장
WAVE_SEQUENCE = ['left', 'right', 'left'] # 흔들기 제스처 패턴 (좌-우-좌)
WAVE_MOVEMENT_THRESHOLD = 30 # 최소 움직임 픽셀 (이 값 이상 움직여야 방향 변화로 인정)
wave_cooldown_active = False
wave_cooldown_start_time = 0
WAVE_COOLDOWN_PERIOD = 2.0 # 제스처 감지 후 2초간 쿨다운

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

    # --- 손 흔들기 제스처 감지 로직 ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기 (디버깅용)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손목 랜드마크 (0번)의 X 좌표를 가져옴
            wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])

            if last_hand_x is not None:
                movement = wrist_x - last_hand_x # 현재 위치 - 이전 위치
                
                if abs(movement) > WAVE_MOVEMENT_THRESHOLD: # 일정 픽셀 이상 움직였을 때만
                    if movement < 0: # 왼쪽으로 움직임
                        if not hand_movement_history or hand_movement_history[-1] != 'left':
                            hand_movement_history.append('left')
                    elif movement > 0: # 오른쪽으로 움직임
                        if not hand_movement_history or hand_movement_history[-1] != 'right':
                            hand_movement_history.append('right')
                    
                    # 히스토리 길이 유지
                    if len(hand_movement_history) > len(WAVE_SEQUENCE):
                        hand_movement_history.pop(0)

                    # 흔들기 패턴과 일치하는지 확인 (쿨다운 중이 아닐 때만)
                    if (hand_movement_history == WAVE_SEQUENCE or hand_movement_history == WAVE_SEQUENCE[::-1]) and not wave_cooldown_active:
                        trigger_special_move(b'u') # 'u' 명령 전송
                        hand_movement_history = [] # 히스토리 초기화
                        wave_cooldown_active = True
                        wave_cooldown_start_time = time.time()
            
            last_hand_x = wrist_x
            break # 첫 번째 감지된 손만 처리
    else:
        # 손이 감지되지 않으면 제스처 감지 상태 초기화
        last_hand_x = None
        hand_movement_history = []
    
    # 쿨다운 관리
    if wave_cooldown_active and (time.time() - wave_cooldown_start_time > WAVE_COOLDOWN_PERIOD):
        wave_cooldown_active = False

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
        if not shaking_active:
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
    cv2.imshow('Face & Hand Tracking', image) # 창 이름 변경

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