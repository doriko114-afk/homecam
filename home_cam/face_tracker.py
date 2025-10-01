import cv2
import numpy as np
import serial
import time

# 시리얼 포트 설정 (사용자 환경에 맞게 수정 필요)
try:
    sp = serial.Serial('COM3', 115200, timeout=1)
except serial.SerialException as e:
    print(f"시리얼 포트 연결 오류: {e}")
    print("COM 포트가 올바른지, 다른 프로그램에서 사용 중이지 않은지 확인하세요.")
    sp = None # 시리얼 포트 연결 실패 시 None으로 설정

def up():
    if sp:
        sp.write(b'w')
        # print("Up command sent") # 디버깅용
def down():
    if sp:
        sp.write(b's')
        # print("Down command sent") # 디버깅용
def left():
    if sp:
        sp.write(b'a')
        # print("Left command sent") # 디버깅용
def right():
    if sp:
        sp.write(b'd')
        # print("Right command sent") # 디버깅용

# 가중치 파일 경로 (스크립트와 같은 디렉토리에 있어야 함)
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

if cascade.empty():
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

while True:
    status, frame = webcam.read()

    if not status:
        print("프레임을 읽을 수 없습니다. 웹캠 연결을 확인하세요.")
        break

    # 그레이 스케일 변환 (얼굴 감지는 그레이스케일 이미지에서 수행)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = cascade.detectMultiScale(gray,            # 입력 이미지
                                     scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                     minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                     minSize=(30,30)  # 탐지 객체 최소 크기 (얼굴 크기에 따라 조정)
                                     )
    
    largest_face = None
    largest_area = 0

    # 가장 큰 얼굴 찾기
    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)

    if largest_face is not None:
        x, y, w, h = largest_face
        # 얼굴 주변에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 얼굴의 중심 좌표 계산
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # 화면 중앙에 얼굴 중심 표시 (디버깅용)
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
        cv2.rectangle(frame, (frame_center_x - margin_x, frame_center_y - margin_y),
                      (frame_center_x + margin_x, frame_center_y + margin_y), (255, 255, 0), 1)


        # 팬/틸트 제어 로직
        if face_center_x < frame_center_x - margin_x:
            left()
            cv2.putText(frame, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif face_center_x > frame_center_x + margin_x:
            right()
            cv2.putText(frame, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # else:
            # print("PAN CENTER")

        if face_center_y < frame_center_y - margin_y:
            up()
            cv2.putText(frame, "UP", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif face_center_y > frame_center_y + margin_y:
            down()
            cv2.putText(frame, "DOWN", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # else:
            # print("TILT CENTER")
        
        # 시리얼 명령 전송 후 잠시 대기하여 과도한 명령 방지
        time.sleep(0.1)
        
        print(f"Face Center: ({face_center_x}, {face_center_y})")

    # 결과 화면 표시
    cv2.imshow('Face Tracking', frame)

    # 'ESC' 키를 누르면 종료
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
if sp:
    sp.close()