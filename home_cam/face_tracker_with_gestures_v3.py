import cv2
import numpy as np
import serial
import time
import mediapipe as mp # MediaPipe 임포트
import mediapipe.python.solutions.drawing_utils as mp_drawing # 랜드마크를 그리기 위한 유틸리티

# 시리얼 포트 설정 (사용자 환경에 맞게 수정 필요)
try:
    sp = serial.Serial('COM3', 115200, timeout=1)
    print("시리얼 포트 COM3가 성공적으로 열렸습니다.")
except serial.SerialException as e:
    print(f"시리얼 포트 열기 오류: {e}")
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

# 웹캠 초기화
webcam = cv2.VideoCapture(1)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not webcam.isOpened():
    print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    exit()

frame_center_x = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
frame_center_y = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

print(f"웹캠 해상도: {webcam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"화면 중앙: ({frame_center_x}, {frame_center_y})")

# 특수 동작 관련 기능은 제거되었으므로 해당 변수 및 함수는 필요 없습니다.
# --- MediaPipe Face Mesh 설정 (시선 추정용) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, # 최대 1개의 얼굴만 감지
    refine_landmarks=True, # 눈동자 랜드마크 정밀화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

while True:
    status, frame = webcam.read()

    if not status:
        print("프레임을 읽을 수 없습니다. 웹캠 연결을 확인하세요.")
        break

    # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB를 선호)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 성능 향상을 위해 이미지를 쓰기 불가능으로 표시
    image.flags.writeable = False

    # --- MediaPipe Face Mesh로 얼굴 감지 및 시선 추정 ---
    try:
        face_results = face_mesh.process(image)
        
        # 이미지에 그리기 위해 쓰기 가능으로 설정
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 다시 BGR로 변환

        # gaze_override_tracking은 이제 필요 없으므로 제거합니다。

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # 얼굴 랜드마크 그리기 (디버깅용)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,255,200), thickness=1, circle_radius=1))

                # 시선 추정을 위한 2D 이미지 포인트 추출
                # MediaPipe Face Mesh 랜드마크 인덱스 (코 끝, 턱, 왼쪽 눈 왼쪽 끝, 오른쪽 눈 오른쪽 끝, 왼쪽 입술 끝, 오른쪽 입술 끝)
                # 이 인덱스들은 MediaPipe Face Mesh 모델에 따라 다를 수 있습니다.
                # 정확한 인덱스는 MediaPipe 문서 또는 예제 코드를 참조해야 합니다.
                # 여기서는 일반적인 랜드마크 위치를 가정합니다.
                # 1: 코 끝 (Nose tip)
                # 152: 턱 (Chin)
                # 33: 왼쪽 눈 왼쪽 끝 (Left eye left corner)
                # 263: 오른쪽 눈 오른쪽 끝 (Right eye right corner)
                # 61: 왼쪽 입술 끝 (Left mouth corner)
                # 291: 오른쪽 입술 끝 (Right mouth corner)
                
                image_points = np.array([
                    (face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]),
                    (face_landmarks.landmark[152].x * frame.shape[1], face_landmarks.landmark[152].y * frame.shape[0]),
                    (face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]), # Left eye left corner
                    (face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]), # Right eye right corner
                    (face_landmarks.landmark[61].x * frame.shape[1], face_landmarks.landmark[61].y * frame.shape[0]),
                    (face_landmarks.landmark[291].x * frame.shape[1], face_landmarks.landmark[291].y * frame.shape[0])
                ], dtype=np.float64)

                # solvePnP를 사용하여 머리 자세 추정
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    face_3d_model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                # 회전 행렬로 변환
                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                # 머리 자세 축 그리기 (시각화)
                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 2) # Z-axis (gaze direction)

                # --- 시선 방향에 따른 팬틸트 제어 ---
                # rotation_vector의 y축 (yaw)과 x축 (pitch) 값을 사용하여 시선 방향 판단
                # yaw: 좌우 회전, pitch: 상하 회전
                # 값의 범위는 라디안이므로 각도로 변환하거나 상대적인 값으로 판단
                
                # 대략적인 각도 값 (라디안 -> 도)
                pitch = np.degrees(rotation_vector[0])
                yaw = np.degrees(rotation_vector[1])
                
                # 시선 제어를 위한 마진 (데드 존)
                gaze_margin_x = 10 # 좌우 시선 마진 (도)
                gaze_margin_y = 10 # 상하 시선 마진 (도)

                # 특수 동작 중이 아닐 때만 시선 제어
                # gaze_override_tracking은 이제 필요 없으므로 제거합니다。
                    
                if yaw < -gaze_margin_x: # 왼쪽을 바라봄
                    left()
                    cv2.putText(image, "GAZE LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif yaw > gaze_margin_x: # 오른쪽을 바라봄
                    right()
                    cv2.putText(image, "GAZE RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(image, "GAZE CENTER X", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if pitch < -gaze_margin_y: # 위를 바라봄
                    up()
                    cv2.putText(image, "GAZE UP", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif pitch > gaze_margin_y: # 아래를 바라봄
                    down()
                    cv2.putText(image, "GAZE DOWN", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(image, "GAZE CENTER Y", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                break # 첫 번째 감지된 얼굴만 처리
    except Exception as e:
        print(f"Error during MediaPipe processing or gaze estimation: {e}")
