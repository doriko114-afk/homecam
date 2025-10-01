# app.py
# MS Windows 11 + OpenCV + Flask 기반 MJPEG 스트리밍 서버
# 카메라 인덱스 1, 320x240 해상도로 스트리밍
# 브라우저: http://localhost:5000  또는 다른 PC: http://<윈도우IP>:5000

import cv2
import time
import atexit
from flask import Flask, Response, render_template_string

# ===== 사용자 설정 =====
CAMERA_INDEX = 0      # 0: 기본 카메라, 1: 두 번째 카메라
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 80     # 0~100 범위, 높을수록 고화질/대용량
HOST = "0.0.0.0"
PORT = 5000
# =====================

app = Flask(__name__)

cap = None

def open_camera():
    """카메라 열기 (Windows에서는 CAP_DSHOW를 권장)"""
    global cap
    if cap is not None and cap.isOpened():
        return cap

    print("카메라(%d) 오픈 시도..." % CAMERA_INDEX)
    # CAP_DSHOW는 Windows DirectShow 백엔드를 사용하여 지연 및 경고를 줄임
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("카메라 오픈 실패: 인덱스 %d" % CAMERA_INDEX)
        return None

    # 해상도 지정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # 자동 노출/포커스가 문제되면 여기서 추가 설정 가능

    # 설정 적용 대기
    time.sleep(0.2)

    # 최종 설정 확인 출력
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("카메라 오픈 성공: %dx%d" % (w, h))
    return cap

def close_camera():
    """프로세스 종료 시 카메라 자원 반납"""
    global cap
    if cap is not None:
        try:
            if cap.isOpened():
                cap.release()
                print("카메라 자원 반납 완료")
        except Exception as e:
            print("카메라 반납 중 예외: %s" % str(e))
        cap = None

atexit.register(close_camera)

HTML_INDEX = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>MJPEG Stream</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif; }
    header { padding: 12px 16px; border-bottom: 1px solid #ddd; }
    #wrap { padding: 16px; }
    img { max-width: 100%%; height: auto; border: 1px solid #ddd; }
    .hint { color: #666; font-size: 14px; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <strong>윈도우 11 OpenCV MJPEG 스트리밍</strong>
  </header>
  <div id="wrap">
    <img src="/video_feed" alt="video stream">
    <div class="hint">
      이 페이지는 <code>/video_feed</code> 엔드포인트의 MJPEG을 표시합니다.
    </div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_INDEX)

@app.route("/health")
def health():
    """간단한 헬스 체크 엔드포인트"""
    c = open_camera()
    if c is None or not c.isOpened():
        return "camera_not_ready", 503
    return "ok", 200

def mjpeg_generator():
    """MJPEG 스트림 생성기"""
    c = open_camera()
    if c is None:
        # 연결 실패 시 빈 스트림 종료
        print("스트리밍 시작 실패: 카메라 없음")
        return

    # Motion JPEG 파트 헤더
    boundary = b"--frame"
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    print("스트리밍 시작: /video_feed")
    while True:
        # 프레임 읽기
        retval, frame = c.read()
        if not retval or frame is None:
            print("프레임 읽기 실패, 0.3초 대기 후 재시도")
            time.sleep(0.3)
            # 재오픈 시도
            c = open_camera()
            if c is None:
                break
            continue

        # 안전을 위해 리사이즈 보장
        try:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        except Exception as e:
            print("리사이즈 예외: %s" % str(e))
            continue

        # JPEG 인코딩
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            print("JPEG 인코딩 실패")
            continue

        jpg_bytes = buffer.tobytes()

        # 멀티파트 응답 전송
        yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n" + \
              b"Content-Length: " + str(len(jpg_bytes)).encode("ascii") + b"\r\n\r\n" + \
              jpg_bytes + b"\r\n"

        # 너무 빠른 전송으로 CPU 점유가 높을 때 약간 쉼
        # 필요 시 주석 처리 가능
        time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # 첫 접속 지연을 줄이기 위해 선오픈 시도
    open_camera()
    print("Flask 서버 시작: http://%s:%d" % ("localhost", PORT))
    print("동일 네트워크의 다른 PC에서는 http://<윈도우IP>:%d" % PORT)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
