# HomeCam: 지능형 팬틸트 모니터링 시스템

## 프로젝트 개요

이 프로젝트는 웹캠, 팬틸트 서보모터, STM32 마이크로컨트롤러를 활용하여 실시간 객체 탐지 및 추적 기능을 갖춘 지능형 모니터링 시스템을 구축합니다. 사용자는 웹 인터페이스를 통해 원격으로 팬틸트를 제어하거나, YOLOv8 기반의 객체 탐지 기능을 통해 특정 대상을 자동으로 추적할 수 있습니다.

## 주요 기능

-   **실시간 웹캠 스트리밍:** 웹 인터페이스를 통해 웹캠 영상을 실시간으로 모니터링합니다.
-   **팬틸트 서보모터 제어:**
    -   **YOLOv8 기반 객체 추적:** 특정 객체(예: 사람)를 자동으로 탐지하고 추적하여 팬틸트를 제어합니다.
    -   **웹 인터페이스 수동 제어:** 웹 페이지의 버튼을 통해 팬틸트의 상하좌우 움직임을 원격으로 조작합니다.
-   **STM32 연동:** 시리얼 통신을 통해 STM32 마이크로컨트롤러가 서보모터를 정밀하게 제어합니다.
-   **모듈화된 코드:** 객체 탐지 로직과 웹 제어 로직이 분리되어 있어 유연한 확장이 가능합니다.

## 기술 스택

-   **하드웨어:**
    -   USB 웹캠
    -   팬틸트 서보모터 (2축)
    -   STM32 마이크로컨트롤러
-   **소프트웨어:**
    -   **Python:**
        -   `OpenCV`: 영상 처리 및 웹캠 제어
        -   `ultralytics`: YOLOv8 객체 탐지 프레임워크
        -   `pyserial`: STM32와의 시리얼 통신
        -   `Flask`: MJPEG 스트리밍 서버 (`mjpg_stream.py`)
    -   **Node.js:** 웹 서버 및 API 처리 (`app.js`)
    -   **HTML/CSS/JavaScript:** 웹 제어 인터페이스 (`public/index.html` - Node.js에서 제공)
    -   **C (STM32 펌웨어):** 서보모터 제어 로직

## 설치 및 실행

### 1. Python 환경 설정

필요한 Python 라이브러리를 설치합니다:

```bash
pip install opencv-python ultralytics pyserial Flask
```

### 2. YOLOv8 모델 가중치 다운로드

`yolo_pan_tilt_tracker.py` 스크립트와 같은 디렉토리에 `yolov8n.pt` 모델 파일을 다운로드합니다:

[YOLOv8n 모델 다운로드](https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt)

### 3. STM32 펌웨어 업로드

STM32 마이크로컨트롤러에 팬틸트 서보모터를 제어하는 펌웨어를 업로드합니다. (예: `main.c` 파일의 서보 제어 로직)

### 4. 실행

#### 4.1. YOLOv8 객체 추적 실행

```bash
python yolo_pan_tilt_tracker.py
```

특정 클래스 ID를 추적하려면 `--class_id` 인자를 사용합니다 (예: 사람 `0`, 노트북 `64`):

```bash
python yolo_pan_tilt_tracker.py --class_id 0
```

#### 4.2. 웹 인터페이스 실행 (Node.js 서버 및 MJPG 스트리머 필요)

웹 인터페이스 (`pan_tilt_web_interface.html`)를 사용하려면 Node.js 서버와 MJPG 스트리머가 필요합니다.

1.  **Node.js 서버 실행:**
    `app.js` 파일을 실행하여 웹 서버를 시작합니다.
    ```bash
    node app.js
    ```
2.  **MJPG 스트리머 실행:**
    웹캠 영상을 스트리밍하기 위해 `mjpg_stream.py` 스크립트를 실행합니다.
    ```bash
    python mjpg_stream.py
    ```
3.  **웹 인터페이스 접속:**
    Node.js 서버와 MJPG 스트리머가 실행 중인 상태에서 웹 브라우저를 열고 `http://localhost:3000`으로 접속합니다. `public/index.html` 파일이 Node.js 서버를 통해 제공되며, 이 인터페이스를 통해 팬틸트를 제어하고 `mjpg_stream.py`에서 제공하는 실시간 웹캠 영상을 확인할 수 있습니다.

## img
<img src = "https://github.com/doriko114-afk/homecam/blob/master/img/homecam.png">
