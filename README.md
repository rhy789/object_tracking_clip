# Object Tracking with YOLOv5 + DeepSORT + CLIP

YOLOv5 객체 탐지, DeepSORT 추적, CLIP 임베딩을 통합한 고급 객체 추적 시스템입니다.

## 🚀 주요 기능

- **YOLOv5**: 실시간 객체 탐지
- **DeepSORT**: 다중 객체 추적 (Kalman Filter + Hungarian Algorithm)
- **CLIP**: 시각적 특징 임베딩을 통한 향상된 재식별

## 📁 프로젝트 구조

```
YCDC/
├── CLIP/                  # OpenAI CLIP 모델
├── deep_sort/             # DeepSORT 추적 알고리즘
├── yolov5/               # YOLOv5 객체 탐지
├── integrated_tracking.py # 통합 추적 시스템
├── performance_comparison.py # 성능 비교 도구
├── test_deepsort_clip.py # DeepSORT + CLIP 테스트
├── Dockerfile            # Docker 설정
├── docker-compose.yml    # Docker Compose 설정
├── requirements.txt      # Python 의존성
└── manual               # 실행 매뉴얼
```

## 🛠️ 설치 방법

### Docker 사용 (권장)

```bash
# Docker 이미지 빌드
docker-compose build

# 컨테이너 시작
docker-compose up -d

# 컨테이너 접속
docker exec -it tracking_system bash
```

### 로컬 설치

```bash
# 저장소 클론
git clone https://github.com/rhy789/object_tracking_clip.git
cd object_tracking_clip

# 의존성 설치
pip install -r requirements.txt

# YOLOv5 모델 다운로드
cd yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cd ..
```

## 📝 사용 방법

### 1. 기본 추적 실행

```bash
python3 integrated_tracking.py --source /path/to/video.mp4
```

### 2. 파라미터 조정

```bash
python3 integrated_tracking.py \
    --source /path/to/video.mp4 \
    --conf-thres 0.5 \
    --iou-thres 0.5 \
    --max-clip-distance 0.6
```

### 3. 결과 저장 위치 지정

```bash
python3 integrated_tracking.py \
    --source /path/to/video.mp4 \
    --save-dir results/my_tracking
```

### 4. 성능 비교

```bash
# 상세 성능 비교
python3 performance_comparison.py --source /path/to/video.mp4

# 간단한 성능 비교
python3 simple_comparison.py
```

## 🐳 Docker 실행 가이드

```bash
# 컨테이너 시작
docker start tracking_system

# 컨테이너 접속
docker exec -it tracking_system bash

# 작업 디렉토리로 이동
cd /workspace

# 추적 실행
python3 integrated_tracking.py --source /workspace/data/people.mp4
```

## 📊 성능 평가

프로젝트에는 추적 성능을 평가하는 도구가 포함되어 있습니다:

- **ID Switch 감소**: CLIP 임베딩을 통한 재식별 향상
- **추적 정확도**: 다양한 IOU 임계값에서의 성능 비교
- **실시간 처리**: FPS 및 처리 속도 측정

## 📄 라이선스

이 프로젝트는 다음 오픈소스 프로젝트를 기반으로 합니다:

- [YOLOv5](https://github.com/ultralytics/yolov5) - GPL-3.0
- [DeepSORT](https://github.com/nwojke/deep_sort) - GPL-3.0
- [CLIP](https://github.com/openai/CLIP) - MIT

## 🤝 기여

기여는 언제나 환영합니다! Pull Request를 제출해주세요.

## 📧 문의

문제가 발생하거나 질문이 있으시면 Issues를 통해 문의해주세요.
