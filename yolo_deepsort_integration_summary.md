# YOLOv5 + DeepSORT 통합 시스템 정리

## 1. 전체 개요

```
영상 입력 → YOLOv5 탐지 → CLIP 임베딩 추출 → DeepSORT 트래킹 → 결과 저장
```

**핵심 특징:**
- ✅ 실시간 처리 (중간 파일 저장 없음)
- ✅ 단일 스크립트로 전체 파이프라인 실행
- ✅ YOLO → CLIP → DeepSORT 직접 연결

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
┌─────────────┐
│  영상 입력   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  YOLOv5 탐지    │ → 바운딩 박스, 신뢰도, 클래스
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  객체 필터링    │ → 사람만, 작은 객체 제거
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  CLIP 임베딩    │ → 각 바운딩 박스에서 512차원 벡터 추출
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  DeepSORT 트래킹│ → 3단계 매칭 (Appearance → IOU → CLIP)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  결과 시각화    │ → 트랙 ID, 바운딩 박스 그리기
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  영상 저장      │ → 트래킹 결과 영상
└─────────────────┘
```

### 2.2 핵심 함수

**파일:** `integrated_tracking.py`

```python
def run_integrated_tracking(
    source,              # 입력 영상 경로
    weights='yolov5s.pt', # YOLO 모델 가중치
    conf_thres=0.4,      # 신뢰도 임계값
    iou_thres=0.4,       # NMS IOU 임계값
    max_clip_distance=0.5, # CLIP 매칭 거리
    save_dir='results/integrated_tracking'  # 저장 위치
):
    # 전체 파이프라인 실행
    ...
```

---

## 3. 단계별 상세 설명

### 3.1 모델 로딩

```python
# 1. YOLOv5 모델 로드
device = select_device('')  # GPU 자동 선택
model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)

# 2. CLIP 모델 로드
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# 3. DeepSORT 트래커 초기화
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2)
tracker_obj = tracker.Tracker(
    metric, 
    max_iou_distance=0.7, 
    max_age=30, 
    n_init=3,
    max_clip_distance=0.5
)
```

**특징:**
- GPU 자동 선택 (CUDA 가능 시 GPU 사용)
- YOLOv5s (경량 모델, 빠른 처리)
- CLIP ViT-B/32 (균형잡힌 성능과 속도)

---

### 3.2 영상 프레임 처리 루프

```python
for path, im, im0s, vid_cap, s in dataset:
    # im: 전처리된 이미지 (640x640, torch tensor)
    # im0s: 원본 이미지 (numpy array)
    
    # 1. YOLOv5 추론
    pred = model(im, augment=False, visualize=False)
    
    # 2. NMS (Non-Maximum Suppression)
    pred = non_max_suppression(
        pred, conf_thres, iou_thres,
        classes=[0],  # 사람만 탐지
        max_det=1000
    )
```

**프로세스:**
1. 이미지 전처리 (letterbox, 크기 조정)
2. YOLO 모델 추론
3. NMS로 중복 탐지 제거

---

### 3.3 객체 필터링

```python
# Rescale boxes (640x640 → 원본 크기)
det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

# 작은 객체 필터링 (이미지 면적의 0.5% 미만 제거)
img_area = im0.shape[0] * im0.shape[1]
bbox_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
area_ratio = bbox_area / img_area
min_area_ratio = 0.005
valid_objects = area_ratio >= min_area_ratio
det = det[valid_objects]
```

**필터링 기준:**
- 클래스: 사람만 (class 0)
- 신뢰도: `conf_thres` 이상
- 크기: 이미지 면적의 0.5% 이상

---

### 3.4 CLIP 임베딩 추출

```python
for *xyxy, conf, cls in det:
    bbox = [int(x) for x in xyxy]  # [x1, y1, x2, y2]
    
    # CLIP 임베딩 추출
    clip_embedding = extract_clip_embedding(
        im0, bbox, clip_model, clip_preprocess, device
    )
    
    if clip_embedding is not None:
        # TLWH 형식으로 변환
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        tlwh = np.array([x1, y1, w, h], dtype=np.float64)
        
        # CLIP 임베딩을 feature로 사용
        feature = clip_embedding  # (512,) shape
        
        # Detection 객체 생성
        detection = Detection(
            tlwh=tlwh,
            confidence=float(conf),
            feature=feature,
            clip_embedding=clip_embedding
        )
        detections.append(detection)
```

**핵심:**
- 각 탐지마다 CLIP 임베딩 추출
- 512차원 벡터로 시각적 특성 표현
- DeepSORT Detection 객체 생성

---

### 3.5 DeepSORT 트래킹

```python
# DeepSORT 업데이트
tracker_obj.predict()  # Kalman 필터로 위치 예측
tracker_obj.update(detections)  # 3단계 매칭 파이프라인
```

**매칭 과정:**
1. **Appearance 매칭**: Feature 벡터 기반
2. **IOU 매칭**: 바운딩 박스 겹침 기반
3. **CLIP 매칭**: 시각적 유사도 기반 ✨

**결과:**
- 매칭된 탐지 → 트랙 업데이트
- 미매칭 탐지 → 새 트랙 생성
- 오래된 트랙 → 삭제

---

### 3.6 결과 시각화

```python
for track in tracker_obj.tracks:
    if not track.is_confirmed() or track.time_since_update > 1:
        continue  # 확인되지 않은 트랙 또는 오래된 트랙 건너뛰기
    
    # 바운딩 박스 좌표
    tlwh = track.to_tlwh()
    x1, y1 = int(tlwh[0]), int(tlwh[1])
    x2, y2 = int(x1 + tlwh[2]), int(y1 + tlwh[3])
    
    # 트랙 ID
    track_id = track.track_id
    
    # 색상 생성 (트랙 ID 기반)
    color = (
        int((track_id * 3) % 255),
        int((track_id * 7) % 255),
        int((track_id * 11) % 255)
    )
    
    # 바운딩 박스 그리기
    cv2.rectangle(im0_with_tracks, (x1, y1), (x2, y2), color, 2)
    
    # 트랙 ID 라벨
    label = f"ID:{track_id}"
    cv2.putText(im0_with_tracks, label, (x1, y1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
```

**시각화 요소:**
- 바운딩 박스: 트랙 ID별 고유 색상
- 트랙 ID 라벨: 바운딩 박스 상단에 표시
- 컬러 구분: 각 트랙은 다른 색상

---

### 3.7 영상 저장

```python
if video_writer is None:
    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_path = save_path / "tracked_video.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

# 매 프레임 저장
video_writer.write(im0_with_tracks)
```

**저장 정보:**
- 원본 FPS, 해상도 유지
- MP4 형식 (mp4v 코덱)
- 트래킹 결과 포함

---

## 4. 실행 방법

### 4.1 기본 실행

```bash
python3 integrated_tracking.py --source /workspace/data/people.mp4
```

### 4.2 모든 옵션 지정

```bash
python3 integrated_tracking.py \
    --source /workspace/data/people.mp4 \
    --weights yolov5s.pt \
    --conf-thres 0.4 \
    --iou-thres 0.4 \
    --max-clip-distance 0.5 \
    --save-dir results/integrated_tracking
```

### 4.3 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--source` | (필수) | 입력 영상 경로 |
| `--weights` | `yolov5s.pt` | YOLOv5 모델 가중치 |
| `--conf-thres` | `0.4` | 신뢰도 임계값 |
| `--iou-thres` | `0.4` | NMS IOU 임계값 |
| `--max-clip-distance` | `0.5` | CLIP 매칭 최대 거리 |
| `--save-dir` | `results/integrated_tracking` | 결과 저장 디렉토리 |

---

## 5. 데이터 흐름

### 5.1 프레임별 처리 흐름

```
프레임 N 입력
    ↓
YOLO 탐지 → [bbox1, bbox2, ...]
    ↓
필터링 → [bbox1, bbox3, ...]  (작은 객체 제거)
    ↓
CLIP 임베딩 → [embed1, embed3, ...]
    ↓
Detection 객체 생성 → [det1, det3, ...]
    ↓
DeepSORT 업데이트
    ↓
Kalman 필터 예측
    ↓
3단계 매칭
    ├─ Appearance 매칭
    ├─ IOU 매칭
    └─ CLIP 매칭
    ↓
트랙 업데이트/생성/삭제
    ↓
시각화 → 프레임 N 출력
```

### 5.2 메모리 효율성

**최적화 방법:**
- 중간 파일 저장 없음 (메모리 직접 전달)
- GPU 메모리 효율적 사용 (torch.no_grad())
- 프레임별 처리 (전체 영상 로드 불필요)

---

## 6. 성능 특징

### 6.1 처리 속도

**예상 성능 (RTX 4070 Laptop):**
- YOLO 탐지: ~10ms/frame
- CLIP 임베딩: ~5ms/frame (탐지당)
- DeepSORT 업데이트: ~1ms/frame
- **총 처리 시간**: ~16ms/frame → **약 60 FPS**

### 6.2 정확도

**개선 사항:**
- ✅ 폐색 처리 (CLIP 임베딩 기반)
- ✅ 초기 트랙 안정화 (3단계 매칭)
- ✅ 512차원 임베딩 (고차원 정보)
- ✅ 히스토리 평균 (노이즈 제거)

---

## 7. 주요 코드 구조

### 7.1 CLIP 임베딩 추출 함수

```python
def extract_clip_embedding(image, bbox, clip_model, clip_preprocess, device):
    """바운딩박스 영역에서 CLIP 임베딩 추출"""
    # 1. 바운딩 박스 좌표 추출
    x1, y1, x2, y2 = map(int, bbox)
    
    # 2. 이미지 크롭
    cropped_image = image[y1:y2, x1:x2]
    pil_image = Image.fromarray(cropped_image)
    
    # 3. CLIP 전처리
    image_tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    
    # 4. CLIP 임베딩 추출
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().flatten().astype(np.float32)
```

### 7.2 DeepSORT Detection 객체 생성

```python
# Detection 객체 생성
detection = Detection(
    tlwh=tlwh,              # 바운딩 박스 (Top-Left-Width-Height)
    confidence=float(conf), # 신뢰도
    feature=clip_embedding, # Feature 벡터 (512차원)
    clip_embedding=clip_embedding  # CLIP 임베딩 (별도 저장)
)
```

### 7.3 트래킹 결과 시각화

```python
for track in tracker_obj.tracks:
    if track.is_confirmed() and track.time_since_update <= 1:
        # 바운딩 박스 그리기
        tlwh = track.to_tlwh()
        x1, y1, x2, y2 = int(tlwh[0]), int(tlwh[1]), \
                         int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
        
        # 색상 및 라벨
        color = (int((track.track_id * 3) % 255),
                 int((track.track_id * 7) % 255),
                 int((track.track_id * 11) % 255))
        
        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im0, f"ID:{track.track_id}", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
```

---

## 8. 출력 결과

### 8.1 저장 파일

```
results/integrated_tracking/
└── tracked_video.mp4  # 트래킹 결과 영상
```

### 8.2 콘솔 출력

```
YOLOv5 모델 로딩 중...
CLIP 모델 로딩 중...
DeepSORT 트래커 초기화 중...

트래킹 시작...
--------------------------------------------------
프레임 1: 탐지 3개 → 트랙 3개
프레임 2: 탐지 3개 → 트랙 3개
프레임 3: 탐지 2개 → 트랙 3개
...
--------------------------------------------------

✅ 트래킹 완료!
총 프레임: 242
결과 저장 위치: results/integrated_tracking
총 추적된 객체 수: 5
```

---

## 9. 핵심 정리

### 9.1 통합 시스템의 장점

1. **실시간 처리**: 중간 파일 저장 없음, 직접 메모리 전달
2. **단일 스크립트**: 하나의 명령으로 전체 파이프라인 실행
3. **높은 정확도**: 3단계 매칭 파이프라인 (Appearance → IOU → CLIP)
4. **폐색 대응**: CLIP 임베딩 기반 시각적 유사도 매칭
5. **확장 가능**: 쉬운 파라미터 조정

### 9.2 전체 흐름 요약

```
입력 영상
    ↓
YOLOv5 탐지 (사람 객체만)
    ↓
필터링 (작은 객체 제거)
    ↓
CLIP 임베딩 추출 (512차원)
    ↓
DeepSORT 트래킹 (3단계 매칭)
    ↓
결과 시각화 (트랙 ID, 바운딩 박스)
    ↓
출력 영상
```

### 9.3 핵심 특징

- **YOLOv5**: 빠른 객체 탐지
- **CLIP**: 고차원 시각적 특징 추출
- **DeepSORT**: 3단계 매칭 기반 다중 객체 트래킹
- **통합**: 하나의 스크립트로 전체 파이프라인 실행

---

## 10. 추가 개선 가능한 부분

### 10.1 성능 최적화
- ✅ 배치 처리로 CLIP 임베딩 속도 향상
- ✅ GPU 메모리 사용량 모니터링
- ✅ 다중 스레드 처리

### 10.2 기능 확장
- ✅ 여러 클래스 지원 (사람 외에도)
- ✅ 실시간 웹캠 입력 지원
- ✅ 실시간 결과 스트리밍

### 10.3 정확도 향상
- ✅ 파라미터 자동 튜닝
- ✅ 신뢰도 가중치 조정
- ✅ 다양한 NMS 전략

---

## 11. 결론

**YOLOv5 + DeepSORT 통합 시스템**은 다음과 같은 특징을 가집니다:

1. **통합된 파이프라인**: YOLO → CLIP → DeepSORT 단일 시스템
2. **실시간 처리**: 중간 파일 없이 직접 메모리 전달
3. **높은 정확도**: 3단계 매칭으로 폐색 상황 대응
4. **사용 용이성**: 단일 명령으로 실행

**최종 목표 달성:**
- ✅ 단일 스크립트로 전체 파이프라인 실행
- ✅ 실시간 처리 (중간 파일 저장 없음)
- ✅ 정확한 다중 객체 트래킹
- ✅ 폐색 상황 대응
