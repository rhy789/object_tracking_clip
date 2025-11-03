#!/usr/bin/env python3
"""
YOLOv5 + DeepSORT + CLIP 통합 트래킹 시스템
실시간으로 YOLO 탐지 → CLIP 임베딩 → DeepSORT 트래킹
"""

import cv2
import numpy as np
import torch
import clip
from PIL import Image
import sys
import argparse
from pathlib import Path

# YOLOv5 경로 추가
YOLO_PATH = Path(__file__).parent / "yolov5"
if str(YOLO_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_PATH))

# DeepSORT 경로 추가
DEEPSORT_PATH = Path(__file__).parent / "deep_sort"
if str(DEEPSORT_PATH) not in sys.path:
    sys.path.insert(0, str(DEEPSORT_PATH))

# YOLOv5 imports (절대 경로 사용)
sys.path.insert(0, str(Path(__file__).parent / "yolov5"))
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.augmentations import letterbox
from utils.dataloaders import LoadImages

# DeepSORT imports
from deep_sort import tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection

# YOLO person class = 0
PERSON_CLASS = 0


def extract_clip_embedding(image, bbox, clip_model, clip_preprocess, device):
    """
    바운딩박스 영역에서 CLIP 임베딩 추출
    
    Args:
        image: 원본 이미지 (numpy array)
        bbox: 바운딩박스 좌표 [x1, y1, x2, y2]
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        device: GPU/CPU 디바이스
    
    Returns:
        embedding: CLIP 임베딩 벡터 (numpy array, float32)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        cropped_image = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_image)
        image_tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten().astype(np.float32)
        
    except Exception as e:
        print(f"CLIP 임베딩 추출 오류: {e}")
        return None


def run_integrated_tracking(
    source,
    weights='yolov5s.pt',
    conf_thres=0.4,
    iou_thres=0.4,
    max_age=30,
    n_init=3,
    max_clip_distance=0.5,
    save_dir='results/integrated_tracking',
    save_video=True,
):
    """
    YOLOv5 + CLIP + DeepSORT 통합 트래킹 실행
    
    Args:
        source: 입력 영상 경로
        weights: YOLOv5 모델 가중치
        conf_thres: YOLO 신뢰도 임계값
        iou_thres: NMS IOU 임계값
        max_age: DeepSORT 최대 추적 연령
        n_init: DeepSORT 초기화 프레임 수
        max_clip_distance: CLIP 매칭 최대 거리
        save_dir: 결과 저장 디렉토리
        save_video: 영상 저장 여부
    """
    
    # 디바이스 설정
    device = select_device('')
    
    # YOLOv5 모델 로드
    print("YOLOv5 모델 로딩 중...")
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    
    # CLIP 모델 로드
    print("CLIP 모델 로딩 중...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # DeepSORT 트래커 초기화
    print("DeepSORT 트래커 초기화 중...")
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2)
    tracker_obj = tracker.Tracker(
        metric, 
        max_iou_distance=0.7, 
        max_age=max_age, 
        n_init=n_init,
        max_clip_distance=max_clip_distance
    )
    
    # 데이터셋 로드
    dataset = LoadImages(source, img_size=640, stride=32, auto=True)
    
    # 결과 저장 디렉토리
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 비디오 작성기 초기화
    video_writer = None
    frame_count = 0
    
    print("\n트래킹 시작...")
    print("-" * 50)
    
    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        
        # 이미지 전처리
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        # YOLOv5 추론
        pred = model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, 
            classes=[PERSON_CLASS],  # 사람만 탐지
            max_det=1000
        )
        
        # DeepSORT Detection 객체 생성
        detections = []
        im0 = im0s.copy()
        
        # 결과 처리
        if len(pred[0]):
            det = pred[0]
            
            # Rescale boxes
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 작은 객체 필터링 (이미지 면적의 0.5% 미만)
            img_area = im0.shape[0] * im0.shape[1]
            bbox_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            area_ratio = bbox_area / img_area
            min_area_ratio = 0.005
            valid_objects = area_ratio >= min_area_ratio
            det = det[valid_objects]
            
            # 각 탐지에 대해 CLIP 임베딩 추출 및 DeepSORT Detection 생성
            for *xyxy, conf, cls in det:
                bbox = [int(x) for x in xyxy]
                
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
                    feature = clip_embedding
                    
                    # Detection 객체 생성
                    detection = Detection(tlwh, float(conf), feature, clip_embedding)
                    detections.append(detection)
        
        # DeepSORT 업데이트
        tracker_obj.predict()
        tracker_obj.update(detections)
        
        # 결과 시각화
        im0_with_tracks = im0.copy()
        
        for track in tracker_obj.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # 바운딩 박스 좌표
            tlwh = track.to_tlwh()
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(x1 + tlwh[2]), int(y1 + tlwh[3])
            
            # 트랙 ID
            track_id = track.track_id
            
            # 색상 생성 (track_id 기반)
            color = (
                int((track_id * 3) % 255),
                int((track_id * 7) % 255),
                int((track_id * 11) % 255)
            )
            
            # 바운딩 박스 그리기
            cv2.rectangle(im0_with_tracks, (x1, y1), (x2, y2), color, 2)
            
            # 트랙 ID 라벨
            label = f"ID:{track_id}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                im0_with_tracks,
                (x1, y1 - label_size[1] - baseline - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                im0_with_tracks,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # 프레임 정보 출력
        print(f"프레임 {frame_count}: 탐지 {len(detections)}개 → 트랙 {len([t for t in tracker_obj.tracks if t.is_confirmed()])}개")
        
        # 비디오 저장
        if save_video:
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
                print(f"비디오 저장 시작: {video_path}")
            
            video_writer.write(im0_with_tracks)
    
    # 정리
    if video_writer is not None:
        video_writer.release()
    
    print("-" * 50)
    print(f"\n✅ 트래킹 완료!")
    print(f"총 프레임: {frame_count}")
    print(f"결과 저장 위치: {save_path}")
    
    # 통계 출력
    total_tracks = len(set(t.track_id for t in tracker_obj.tracks if t.is_confirmed()))
    print(f"총 추적된 객체 수: {total_tracks}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 + DeepSORT + CLIP 통합 트래킹')
    parser.add_argument('--source', type=str, required=True, help='입력 영상 경로')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='YOLOv5 모델 가중치')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IOU 임계값')
    parser.add_argument('--max-clip-distance', type=float, default=0.5, help='CLIP 매칭 최대 거리')
    parser.add_argument('--save-dir', type=str, default='results/integrated_tracking', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    run_integrated_tracking(
        source=args.source,
        weights=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_clip_distance=args.max_clip_distance,
        save_dir=args.save_dir,
        save_video=True
    )
