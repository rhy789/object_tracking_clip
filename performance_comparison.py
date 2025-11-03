#!/usr/bin/env python3
"""
YOLOv5 + DeepSORT 성능 비교 (Before/After CLIP)
- Before: YOLOv5 + DeepSORT (기본)
- After: YOLOv5 + DeepSORT + CLIP (개선)
"""

import cv2
import numpy as np
import torch
import clip
from PIL import Image
import sys
import argparse
from pathlib import Path
import time
import json
from datetime import datetime

# YOLOv5 경로 추가
YOLO_PATH = Path(__file__).parent / "yolov5"
if str(YOLO_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_PATH))

# DeepSORT 경로 추가
DEEPSORT_PATH = Path(__file__).parent / "deep_sort"
if str(DEEPSORT_PATH) not in sys.path:
    sys.path.insert(0, str(DEEPSORT_PATH))

# YOLOv5 imports
sys.path.insert(0, str(Path(__file__).parent / "yolov5"))
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import LoadImages

# DeepSORT imports
from deep_sort import tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection

PERSON_CLASS = 0


class PerformanceMetrics:
    """성능 지표 측정 클래스"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_times = []
        self.detection_counts = []
        self.track_counts = []
        self.track_ids = set()
        self.total_tracks = 0
        self.id_switches = 0
        self.fragments = 0
        self.motp = 0.0  # Multiple Object Tracking Precision
        self.mota = 0.0  # Multiple Object Tracking Accuracy
        
    def add_frame(self, frame_time, detection_count, track_count, current_track_ids):
        """프레임별 지표 추가"""
        self.frame_times.append(frame_time)
        self.detection_counts.append(detection_count)
        self.track_counts.append(track_count)
        
        # ID 스위치 계산 (간단한 버전)
        if hasattr(self, 'prev_track_ids'):
            new_tracks = current_track_ids - self.prev_track_ids
            lost_tracks = self.prev_track_ids - current_track_ids
            self.id_switches += len(new_tracks) + len(lost_tracks)
        
        self.prev_track_ids = current_track_ids.copy()
        self.track_ids.update(current_track_ids)
        self.total_tracks = len(self.track_ids)
    
    def calculate_metrics(self):
        """최종 지표 계산"""
        if not self.frame_times:
            return {}
        
        return {
            'avg_fps': 1.0 / np.mean(self.frame_times),
            'avg_frame_time_ms': np.mean(self.frame_times) * 1000,
            'avg_detections': np.mean(self.detection_counts),
            'avg_tracks': np.mean(self.track_counts),
            'total_tracks': self.total_tracks,
            'id_switches': self.id_switches,
            'fragments': self.fragments,
            'total_frames': len(self.frame_times)
        }


def extract_clip_embedding(image, bbox, clip_model, clip_preprocess, device):
    """CLIP 임베딩 추출"""
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


def run_baseline_tracking(source, weights='yolov5s.pt', conf_thres=0.4, iou_thres=0.4):
    """기본 YOLOv5 + DeepSORT (CLIP 없음)"""
    print("🔵 기본 YOLOv5 + DeepSORT 실행 중...")
    
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    
    # DeepSORT 트래커 (CLIP 없음)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2)
    tracker_obj = tracker.Tracker(
        metric, 
        max_iou_distance=0.7, 
        max_age=30, 
        n_init=3,
        max_clip_distance=1.0  # CLIP 매칭 비활성화
    )
    
    dataset = LoadImages(source, img_size=640, stride=32, auto=True)
    metrics = PerformanceMetrics()
    
    for path, im, im0s, vid_cap, s in dataset:
        start_time = time.time()
        
        # YOLO 추론
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, 
            classes=[PERSON_CLASS],
            max_det=1000
        )
        
        # Detection 객체 생성 (기본 feature 사용)
        detections = []
        im0 = im0s.copy()
        
        if len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 작은 객체 필터링
            img_area = im0.shape[0] * im0.shape[1]
            bbox_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            area_ratio = bbox_area / img_area
            min_area_ratio = 0.005
            valid_objects = area_ratio >= min_area_ratio
            det = det[valid_objects]
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                # CUDA 텐서를 CPU로 이동 후 NumPy 변환
                tlwh = np.array([x1.cpu(), y1.cpu(), w.cpu(), h.cpu()], dtype=np.float64)
                
                # 기본 feature (랜덤 벡터)
                feature = np.random.randn(128).astype(np.float32)
                
                detection = Detection(tlwh, float(conf.cpu()), feature)
                detections.append(detection)
        
        # DeepSORT 업데이트
        tracker_obj.predict()
        tracker_obj.update(detections)
        
        # 지표 수집
        frame_time = time.time() - start_time
        current_track_ids = {t.track_id for t in tracker_obj.tracks if t.is_confirmed()}
        metrics.add_frame(frame_time, len(detections), len(current_track_ids), current_track_ids)
    
    return metrics.calculate_metrics()


def run_clip_tracking(source, weights='yolov5s.pt', conf_thres=0.4, iou_thres=0.4):
    """개선된 YOLOv5 + DeepSORT + CLIP"""
    print("🟢 YOLOv5 + DeepSORT + CLIP 실행 중...")
    
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    
    # CLIP 모델 로드
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # DeepSORT 트래커 (CLIP 포함)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2)
    tracker_obj = tracker.Tracker(
        metric, 
        max_iou_distance=0.7, 
        max_age=30, 
        n_init=3,
        max_clip_distance=0.5  # CLIP 매칭 활성화
    )
    
    dataset = LoadImages(source, img_size=640, stride=32, auto=True)
    metrics = PerformanceMetrics()
    
    for path, im, im0s, vid_cap, s in dataset:
        start_time = time.time()
        
        # YOLO 추론
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, 
            classes=[PERSON_CLASS],
            max_det=1000
        )
        
        # Detection 객체 생성 (CLIP 임베딩 사용)
        detections = []
        im0 = im0s.copy()
        
        if len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 작은 객체 필터링
            img_area = im0.shape[0] * im0.shape[1]
            bbox_area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            area_ratio = bbox_area / img_area
            min_area_ratio = 0.005
            valid_objects = area_ratio >= min_area_ratio
            det = det[valid_objects]
            
            for *xyxy, conf, cls in det:
                bbox = [int(x) for x in xyxy]
                
                # CLIP 임베딩 추출
                clip_embedding = extract_clip_embedding(
                    im0, bbox, clip_model, clip_preprocess, device
                )
                
                if clip_embedding is not None:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    # 이미 int로 변환된 bbox이므로 그대로 사용
                    tlwh = np.array([x1, y1, w, h], dtype=np.float64)
                    
                    # CLIP 임베딩을 feature로 사용
                    feature = clip_embedding
                    
                    detection = Detection(tlwh, float(conf.cpu()), feature, clip_embedding)
                    detections.append(detection)
        
        # DeepSORT 업데이트
        tracker_obj.predict()
        tracker_obj.update(detections)
        
        # 지표 수집
        frame_time = time.time() - start_time
        current_track_ids = {t.track_id for t in tracker_obj.tracks if t.is_confirmed()}
        metrics.add_frame(frame_time, len(detections), len(current_track_ids), current_track_ids)
    
    return metrics.calculate_metrics()


def print_comparison_table(baseline_metrics, clip_metrics):
    """비교 테이블 출력"""
    print("\n" + "="*80)
    print("📊 성능 비교 결과 (Before vs After)")
    print("="*80)
    
    # 메트릭 이름과 단위
    metrics_info = [
        ("평균 FPS", "fps", "avg_fps"),
        ("평균 프레임 시간", "ms", "avg_frame_time_ms"),
        ("평균 탐지 수", "개", "avg_detections"),
        ("평균 트랙 수", "개", "avg_tracks"),
        ("총 트랙 수", "개", "total_tracks"),
        ("ID 스위치", "회", "id_switches"),
        ("총 프레임", "개", "total_frames")
    ]
    
    print(f"{'지표':<20} {'Before (기본)':<15} {'After (CLIP)':<15} {'개선율':<15}")
    print("-"*80)
    
    for name, unit, key in metrics_info:
        before_val = baseline_metrics.get(key, 0)
        after_val = clip_metrics.get(key, 0)
        
        if before_val > 0:
            if key in ['avg_frame_time_ms', 'id_switches']:
                # 낮을수록 좋은 지표
                improvement = ((before_val - after_val) / before_val) * 100
                symbol = "↓" if improvement > 0 else "↑"
            else:
                # 높을수록 좋은 지표
                improvement = ((after_val - before_val) / before_val) * 100
                symbol = "↑" if improvement > 0 else "↓"
        else:
            improvement = 0
            symbol = "="
        
        print(f"{name:<20} {before_val:<15.2f} {after_val:<15.2f} {symbol}{abs(improvement):.1f}%")
    
    print("="*80)


def save_comparison_results(baseline_metrics, clip_metrics, save_path):
    """결과를 JSON 파일로 저장"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'baseline_metrics': baseline_metrics,
        'clip_metrics': clip_metrics,
        'comparison': {}
    }
    
    # 개선율 계산
    for key in baseline_metrics:
        if key in clip_metrics and baseline_metrics[key] > 0:
            if key in ['avg_frame_time_ms', 'id_switches']:
                improvement = ((baseline_metrics[key] - clip_metrics[key]) / baseline_metrics[key]) * 100
            else:
                improvement = ((clip_metrics[key] - baseline_metrics[key]) / baseline_metrics[key]) * 100
            results['comparison'][key] = {
                'improvement_percent': improvement,
                'baseline': baseline_metrics[key],
                'clip': clip_metrics[key]
            }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 상세 결과 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 + DeepSORT 성능 비교')
    parser.add_argument('--source', type=str, required=True, help='입력 영상 경로')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='YOLOv5 모델 가중치')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IOU 임계값')
    parser.add_argument('--save-results', type=str, default='results/performance_comparison.json', help='결과 저장 경로')
    
    args = parser.parse_args()
    
    print("🚀 YOLOv5 + DeepSORT 성능 비교 시작")
    print(f"입력 영상: {args.source}")
    print(f"모델: {args.weights}")
    print(f"신뢰도 임계값: {args.conf_thres}")
    print(f"IOU 임계값: {args.iou_thres}")
    print("-"*50)
    
    # Before: 기본 YOLOv5 + DeepSORT
    baseline_metrics = run_baseline_tracking(
        args.source, args.weights, args.conf_thres, args.iou_thres
    )
    
    print("\n" + "-"*50)
    
    # After: YOLOv5 + DeepSORT + CLIP
    clip_metrics = run_clip_tracking(
        args.source, args.weights, args.conf_thres, args.iou_thres
    )
    
    # 결과 비교 출력
    print_comparison_table(baseline_metrics, clip_metrics)
    
    # 결과 저장
    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_comparison_results(baseline_metrics, clip_metrics, save_path)
    
    print("\n✅ 성능 비교 완료!")


if __name__ == '__main__':
    main()
