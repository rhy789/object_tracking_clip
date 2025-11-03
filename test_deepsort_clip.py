#!/usr/bin/env python3
"""
DeepSORT + CLIP 트래킹 테스트
저장된 CLIP 임베딩을 로드하여 트래킹 수행
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys
import os

# 경로 설정
sys.path.append('/workspace/deep_sort')
sys.path.insert(0, '/workspace/deep_sort')

from deep_sort import tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection


def load_embeddings(file_path):
    """저장된 임베딩 파일 로드"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 리스트를 numpy 배열로 변환
    for frame_data in data['frames']:
        for detection in frame_data['detections']:
            if detection['clip_embedding'] is not None:
                detection['clip_embedding'] = np.array(detection['clip_embedding'])
    
    return data


def create_detection_from_data(detection_data):
    """JSON 데이터로부터 Detection 객체 생성"""
    bbox = detection_data['bbox']
    confidence = detection_data['confidence']
    
    # TLWH 형식으로 변환 (x1, y1, x2, y2) -> (x, y, w, h)
    tlwh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    # CLIP embedding
    clip_embedding = detection_data.get('clip_embedding')
    
    # CLIP 임베딩을 feature로 사용 (512차원)
    # DeepSORT는 feature를 사용하여 appearance matching을 수행하므로
    # CLIP 임베딩을 feature로 직접 사용
    if clip_embedding is not None:
        feature = np.array(clip_embedding)
        # dtype를 float32로 변환 (DeepSORT 요구사항)
        feature = feature.astype(np.float32)
    else:
        # CLIP 임베딩이 없는 경우 fallback
        feature = np.random.randn(512).astype(np.float32)
        feature = feature / (np.linalg.norm(feature) + 1e-8)  # 정규화
        clip_embedding = None
    
    return Detection(tlwh, confidence, feature, clip_embedding)


def run_tracking():
    """메인 트래킹 함수"""
    print("=== DeepSORT + CLIP 트래킹 테스트 ===")
    
    # 경로 설정
    embeddings_file = "/workspace/results/clip_embedding_0.4_iou/embeddings.json"
    video_file = "/workspace/results/clip_embedding_0.4_iou/people.mp4"
    output_dir = Path("/workspace/results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"임베딩 파일: {embeddings_file}")
    print(f"입력 영상: {video_file}")
    
    # 임베딩 데이터 로드
    print("\n임베딩 데이터 로딩 중...")
    embeddings_data = load_embeddings(embeddings_file)
    print(f"총 프레임 수: {len(embeddings_data['frames'])}")
    
    # 영상 열기
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 영상 설정
    output_file = str(output_dir / "deepsort_clip_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # DeepSORT 설정
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2)
    my_tracker = tracker.Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3, max_clip_distance=0.5)
    
    frame_idx = 0
    print("\n트래킹 시작...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 해당 프레임의 탐지 데이터 가져오기
        if frame_idx < len(embeddings_data['frames']):
            frame_data = embeddings_data['frames'][frame_idx]
            detections = []
            
            for detection_data in frame_data['detections']:
                detection = create_detection_from_data(detection_data)
                detections.append(detection)
            
            # DeepSORT 업데이트
            my_tracker.predict()
            my_tracker.update(detections)
            
            # 디버깅: CLIP 임베딩 사용 확인
            if frame_idx == 0:
                print(f"\n첫 프레임 디버깅:")
                print(f"- 탐지 수: {len(detections)}")
                for i, det in enumerate(detections):
                    has_clip = det.clip_embedding is not None
                    print(f"  탐지 {i}: feature shape={det.feature.shape}, CLIP 임베딩={'있음' if has_clip else '없음'}")
            
            # 결과 시각화
            for track in my_tracker.tracks:
                if track.is_confirmed():
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track.track_id}", 
                              (int(bbox[0]), int(bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 프레임 정보 표시
            cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len([t for t in my_tracker.tracks if t.is_confirmed()])}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 영상 출력 (GUI 환경이 아니면 주석 처리)
            # cv2.imshow('DeepSORT + CLIP Tracking', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # 결과 저장
            out.write(frame)
            
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                print(f"프레임 {frame_idx} 처리 완료...")
    
    # 정리
    cap.release()
    out.release()
    # cv2.destroyAllWindows()  # GUI 환경이 아니면 주석 처리
    
    print(f"\n✅ 트래킹 완료!")
    print(f"결과 영상: {output_file}")
    
    # 최종 통계
    total_tracks = len(set([t.track_id for t in my_tracker.tracks if t.is_confirmed()]))
    print(f"총 트랙 수: {total_tracks}")


if __name__ == "__main__":
    run_tracking()
