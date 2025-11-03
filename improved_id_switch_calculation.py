#!/usr/bin/env python3
"""
개선된 ID 스위치 계산 방법
"""

import numpy as np
from collections import defaultdict

class ImprovedPerformanceMetrics:
    """개선된 성능 지표 측정 클래스"""
    
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
        
        # ID 스위치 계산을 위한 추가 변수
        self.track_history = []  # 각 프레임의 트랙 ID 리스트
        self.track_lifetimes = defaultdict(int)  # 각 트랙의 생존 시간
        self.track_positions = {}  # 각 트랙의 마지막 위치
        
    def add_frame(self, frame_time, detection_count, track_count, current_track_ids):
        """프레임별 지표 추가"""
        self.frame_times.append(frame_time)
        self.detection_counts.append(detection_count)
        self.track_counts.append(track_count)
        
        # 트랙 히스토리 저장
        self.track_history.append(set(current_track_ids))
        
        # ID 스위치 계산 (개선된 버전)
        if hasattr(self, 'prev_track_ids') and len(self.prev_track_ids) > 0:
            self._calculate_id_switches(current_track_ids)
        
        self.prev_track_ids = current_track_ids.copy()
        self.track_ids.update(current_track_ids)
        self.total_tracks = len(self.track_ids)
    
    def _calculate_id_switches(self, current_track_ids):
        """ID 스위치 계산 (개선된 버전)"""
        if not hasattr(self, 'prev_track_ids'):
            return
        
        # 1. 연속된 트랙 찾기 (같은 객체가 다른 ID로 매칭된 경우)
        for prev_id in self.prev_track_ids:
            if prev_id in current_track_ids:
                # 트랙이 유지됨 - ID 스위치 아님
                continue
            else:
                # 트랙이 사라짐 - 잠재적 ID 스위치 후보
                pass
        
        # 2. 새 트랙 중에서 기존 트랙과 매칭 가능한 것 찾기
        for current_id in current_track_ids:
            if current_id not in self.prev_track_ids:
                # 새 트랙 - 잠재적 ID 스위치 후보
                pass
        
        # 간단한 버전: 새 트랙 + 사라진 트랙의 일부를 ID 스위치로 계산
        new_tracks = current_track_ids - self.prev_track_ids
        lost_tracks = self.prev_track_ids - current_track_ids
        
        # ID 스위치 = min(새 트랙 수, 사라진 트랙 수)
        # (모든 새 트랙이 ID 스위치는 아니지만, 일부는 그럴 가능성이 높음)
        potential_switches = min(len(new_tracks), len(lost_tracks))
        self.id_switches += potential_switches
    
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


def demonstrate_id_switch_calculation():
    """ID 스위치 계산 예시"""
    print("🔍 ID 스위치 계산 예시")
    print("="*50)
    
    # 시뮬레이션 데이터
    track_sequences = [
        [1, 2, 3],      # 프레임 1
        [1, 2, 4],      # 프레임 2 (트랙 3 → 4로 변경)
        [1, 2, 4],      # 프레임 3 (안정)
        [1, 5, 4],      # 프레임 4 (트랙 2 → 5로 변경)
        [1, 5, 6],      # 프레임 5 (트랙 4 → 6으로 변경)
    ]
    
    print("프레임별 트랙 ID 변화:")
    for i, tracks in enumerate(track_sequences, 1):
        print(f"프레임 {i}: {tracks}")
    
    print("\n현재 방식 (간단한 버전):")
    metrics = ImprovedPerformanceMetrics()
    
    for i, tracks in enumerate(track_sequences):
        current_track_ids = set(tracks)
        metrics.add_frame(0.1, len(tracks), len(tracks), current_track_ids)
        print(f"프레임 {i+1} 후 ID 스위치: {metrics.id_switches}")
    
    print(f"\n총 ID 스위치: {metrics.id_switches}")
    
    print("\n실제 ID 스위치 분석:")
    print("프레임 1→2: 트랙 3이 사라지고 트랙 4가 생성됨")
    print("  - 가능성 1: 같은 객체, ID 스위치 (3→4)")
    print("  - 가능성 2: 다른 객체, 정상적인 변화")
    print("프레임 3→4: 트랙 2가 사라지고 트랙 5가 생성됨")
    print("  - 가능성 1: 같은 객체, ID 스위치 (2→5)")
    print("  - 가능성 2: 다른 객체, 정상적인 변화")
    print("프레임 4→5: 트랙 4가 사라지고 트랙 6이 생성됨")
    print("  - 가능성 1: 같은 객체, ID 스위치 (4→6)")
    print("  - 가능성 2: 다른 객체, 정상적인 변화")


if __name__ == '__main__':
    demonstrate_id_switch_calculation()

