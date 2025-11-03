#!/bin/bash

# CLIP + DeepSORT + YOLOv5 통합 추적 시스템 도커 실행 스크립트

echo "🚀 CLIP + DeepSORT + YOLOv5 통합 추적 시스템 시작"
echo "================================================"

# X11 포워딩 설정 (GUI 표시용)
echo "🔧 X11 포워딩 설정 중..."
xhost +local:root

# 도커 컨테이너 실행
echo "🏃 도커 컨테이너 실행 중..."
docker-compose up

echo "✅ 도커 컨테이너 실행 완료!"
echo ""
echo "📋 사용법:"
echo "  컨테이너 내부에서:"
echo "    python3 main.py --input data/people.mp4 --output results/tracked_video.mp4"
echo ""
echo "  종료하려면: Ctrl+C"

