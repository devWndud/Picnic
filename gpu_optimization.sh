#!/bin/bash

# RTX 5000 Ada Generation GPU 최적화 스크립트

echo "🚀 RTX 5000 Ada Generation GPU 최적화 설정 시작..."

# 1. NVIDIA 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/cuda_cache
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB 캐시

# 2. GPU 메모리 최적화 환경 변수 (안전한 버전)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 4. cuDNN 최적화
export CUDNN_BENCHMARK=1
export CUDNN_DETERMINISTIC=0

# 5. GPU 메모리 풀링 설정
export CUDA_MEMORY_POOL_SIZE=26843545600  # 25GB (32GB 중 25GB 사용)

# 6. 시스템 메모리 설정
echo 1 > /proc/sys/vm/drop_caches
echo 3 > /proc/sys/vm/drop_caches

# 7. GPU 전력 관리 설정 (최대 성능)
sudo nvidia-smi -pm 1  # 지속적 성능 모드
sudo nvidia-smi -ac 1215,1410  # 메모리 및 그래픽 클럭 최대

# 8. GPU 메모리 정보 출력
echo "📊 GPU 정보:"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

# 9. 현재 설정된 환경 변수 출력
echo "🔧 설정된 환경 변수:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "CUDNN_BENCHMARK: $CUDNN_BENCHMARK"

echo "✅ GPU 최적화 설정 완료!"
echo "이제 Python 스크립트를 실행하세요:"
echo "source gpu_optimization.sh && python flask_server.py"
