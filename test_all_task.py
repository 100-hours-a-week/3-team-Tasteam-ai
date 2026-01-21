"""
RunPod Pod 서버 API 전체 기능 통합 테스트 스크립트 (다중 모델 지원)

이 스크립트는 RunPod Pod에서 실행 중인 FastAPI 서버를 대상으로 테스트를 수행합니다.
성능 측정, 정확도 측정, 여러 모델 비교를 지원합니다.

사용 방법:
    ========================================
    1. 모델
    ========================================
    
    # jinsoo1218/runpod_vllm:latest
    # runpod_env
    
    llm:
    Qwen/Qwen2.5-7B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    google/gemma-2-9b-it

    Embedding:

    jhgan/ko-sbert-multitask
    dragonkue/BGE-m3-ko
    upskyy/bge-m3-korean
    
    ========================================
    2. 성능 측정 모드 (벤치마크)
    ========================================
    
    python test_openai_all.py --benchmark
    python test_openai_all.py --benchmark --iterations 10
    
    ========================================
    4. 결과 저장
    ========================================
    
    초기 셋팅
    qwen, kakao-app, sbert, 
    
    1. llm 모델 비교
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen.json
    python test_openai_all.py --benchmark --iterations 3 --save-results llama.json
    python test_openai_all.py --benchmark --iterations 3 --save-results gemma.json
    
    3. llm 모델 고정, sentiment 모델 비교
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_Kakao-app.json
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_klue-roberta.json
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_kcelectra.json
    
    3. llm 모델 고정, sentiment 모델 고정, embedding 모델 비교
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_Kakao-app_sbert.json
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_Kakao-app_bge-m3-drag.json
    python test_openai_all.py --benchmark --iterations 3 --save-results qwen_Kakao-app_bge-m3-upsky.json
    
    ========================================
    주요 옵션
    ========================================
    
    --benchmark: 성능 측정 모드 활성화 (처리 시간, TTFT, TPS 등)
    --compare-models: 여러 모델 비교 모드
    --models: 비교할 모델명 리스트 (--compare-models와 함께 사용)
    --provider: LLM 제공자 (openai, local, runpod)
    --iterations: 성능 측정 반복 횟수 (기본값: 5)
    --save-results: 결과를 저장할 JSON 파일 경로
    --generate-report: 모델 비교 리포트 생성
    
    ========================================
    측정 지표
    ========================================
    
    성능 지표:
    - 처리 시간 (평균, P95, P99)
    - TTFT (Time To First Token)
    - TPS (Tokens Per Second)
    - 처리량 (req/s)
    
    정확도 지표:
    - BLEU Score (요약)
    - ROUGE Score (요약)
    - Precision@K (강점 추출)
    - MAE (감성 분석)
"""

import os
import sys
import json
import time
import requests
import subprocess
import tempfile
import argparse
import statistics
import sqlite3
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.metrics_collector import MetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    METRICS_COLLECTOR_AVAILABLE = False

try:
    from scripts.gpu_monitor import GPUMonitor
    GPU_MONITOR_AVAILABLE = True
except ImportError:
    GPU_MONITOR_AVAILABLE = False

# 색상 출력을 위한 ANSI 코드
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

try:
    from scripts.evaluate_sentiment_analysis import SentimentAnalysisEvaluator
    from scripts.evaluate_summary import SummaryEvaluator
    from scripts.evaluate_strength_extraction import StrengthExtractionEvaluator
    from scripts.evaluate_vector_search import PrecisionAtKEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print_warning("평가 스크립트를 import할 수 없습니다. 정확도 측정이 비활성화됩니다.")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

# jinsoo1218/runpod_vllm:latest
# runpod_env
# 테스트 설정
# RunPod Pod 서버 URL (환경 변수로 오버라이드 가능)
#BASE_URL = "http://213.192.2.74:40162"  # RunPod Pod IP:포트로 변경 (예: http://213.192.2.68:40183)
BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"
METRICS_DB_PATH = "metrics.db"

# 샘플 데이터 (데이터 생성 후 업데이트됨)
SAMPLE_RESTAURANT_ID = 1
SAMPLE_REVIEWS = []

# 테스트 메트릭 수집용 전역 딕셔너리 (JSON 저장용)
test_metrics: Dict[str, Any] = {}


def safe_json_response(response, error_msg="응답 처리 실패", allow_404=False):
    """안전하게 JSON 응답 파싱 (runpod_pod_all_test.py 참고)"""
    try:
        # 404는 비즈니스 로직상 정상 응답일 수 있음 (데이터 없음 등)
        if response.status_code == 404 and allow_404:
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    print(f"   ℹ️ 정보: {error_detail['detail']}")
                    return error_detail  # 404 응답도 반환
            except:
                pass
        
        response.raise_for_status()  # HTTP 오류 확인
        if not response.text:
            print(f"   ⚠️ 빈 응답 반환")
            return None
        return response.json()
    except requests.exceptions.HTTPError as e:
        # 404는 비즈니스 로직상 정상일 수 있으므로 별도 처리
        if response.status_code == 404:
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    print(f"   ℹ️ 정보: {error_detail['detail']}")
                    if allow_404:
                        return error_detail
                    else:
                        print(f"   ⚠️ 리소스를 찾을 수 없습니다 (정상일 수 있음)")
                        return None
            except:
                pass
        
        print(f"   ⚠️ HTTP 오류: {e}")
        print(f"   상태 코드: {response.status_code}")
        
        # 상세 오류 메시지 추출 시도
        try:
            error_detail = response.json()
            if "detail" in error_detail:
                print(f"   오류 상세: {error_detail['detail']}")
            else:
                print(f"   응답 내용: {json.dumps(error_detail, ensure_ascii=False, indent=2)[:500]}")
        except:
            print(f"   응답 내용 (텍스트): {response.text[:500]}")
        
        return None
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON 파싱 오류: {e}")
        print(f"   응답 내용: {response.text[:500]}")
        print(f"   상태 코드: {response.status_code}")
        return None
    except Exception as e:
        print(f"   ⚠️ {error_msg}: {e}")
        return None


def check_server_health():
    """서버 헬스 체크 (RunPod Pod 서버용)"""
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        elapsed_time = time.time() - start_time
        result = safe_json_response(response, "헬스 체크 실패")
        if result:
            print_success(f"서버 연결 성공: {result}")
            print_info(f"   ⏱️ 응답 시간: {elapsed_time:.2f}초")
            print_info(f"   서버 URL: {BASE_URL}")
            return True
        else:
            print_error("헬스 체크 실패: 응답을 파싱할 수 없습니다")
            return False
    except Exception as e:
        print_error(f"헬스 체크 실패: {e}")
        print_info(f"서버 URL: {BASE_URL}")
        print_info("서버가 실행 중인지 확인하세요 (RunPod Pod에서 FastAPI 서버 확인)")
        return False


def generate_test_data(
    generate_from_kr3: bool = False,
    kr3_sample: Optional[int] = None,
    kr3_restaurants: Optional[int] = None,
):
    """
    테스트 데이터 로드 또는 생성
    
    Args:
        generate_from_kr3: kr3.tsv에서 데이터 생성 여부
        kr3_sample: kr3.tsv에서 샘플링할 리뷰 수
        kr3_restaurants: 생성할 레스토랑 수
    """
    # kr3.tsv에서 데이터 생성 모드
    if generate_from_kr3:
        return generate_test_data_from_kr3(kr3_sample, kr3_restaurants)
    
    # 기본: test_data_sample.json 파일에서 테스트 데이터 로드
    print_header("테스트 데이터 로드")
    
    # test_data_sample.json 파일 경로
    test_data_path = project_root / "data" / "test_data_sample.json"
    
    if not test_data_path.exists():
        print_warning(f"테스트 데이터 파일이 없습니다: {test_data_path}")
        print_info("대체 방법: --generate-from-kr3 옵션으로 kr3.tsv에서 데이터를 생성할 수 있습니다.")
        return None
    
    try:
        # JSON 파일 읽기
        print_info(f"테스트 데이터 파일 로드 중: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        restaurants_count = len(data.get('restaurants', []))
        print_success(f"테스트 데이터 로드 완료: {restaurants_count}개 레스토랑")
        
        # 총 리뷰 수 계산
        total_reviews = sum(
            len(restaurant.get('reviews', []))
            for restaurant in data.get('restaurants', [])
        )
        print_info(f"  - 총 리뷰 수: {total_reviews}개")
        
        # 임시 파일 경로는 None 반환 (더 이상 필요 없음)
        return data, None
        
    except json.JSONDecodeError as e:
        print_error(f"JSON 파일 파싱 오류: {str(e)}")
        return None
    except Exception as e:
        print_error(f"테스트 데이터 로드 중 오류: {str(e)}")
        return None


def generate_test_data_from_kr3(
    sample: Optional[int] = None,
    restaurants: Optional[int] = None,
):
    """kr3.tsv 파일에서 테스트 데이터 생성"""
    print_header("kr3.tsv에서 테스트 데이터 생성")
    
    # kr3.tsv 파일 확인
    kr3_path = project_root / "data" / "kr3.tsv"
    if not kr3_path.exists():
        print_error(f"kr3.tsv 파일이 없습니다: {kr3_path}")
        return None
    
    # 임시 JSON 파일 생성
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    temp_json_path = temp_file.name
    
    try:
        # convert_kr3_tsv.py 실행
        print_info("kr3.tsv에서 테스트 데이터 생성 중...")
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "convert_kr3_tsv.py"),
            "--input", str(kr3_path),
            "--output", temp_json_path,
        ]
        
        # 샘플링 옵션 추가
        if sample:
            cmd.extend(["--sample", str(sample)])
        
        # 레스토랑 수 옵션 추가
        if restaurants:
            cmd.extend(["--restaurants", str(restaurants)])
        
        print_info(f"실행 명령: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print_error(f"데이터 생성 실패: {result.stderr}")
            if os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
            return None
        
        # 생성된 JSON 파일 읽기
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        restaurants_count = len(data.get('restaurants', []))
        print_success(f"테스트 데이터 생성 완료: {restaurants_count}개 레스토랑")
        
        # 총 리뷰 수 계산
        total_reviews = sum(
            len(restaurant.get('reviews', []))
            for restaurant in data.get('restaurants', [])
        )
        print_info(f"  - 총 리뷰 수: {total_reviews}개")
        
        return data, temp_json_path
        
    except subprocess.TimeoutExpired:
        print_error("데이터 생성 시간 초과 (300초)")
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return None
    except Exception as e:
        print_error(f"데이터 생성 중 오류: {str(e)}")
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return None


def upload_data_to_qdrant(data: Dict[str, Any]):
    """생성된 데이터를 Qdrant에 upload"""
    print_header("Qdrant에 데이터 Upload")
    
    if not data or "restaurants" not in data:
        print_warning("Upload할 데이터가 없습니다.")
        return False
    
    url = f"{BASE_URL}{API_PREFIX}/vector/upload"
    
    # 모든 리뷰와 레스토랑 정보를 수집
    all_reviews = []
    all_restaurants = []
    
    for restaurant_data in data["restaurants"]:
        # 레스토랑 정보 추가
        restaurant_id = restaurant_data.get("restaurant_id")
        restaurant_info = {
            "id": int(restaurant_id) if isinstance(restaurant_id, (int, str)) and str(restaurant_id).isdigit() else restaurant_id,
            "name": restaurant_data.get("restaurant_name", f"Test Restaurant {restaurant_id}"),
            "full_address": None,
            "location": None,
            "created_at": None,
            "deleted_at": None
        }
        all_restaurants.append(restaurant_info)
        
        # 리뷰 정보 추가 (restaurant_id를 int로 변환)
        reviews = restaurant_data.get("reviews", [])
        for review in reviews:
            # restaurant_id를 int로 변환 (ReviewModel이 int를 기대)
            review_copy = review.copy()
            if "restaurant_id" in review_copy:
                review_copy["restaurant_id"] = int(review_copy["restaurant_id"]) if isinstance(review_copy["restaurant_id"], str) and str(review_copy["restaurant_id"]).isdigit() else review_copy["restaurant_id"]
            all_reviews.append(review_copy)
    
    try:
        payload = {
            "reviews": all_reviews,
            "restaurants": all_restaurants
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)  # 대용량 데이터를 위해 타임아웃 증가
        elapsed_time = time.time() - start_time
        result = safe_json_response(response, "업로드 실패")
        
        if result:
            points_count = result.get("points_count", 0)
            print_success(f"총 {points_count}개 포인트가 Qdrant에 upload되었습니다.")
            print_info(f"  - 리뷰: {len(all_reviews)}개")
            print_info(f"  - 레스토랑: {len(all_restaurants)}개")
            print_info(f"  ⏱️ 응답 시간: {elapsed_time:.2f}초")
            return True
        else:
            print_warning("Upload 실패")
            print_info("💡 해결 방법:")
            print_info("   1. RUNPOD 환경 변수에 QDRANT_URL=:memory: 설정 (인메모리 사용)")
            print_info("   2. 또는 외부 Qdrant 서버 URL 설정")
            print_info("   3. 서버 로그 확인: docker logs 또는 RUNPOD 로그 뷰어")
            return False
            
    except Exception as e:
        print_warning(f"Upload 중 오류: {str(e)}")
        return False


def calculate_percentile(data: List[float], percentile: float) -> float:
    """퍼센타일 계산"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def measure_performance(
    endpoint: str,
    payload: Dict[str, Any],
    num_iterations: int = 5,
    warmup_iterations: int = 1,
    timeout: int = 60
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    성능 측정 (여러 번 반복 실행하여 통계 수집)
    
    Returns:
        (성공 여부, 성능 통계 딕셔너리)
    """
    # endpoint가 이미 전체 URL인지 확인 (http:// 또는 https://로 시작)
    if endpoint.startswith(("http://", "https://")):
        url = endpoint
    else:
        url = f"{BASE_URL}{endpoint}"
    latencies = []
    success_count = 0
    error_count = 0
    error_4xx_count = 0
    error_5xx_count = 0
    status_codes = []
    
    # GPU 모니터 초기화 (가능한 경우)
    gpu_monitor = None
    gpu_metrics_before = None
    gpu_metrics_after = None
    if GPU_MONITOR_AVAILABLE:
        try:
            gpu_monitor = GPUMonitor(device_index=0)
            gpu_metrics_before = gpu_monitor.get_metrics()
        except Exception:
            pass
    
    # CPU/메모리 메트릭 수집 시작
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory()
    
    # 워밍업
    for i in range(warmup_iterations):
        try:
            requests.post(url, json=payload, timeout=timeout)
        except Exception:
            pass
    
    # 실제 측정
    measurement_start_time = time.perf_counter()
    last_successful_response = None  # 정확도 평가를 위해 마지막 성공 응답 저장
    for i in range(num_iterations):
        try:
            start_time = time.perf_counter()
            response = requests.post(url, json=payload, timeout=timeout)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            status_codes.append(response.status_code)
            
            if response.status_code == 200:
                latencies.append(latency)
                success_count += 1
                try:
                    last_successful_response = response.json()  # 마지막 성공 응답 저장
                except:
                    pass
            elif 400 <= response.status_code < 500:
                error_4xx_count += 1
                error_count += 1
                # 첫 번째 요청 실패 시 상세 출력
                if i == 0:
                    try:
                        error_detail = response.json()
                        detail_msg = error_detail.get('detail', response.text[:200])
                        print_warning(f"요청 {i+1}/{num_iterations} 실패 (4xx): {detail_msg}")
                    except:
                        print_warning(f"요청 {i+1}/{num_iterations} 실패 (4xx): {response.status_code} - {response.text[:200]}")
            elif 500 <= response.status_code < 600:
                error_5xx_count += 1
                error_count += 1
                # 첫 번째 요청 실패 시 상세 출력
                if i == 0:
                    try:
                        error_detail = response.json()
                        detail_msg = error_detail.get('detail', response.text[:200])
                        print_warning(f"요청 {i+1}/{num_iterations} 실패 (5xx): {detail_msg}")
                    except:
                        print_warning(f"요청 {i+1}/{num_iterations} 실패 (5xx): {response.status_code} - {response.text[:200]}")
            else:
                error_count += 1
        except requests.exceptions.Timeout:
            error_count += 1
            if i == 0:
                print_error(f"요청 {i+1}/{num_iterations} 타임아웃 (timeout={timeout}초)")
        except requests.exceptions.ConnectionError as e:
            error_count += 1
            if i == 0:
                print_error(f"요청 {i+1}/{num_iterations} 연결 실패: {str(e)}")
        except Exception as e:
            error_count += 1
            if i == 0:  # 첫 번째 요청만 상세 출력
                print_error(f"요청 {i+1}/{num_iterations} 예외 발생: {type(e).__name__}: {str(e)}")
    measurement_end_time = time.perf_counter()
    
    # CPU/메모리 메트릭 수집 종료
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory()
    
    # GPU 메트릭 수집 종료
    if gpu_monitor:
        try:
            gpu_metrics_after = gpu_monitor.get_metrics()
        except Exception:
            pass
    
    if not latencies:
        # 실패 원인 상세 출력
        print_error(f"성능 측정 실패: 성공한 요청이 없습니다.")
        if status_codes:
            print_info(f"  상태 코드 분포: {status_codes}")
            print_info(f"  4xx 오류: {error_4xx_count}개, 5xx 오류: {error_5xx_count}개")
        else:
            print_info(f"  모든 요청이 예외로 실패했습니다. (총 {error_count}개)")
        return False, None
    
    # 통계 계산
    total_time = measurement_end_time - measurement_start_time
    throughput_req_per_sec = len(latencies) / total_time if total_time > 0 else 0
    
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = calculate_percentile(latencies, 95)
    p99_latency = calculate_percentile(latencies, 99)
    
    stats = {
        "avg_latency_sec": avg_latency,
        "min_latency_sec": min_latency,
        "max_latency_sec": max_latency,
        "p95_latency_sec": p95_latency,
        "p99_latency_sec": p99_latency,
        "success_count": success_count,
        "error_count": error_count,
        "error_4xx_count": error_4xx_count,
        "error_5xx_count": error_5xx_count,
        "total_iterations": num_iterations,
        "success_rate": (success_count / num_iterations) * 100 if num_iterations > 0 else 0,
        "throughput_req_per_sec": throughput_req_per_sec,
        "total_measurement_time_sec": total_time,
        "last_successful_response": last_successful_response,  # 정확도 평가용
    }
    
    # CPU/메모리 메트릭 추가
    if cpu_after is not None:
        stats["cpu_usage_percent"] = cpu_after
    if mem_after is not None:
        stats["memory_usage_percent"] = mem_after.percent
        stats["memory_used_mb"] = mem_after.used / (1024 ** 2)
        stats["memory_total_mb"] = mem_after.total / (1024 ** 2)
    
    # GPU 메트릭 추가
    if gpu_metrics_after:
        stats["gpu_utilization_percent"] = gpu_metrics_after.get("gpu_util_percent", 0)
        stats["gpu_memory_usage_percent"] = gpu_metrics_after.get("memory_util_percent", 0)
        stats["gpu_memory_used_mb"] = gpu_metrics_after.get("memory_used_mb", 0)
        stats["gpu_memory_total_mb"] = gpu_metrics_after.get("memory_total_mb", 0)
    
    return True, stats


def load_test(
    endpoint: str,
    payload: Dict[str, Any],
    total_requests: int = 100,
    concurrent_users: int = 10,
    timeout: int = 60,
    ramp_up_seconds: int = 0
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    부하테스트 (동시 요청 처리 능력 측정)
    
    Args:
        endpoint: API 엔드포인트 경로
        payload: 요청 페이로드
        total_requests: 총 요청 수
        concurrent_users: 동시 사용자 수 (동시 실행할 요청 수)
        timeout: 요청 타임아웃 (초)
        ramp_up_seconds: 점진적 부하 증가 시간 (초, 0이면 즉시 시작)
    
    Returns:
        (성공 여부, 부하테스트 통계 딕셔너리)
    """
    # endpoint가 이미 전체 URL인지 확인
    if endpoint.startswith(("http://", "https://")):
        url = endpoint
    else:
        url = f"{BASE_URL}{endpoint}"
    
    latencies = []
    success_count = 0
    error_count = 0
    error_4xx_count = 0
    error_5xx_count = 0
    status_codes = []
    request_timestamps = []
    
    def make_request(request_id: int) -> Tuple[int, float, int, Optional[Dict[str, Any]]]:
        """단일 요청 실행"""
        try:
            start_time = time.perf_counter()
            response = requests.post(url, json=payload, timeout=timeout)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            status_code = response.status_code
            
            result = None
            if response.status_code == 200:
                try:
                    result = response.json()
                except:
                    pass
            
            return request_id, latency, status_code, result
        except Exception as e:
            # 에러는 나중에 집계
            return request_id, -1, 0, None
    
    # GPU 모니터 초기화
    gpu_monitor = None
    gpu_metrics_before = None
    gpu_metrics_after = None
    if GPU_MONITOR_AVAILABLE:
        try:
            gpu_monitor = GPUMonitor(device_index=0)
            gpu_metrics_before = gpu_monitor.get_metrics()
        except Exception:
            pass
    
    # CPU/메모리 메트릭 수집 시작
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory()
    
    print_info(f"부하테스트 시작: 총 {total_requests}개 요청, 동시 사용자 {concurrent_users}명")
    if ramp_up_seconds > 0:
        print_info(f"점진적 부하 증가: {ramp_up_seconds}초 동안 부하 증가")
    
    # 부하테스트 실행
    test_start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # 요청 제출
        futures = []
        for i in range(total_requests):
            # 점진적 부하 증가 (ramp-up)
            if ramp_up_seconds > 0:
                delay = (ramp_up_seconds / total_requests) * i
                if delay > 0:
                    time.sleep(delay)
            
            future = executor.submit(make_request, i)
            futures.append(future)
        
        # 결과 수집
        for future in as_completed(futures):
            try:
                request_id, latency, status_code, result = future.result()
                request_timestamps.append(time.perf_counter())
                
                if latency >= 0:
                    latencies.append(latency)
                    status_codes.append(status_code)
                    
                    if status_code == 200:
                        success_count += 1
                    elif 400 <= status_code < 500:
                        error_4xx_count += 1
                        error_count += 1
                    elif 500 <= status_code < 600:
                        error_5xx_count += 1
                        error_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
    
    test_end_time = time.perf_counter()
    
    # CPU/메모리 메트릭 수집 종료
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory()
    
    # GPU 메트릭 수집 종료
    if gpu_monitor:
        try:
            gpu_metrics_after = gpu_monitor.get_metrics()
        except Exception:
            pass
    
    if not latencies:
        print_error(f"부하테스트 실패: 성공한 요청이 없습니다.")
        if status_codes:
            print_info(f"  상태 코드 분포: {status_codes}")
            print_info(f"  4xx 오류: {error_4xx_count}개, 5xx 오류: {error_5xx_count}개")
        return False, None
    
    # 통계 계산
    total_time = test_end_time - test_start_time
    throughput_req_per_sec = len(latencies) / total_time if total_time > 0 else 0
    
    # 요청 간격 계산 (RPS 측정용)
    if len(request_timestamps) > 1:
        intervals = [request_timestamps[i] - request_timestamps[i-1] for i in range(1, len(request_timestamps))]
        avg_interval = statistics.mean(intervals) if intervals else 0
        actual_rps = 1.0 / avg_interval if avg_interval > 0 else 0
    else:
        actual_rps = throughput_req_per_sec
    
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = calculate_percentile(latencies, 50)
    p95_latency = calculate_percentile(latencies, 95)
    p99_latency = calculate_percentile(latencies, 99)
    
    # 동시 처리 능력 계산
    if len(request_timestamps) > 1:
        # 시간 윈도우에서 최대 동시 요청 수 추정
        time_window = 1.0  # 1초 윈도우
        max_concurrent = 0
        for ts in request_timestamps:
            window_end = ts + time_window
            concurrent_count = sum(1 for t in request_timestamps if ts <= t < window_end)
            max_concurrent = max(max_concurrent, concurrent_count)
    else:
        max_concurrent = 1
    
    stats = {
        "total_requests": total_requests,
        "concurrent_users": concurrent_users,
        "success_count": success_count,
        "error_count": error_count,
        "error_4xx_count": error_4xx_count,
        "error_5xx_count": error_5xx_count,
        "success_rate": (success_count / total_requests) * 100 if total_requests > 0 else 0,
        "avg_latency_sec": avg_latency,
        "min_latency_sec": min_latency,
        "max_latency_sec": max_latency,
        "p50_latency_sec": p50_latency,
        "p95_latency_sec": p95_latency,
        "p99_latency_sec": p99_latency,
        "throughput_req_per_sec": throughput_req_per_sec,
        "actual_rps": actual_rps,
        "max_concurrent_requests": max_concurrent,
        "total_test_time_sec": total_time,
        "ramp_up_seconds": ramp_up_seconds,
    }
    
    # CPU/메모리 메트릭 추가
    if cpu_after is not None:
        stats["cpu_usage_percent"] = cpu_after
    if mem_after is not None:
        stats["memory_usage_percent"] = mem_after.percent
        stats["memory_used_mb"] = mem_after.used / (1024 ** 2)
        stats["memory_total_mb"] = mem_after.total / (1024 ** 2)
    
    # GPU 메트릭 추가
    if gpu_metrics_after:
        stats["gpu_utilization_percent"] = gpu_metrics_after.get("gpu_util_percent", 0)
        stats["gpu_memory_usage_percent"] = gpu_metrics_after.get("memory_util_percent", 0)
        stats["gpu_memory_used_mb"] = gpu_metrics_after.get("memory_used_mb", 0)
        stats["gpu_memory_total_mb"] = gpu_metrics_after.get("memory_total_mb", 0)
    
    return True, stats


def query_metrics_from_db(analysis_type: str, limit: int = 10) -> Optional[Dict[str, Any]]:
    """SQLite에서 최근 메트릭 조회 (확장된 지표 포함)"""
    if not Path(METRICS_DB_PATH).exists():
        return None
    
    try:
        conn = sqlite3.connect(METRICS_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # analysis_metrics 조회 (최소/최대 포함)
        cursor.execute("""
            SELECT 
                AVG(processing_time_ms) as avg_processing_time_ms,
                MIN(processing_time_ms) as min_processing_time_ms,
                MAX(processing_time_ms) as max_processing_time_ms,
                AVG(tokens_used) as avg_tokens_used,
                MIN(tokens_used) as min_tokens_used,
                MAX(tokens_used) as max_tokens_used,
                COUNT(*) as total_requests,
                SUM(error_count) as total_errors,
                (SUM(error_count) * 100.0 / COUNT(*)) as error_rate_percent,
                (COUNT(*) - SUM(error_count)) * 100.0 / COUNT(*) as success_rate_percent
            FROM analysis_metrics
            WHERE analysis_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (analysis_type, limit))
        
        analysis_result = cursor.fetchone()
        
        # vllm_metrics 조회 (TTFT P95/P99 포함)
        cursor.execute("""
            SELECT 
                AVG(ttft_ms) as avg_ttft_ms,
                MIN(ttft_ms) as min_ttft_ms,
                MAX(ttft_ms) as max_ttft_ms,
                AVG(tps) as avg_tps,
                AVG(tpot_ms) as avg_tpot_ms,
                COUNT(*) as total_vllm_requests
            FROM vllm_metrics
            WHERE analysis_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (analysis_type, limit))
        
        vllm_result = cursor.fetchone()
        
        # TTFT P95/P99 계산
        cursor.execute("""
            SELECT ttft_ms
            FROM vllm_metrics
            WHERE analysis_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (analysis_type, limit))
        
        ttft_values = [row[0] for row in cursor.fetchall() if row[0] is not None]
        p95_ttft_ms = None
        p99_ttft_ms = None
        if ttft_values:
            sorted_ttft = sorted(ttft_values)
            p95_index = int(len(sorted_ttft) * 0.95)
            p99_index = int(len(sorted_ttft) * 0.99)
            p95_ttft_ms = sorted_ttft[min(p95_index, len(sorted_ttft) - 1)]
            p99_ttft_ms = sorted_ttft[min(p99_index, len(sorted_ttft) - 1)]
        
        conn.close()
        
        metrics = {}
        if analysis_result:
            metrics.update({
                "avg_processing_time_ms": analysis_result["avg_processing_time_ms"],
                "min_processing_time_ms": analysis_result["min_processing_time_ms"],
                "max_processing_time_ms": analysis_result["max_processing_time_ms"],
                "avg_tokens_used": analysis_result["avg_tokens_used"],
                "min_tokens_used": analysis_result["min_tokens_used"],
                "max_tokens_used": analysis_result["max_tokens_used"],
                "total_requests": analysis_result["total_requests"],
                "total_errors": analysis_result["total_errors"],
                "error_rate_percent": analysis_result["error_rate_percent"],
                "success_rate_percent": analysis_result["success_rate_percent"]
            })
        if vllm_result:
            metrics.update({
                "avg_ttft_ms": vllm_result["avg_ttft_ms"],
                "min_ttft_ms": vllm_result["min_ttft_ms"],
                "max_ttft_ms": vllm_result["max_ttft_ms"],
                "p95_ttft_ms": p95_ttft_ms,
                "p99_ttft_ms": p99_ttft_ms,
                "avg_tps": vllm_result["avg_tps"],
                "avg_tpot_ms": vllm_result["avg_tpot_ms"],
                "total_vllm_requests": vllm_result["total_vllm_requests"]
            })
        
        return metrics if metrics else None
    except Exception as e:
        print_warning(f"메트릭 조회 실패: {str(e)}")
        return None


def get_goodput_stats() -> Optional[Dict[str, Any]]:
    """Goodput 통계 조회"""
    if not METRICS_COLLECTOR_AVAILABLE:
        return None
    
    try:
        metrics = MetricsCollector()
        goodput_stats = metrics.get_goodput_stats()
        return goodput_stats
    except Exception as e:
        print_warning(f"Goodput 통계 조회 실패: {str(e)}")
        return None


def evaluate_accuracy(
    analysis_type: str,
    restaurant_id: int,
    api_result: Dict[str, Any],
    ground_truth_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    정확도 평가 (Ground Truth 비교)
    
    Args:
        analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
        restaurant_id: 레스토랑 ID
        api_result: API 호출 결과
        ground_truth_path: Ground Truth 파일 경로
        
    Returns:
        정확도 메트릭 딕셔너리 또는 None
    """
    if not EVALUATION_AVAILABLE:
        return None
    
    if not ground_truth_path or not Path(ground_truth_path).exists():
        return None
    
    try:
        if analysis_type == "sentiment":
            evaluator = SentimentAnalysisEvaluator(
                base_url=BASE_URL,
                ground_truth_path=ground_truth_path
            )
            if not evaluator.ground_truth:
                return None
            
            # Ground Truth에서 해당 레스토랑 찾기
            restaurants = evaluator.ground_truth.get("restaurants", [])
            gt_restaurant = None
            for r in restaurants:
                if r.get("restaurant_id") == restaurant_id:
                    gt_restaurant = r
                    break
            
            if not gt_restaurant:
                return None
            
            # 정확도 계산
            gt_positive_ratio = gt_restaurant.get("positive_ratio", 0)
            gt_negative_ratio = gt_restaurant.get("negative_ratio", 0)
            predicted_positive_ratio = api_result.get("positive_ratio", 0)
            predicted_negative_ratio = api_result.get("negative_ratio", 0)
            
            ratio_error_positive = abs(predicted_positive_ratio - gt_positive_ratio)
            ratio_error_negative = abs(predicted_negative_ratio - gt_negative_ratio)
            
            return {
                "mae_positive_ratio": ratio_error_positive,
                "mae_negative_ratio": ratio_error_negative,
                "avg_ratio_error": (ratio_error_positive + ratio_error_negative) / 2,
                "ground_truth_positive_ratio": gt_positive_ratio,
                "ground_truth_negative_ratio": gt_negative_ratio,
            }
        
        elif analysis_type == "summary":
            evaluator = SummaryEvaluator(
                base_url=BASE_URL,
                ground_truth_path=ground_truth_path
            )
            if not evaluator.ground_truth:
                return None
            
            # Ground Truth에서 해당 레스토랑 찾기
            restaurants = evaluator.ground_truth.get("restaurants", [])
            gt_restaurant = None
            for r in restaurants:
                if r.get("restaurant_id") == restaurant_id:
                    gt_restaurant = r
                    break
            
            if not gt_restaurant:
                return None
            
            # BLEU Score 계산
            predicted_summary = api_result.get("overall_summary", "")
            gt_summary = gt_restaurant.get("overall_summary", "")
            
            if predicted_summary and gt_summary:
                bleu_score = evaluator.calculate_bleu_score(predicted_summary, gt_summary)
                rouge_scores = evaluator.calculate_rouge_scores(predicted_summary, gt_summary)
                
                return {
                    "bleu_score": bleu_score,
                    "rouge1": rouge_scores.get("rouge1", 0),
                    "rouge2": rouge_scores.get("rouge2", 0),
                    "rougeL": rouge_scores.get("rougeL", 0),
                }
        
        elif analysis_type == "strength":
            evaluator = StrengthExtractionEvaluator(
                base_url=BASE_URL,
                ground_truth_path=ground_truth_path
            )
            if not evaluator.ground_truth:
                return None
            
            # Ground Truth에서 해당 레스토랑 찾기
            restaurants = evaluator.ground_truth.get("restaurants", [])
            gt_restaurant = None
            for r in restaurants:
                if r.get("restaurant_id") == restaurant_id:
                    gt_restaurant = r
                    break
            
            if not gt_restaurant:
                return None
            
            # Precision@K, Recall@K 계산 (k=1, 3, 5, 10)
            predicted_strengths = api_result.get("strengths", [])
            gt_strengths = gt_restaurant.get("ground_truth_strengths", {})
            gt_all = gt_strengths.get("representative", []) + gt_strengths.get("distinct", [])
            
            if predicted_strengths and gt_all:
                k_values = [1, 3, 5, 10]
                precision_at_k = {}
                recall_at_k = {}
                
                # 각 k 값에 대해 Precision@k, Recall@k 계산
                for k in k_values:
                    precision_at_k[f"P@{k}"] = evaluator.calculate_precision_at_k(
                        predicted_strengths=predicted_strengths[:k],
                        ground_truth_strengths=gt_all,
                        k=k
                    )
                    recall_at_k[f"R@{k}"] = evaluator.calculate_recall_at_k(
                        predicted_strengths=predicted_strengths[:k],
                        ground_truth_strengths=gt_all,
                        k=k
                    )
                
                coverage = evaluator.calculate_coverage(
                    predicted_strengths=predicted_strengths,
                    ground_truth_strengths=gt_all
                )
                
                # coverage가 딕셔너리인 경우 coverage 값만 추출
                coverage_value = coverage.get("coverage", 0.0) if isinstance(coverage, dict) else coverage
                
                return {
                    "k_values": k_values,
                    "precision_at_k": precision_at_k,
                    "recall_at_k": recall_at_k,
                    "precision_at_5": precision_at_k.get("P@5", 0.0),  # 하위 호환성 유지
                    "recall_at_5": recall_at_k.get("R@5", 0.0),  # 하위 호환성 유지
                    "coverage": coverage_value,
                }
        
        return None
    except Exception as e:
        print_warning(f"정확도 평가 실패: {str(e)}")
        return None


def test_sentiment_analysis(enable_benchmark: bool = False, num_iterations: int = 5):
    """감성 분석 테스트"""
    print_header("1. 감성 분석 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/sentiment/analyze"
    payload = {
        "restaurant_id": SAMPLE_RESTAURANT_ID,
        "reviews": SAMPLE_REVIEWS
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1)
            api_result = None  # API 결과 저장용
            
            if success and stats:
                print_success(f"감성 분석 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print(f"  - 평균 처리 시간: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95 처리 시간: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99 처리 시간: {stats['p99_latency_sec']:.3f}초")
                print(f"  - 최소/최대: {stats['min_latency_sec']:.3f}초 / {stats['max_latency_sec']:.3f}초")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("sentiment", limit=5)
                if db_metrics:
                    print_info("SQLite 메트릭 (최근 5개 요청):")
                    if db_metrics.get("avg_processing_time_ms"):
                        print(f"  - 평균 처리 시간: {db_metrics['avg_processing_time_ms']:.2f}ms")
                    if db_metrics.get("avg_tokens_used"):
                        print(f"  - 평균 토큰 사용량: {db_metrics['avg_tokens_used']:.0f} tokens")
                    if db_metrics.get("avg_ttft_ms"):
                        print(f"  - 평균 TTFT: {db_metrics['avg_ttft_ms']:.2f}ms")
                        sla_status = "✓" if db_metrics['avg_ttft_ms'] < 2000 else "✗"
                        print(f"  - SLA 준수 (TTFT < 2초): {sla_status}")
                    if db_metrics.get("avg_tps"):
                        print(f"  - 평균 TPS: {db_metrics['avg_tps']:.2f} tokens/sec")
                    if db_metrics.get("error_rate_percent"):
                        print(f"  - 에러율: {db_metrics['error_rate_percent']:.2f}%")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 평균 1.2초, P95 3.2초, P99 6.8초)
                target_avg = 1.2
                target_p95 = 3.2
                target_p99 = 6.8
                
                avg_time = stats['avg_latency_sec']
                p95_time = stats['p95_latency_sec']
                p99_time = stats['p99_latency_sec']
                
                targets_met = []
                if avg_time <= target_avg:
                    targets_met.append(f"평균 ({avg_time:.2f}초 ≤ {target_avg}초)")
                else:
                    print_warning(f"  ⚠ 평균 목표 미달성 (목표: {target_avg}초, 실제: {avg_time:.2f}초)")
                
                if p95_time <= target_p95:
                    targets_met.append(f"P95 ({p95_time:.2f}초 ≤ {target_p95}초)")
                else:
                    print_warning(f"  ⚠ P95 목표 미달성 (목표: {target_p95}초, 실제: {p95_time:.2f}초)")
                
                if p99_time <= target_p99:
                    targets_met.append(f"P99 ({p99_time:.2f}초 ≤ {target_p99}초)")
                else:
                    print_warning(f"  ⚠ P99 목표 미달성 (목표: {target_p99}초, 실제: {p99_time:.2f}초)")
                
                if len(targets_met) == 3:
                    print_success(f"  ✓ 모든 목표 달성: {', '.join(targets_met)}")
                
                # 정확도 평가 (Ground Truth 비교, 벤치마크 모드에서도 수행)
                accuracy_metrics = None
                if stats.get("last_successful_response"):
                    ground_truth_path = str(project_root / "scripts" / "Ground_truth_sentiment.json")
                    accuracy_metrics = evaluate_accuracy(
                        analysis_type="sentiment",
                        restaurant_id=SAMPLE_RESTAURANT_ID,
                        api_result=stats.get("last_successful_response", {}),
                        ground_truth_path=ground_truth_path
                    )
                    if accuracy_metrics:
                        print_info("정확도 평가 (Ground Truth 비교):")
                        if accuracy_metrics.get("mae_positive_ratio") is not None:
                            mae_positive = accuracy_metrics['mae_positive_ratio']
                            if isinstance(mae_positive, (int, float)):
                                print(f"  - MAE (Positive Ratio): {float(mae_positive):.2f}%")
                        if accuracy_metrics.get("mae_negative_ratio") is not None:
                            mae_negative = accuracy_metrics['mae_negative_ratio']
                            if isinstance(mae_negative, (int, float)):
                                print(f"  - MAE (Negative Ratio): {float(mae_negative):.2f}%")
                
                # JSON 저장용 메트릭 수집
                test_metrics["감성 분석"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_avg_sec": target_avg,
                        "target_p95_sec": target_p95,
                        "target_p99_sec": target_p99,
                        "target_avg_achieved": avg_time <= target_avg,
                        "target_p95_achieved": p95_time <= target_p95,
                        "target_p99_achieved": p99_time <= target_p99,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": accuracy_metrics if accuracy_metrics else None,
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"감성 분석 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 긍정 비율: {data.get('positive_ratio', 'N/A')}%")
                print(f"  - 부정 비율: {data.get('negative_ratio', 'N/A')}%")
                print(f"  - 긍정 개수: {data.get('positive_count', 'N/A')}")
                print(f"  - 부정 개수: {data.get('negative_count', 'N/A')}")
                print(f"  - 전체 개수: {data.get('total_count', 'N/A')}")
                if data.get('debug'):
                    print(f"  - Request ID: {data['debug'].get('request_id', 'N/A')}")
                    print(f"  - 처리 시간: {data['debug'].get('processing_time_ms', 'N/A')}ms")
                return True
            else:
                print_error(f"감성 분석 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"감성 분석 중 오류: {str(e)}")
        return False


def test_sentiment_analysis_batch(enable_benchmark: bool = False, num_iterations: int = 5):
    """배치 감성 분석 테스트"""
    print_header("2. 배치 감성 분석 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/sentiment/analyze/batch"
    # 10개 레스토랑 배치 생성 (QUANTITATIVE_METRICS.md 요구사항)
    restaurants_payload = []
    for i in range(10):
        restaurants_payload.append({
            "restaurant_id": SAMPLE_RESTAURANT_ID + i,
            "reviews": SAMPLE_REVIEWS  # 모든 레스토랑에 동일한 리뷰 사용
        })
    
    payload = {
        "restaurants": restaurants_payload
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=120)
            
            if success and stats:
                print_success(f"배치 감성 분석 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("배치 처리 시간 통계 (10개 레스토랑):")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                print(f"  - 최소/최대: {stats['min_latency_sec']:.3f}초 / {stats['max_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("sentiment", limit=5)
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 5-10초)
                target_min = 5.0
                target_max = 10.0
                avg_time = stats['avg_latency_sec']
                if target_min <= avg_time <= target_max:
                    print_success(f"  ✓ 목표 범위 달성 ({target_min}-{target_max}초)")
                else:
                    print_warning(f"  ⚠ 목표 범위 미달성 (목표: {target_min}-{target_max}초, 실제: {avg_time:.2f}초)")
                
                # JSON 저장용 메트릭 수집
                test_metrics["배치 감성 분석"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_min_sec": target_min,
                        "target_max_sec": target_max,
                        "target_achieved": target_min <= avg_time <= target_max,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": None,  # 배치 감성 분석은 정확도 평가 없음
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=120)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"배치 감성 분석 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 처리된 레스토랑 수: {len(data.get('results', []))}")
                for result in data.get('results', [])[:5]:  # 상위 5개만 출력
                    print(f"    레스토랑 {result.get('restaurant_id')}: "
                          f"긍정 {result.get('positive_ratio', 'N/A')}%, "
                          f"부정 {result.get('negative_ratio', 'N/A')}%")
                return True
            else:
                print_error(f"배치 감성 분석 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"배치 감성 분석 중 오류: {str(e)}")
        return False


def test_summarize(enable_benchmark: bool = False, num_iterations: int = 5):
    """리뷰 요약 테스트"""
    print_header("3. 리뷰 요약 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/llm/summarize"
    payload = {
        "restaurant_id": str(SAMPLE_RESTAURANT_ID),
        "positive_query": "맛있다 좋다 만족",
        "negative_query": "맛없다 별로 불만",
        "limit": 10,
        "min_score": 0.0
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=120)
            
            if success and stats:
                print_success(f"리뷰 요약 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("summary", limit=5)
                if db_metrics:
                    print_info("SQLite 메트릭 (최근 5개 요청):")
                    if db_metrics.get("avg_processing_time_ms"):
                        print(f"  - 평균 처리 시간: {db_metrics['avg_processing_time_ms']:.2f}ms")
                    if db_metrics.get("avg_tokens_used"):
                        print(f"  - 평균 토큰 사용량: {db_metrics['avg_tokens_used']:.0f} tokens")
                    if db_metrics.get("avg_ttft_ms"):
                        print(f"  - 평균 TTFT: {db_metrics['avg_ttft_ms']:.2f}ms")
                        if db_metrics.get("p95_ttft_ms"):
                            print(f"  - P95 TTFT: {db_metrics['p95_ttft_ms']:.2f}ms")
                        if db_metrics.get("p99_ttft_ms"):
                            print(f"  - P99 TTFT: {db_metrics['p99_ttft_ms']:.2f}ms")
                        sla_status = "✓" if db_metrics['avg_ttft_ms'] < 2000 else "✗"
                        print(f"  - SLA 준수 (TTFT < 2초): {sla_status}")
                    if db_metrics.get("avg_tps"):
                        print(f"  - 평균 TPS: {db_metrics['avg_tps']:.2f} tokens/sec")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 평균 2.5초, P95 4.8초, P99 9.5초)
                target_avg = 2.5
                target_p95 = 4.8
                target_p99 = 9.5
                
                avg_time = stats['avg_latency_sec']
                p95_time = stats['p95_latency_sec']
                p99_time = stats['p99_latency_sec']
                
                targets_met = []
                if avg_time <= target_avg:
                    targets_met.append(f"평균 ({avg_time:.2f}초 ≤ {target_avg}초)")
                else:
                    print_warning(f"  ⚠ 평균 목표 미달성 (목표: {target_avg}초, 실제: {avg_time:.2f}초)")
                
                if p95_time <= target_p95:
                    targets_met.append(f"P95 ({p95_time:.2f}초 ≤ {target_p95}초)")
                else:
                    print_warning(f"  ⚠ P95 목표 미달성 (목표: {target_p95}초, 실제: {p95_time:.2f}초)")
                
                if p99_time <= target_p99:
                    targets_met.append(f"P99 ({p99_time:.2f}초 ≤ {target_p99}초)")
                else:
                    print_warning(f"  ⚠ P99 목표 미달성 (목표: {target_p99}초, 실제: {p99_time:.2f}초)")
                
                if len(targets_met) == 3:
                    print_success(f"  ✓ 모든 목표 달성: {', '.join(targets_met)}")
                
                # 정확도 평가 (Ground Truth 비교, 벤치마크 모드에서도 수행)
                accuracy_metrics = None
                if stats.get("last_successful_response"):
                    ground_truth_path = str(project_root / "scripts" / "Ground_truth_summary.json")
                    accuracy_metrics = evaluate_accuracy(
                        analysis_type="summary",
                        restaurant_id=SAMPLE_RESTAURANT_ID,
                        api_result=stats.get("last_successful_response", {}),
                        ground_truth_path=ground_truth_path
                    )
                    if accuracy_metrics:
                        print_info("정확도 평가 (Ground Truth 비교):")
                        if accuracy_metrics.get("bleu_score") is not None:
                            bleu_score = accuracy_metrics['bleu_score']
                            if isinstance(bleu_score, (int, float)):
                                print(f"  - BLEU Score: {float(bleu_score):.4f}")
                        if accuracy_metrics.get("rouge1") is not None:
                            rouge1 = accuracy_metrics['rouge1']
                            if isinstance(rouge1, (int, float)):
                                print(f"  - ROUGE-1: {float(rouge1):.4f}")
                        if accuracy_metrics.get("rouge2") is not None:
                            rouge2 = accuracy_metrics['rouge2']
                            if isinstance(rouge2, (int, float)):
                                print(f"  - ROUGE-2: {float(rouge2):.4f}")
                        if accuracy_metrics.get("rougeL") is not None:
                            rougeL = accuracy_metrics['rougeL']
                            if isinstance(rougeL, (int, float)):
                                print(f"  - ROUGE-L: {float(rougeL):.4f}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["리뷰 요약"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_avg_sec": target_avg,
                        "target_p95_sec": target_p95,
                        "target_p99_sec": target_p99,
                        "target_avg_achieved": avg_time <= target_avg,
                        "target_p95_achieved": p95_time <= target_p95,
                        "target_p99_achieved": p99_time <= target_p99,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": accuracy_metrics if accuracy_metrics else None,
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=120)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"리뷰 요약 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 전체 요약: {data.get('overall_summary', 'N/A')[:100]}...")
                print(f"  - 긍정 aspect 수: {len(data.get('positive_aspects', []))}")
                print(f"  - 부정 aspect 수: {len(data.get('negative_aspects', []))}")
                print(f"  - 긍정 리뷰 수: {data.get('positive_count', 'N/A')}")
                print(f"  - 부정 리뷰 수: {data.get('negative_count', 'N/A')}")
                
                # 정확도 평가 (Ground Truth 비교, 기본 모드에서도 수행)
                ground_truth_path = str(project_root / "scripts" / "Ground_truth_summary.json")
                accuracy_metrics = evaluate_accuracy(
                    analysis_type="summary",
                    restaurant_id=SAMPLE_RESTAURANT_ID,
                    api_result=data,
                    ground_truth_path=ground_truth_path
                )
                if accuracy_metrics:
                    print_info("정확도 평가 (Ground Truth 비교):")
                    if accuracy_metrics.get("bleu_score") is not None:
                        print(f"  - BLEU Score: {accuracy_metrics['bleu_score']:.4f}")
                
                return True
            else:
                print_error(f"리뷰 요약 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"리뷰 요약 중 오류: {str(e)}")
        return False


def test_summarize_batch(enable_benchmark: bool = False, num_iterations: int = 5):
    """배치 리뷰 요약 테스트"""
    print_header("4. 배치 리뷰 요약 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/llm/summarize/batch"
    payload = {
        "restaurants": [
            {
                "restaurant_id": SAMPLE_RESTAURANT_ID,
                "positive_query": "맛있다 좋다 만족",
                "negative_query": "맛없다 별로 불만",
                "limit": 10,
                "min_score": 0.0
            },
            {
                "restaurant_id": SAMPLE_RESTAURANT_ID + 1,
                "positive_query": "맛있다 좋다 만족",
                "negative_query": "맛없다 별로 불만",
                "limit": 10,
                "min_score": 0.0
            }
        ]
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=180)
            
            if success and stats:
                print_success(f"배치 리뷰 요약 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("summary", limit=5)
                if db_metrics:
                    print_info("SQLite 메트릭 (최근 5개 요청):")
                    if db_metrics.get("avg_processing_time_ms"):
                        print(f"  - 평균 처리 시간: {db_metrics['avg_processing_time_ms']:.2f}ms")
                    if db_metrics.get("avg_tokens_used"):
                        print(f"  - 평균 토큰 사용량: {db_metrics['avg_tokens_used']:.0f} tokens")
                    if db_metrics.get("avg_ttft_ms"):
                        print(f"  - 평균 TTFT: {db_metrics['avg_ttft_ms']:.2f}ms")
                        if db_metrics.get("p95_ttft_ms"):
                            print(f"  - P95 TTFT: {db_metrics['p95_ttft_ms']:.2f}ms")
                        if db_metrics.get("p99_ttft_ms"):
                            print(f"  - P99 TTFT: {db_metrics['p99_ttft_ms']:.2f}ms")
                        sla_status = "✓" if db_metrics['avg_ttft_ms'] < 2000 else "✗"
                        print(f"  - SLA 준수 (TTFT < 2초): {sla_status}")
                    if db_metrics.get("avg_tps"):
                        print(f"  - 평균 TPS: {db_metrics['avg_tps']:.2f} tokens/sec")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 배치 처리 5-10초)
                target_min = 5.0
                target_max = 10.0
                avg_time = stats['avg_latency_sec']
                if target_min <= avg_time <= target_max:
                    print_success(f"  ✓ 목표 범위 달성 ({target_min}-{target_max}초)")
                else:
                    print_warning(f"  ⚠ 목표 범위 미달성 (목표: {target_min}-{target_max}초, 실제: {avg_time:.2f}초)")
                
                # 정확도 평가 (Ground Truth 비교, 벤치마크 모드에서도 수행)
                accuracy_metrics = None
                if stats.get("last_successful_response"):
                    ground_truth_path = str(project_root / "scripts" / "Ground_truth_summary.json")
                    accuracy_metrics = evaluate_accuracy(
                        analysis_type="summary",
                        restaurant_id=SAMPLE_RESTAURANT_ID,
                        api_result=stats.get("last_successful_response", {}),
                        ground_truth_path=ground_truth_path
                    )
                    if accuracy_metrics:
                        print_info("정확도 평가 (Ground Truth 비교):")
                        if accuracy_metrics.get("bleu_score") is not None:
                            bleu_score = accuracy_metrics['bleu_score']
                            if isinstance(bleu_score, (int, float)):
                                print(f"  - BLEU Score: {float(bleu_score):.4f}")
                        if accuracy_metrics.get("rouge1") is not None:
                            rouge1 = accuracy_metrics['rouge1']
                            if isinstance(rouge1, (int, float)):
                                print(f"  - ROUGE-1: {float(rouge1):.4f}")
                        if accuracy_metrics.get("rouge2") is not None:
                            rouge2 = accuracy_metrics['rouge2']
                            if isinstance(rouge2, (int, float)):
                                print(f"  - ROUGE-2: {float(rouge2):.4f}")
                        if accuracy_metrics.get("rougeL") is not None:
                            rougeL = accuracy_metrics['rougeL']
                            if isinstance(rougeL, (int, float)):
                                print(f"  - ROUGE-L: {float(rougeL):.4f}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["배치 리뷰 요약"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_min_sec": target_min,
                        "target_max_sec": target_max,
                        "target_achieved": target_min <= avg_time <= target_max,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": accuracy_metrics if accuracy_metrics else None,
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=180)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"배치 리뷰 요약 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 처리된 레스토랑 수: {len(data.get('results', []))}")
                for result in data.get('results', []):
                    print(f"    레스토랑 {result.get('restaurant_id')}: "
                          f"요약 완료 ({len(result.get('positive_aspects', []))}개 긍정, "
                          f"{len(result.get('negative_aspects', []))}개 부정 aspect)")
                return True
            else:
                print_error(f"배치 리뷰 요약 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"배치 리뷰 요약 중 오류: {str(e)}")
        return False


def test_extract_strengths(enable_benchmark: bool = False, num_iterations: int = 5):
    """강점 추출 테스트"""
    print_header("5. 강점 추출 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/llm/extract/strengths"
    payload = {
        "restaurant_id": SAMPLE_RESTAURANT_ID,
        "strength_type": "both",  # representative + distinct
        "category_filter": None,  # None이면 모든 레스토랑과 비교 (비교군 찾기 가능)
        "region_filter": None,
        "price_band_filter": None,
        "top_k": 5,
        "max_candidates": 100,
        "months_back": 24,  # 테스트 데이터 대응을 위해 24개월로 확대
        "min_support": 1  # 테스트 데이터 대응을 위해 1로 낮춤 (최소 1개 리뷰만 있어도 통과)
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=180)
            
            if success and stats:
                print_success(f"강점 추출 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("strength", limit=5)
                if db_metrics:
                    print_info("SQLite 메트릭 (최근 5개 요청):")
                    if db_metrics.get("avg_processing_time_ms"):
                        print(f"  - 평균 처리 시간: {db_metrics['avg_processing_time_ms']:.2f}ms")
                    if db_metrics.get("avg_tokens_used"):
                        print(f"  - 평균 토큰 사용량: {db_metrics['avg_tokens_used']:.0f} tokens")
                    if db_metrics.get("avg_ttft_ms"):
                        print(f"  - 평균 TTFT: {db_metrics['avg_ttft_ms']:.2f}ms")
                        if db_metrics.get("p95_ttft_ms"):
                            print(f"  - P95 TTFT: {db_metrics['p95_ttft_ms']:.2f}ms")
                        if db_metrics.get("p99_ttft_ms"):
                            print(f"  - P99 TTFT: {db_metrics['p99_ttft_ms']:.2f}ms")
                        sla_status = "✓" if db_metrics['avg_ttft_ms'] < 2000 else "✗"
                        print(f"  - SLA 준수 (TTFT < 2초): {sla_status}")
                    if db_metrics.get("avg_tps"):
                        print(f"  - 평균 TPS: {db_metrics['avg_tps']:.2f} tokens/sec")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 평균 3.0초, P95 5.5초, P99 11.2초)
                target_avg = 3.0
                target_p95 = 5.5
                target_p99 = 11.2
                
                avg_time = stats['avg_latency_sec']
                p95_time = stats['p95_latency_sec']
                p99_time = stats['p99_latency_sec']
                
                targets_met = []
                if avg_time <= target_avg:
                    targets_met.append(f"평균 ({avg_time:.2f}초 ≤ {target_avg}초)")
                else:
                    print_warning(f"  ⚠ 평균 목표 미달성 (목표: {target_avg}초, 실제: {avg_time:.2f}초)")
                
                if p95_time <= target_p95:
                    targets_met.append(f"P95 ({p95_time:.2f}초 ≤ {target_p95}초)")
                else:
                    print_warning(f"  ⚠ P95 목표 미달성 (목표: {target_p95}초, 실제: {p95_time:.2f}초)")
                
                if p99_time <= target_p99:
                    targets_met.append(f"P99 ({p99_time:.2f}초 ≤ {target_p99}초)")
                else:
                    print_warning(f"  ⚠ P99 목표 미달성 (목표: {target_p99}초, 실제: {p99_time:.2f}초)")
                
                if len(targets_met) == 3:
                    print_success(f"  ✓ 모든 목표 달성: {', '.join(targets_met)}")
                
                # 정확도 평가 (Ground Truth 비교)
                accuracy_metrics = None
                ground_truth_path = str(project_root / "scripts" / "Ground_truth_strength.json")
                accuracy_metrics = evaluate_accuracy(
                    analysis_type="strength",
                    restaurant_id=SAMPLE_RESTAURANT_ID,
                    api_result=stats.get("last_successful_response", {}),
                    ground_truth_path=ground_truth_path
                )
                if accuracy_metrics:
                    print_info("정확도 평가 (Ground Truth 비교):")
                    
                    # k_values 전체에 대한 Precision/Recall 출력
                    if accuracy_metrics.get("precision_at_k"):
                        print_info("  Precision@k:")
                        precision_at_k = accuracy_metrics.get("precision_at_k", {})
                        k_values = accuracy_metrics.get("k_values", [1, 3, 5, 10])
                        for k in k_values:
                            k_key = f"P@{k}"
                            precision = precision_at_k.get(k_key, 0.0)
                            if isinstance(precision, (int, float)):
                                print(f"    - {k_key}: {float(precision):.4f} ({float(precision)*100:.2f}%)")
                    
                    if accuracy_metrics.get("recall_at_k"):
                        print_info("  Recall@k:")
                        recall_at_k = accuracy_metrics.get("recall_at_k", {})
                        k_values = accuracy_metrics.get("k_values", [1, 3, 5, 10])
                        for k in k_values:
                            k_key = f"R@{k}"
                            recall = recall_at_k.get(k_key, 0.0)
                            if isinstance(recall, (int, float)):
                                print(f"    - {k_key}: {float(recall):.4f} ({float(recall)*100:.2f}%)")
                    
                    # 하위 호환성: precision_at_5, recall_at_5 개별 출력도 지원
                    if accuracy_metrics.get("precision_at_5") is not None and not accuracy_metrics.get("precision_at_k"):
                        precision_at_5 = accuracy_metrics['precision_at_5']
                        if isinstance(precision_at_5, (int, float)):
                            print(f"  - Precision@5: {float(precision_at_5):.4f}")
                    if accuracy_metrics.get("recall_at_5") is not None and not accuracy_metrics.get("recall_at_k"):
                        recall_at_5 = accuracy_metrics["recall_at_5"]
                        if isinstance(recall_at_5, (int, float)):
                            print(f"  - Recall@5: {float(recall_at_5):.4f}")
                    
                    if accuracy_metrics.get("coverage") is not None:
                        coverage = accuracy_metrics['coverage']
                        # coverage가 딕셔너리일 경우를 대비해 숫자로 변환
                        if isinstance(coverage, (int, float)):
                            print(f"  - Coverage: {float(coverage):.4f}")
                        elif isinstance(coverage, dict):
                            # coverage가 딕셔너리인 경우 (calculate_coverage 반환값)
                            coverage_value = coverage.get("coverage", 0.0)
                            if isinstance(coverage_value, (int, float)):
                                print(f"  - Coverage: {float(coverage_value):.4f}")
                    
                    target_accuracy = 0.88
                    # precision_at_k에서 P@5 값을 우선 사용
                    precision_at_5_value = accuracy_metrics.get("precision_at_k", {}).get("P@5") or accuracy_metrics.get("precision_at_5", 0)
                    if isinstance(precision_at_5_value, (int, float)) and float(precision_at_5_value) >= target_accuracy:
                        print_success(f"  ✓ 목표 달성 (목표: {target_accuracy}, 실제: {float(precision_at_5_value):.4f})")
                    elif isinstance(precision_at_5_value, (int, float)):
                        print_warning(f"  ⚠ 목표 미달성 (목표: {target_accuracy}, 실제: {float(precision_at_5_value):.4f})")
                
                # JSON 저장용 메트릭 수집
                test_metrics["강점 추출"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_avg_sec": target_avg,
                        "target_p95_sec": target_p95,
                        "target_p99_sec": target_p99,
                        "target_avg_achieved": avg_time <= target_avg,
                        "target_p95_achieved": p95_time <= target_p95,
                        "target_p99_achieved": p99_time <= target_p99,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": accuracy_metrics if accuracy_metrics else None,
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=180)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"강점 추출 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 추출된 강점 수: {len(data.get('strengths', []))}")
                print(f"  - 후보 수: {data.get('total_candidates', 'N/A')}")
                print(f"  - 검증 통과 수: {data.get('validated_count', 'N/A')}")
                
                # 상위 3개 강점 출력
                for i, strength in enumerate(data.get('strengths', [])[:3], 1):
                    print(f"\n  강점 {i}:")
                    print(f"    - Aspect: {strength.get('aspect', 'N/A')}")
                    print(f"    - Claim: {strength.get('claim', 'N/A')[:50]}...")
                    print(f"    - Support Count: {strength.get('support_count', 'N/A')}")
                    if strength.get('distinct_score') is not None:
                        print(f"    - Distinct Score: {strength.get('distinct_score', 'N/A')}")
                
                # 정확도 평가 (Ground Truth 비교, 기본 모드에서도 수행)
                ground_truth_path = str(project_root / "scripts" / "Ground_truth_strength.json")
                accuracy_metrics = evaluate_accuracy(
                    analysis_type="strength",
                    restaurant_id=SAMPLE_RESTAURANT_ID,
                    api_result=data,
                    ground_truth_path=ground_truth_path
                )
                if accuracy_metrics:
                    print_info("정확도 평가 (Ground Truth 비교):")
                    
                    # k_values 전체에 대한 Precision/Recall 출력
                    if accuracy_metrics.get("precision_at_k"):
                        print_info("  Precision@k:")
                        precision_at_k = accuracy_metrics.get("precision_at_k", {})
                        k_values = accuracy_metrics.get("k_values", [1, 3, 5, 10])
                        for k in k_values:
                            k_key = f"P@{k}"
                            precision = precision_at_k.get(k_key, 0.0)
                            if isinstance(precision, (int, float)):
                                print(f"    - {k_key}: {float(precision):.4f} ({float(precision)*100:.2f}%)")
                    
                    if accuracy_metrics.get("recall_at_k"):
                        print_info("  Recall@k:")
                        recall_at_k = accuracy_metrics.get("recall_at_k", {})
                        k_values = accuracy_metrics.get("k_values", [1, 3, 5, 10])
                        for k in k_values:
                            k_key = f"R@{k}"
                            recall = recall_at_k.get(k_key, 0.0)
                            if isinstance(recall, (int, float)):
                                print(f"    - {k_key}: {float(recall):.4f} ({float(recall)*100:.2f}%)")
                    
                    # 하위 호환성: precision_at_5, recall_at_5 개별 출력도 지원
                    if accuracy_metrics.get("precision_at_5") is not None and not accuracy_metrics.get("precision_at_k"):
                        precision_at_5 = accuracy_metrics['precision_at_5']
                        if isinstance(precision_at_5, (int, float)):
                            print(f"  - Precision@5: {float(precision_at_5):.4f}")
                    if accuracy_metrics.get("recall_at_5") is not None and not accuracy_metrics.get("recall_at_k"):
                        recall_at_5 = accuracy_metrics["recall_at_5"]
                        if isinstance(recall_at_5, (int, float)):
                            print(f"  - Recall@5: {float(recall_at_5):.4f}")
                    
                    if accuracy_metrics.get("coverage") is not None:
                        coverage = accuracy_metrics['coverage']
                        # coverage가 딕셔너리일 경우를 대비해 숫자로 변환
                        if isinstance(coverage, (int, float)):
                            print(f"  - Coverage: {float(coverage):.4f}")
                        elif isinstance(coverage, dict):
                            # coverage가 딕셔너리인 경우 (calculate_coverage 반환값)
                            coverage_value = coverage.get("coverage", 0.0)
                            if isinstance(coverage_value, (int, float)):
                                print(f"  - Coverage: {float(coverage_value):.4f}")
                
                return True
            else:
                print_error(f"강점 추출 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"강점 추출 중 오류: {str(e)}")
        return False


def test_vector_search(enable_benchmark: bool = False, num_iterations: int = 5):
    """벡터 검색 테스트"""
    print_header("6. 벡터 검색 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/vector/search/similar"
    payload = {
        "query_text": "맛있다 좋다 만족",
        "restaurant_id": SAMPLE_RESTAURANT_ID,
        "limit": 5,
        "min_score": 0.0
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=30)
            
            if success and stats:
                print_success(f"벡터 검색 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 평균 1.5초, P95 3.0초, P99 6.0초)
                target_avg = 1.5
                target_p95 = 3.0
                target_p99 = 6.0
                
                avg_time = stats['avg_latency_sec']
                p95_time = stats['p95_latency_sec']
                p99_time = stats['p99_latency_sec']
                
                targets_met = []
                if avg_time <= target_avg:
                    targets_met.append(f"평균 ({avg_time:.2f}초 ≤ {target_avg}초)")
                else:
                    print_warning(f"  ⚠ 평균 목표 미달성 (목표: {target_avg}초, 실제: {avg_time:.2f}초)")
                
                if p95_time <= target_p95:
                    targets_met.append(f"P95 ({p95_time:.2f}초 ≤ {target_p95}초)")
                else:
                    print_warning(f"  ⚠ P95 목표 미달성 (목표: {target_p95}초, 실제: {p95_time:.2f}초)")
                
                if p99_time <= target_p99:
                    targets_met.append(f"P99 ({p99_time:.2f}초 ≤ {target_p99}초)")
                else:
                    print_warning(f"  ⚠ P99 목표 미달성 (목표: {target_p99}초, 실제: {p99_time:.2f}초)")
                
                if len(targets_met) == 3:
                    print_success(f"  ✓ 모든 목표 달성: {', '.join(targets_met)}")
                
                # Precision@k 평가 (임베딩 모델 정확도 측정)
                precision_metrics = None
                if EVALUATION_AVAILABLE:
                    try:
                        ground_truth_path = str(project_root / "scripts" / "Ground_truth_vector_search.json")
                        if Path(ground_truth_path).exists():
                            evaluator = PrecisionAtKEvaluator(
                                base_url=BASE_URL,
                                ground_truth_path=ground_truth_path
                            )
                            
                            # Precision@k 평가 수행 (k=1, 3, 5, 10)
                            k_values = [1, 3, 5, 10]
                            precision_result = evaluator.evaluate(
                                k_values=k_values,
                                limit=10,
                                min_score=0.0
                            )
                            
                            if precision_result:
                                avg_precisions = precision_result.get("average_precisions", {})
                                if avg_precisions:
                                    print_info("Precision@k 평가 (임베딩 모델 정확도):")
                                    for k in k_values:
                                        k_key = f"P@{k}"
                                        precision = avg_precisions.get(k_key, 0.0)
                                        if isinstance(precision, (int, float)):
                                            print(f"  - {k_key}: {float(precision):.4f} ({float(precision)*100:.2f}%)")
                                    
                                    avg_recalls = precision_result.get("average_recalls", {})
                                    if avg_recalls:
                                        print_info("Recall@k 평가 (임베딩 모델 정확도):")
                                        for k in k_values:
                                            k_key = f"R@{k}"
                                            recall = avg_recalls.get(k_key, 0.0)
                                            if isinstance(recall, (int, float)):
                                                print(f"  - {k_key}: {float(recall):.4f} ({float(recall)*100:.2f}%)")
                                    
                                    precision_metrics = {
                                        "k_values": k_values,
                                        "average_precisions": avg_precisions,
                                        "average_recalls": avg_recalls if avg_recalls else None,
                                        "total_queries": precision_result.get("total_queries", 0),
                                        "evaluated_queries": precision_result.get("evaluated_queries", 0),
                                    }
                        else:
                            print_warning(f"Ground Truth 파일을 찾을 수 없습니다: {ground_truth_path}")
                    except Exception as e:
                        print_warning(f"Precision@k 평가 실패: {str(e)}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["벡터 검색"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_avg_sec": target_avg,
                        "target_p95_sec": target_p95,
                        "target_p99_sec": target_p99,
                        "target_avg_achieved": avg_time <= target_avg,
                        "target_p95_achieved": p95_time <= target_p95,
                        "target_p99_achieved": p99_time <= target_p99,
                    },
                    "sqlite_metrics": None,
                    "accuracy": precision_metrics if precision_metrics else None,  # Precision@k 메트릭
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"벡터 검색 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 검색 결과 수: {len(data.get('results', []))}")
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    print(f"  결과 {i}:")
                    score = result.get('score', 'N/A')
                    if isinstance(score, (int, float)):
                        print(f"    - 유사도: {score:.3f}")
                    else:
                        print(f"    - 유사도: {score}")
                    # VectorSearchResult 구조: {"review": {...}, "score": ...}
                    review = result.get('review', {})
                    content = review.get('content', 'N/A')
                    if isinstance(content, str) and len(content) > 50:
                        print(f"    - 리뷰 내용: {content[:50]}...")
                    else:
                        print(f"    - 리뷰 내용: {content}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["벡터 검색"] = {
                    "performance": {
                        "elapsed_time_sec": elapsed_time,
                        "result_count": len(data.get('results', [])),
                    },
                    "sqlite_metrics": None,
                    "accuracy": None,
                }
                
                return True
            else:
                print_error(f"벡터 검색 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"벡터 검색 중 오류: {str(e)}")
        return False


def test_review_image_search(enable_benchmark: bool = False, num_iterations: int = 5):
    """리뷰 이미지 검색 테스트"""
    print_header("7. 리뷰 이미지 검색 테스트")
    
    url = f"{BASE_URL}{API_PREFIX}/vector/search/review-images"
    payload = {
        "query": "맛있다 좋다 만족",
        "restaurant_id": SAMPLE_RESTAURANT_ID,
        "limit": 5,
        "min_score": 0.0,
        "expand_query": None  # 자동 판단
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=60)
            
            if success and stats:
                print_success(f"리뷰 이미지 검색 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # 목표값 비교 (QUANTITATIVE_METRICS.md: 평균 2.0초, P95 4.0초, P99 8.0초)
                target_avg = 2.0
                target_p95 = 4.0
                target_p99 = 8.0
                
                avg_time = stats['avg_latency_sec']
                p95_time = stats['p95_latency_sec']
                p99_time = stats['p99_latency_sec']
                
                targets_met = []
                if avg_time <= target_avg:
                    targets_met.append(f"평균 ({avg_time:.2f}초 ≤ {target_avg}초)")
                else:
                    print_warning(f"  ⚠ 평균 목표 미달성 (목표: {target_avg}초, 실제: {avg_time:.2f}초)")
                
                if p95_time <= target_p95:
                    targets_met.append(f"P95 ({p95_time:.2f}초 ≤ {target_p95}초)")
                else:
                    print_warning(f"  ⚠ P95 목표 미달성 (목표: {target_p95}초, 실제: {p95_time:.2f}초)")
                
                if p99_time <= target_p99:
                    targets_met.append(f"P99 ({p99_time:.2f}초 ≤ {target_p99}초)")
                else:
                    print_warning(f"  ⚠ P99 목표 미달성 (목표: {target_p99}초, 실제: {p99_time:.2f}초)")
                
                if len(targets_met) == 3:
                    print_success(f"  ✓ 모든 목표 달성: {', '.join(targets_met)}")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("image_search", limit=5)
                if db_metrics:
                    print_info("SQLite 메트릭 (최근 5개 요청):")
                    if db_metrics.get("avg_processing_time_ms"):
                        print(f"  - 평균 처리 시간: {db_metrics['avg_processing_time_ms']:.2f}ms")
                    if db_metrics.get("avg_tokens_used"):
                        print(f"  - 평균 토큰 사용량: {db_metrics['avg_tokens_used']:.0f} tokens")
                
                # Precision@k / Recall@k 평가 (임베딩 모델 정확도 측정)
                precision_metrics = None
                if EVALUATION_AVAILABLE:
                    try:
                        ground_truth_path = str(project_root / "scripts" / "Ground_truth_vector_search.json")
                        if Path(ground_truth_path).exists():
                            evaluator = PrecisionAtKEvaluator(
                                base_url=BASE_URL,
                                ground_truth_path=ground_truth_path
                            )
                            
                            # 이미지 검색 결과에서 review_id 추출
                            last_response = stats.get("last_successful_response", {})
                            if last_response and last_response.get("results"):
                                # 이미지 검색 결과를 벡터 검색 형식으로 변환하여 평가
                                # 쿼리와 레스토랑 ID를 사용하여 Ground Truth와 매칭
                                query_text = payload.get("query", "")
                                restaurant_id = payload.get("restaurant_id")
                                
                                # Precision@k 평가 수행 (k=1, 3, 5, 10)
                                k_values = [1, 3, 5, 10]
                                
                                # 이미지 검색 결과에서 review_id 리스트 추출
                                retrieved_review_ids = []
                                for result in last_response.get("results", []):
                                    review_id = result.get("review_id")
                                    if review_id is not None:
                                        try:
                                            retrieved_review_ids.append(int(review_id))
                                        except (ValueError, TypeError):
                                            continue
                                
                                if retrieved_review_ids and evaluator.ground_truth:
                                    # Ground Truth에서 해당 쿼리와 레스토랑 ID로 관련 문서 찾기
                                    queries = evaluator.ground_truth.get("queries", [])
                                    relevant_ids = set()
                                    for query_data in queries:
                                        if (query_data.get("query") == query_text or 
                                            (restaurant_id and query_data.get("restaurant_id") == restaurant_id)):
                                            relevant_ids.update(query_data.get("relevant_review_ids", []))
                                    
                                    if relevant_ids:
                                        # Precision@k, Recall@k 계산
                                        precision_at_k = {}
                                        recall_at_k = {}
                                        for k in k_values:
                                            precision_at_k[f"P@{k}"] = evaluator.calculate_precision_at_k(
                                                retrieved_ids=retrieved_review_ids,
                                                relevant_ids=relevant_ids,
                                                k=k
                                            )
                                            recall_at_k[f"R@{k}"] = evaluator.calculate_recall_at_k(
                                                retrieved_ids=retrieved_review_ids,
                                                relevant_ids=relevant_ids,
                                                k=k
                                            )
                                        
                                        if precision_at_k or recall_at_k:
                                            print_info("Precision@k / Recall@k 평가 (임베딩 모델 정확도):")
                                            for k in k_values:
                                                k_key_p = f"P@{k}"
                                                k_key_r = f"R@{k}"
                                                precision = precision_at_k.get(k_key_p, 0.0)
                                                recall = recall_at_k.get(k_key_r, 0.0)
                                                if isinstance(precision, (int, float)):
                                                    print(f"  - {k_key_p}: {float(precision):.4f} ({float(precision)*100:.2f}%)")
                                                if isinstance(recall, (int, float)):
                                                    print(f"  - {k_key_r}: {float(recall):.4f} ({float(recall)*100:.2f}%)")
                                        
                                        precision_metrics = {
                                            "k_values": k_values,
                                            "precision_at_k": precision_at_k,
                                            "recall_at_k": recall_at_k,
                                            "total_queries": 1,
                                            "evaluated_queries": 1 if relevant_ids else 0,
                                        }
                        else:
                            print_warning(f"Ground Truth 파일을 찾을 수 없습니다: {ground_truth_path}")
                    except Exception as e:
                        print_warning(f"Precision@k / Recall@k 평가 실패: {str(e)}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["리뷰 이미지 검색"] = {
                    "performance": {
                        "avg_latency_sec": stats.get("avg_latency_sec"),
                        "min_latency_sec": stats.get("min_latency_sec"),
                        "max_latency_sec": stats.get("max_latency_sec"),
                        "p95_latency_sec": stats.get("p95_latency_sec"),
                        "p99_latency_sec": stats.get("p99_latency_sec"),
                        "success_rate": stats.get("success_rate"),
                        "success_count": stats.get("success_count"),
                        "total_iterations": stats.get("total_iterations"),
                        "throughput_req_per_sec": stats.get("throughput_req_per_sec"),
                        "target_avg_sec": target_avg,
                        "target_p95_sec": target_p95,
                        "target_p99_sec": target_p99,
                        "target_avg_achieved": avg_time <= target_avg,
                        "target_p95_achieved": p95_time <= target_p95,
                        "target_p99_achieved": p99_time <= target_p99,
                    },
                    "sqlite_metrics": db_metrics if db_metrics else None,
                    "accuracy": precision_metrics if precision_metrics else None,  # Precision@k / Recall@k 메트릭
                }
                
                return True
            else:
                print_error("성능 측정 실패")
                return False
        else:
            # 기본 테스트 모드
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"리뷰 이미지 검색 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 검색 결과 수: {len(data.get('results', []))}")
                print(f"  - 총 결과 수: {data.get('total', 0)}")
                
                # 상위 3개 결과 출력
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    print(f"  결과 {i}:")
                    print(f"    - 레스토랑 ID: {result.get('restaurant_id', 'N/A')}")
                    print(f"    - 리뷰 ID: {result.get('review_id', 'N/A')}")
                    print(f"    - 이미지 URL: {result.get('image_url', 'N/A')[:50]}...")
                    review = result.get('review', {})
                    content = review.get('content', 'N/A')
                    if isinstance(content, str) and len(content) > 50:
                        print(f"    - 리뷰 내용: {content[:50]}...")
                    else:
                        print(f"    - 리뷰 내용: {content}")
                
                # JSON 저장용 메트릭 수집
                test_metrics["리뷰 이미지 검색"] = {
                    "performance": {
                        "elapsed_time_sec": elapsed_time,
                        "result_count": len(data.get('results', [])),
                        "total_count": data.get('total', 0),
                    },
                    "sqlite_metrics": None,
                    "accuracy": None,
                }
                
                return True
            else:
                print_error(f"리뷰 이미지 검색 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"리뷰 이미지 검색 중 오류: {str(e)}")
        return False


def run_tests_for_model(
    model_name: str,
    provider: str,
    enable_benchmark: bool = False,
    iterations: int = 5,
    tests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    특정 모델에 대한 테스트 실행
    
    Args:
        model_name: 테스트할 모델명
        provider: LLM 제공자 ("openai", "local", "runpod")
        enable_benchmark: 성능 측정 모드 활성화 여부
        iterations: 성능 측정 반복 횟수
        
    Returns:
        테스트 결과 딕셔너리
    """
    # 환경 변수 설정
    original_provider = os.getenv("LLM_PROVIDER")
    original_model = os.getenv("OPENAI_MODEL") if provider == "openai" else os.getenv("LLM_MODEL")
    
    try:
        os.environ["LLM_PROVIDER"] = provider
        if provider == "openai":
            os.environ["OPENAI_MODEL"] = model_name
        else:
            os.environ["LLM_MODEL"] = model_name
        
        print_header(f"모델 테스트: {model_name} ({provider})")
        print_info(f"서버 URL: {BASE_URL}")
        
        # test_metrics 초기화 (모델별로 독립적으로 관리)
        global test_metrics
        original_test_metrics = test_metrics.copy()
        test_metrics.clear()
        
        # 테스트 실행
        selected_tests = tests or ["summarize", "summarize_batch"]
        if "all" in selected_tests:
            selected_tests = ["sentiment", "sentiment_batch", "summarize", "summarize_batch", "strength", "vector", "image_search"]

        test_registry = {
            "sentiment": ("감성 분석", lambda: test_sentiment_analysis(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "sentiment_batch": ("배치 감성 분석", lambda: test_sentiment_analysis_batch(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "summarize": ("리뷰 요약", lambda: test_summarize(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "summarize_batch": ("배치 리뷰 요약", lambda: test_summarize_batch(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "strength": ("강점 추출", lambda: test_extract_strengths(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "vector": ("벡터 검색", lambda: test_vector_search(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "image_search": ("리뷰 이미지 검색", lambda: test_review_image_search(enable_benchmark=enable_benchmark, num_iterations=iterations)),
        }

        results = []
        for key in selected_tests:
            if key not in test_registry:
                print_warning(f"알 수 없는 테스트 항목: {key} (skip)")
                continue
            label, fn = test_registry[key]
            results.append((label, fn()))
        
        # test_metrics 저장 (모델별로)
        model_test_metrics = test_metrics.copy()
        
        # results를 qwen.json과 유사한 구조로도 제공 (형식 변환 X, 반환값에만 포함)
        # - compare_models 저장 시 이 구조를 그대로 덤프하면 모든 메트릭이 포함됨
        test_results: Dict[str, Any] = {}
        for test_name, ok in results:
            test_result_dict: Dict[str, Any] = {
                "status": "passed" if ok else "failed",
                "success": ok,
            }
            if test_name in model_test_metrics:
                # performance/sqlite_metrics/accuracy 등 모든 메트릭 포함
                test_result_dict.update(model_test_metrics[test_name])
            test_results[test_name] = test_result_dict
        
        # 결과 집계
        success_count = sum(1 for _, result in results if result)
        total_count = len(results)
        
        # test_metrics 복원
        test_metrics.clear()
        test_metrics.update(original_test_metrics)
        
        return {
            "model_name": model_name,
            "provider": provider,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
            "results": results,  # 기존 호환
            "test_results": test_results,  # 권장: 테스트별 + 메트릭까지 포함된 구조
            "test_metrics": model_test_metrics,  # 모든 메트릭 원본 (디버깅/후처리용)
        }
    finally:
        # 환경 변수 복원
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)
        
        if provider == "openai":
            if original_model:
                os.environ["OPENAI_MODEL"] = original_model
            else:
                os.environ.pop("OPENAI_MODEL", None)
        else:
            if original_model:
                os.environ["LLM_MODEL"] = original_model
            else:
                os.environ.pop("LLM_MODEL", None)


def compare_models(
    models: List[str],
    provider: str,
    enable_benchmark: bool = False,
    iterations: int = 5,
    save_results: Optional[str] = None,
    generate_report: bool = False,
    tests: Optional[List[str]] = None,
    base_ports: Optional[List[int]] = None,
    test_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    여러 모델 비교 테스트
    
    Args:
        models: 비교할 모델명 리스트
        provider: LLM 제공자 ("openai", "local", "runpod")
        enable_benchmark: 성능 측정 모드 활성화 여부
        iterations: 성능 측정 반복 횟수
        save_results: 결과를 저장할 JSON 파일 경로
        generate_report: 비교 리포트 생성 여부
        base_ports: 각 모델별 서버 포트 리스트 (None이면 자동 할당: 8001부터 시작)
        test_data: 업로드할 테스트 데이터 (각 포트별로 업로드)
        
    Returns:
        비교 결과 딕셔너리
    """
    global BASE_URL
    
    # 포트 자동 할당 (지정되지 않은 경우)
    if base_ports is None:
        base_ports = [8001 + i for i in range(len(models))]
    
    if len(base_ports) != len(models):
        print_error(f"포트 개수({len(base_ports)})와 모델 개수({len(models)})가 일치하지 않습니다.")
        sys.exit(1)
    
    print_header(f"여러 모델 비교 테스트 ({len(models)}개 모델)")
    print_info(f"제공자: {provider}")
    print_info(f"모델 목록: {', '.join(models)}")
    print_info("\n각 모델은 별도 포트에서 실행 중이어야 합니다:")
    for model, port in zip(models, base_ports):
        print_info(f"  - {model}: http://localhost:{port}")
    
    all_results = {}
    original_base_url = BASE_URL
    
    for i, (model_name, port) in enumerate(zip(models, base_ports), 1):
        print(f"\n{'='*60}")
        print(f"모델 {i}/{len(models)}: {model_name} (포트: {port})")
        print(f"{'='*60}\n")
        
        # BASE_URL을 해당 모델의 포트로 임시 변경
        BASE_URL = f"http://localhost:{port}"
        
        try:
            # 서버 연결 확인
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code != 200:
                    print_warning(f"포트 {port}의 서버가 응답하지 않습니다. (상태 코드: {response.status_code})")
            except Exception as e:
                print_error(f"포트 {port}의 서버에 연결할 수 없습니다: {e}")
                print_info(f"다음 명령으로 서버를 시작하세요:")
                if provider == "openai":
                    print_info(f"  LLM_PROVIDER={provider} OPENAI_MODEL={model_name} uvicorn app:app --port {port}")
                else:
                    print_info(f"  LLM_PROVIDER={provider} LLM_MODEL={model_name} uvicorn app:app --port {port}")
                all_results[model_name] = {
                    "model_name": model_name,
                    "provider": provider,
                    "success_count": 0,
                    "total_count": 0,
                    "success_rate": 0,
                    "results": [],
                    "error": f"서버 연결 실패 (포트 {port})"
                }
                continue
            
            # 각 포트별로 데이터 업로드
            if test_data:
                print_info(f"포트 {port}에 테스트 데이터 업로드 중...")
                if upload_data_to_qdrant(test_data):
                    print_success(f"포트 {port}에 데이터 업로드 완료")
                else:
                    print_warning(f"포트 {port}에 데이터 업로드 실패. 테스트가 실패할 수 있습니다.")
            
            result = run_tests_for_model(
                model_name=model_name,
                provider=provider,
                enable_benchmark=enable_benchmark,
                iterations=iterations,
                tests=tests,
            )
            all_results[model_name] = result
        finally:
            # BASE_URL 복원
            BASE_URL = original_base_url
        
        # 모델 간 간격
        if i < len(models):
            print_info("다음 모델 테스트로 이동...")
            time.sleep(2)  # 짧은 대기
    
    # 비교 리포트 생성
    if generate_report:
        print_header("모델 비교 리포트")
        print("\n성공률 비교:")
        for model_name, result in all_results.items():
            success_rate = result.get("success_rate", 0)
            status = "✓" if success_rate == 100 else "⚠" if success_rate >= 50 else "✗"
            print(f"  {status} {model_name}: {success_rate:.1f}% ({result['success_count']}/{result['total_count']})")
    
    # 결과 저장 (형식 변환 없이, 반환값 그대로 저장)
    if save_results:
        with open(save_results, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print_success(f"결과 저장 완료: {save_results}")
    
    return all_results


def main():
    """
    메인 테스트 실행
    
    여러 모델 테스트 지원:
    - 단일 모델: --model 옵션 사용
    - 여러 모델 비교: --compare-models 옵션 사용
    - 환경 변수 기반: 환경 변수만 설정하여 실행
    """
    parser = argparse.ArgumentParser(
        description="전체 기능 통합 테스트 (다중 모델 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 단일 모델 테스트
  python test_openai_all.py --model "gpt-4o-mini" --provider openai
  
  # 여러 모델 비교
  python test_openai_all.py --compare-models \\
      --models "gpt-4o-mini" "gpt-3.5-turbo" \\
      --provider openai \\
      --benchmark \\
      --save-results results.json
  
  # 부하테스트 (baseline(대표 성능) 측정)

  python test_openai_all.py --load-test \
      --total-requests 500 \
      --concurrent-users 5 \
      --ramp-up 20 \
      --save-results load_test_baseline_results.json
      
# 부하테스트 (stress(한계 확인) 측정)

  python test_openai_all.py --load-test \
      --total-requests 1000 \
      --concurrent-users 15 \
      --ramp-up 30 \
      --save-results load_test_stress_results.json
  
  # 환경 변수 기반 (기존 방식)
  export LLM_PROVIDER="openai"
  export OPENAI_MODEL="gpt-4o-mini"
  python test_openai_all.py --benchmark
        """
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="성능 측정 모드 활성화 (QUANTITATIVE_METRICS.md 지표 측정)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="성능 측정 반복 횟수 (기본값: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="테스트할 모델명 (예: 'gpt-4o-mini', 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "local", "runpod"],
        help="LLM 제공자 선택 (openai, local, runpod)"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="여러 모델 비교 테스트 모드"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="비교할 모델명 리스트 (--compare-models와 함께 사용)"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="결과를 저장할 JSON 파일 경로"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="모델 비교 리포트 생성 (--compare-models와 함께 사용)"
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="부하테스트 모드 활성화 (동시 요청 처리 능력 측정)"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        choices=["all", "sentiment", "sentiment_batch", "summarize", "summarize_batch", "strength", "vector", "image_search"],
        help="실행할 테스트 선택 (기본값: summarize summarize_batch). 예: --tests summarize summarize_batch",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=100,
        help="부하테스트 총 요청 수 (기본값: 100)"
    )
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=10,
        help="부하테스트 동시 사용자 수 (기본값: 10)"
    )
    parser.add_argument(
        "--ramp-up",
        type=int,
        default=0,
        help="부하테스트 점진적 부하 증가 시간(초) (기본값: 0, 즉시 시작)"
    )
    parser.add_argument(
        "--generate-from-kr3",
        action="store_true",
        help="kr3.tsv에서 테스트 데이터 생성 (기본값: test_data_sample.json 사용)"
    )
    parser.add_argument(
        "--kr3-sample",
        type=int,
        default=None,
        help="kr3.tsv에서 샘플링할 리뷰 수 (--generate-from-kr3와 함께 사용)"
    )
    parser.add_argument(
        "--kr3-restaurants",
        type=int,
        default=None,
        help="생성할 레스토랑 수 (--generate-from-kr3와 함께 사용)"
    )
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        help="각 모델별 서버 포트 리스트 (--compare-models와 함께 사용). 예: --ports 8001 8002 8003. 지정하지 않으면 8001부터 자동 할당"
    )
    args = parser.parse_args()
    
    print_header("RunPod Pod 서버 API 전체 기능 통합 테스트")
    print_info(f"서버 URL: {BASE_URL}")
    print_info("RunPod Pod에서 실행 중인 FastAPI 서버를 테스트합니다")
    
    if args.benchmark:
        print_info("성능 측정 모드 활성화 (QUANTITATIVE_METRICS.md 지표 측정)")
        print_info(f"반복 횟수: {args.iterations}")
    
    # 환경 변수 설정 (--provider 옵션이 있으면 적용)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        print_info(f"LLM_PROVIDER 설정: {args.provider}")
    
    # 환경 변수 확인
    llm_provider = os.getenv("LLM_PROVIDER", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # --model 옵션이 있으면 환경 변수 설정
    if args.model:
        if args.provider == "openai" or (not args.provider and llm_provider == "openai"):
            os.environ["OPENAI_MODEL"] = args.model
            print_info(f"OPENAI_MODEL 설정: {args.model}")
        else:
            os.environ["LLM_MODEL"] = args.model
            print_info(f"LLM_MODEL 설정: {args.model}")
    
    if llm_provider == "local":
        llm_model = os.getenv("LLM_MODEL", "")
        if llm_model == "Qwen/Qwen2.5-7B-Instruct":
            print_info("Qwen/Qwen2.5-7B-Instruct 모델 사용")
        elif llm_model == "meta-llama/Llama-3.1-8B-Instruct":
            print_info("meta-llama/Llama-3.1-8B-Instruct 모델 사용")
        elif llm_model == "google/gemma-2-9b-it":
            print_info("google/gemma-2-9b-it 모델 사용")
        elif llm_model == "LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF":
            print_info("LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF 모델 사용")
        elif llm_model == "unsloth/DeepSeek-R1-GGUF":
            print_info("unsloth/DeepSeek-R1-GGUF 모델 사용")
        elif llm_model:
            print_info(f"로컬 모델 사용: {llm_model}")
    
    # OpenAI 모델 확인
    openai_model = os.getenv("OPENAI_MODEL", "")
    if openai_model:
        print_info(f"OpenAI 모델 사용: {openai_model}")
    
    if llm_provider and llm_provider != "openai":
        print_info(f"LLM_PROVIDER: {llm_provider}")
    
    if not openai_key:
        print_warning("OPENAI_API_KEY가 설정되지 않았습니다.")
        print_info("다음 명령으로 설정하세요: export OPENAI_API_KEY='your_api_key'")
    # OpenAI API 키 확인 메시지 제거
    
    # 서버 헬스 체크
    if not check_server_health():
        sys.exit(1)
    
    # 테스트 데이터 생성
    data_result = generate_test_data(
        generate_from_kr3=args.generate_from_kr3,
        kr3_sample=args.kr3_sample,
        kr3_restaurants=args.kr3_restaurants,
    )
    temp_json_path = None
    test_data = None
    
    if data_result:
        data, temp_json_path = data_result
        test_data = data  # compare_models에 전달할 데이터 저장
        
        # SAMPLE_RESTAURANT_ID와 SAMPLE_REVIEWS를 실제 데이터로 업데이트
        if data.get("restaurants"):
            global SAMPLE_RESTAURANT_ID, SAMPLE_REVIEWS
            first_restaurant = data["restaurants"][0]
            SAMPLE_RESTAURANT_ID = first_restaurant.get("restaurant_id", 1)
            SAMPLE_REVIEWS = first_restaurant.get("reviews", [])
            print_info(f"테스트 레스토랑 ID: {SAMPLE_RESTAURANT_ID}")
            print_info(f"테스트 리뷰 수: {len(SAMPLE_REVIEWS)}개")
    
    # 모델 비교 모드 처리
    if args.compare_models:
        if not args.models or not args.provider:
            print_error("--compare-models 모드에서는 --models와 --provider 옵션이 필요합니다.")
            print_info("사용 예: python test_all_task.py --compare-models --models 'model1' 'model2' --provider openai --benchmark --save-results results.json")
            print_info("포트 지정 예: python test_all_task.py --compare-models --models 'model1' 'model2' --provider local --ports 8001 8002")
            sys.exit(1)
        
        # 포트 검증
        if args.ports and len(args.ports) != len(args.models):
            print_error(f"포트 개수({len(args.ports)})와 모델 개수({len(args.models)})가 일치하지 않습니다.")
            sys.exit(1)
        
        # compare_models() 함수 호출
        comparison_results = compare_models(
            models=args.models,
            provider=args.provider,
            enable_benchmark=args.benchmark,
            iterations=args.iterations,
            save_results=args.save_results,
            generate_report=args.generate_report,
            tests=args.tests,
            base_ports=args.ports,
            test_data=test_data,
        )
        
        # 결과 요약 출력
        print_header("모델 비교 테스트 완료")
        if args.save_results:
            print_success(f"결과가 저장되었습니다: {args.save_results}")
        
        # 임시 파일 정리
        if temp_json_path and os.path.exists(temp_json_path):
            try:
                os.unlink(temp_json_path)
            except Exception:
                pass
        
        sys.exit(0)
    
    # 일반 모드: 데이터 업로드 (compare_models 모드가 아닐 때만)
    if test_data:
        if upload_data_to_qdrant(test_data):
            print_success("테스트 데이터 준비 완료")
        else:
            print_warning("Qdrant upload 실패. 일부 테스트가 실패할 수 있습니다.")
    else:
        print_warning("테스트 데이터 생성 실패. 일부 테스트가 실패할 수 있습니다.")
    
    # 임시 파일 정리
    if temp_json_path and os.path.exists(temp_json_path):
        try:
            os.unlink(temp_json_path)
        except Exception:
            pass
    
    # 부하테스트 모드 처리
    if args.load_test:
        print_header("부하테스트 모드")
        print_info(f"총 요청 수: {args.total_requests}")
        print_info(f"동시 사용자 수: {args.concurrent_users}")
        if args.ramp_up > 0:
            print_info(f"점진적 부하 증가: {args.ramp_up}초")
        
        # 각 엔드포인트에 대해 부하테스트 실행
        load_test_results = {}
        
        # 1. 배치 감성 분석 부하테스트
        print_header("1. 배치 감성 분석 부하테스트")
        url = f"{BASE_URL}{API_PREFIX}/sentiment/analyze/batch"
        # 10개 레스토랑 배치 생성
        restaurants_payload = []
        for i in range(10):
            restaurants_payload.append({
                "restaurant_id": SAMPLE_RESTAURANT_ID + i,
                "reviews": SAMPLE_REVIEWS  # 모든 레스토랑에 동일한 리뷰 사용
            })
        payload = {
            "restaurants": restaurants_payload
        }
        success, stats = load_test(
            endpoint=url,
            payload=payload,
            total_requests=args.total_requests,
            concurrent_users=args.concurrent_users,
            timeout=120,
            ramp_up_seconds=args.ramp_up
        )
        if success and stats:
            print_success("배치 감성 분석 부하테스트 완료")
            print(f"  - 평균 응답 시간: {stats['avg_latency_sec']:.3f}초")
            print(f"  - P50 응답 시간: {stats.get('p50_latency_sec', 'N/A'):.3f}초" if stats.get('p50_latency_sec') else "  - P50 응답 시간: N/A")
            print(f"  - P95 응답 시간: {stats['p95_latency_sec']:.3f}초")
            print(f"  - P99 응답 시간: {stats['p99_latency_sec']:.3f}초")
            print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
            print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_requests']})")
            print(f"  - 최대 동시 요청 수: {stats.get('max_concurrent_requests', 'N/A')}")
            load_test_results["배치 감성 분석"] = stats
        
        # 2. 배치 리뷰 요약 부하테스트
        print_header("2. 배치 리뷰 요약 부하테스트")
        url = f"{BASE_URL}{API_PREFIX}/llm/summarize/batch"
        payload = {
            "restaurants": [
                {
                    "restaurant_id": SAMPLE_RESTAURANT_ID,
                    "positive_query": "맛있다 좋다 만족",
                    "negative_query": "맛없다 별로 불만",
                    "limit": 10,
                    "min_score": 0.0
                },
                {
                    "restaurant_id": SAMPLE_RESTAURANT_ID + 1,
                    "positive_query": "맛있다 좋다 만족",
                    "negative_query": "맛없다 별로 불만",
                    "limit": 10,
                    "min_score": 0.0
                }
            ]
        }
        success, stats = load_test(
            endpoint=url,
            payload=payload,
            total_requests=args.total_requests,
            concurrent_users=args.concurrent_users,
            timeout=180,
            ramp_up_seconds=args.ramp_up
        )
        if success and stats:
            print_success("배치 리뷰 요약 부하테스트 완료")
            print(f"  - 평균 응답 시간: {stats['avg_latency_sec']:.3f}초")
            print(f"  - P50 응답 시간: {stats.get('p50_latency_sec', 'N/A'):.3f}초" if stats.get('p50_latency_sec') else "  - P50 응답 시간: N/A")
            print(f"  - P95 응답 시간: {stats['p95_latency_sec']:.3f}초")
            print(f"  - P99 응답 시간: {stats['p99_latency_sec']:.3f}초")
            print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
            print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_requests']})")
            print(f"  - 최대 동시 요청 수: {stats.get('max_concurrent_requests', 'N/A')}")
            load_test_results["배치 리뷰 요약"] = stats
        
        # 결과 저장
        if args.save_results:
            load_test_output = {
                "timestamp": datetime.now().isoformat(),
                "server_url": BASE_URL,
                "load_test_mode": True,
                "total_requests": args.total_requests,
                "concurrent_users": args.concurrent_users,
                "ramp_up_seconds": args.ramp_up,
                "test_results": load_test_results,
            }
            with open(args.save_results, 'w', encoding='utf-8') as f:
                json.dump(load_test_output, f, ensure_ascii=False, indent=2)
            print_success(f"\n부하테스트 결과가 저장되었습니다: {args.save_results}")
        
        sys.exit(0)
    
    # 단일 모델 테스트 (기존 로직)
    results = []
    results_dict = {}  # JSON 저장용
    test_metrics.clear()  # 테스트 메트릭 초기화
    
    selected_tests = args.tests or ["summarize", "summarize_batch"]
    if "all" in selected_tests:
        selected_tests = ["sentiment", "sentiment_batch", "summarize", "summarize_batch", "strength", "vector", "image_search"]

    test_registry = {
        "sentiment": ("감성 분석", lambda: test_sentiment_analysis(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "sentiment_batch": ("배치 감성 분석", lambda: test_sentiment_analysis_batch(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "summarize": ("리뷰 요약", lambda: test_summarize(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "summarize_batch": ("배치 리뷰 요약", lambda: test_summarize_batch(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "strength": ("강점 추출", lambda: test_extract_strengths(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "vector": ("벡터 검색", lambda: test_vector_search(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
        "image_search": ("리뷰 이미지 검색", lambda: test_review_image_search(enable_benchmark=args.benchmark, num_iterations=args.iterations)),
    }

    for key in selected_tests:
        if key not in test_registry:
            print_warning(f"알 수 없는 테스트 항목: {key} (skip)")
            continue
        label, fn = test_registry[key]
        results.append((label, fn()))
    
    # 결과 요약
    print_header("테스트 결과 요약")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    # JSON 저장용 결과 딕셔너리 구성
    if args.save_results:
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "server_url": BASE_URL,
            "benchmark_mode": args.benchmark,
            "iterations": args.iterations if args.benchmark else None,
            "model_info": {
                "llm_provider": "local",  # 결과 저장 시 항상 local로 설정
                "llm_model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "")),
                "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
                "sentiment_model": os.getenv("SENTIMENT_MODEL", ""),
            },
            "test_results": {},
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
            }
        }
        
        # 각 테스트 결과 추가
        for name, result in results:
            test_result = {
                "status": "passed" if result else "failed",
                "success": result
            }
            # test_metrics에서 해당 테스트의 성능/정확도 메트릭 추가
            if name in test_metrics:
                test_result.update(test_metrics[name])
            results_dict["test_results"][name] = test_result
    
    for name, result in results:
        if result:
            print_success(f"{name}: 통과")
        else:
            print_error(f"{name}: 실패")
    
    print(f"\n{Colors.BOLD}총 {passed}/{total} 테스트 통과{Colors.RESET}")
    
    if args.benchmark:
        print_info("\n성능 측정 모드로 실행되었습니다.")
        print_info("더 자세한 메트릭은 SQLite 데이터베이스를 확인하세요:")
        print_info(f"  sqlite3 {METRICS_DB_PATH}")
        print_info("\nQUANTITATIVE_METRICS.md의 SQL 쿼리를 사용하여 추가 분석이 가능합니다.")
    
    # 결과 저장
    if args.save_results and results_dict:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print_success(f"\n결과가 저장되었습니다: {args.save_results}")
    
    if passed == total:
        print_success("모든 테스트 통과!")
        sys.exit(0)
    else:
        print_error(f"{total - passed}개 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
