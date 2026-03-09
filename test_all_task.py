"""
RunPod Pod / 로컬 FastAPI 서버 API 전체 기능 통합 테스트 스크립트 (다중 모델 지원)

이 스크립트는 RunPod Pod 또는 로컬에서 실행 중인 FastAPI 서버(src 기반)를 대상으로
HTTP API 호출만으로 테스트를 수행합니다. hybrid_search/final_pipeline 등 비-src 모듈에 의존하지 않습니다.

테스트 대상 API:
    - 감성 분석: /api/v1/sentiment/analyze, /api/v1/sentiment/analyze/batch
    - 요약: /api/v1/llm/summarize, /api/v1/llm/summarize/batch (서버 SPARK_SERVICE_URL 설정 시 recall seeds·전체 평균은 Spark MSA HTTP 호출)
    - 비교: /api/v1/llm/comparison, /api/v1/llm/comparison/batch (Kiwi+lift. 서버 SPARK_SERVICE_URL 설정 시 Spark MSA 사용)
    - 벡터: /api/v1/vector/upload (업로드만. vector/search/similar API는 제거됨)

--benchmark 시 (메트릭 + CPU + GPU 모두 활성화, 기존 동작):
    - 서버 요청 메트릭: X-Benchmark → logs/debug.log, metrics.db
    - 서버 CPU 모니터: X-Enable-CPU-Monitor → logs/cpu_usage.log
    - 서버 GPU 모니터: X-Enable-GPU-Monitor → logs/gpu_usage.log
    분리 옵션 (각각 따로 켜기):
    - --benchmark-metrics: 서버 요청 메트릭만 (X-Benchmark)
    - --benchmark-cpu: 서버 CPU 모니터만 (X-Enable-CPU-Monitor)
    - --benchmark-gpu: 서버 GPU 모니터만 (X-Enable-GPU-Monitor → logs/gpu_usage.log)

사용 예:
    # 기본 테스트 (BASE_URL 환경 변수 또는 스크립트 내 BASE_URL 확인)
    # LLM 추론: API 서버가 USE_POD_VLLM 시 기본으로 213.173.108.29:16366 (VLLM_POD_BASE_URL)로 요청
    python test_all_task.py

    # 성능 측정 모드 (메트릭·CPU 모니터링 활성화)
    python test_all_task.py --benchmark
    python test_all_task.py --benchmark --iterations 10

    # 특정 테스트만 실행
    python test_all_task.py --tests sentiment summarize comparison

    # 결과 JSON 저장
    python test_all_task.py --benchmark --save-results result.json

    # 여러 모델 비교 (8001, 8002, 8003 등에 동시 요청)
    python test_all_task.py --compare-models --models "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" \\
        --ports 8001 8002 --benchmark --save-results compare.json

    # 부하테스트
    python test_all_task.py --load-test --total-requests 500 --concurrent-users 10 --ramp-up 20

    # kr3.tsv에서 테스트 데이터 생성 후 테스트
    python test_all_task.py --generate-from-kr3 --kr3-sample 500 --kr3-restaurants 10

    # 다른 서버 URL 지정
    python test_all_task.py --base-url http://192.168.1.100:8001

주요 옵션:
    --benchmark         성능 측정 전체 (메트릭 + CPU + GPU, 기존 동작)
    --benchmark-metrics 서버 요청 메트릭만 (X-Benchmark)
    --benchmark-cpu     서버 CPU 모니터만 (X-Enable-CPU-Monitor)
    --benchmark-gpu     서버 GPU 모니터만 (logs/gpu_usage.log)
    --iterations N      벤치마크 반복 횟수 (기본 5)
    --tests T1 [T2...]  실행할 테스트: all|sentiment|sentiment_batch|summarize|summarize_batch|comparison|comparison_batch|vector (vector=업로드만)
    --save-results PATH 결과 JSON 저장 경로
    --provider P        LLM 제공자: openai|local|runpod
    --model M           테스트할 LLM 모델명
    --compare-models    여러 모델 비교 모드 (--models, --ports와 함께)
    --load-test         부하테스트 (처리량/지연 측정). --total-requests, --concurrent-users, --ramp-up
    --load-test-data    부하테스트용 대형 JSON (예: real_service_simul_review_data_640k.json). 미지정 시 기본 테스트 데이터
    --generate-from-kr3 kr3.tsv 기반 테스트 데이터 생성 (--kr3-sample, --kr3-restaurants)
    --test-data-max-reviews N  test_data_sample.json 로드 시 최대 리뷰 수 제한 (예: 2000)
    --base-url URL             API 서버 베이스 URL (예: http://192.168.1.100:8001)

측정 지표 (--benchmark / QUANTITATIVE_METRICS.md):
    성능: 처리 시간(평균/P95/P99), TTFT, TPS, 처리량(req/s)
    정확도: BLEU/ROUGE(요약), Precision@K(비교), MAE(감성)
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
try:
    import psutil
except ImportError:
    psutil = None  # optional: CPU/메모리 통계는 생략
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.metrics_collector import MetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    METRICS_COLLECTOR_AVAILABLE = False

# --save-results model_info: env 미설정 시 서버 기본값 표시 (src/config.py와 동일)
try:
    from src.config import (
        DEFAULT_LLM_MODEL,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_SENTIMENT_MODEL,
        DEFAULT_SPARSE_EMBEDDING_MODEL,
    )
    SERVER_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # config.Config.OPENAI_MODEL 기본값
except ImportError:
    DEFAULT_LLM_MODEL = ""
    DEFAULT_EMBEDDING_MODEL = ""
    DEFAULT_SENTIMENT_MODEL = ""
    DEFAULT_SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
    SERVER_DEFAULT_OPENAI_MODEL = ""

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
# BASE_URL: 테스트가 요청을 보내는 API 서버 주소 (FastAPI 앱).
# LLM 추론은 API 서버가 수행하며, RunPod Pod 사용 시 API 서버가 VLLM_POD_BASE_URL(기본 http://213.173.108.29:16366/v1)로 요청함.
#BASE_URL = "http://213.192.2.74:40162"  # RunPod Pod IP:포트로 변경 (예: http://213.192.2.68:40183)
BASE_URL = "http://localhost:8001"
# 스레드별 base_url (compare_models 병렬 실행 시 포트별로 구분)
_thread_local = threading.local()

def get_base_url() -> str:
    """현재 스레드의 base_url이 있으면 반환, 없으면 전역 BASE_URL 반환"""
    return getattr(_thread_local, "base_url", None) or BASE_URL

API_PREFIX = "/api/v1"
METRICS_DB_PATH = "metrics.db"

# --benchmark* 시 서버 헤더: X-Benchmark(요청 메트릭), X-Enable-CPU-Monitor(CPU), X-Enable-GPU-Monitor(GPU). main()에서 설정.
BENCHMARK_HEADERS: Dict[str, str] = {}

# 샘플 데이터 (데이터 생성 후 업데이트됨)
SAMPLE_RESTAURANT_ID = 1
SAMPLE_REVIEWS = []

# 비교: comparison_in_aspect와 맞추기 위해 restaurant_id=4 우선 사용 (test_data_sample에 4가 있으면)
STRENGTH_TARGET_RESTAURANT_ID: Optional[int] = None

# 로드된 테스트 데이터의 레스토랑 ID 목록 (배치 테스트에서 사용, 미설정 시 SAMPLE_RESTAURANT_ID + i 폴백)
BATCH_RESTAURANT_IDS: List[int] = []

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


def get_request_headers() -> Dict[str, str]:
    """API 요청 시 사용할 헤더 (X-Benchmark=요청 메트릭, X-Enable-CPU-Monitor=CPU, X-Enable-GPU-Monitor=GPU)"""
    return dict(BENCHMARK_HEADERS)


def check_server_health():
    """서버 헬스 체크 (RunPod Pod 서버용)"""
    try:
        start_time = time.time()
        response = requests.get(f"{get_base_url()}/health", timeout=10, headers=get_request_headers())
        elapsed_time = time.time() - start_time
        result = safe_json_response(response, "헬스 체크 실패")
        if result:
            print_success(f"서버 연결 성공: {result}")
            print_info(f"   ⏱️ 응답 시간: {elapsed_time:.2f}초")
            print_info(f"   서버 URL: {get_base_url()}")
            return True
        else:
            print_error("헬스 체크 실패: 응답을 파싱할 수 없습니다")
            return False
    except Exception as e:
        print_error(f"헬스 체크 실패: {e}")
        print_info(f"서버 URL: {get_base_url()}")
        print_info("서버가 실행 중인지 확인하세요 (RunPod Pod에서 FastAPI 서버 확인)")
        return False


def generate_test_data(
    generate_from_kr3: bool = False,
    kr3_sample: Optional[int] = None,
    kr3_restaurants: Optional[int] = None,
    max_reviews: Optional[int] = None,
):
    """
    테스트 데이터 로드 또는 생성
    
    Args:
        generate_from_kr3: kr3.tsv에서 데이터 생성 여부
        kr3_sample: kr3.tsv에서 샘플링할 리뷰 수
        kr3_restaurants: 생성할 레스토랑 수
        max_reviews: test_data_sample.json 로드 시 사용할 최대 리뷰 수 (미지정 시 전체)
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
        
        # 데이터 형식 확인 및 변환
        # {"reviews": [...]} 형식 (test_data_sample.json): reviews가 있으면 레스토랑별로 그룹화하여 사용
        # (파일에 "restaurants"만 있고 리뷰가 없을 수 있으므로, reviews 우선 사용)
        if isinstance(data, dict) and data.get("reviews"):
            review_list = data["reviews"]
            print_info("reviews 배열 형식 데이터 감지, 레스토랑별로 그룹화 중...")
            restaurants_dict = {}
            for review in review_list:
                if not isinstance(review, dict):
                    continue
                restaurant_id = review.get('restaurant_id')
                if restaurant_id is None:
                    continue
                restaurant_id_str = str(restaurant_id)
                if restaurant_id_str not in restaurants_dict:
                    restaurants_dict[restaurant_id_str] = {
                        'restaurant_id': restaurant_id,
                        'restaurant_name': review.get('restaurant_name', f'Restaurant {restaurant_id}'),
                        'food_category_id': review.get('food_category_id'),
                        'food_category_name': review.get('food_category_name'),
                        'reviews': []
                    }
                review_id = review.get('id')
                if review_id is None:
                    review_id = hash((restaurant_id, review.get('content', ''))) % (10 ** 9) or 1
                review_data = {
                    'id': review_id,
                    'restaurant_id': review.get('restaurant_id'),
                    'content': review.get('content', ''),
                    'created_at': review.get('created_at') or datetime.now().isoformat(),
                }
                restaurants_dict[restaurant_id_str]['reviews'].append(review_data)
            restaurants_list = list(restaurants_dict.values())
            # 기존 data["restaurants"]에 id/name이 있으면 이름 매칭
            name_by_id = {}
            for r in (data.get("restaurants") or []):
                rid = r.get("id") or r.get("restaurant_id")
                if rid is not None and r.get("name"):
                    name_by_id[int(rid) if isinstance(rid, (int, str)) and str(rid).isdigit() else rid] = r["name"]
            for r in restaurants_list:
                rid = r.get("restaurant_id")
                if rid in name_by_id:
                    r["restaurant_name"] = name_by_id[rid]
            data = {'restaurants': restaurants_list}
            print_info(f"  - {len(data['restaurants'])}개 레스토랑으로 그룹화 완료")
            if max_reviews is not None and max_reviews > 0:
                data = _trim_load_test_data_to_max_reviews(data, max_reviews)
                total_after = sum(len(r.get("reviews") or []) for r in data.get("restaurants", []))
                print_info(f"  - 테스트 데이터 제한: 최대 {max_reviews}건 리뷰 사용 (실제 {total_after}건)")
        elif isinstance(data, list):
            # 리스트 형식: 리뷰 리스트를 레스토랑별로 그룹화
            print_info("리스트 형식 데이터 감지, 레스토랑별로 그룹화 중...")
            restaurants_dict = {}
            for review in data:
                if not isinstance(review, dict):
                    continue
                restaurant_id = review.get('restaurant_id')
                if restaurant_id is None:
                    continue
                
                restaurant_id_str = str(restaurant_id)
                if restaurant_id_str not in restaurants_dict:
                    restaurants_dict[restaurant_id_str] = {
                        'restaurant_id': restaurant_id,
                        'restaurant_name': review.get('restaurant_name', f'Restaurant {restaurant_id}'),
                        'food_category_id': review.get('food_category_id'),
                        'food_category_name': review.get('food_category_name'),
                        'reviews': []
                    }
                
                # 리뷰 추가 (Qdrant 업로드 시 point_id 구분을 위해 id 필수, created_at 필수)
                review_id = review.get('id')
                if review_id is None:
                    review_id = hash((restaurant_id, review.get('content', ''))) % (10 ** 9) or 1
                review_data = {
                    'id': review_id,
                    'restaurant_id': review.get('restaurant_id'),
                    'content': review.get('content', ''),
                    'created_at': review.get('created_at') or datetime.now().isoformat(),
                }
                restaurants_dict[restaurant_id_str]['reviews'].append(review_data)
            
            # 딕셔너리 형식으로 변환
            data = {
                'restaurants': list(restaurants_dict.values())
            }
            print_info(f"  - {len(data['restaurants'])}개 레스토랑으로 그룹화 완료")
            if max_reviews is not None and max_reviews > 0:
                data = _trim_load_test_data_to_max_reviews(data, max_reviews)
                total_after = sum(len(r.get("reviews") or []) for r in data.get("restaurants", []))
                print_info(f"  - 테스트 데이터 제한: 최대 {max_reviews}건 리뷰 사용 (실제 {total_after}건)")
        
        # 딕셔너리 형식 처리 (nested만 있는 파일은 여기서 max_reviews 적용)
        if isinstance(data, dict):
            if max_reviews is not None and max_reviews > 0 and data.get("restaurants") and not data.get("reviews"):
                # nested만 있고 아직 trim 안 된 경우 (파일에 "restaurants"만 있을 때)
                data = _trim_load_test_data_to_max_reviews(data, max_reviews)
            restaurants_count = len(data.get('restaurants', []))
            print_success(f"테스트 데이터 로드 완료: {restaurants_count}개 레스토랑")
            
            # 총 리뷰 수 계산
            total_reviews = sum(
                len(restaurant.get('reviews', []))
                for restaurant in data.get('restaurants', [])
            )
            print_info(f"  - 총 리뷰 수: {total_reviews}개")
            
            # 전역 변수 업데이트
            global SAMPLE_RESTAURANT_ID, SAMPLE_REVIEWS, STRENGTH_TARGET_RESTAURANT_ID
            if data.get('restaurants'):
                first_restaurant = data['restaurants'][0]
                SAMPLE_RESTAURANT_ID = first_restaurant.get('restaurant_id', 1)
                # comparison_in_aspect와 맞추기: restaurant_id=4가 있으면 비교에서 4 사용
                STRENGTH_TARGET_RESTAURANT_ID = 4 if any((r.get('restaurant_id') or 0) == 4 for r in data.get('restaurants', [])) else None
                # ReviewModel 형식으로 변환
                SAMPLE_REVIEWS = []
                for i, review in enumerate(first_restaurant.get('reviews', [])):
                    if isinstance(review, dict) and review.get('content'):
                        review_obj = {
                            'id': review.get('id') or (i + 1),
                            'restaurant_id': review.get('restaurant_id', SAMPLE_RESTAURANT_ID),
                            'content': review.get('content', ''),
                            'created_at': review.get('created_at') or datetime.now().isoformat(),
                        }
                        SAMPLE_REVIEWS.append(review_obj)
                print_info(f"  - 샘플 레스토랑 ID: {SAMPLE_RESTAURANT_ID}")
                print_info(f"  - 샘플 리뷰 수: {len(SAMPLE_REVIEWS)}개")
            
            # 임시 파일 경로는 None 반환 (더 이상 필요 없음)
            return data, None
        else:
            print_error(f"지원하지 않는 데이터 형식: {type(data)}")
            return None
        
    except json.JSONDecodeError as e:
        print_error(f"JSON 파일 파싱 오류: {str(e)}")
        return None
    except Exception as e:
        print_error(f"테스트 데이터 로드 중 오류: {str(e)}")
        import traceback
        print_error(f"상세 오류: {traceback.format_exc()}")
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
    
    url = f"{get_base_url()}{API_PREFIX}/vector/upload"
    
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
        response = requests.post(url, json=payload, timeout=300, headers=get_request_headers())  # 대용량 데이터를 위해 타임아웃 증가
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


def _trim_load_test_data_to_max_reviews(data: Dict[str, Any], max_reviews: int) -> Dict[str, Any]:
    """
    load_test_data를 최대 max_reviews건 리뷰만 남기도록 잘라서 반환.
    레스토랑당 10개씩 채우고, 나머지는 마지막 레스토랑에 할당.
    예: max_reviews=205 → 20개 레스토랑은 10개씩, 1개 레스토랑은 5개 → 총 21개 레스토랑.
    - Nested: 앞에서부터 레스토랑당 10개(마지막만 1~10개).
    - Flat: restaurant_id별 그룹 후 동일.
    """
    if not data or max_reviews <= 0:
        return data or {}
    # 사용할 레스토랑 수 = ceil(max_reviews/10), 단 기존 레스토랑 수를 넘지 않음
    num_restaurants_cap = (max_reviews + 9) // 10
    # Nested: {"restaurants": [{restaurant_id, reviews: [...]}, ...]}
    restaurants = data.get("restaurants")
    if restaurants is not None and (not data.get("reviews")):
        n_rest = len(restaurants)
        num_restaurants = min(num_restaurants_cap, n_rest)
        new_restaurants: List[Dict[str, Any]] = []
        for i in range(num_restaurants):
            r = restaurants[i]
            quota = 10 if i < num_restaurants - 1 else (max_reviews - 10 * (num_restaurants - 1))
            reviews = r.get("reviews") or []
            take = min(quota, len(reviews))
            new_reviews = reviews[:take]
            if new_reviews:
                new_restaurants.append({
                    **r,
                    "reviews": new_reviews,
                })
        return {"restaurants": new_restaurants}
    # Flat: {"reviews": [...], "restaurants": [...]} — 레스토랑당 10개씩 채우기
    flat_reviews = data.get("reviews") or []
    flat_restaurants = data.get("restaurants") or []
    if flat_reviews and flat_restaurants:
        by_rid_flat: Dict[Any, List[Dict[str, Any]]] = {}
        for rev in flat_reviews:
            rid = rev.get("restaurant_id")
            if rid is not None:
                if rid not in by_rid_flat:
                    by_rid_flat[rid] = []
                by_rid_flat[rid].append(rev)
        n_rest = len(by_rid_flat)
        num_restaurants = min(num_restaurants_cap, n_rest)
        trimmed_reviews: List[Dict[str, Any]] = []
        for i, (rid, revs) in enumerate(by_rid_flat.items()):
            if i >= num_restaurants:
                break
            quota = 10 if i < num_restaurants - 1 else (max_reviews - 10 * (num_restaurants - 1))
            take = min(quota, len(revs))
            trimmed_reviews.extend(revs[:take])
        trimmed_reviews = trimmed_reviews[:max_reviews]
        rids_in_use = {r.get("restaurant_id") for r in trimmed_reviews}
        filtered_restaurants = []
        for r in flat_restaurants:
            rid = r.get("id") or r.get("restaurant_id")
            if rid is not None:
                rid = int(rid) if isinstance(rid, str) and str(rid).isdigit() else rid
                if rid in rids_in_use:
                    filtered_restaurants.append(r)
        # 감성 배치용으로 nested 형태로 변환: restaurant_id별로 리뷰 그룹
        by_rid: Dict[Any, List[Dict[str, Any]]] = {}
        for rev in trimmed_reviews:
            rid = rev.get("restaurant_id")
            if rid not in by_rid:
                by_rid[rid] = []
            by_rid[rid].append(rev)
        new_restaurants = []
        for r in filtered_restaurants:
            rid = r.get("id") or r.get("restaurant_id")
            if rid is not None:
                rid = int(rid) if isinstance(rid, str) and str(rid).isdigit() else rid
                revs = by_rid.get(rid, [])
                if revs:
                    new_restaurants.append({
                        "restaurant_id": rid,
                        "restaurant_name": r.get("name") or r.get("restaurant_name") or f"Restaurant {rid}",
                        "reviews": revs,
                    })
        return {"restaurants": new_restaurants}
    return data


def _build_upload_payload_from_load_test_data(
    data: Dict[str, Any],
    max_reviews: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    load_test_data 형식을 벡터 업로드 API payload로 변환.
    지원 형식:
      - {"restaurants": [{restaurant_id, reviews: [...]}]}  (nested)
      - {"reviews": [...], "restaurants": [{id, name}]}    (flat, run_all_restaurants_api 형식)
    max_reviews: 지정 시 업로드할 최대 리뷰 수 (64만 건 중 5000건만 등). None이면 전체.
    Returns: {"reviews": [...], "restaurants": [...]} 또는 None
    """
    if not data:
        return None
    # Flat 형식: reviews + restaurants (run_all_restaurants_api 형식)
    flat_reviews = data.get("reviews") or []
    flat_restaurants = data.get("restaurants") or []
    if flat_reviews and flat_restaurants:
        reviews_ok = []
        for r in flat_reviews:
            if not isinstance(r, dict) or not r.get("content"):
                continue
            reviews_ok.append({
                "id": r.get("id"),
                "restaurant_id": r.get("restaurant_id"),
                "content": r.get("content", ""),
                "created_at": r.get("created_at") or "",
            })
            if max_reviews is not None and len(reviews_ok) >= max_reviews:
                reviews_ok = reviews_ok[:max_reviews]
                break
        if reviews_ok:
            rids_in_use = {r.get("restaurant_id") for r in reviews_ok}
            rests = []
            for r in flat_restaurants:
                rid = r.get("id") or r.get("restaurant_id")
                if rid is not None:
                    rid = int(rid) if isinstance(rid, str) and str(rid).isdigit() else rid
                    if rid in rids_in_use:
                        rests.append({"id": rid, "name": r.get("name", "")})
            if rests:
                return {"reviews": reviews_ok, "restaurants": rests}
    # Nested 형식: restaurants 내 reviews 포함
    all_reviews: List[Dict[str, Any]] = []
    all_restaurants: List[Dict[str, Any]] = []
    for r in data.get("restaurants") or []:
        if max_reviews is not None and len(all_reviews) >= max_reviews:
            break
        rid = r.get("restaurant_id")
        if rid is None:
            continue
        rid = int(rid) if isinstance(rid, str) and str(rid).isdigit() else rid
        rest_info = {
            "id": rid,
            "name": r.get("restaurant_name") or r.get("name") or f"Restaurant {rid}",
        }
        added = 0
        for rev in r.get("reviews") or []:
            if max_reviews is not None and len(all_reviews) >= max_reviews:
                break
            if isinstance(rev, dict) and rev.get("content") is not None:
                rev_copy = {
                    "id": rev.get("id"),
                    "restaurant_id": rev.get("restaurant_id", rid),
                    "content": rev.get("content", ""),
                    "created_at": rev.get("created_at") or "",
                }
                all_reviews.append(rev_copy)
                added += 1
        if added:
            all_restaurants.append(rest_info)
    if max_reviews is not None and len(all_reviews) > max_reviews:
        all_reviews = all_reviews[:max_reviews]
    if not all_reviews:
        return None
    return {"reviews": all_reviews, "restaurants": all_restaurants}


def upload_load_test_data_to_ports(
    load_test_data: Dict[str, Any],
    ports: Optional[List[int]],
    timeout: int = 3600,
    max_reviews: Optional[int] = None,
) -> bool:
    """
    부하테스트용 데이터를 벡터 DB에 업로드.
    ports가 있으면 각 포트에, 없으면 get_base_url()에 1회 업로드.
    max_reviews: 지정 시 업로드할 최대 리뷰 수 (64만 건 중 5000건만 등). None이면 전체.
    Returns: 모두 성공 시 True, 하나라도 실패 시 False
    """
    payload = _build_upload_payload_from_load_test_data(load_test_data, max_reviews=max_reviews)
    if not payload:
        print_warning("업로드할 리뷰가 없습니다.")
        return False
    n_reviews = len(payload["reviews"])
    n_restaurants = len(payload["restaurants"])
    if ports:
        urls = [f"http://localhost:{p}" for p in ports]
        print_header(f"벡터 DB 업로드 (리뷰 {n_reviews}개, 레스토랑 {n_restaurants}개) → {len(urls)}개 포트")
    else:
        urls = [get_base_url()]
        print_header(f"벡터 DB 업로드 (리뷰 {n_reviews}개, 레스토랑 {n_restaurants}개)")
    all_ok = True
    for url in urls:
        full_url = f"{url.rstrip('/')}{API_PREFIX}/vector/upload"
        try:
            start = time.time()
            resp = requests.post(full_url, json=payload, timeout=timeout, headers=get_request_headers())
            elapsed = time.time() - start
            if resp.status_code == 200:
                data = safe_json_response(resp, "")
                pts = data.get("points_count", 0) if data else 0
                print_success(f"  {url} 업로드 완료: {pts}개 포인트 ({elapsed:.1f}초)")
            else:
                print_warning(f"  {url} 업로드 실패: {resp.status_code}")
                all_ok = False
        except Exception as e:
            print_warning(f"  {url} 업로드 오류: {e}")
            all_ok = False
    return all_ok


# 배치 테스트별 레스토랑 수 (--save-results data_info용). load_test 시나리오 없을 때 요약/비교도 10개 전체 사용
BATCH_RESTAURANTS = {"sentiment_batch": 10, "summarize_batch": 10, "comparison_batch": 10}


def _effective_model_info() -> Dict[str, str]:
    """--save-results용 model_info. env 미설정 시 서버 기본값(src/config.py) 사용."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    if provider == "openai":
        llm = (os.getenv("OPENAI_MODEL") or "").strip() or SERVER_DEFAULT_OPENAI_MODEL
    else:
        llm = (os.getenv("LLM_MODEL") or "").strip() or DEFAULT_LLM_MODEL
    dense_emb = (os.getenv("EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL
    sparse_emb = (os.getenv("SPARSE_EMBEDDING_MODEL") or "").strip() or DEFAULT_SPARSE_EMBEDDING_MODEL
    return {
        "llm_provider": provider,
        "llm_model": llm,
        "dense_embedding_model": dense_emb,
        "sparse_embedding_model": sparse_emb,
        "sentiment_model": (os.getenv("SENTIMENT_MODEL") or "").strip() or DEFAULT_SENTIMENT_MODEL,
    }


def build_data_info(
    test_data: Optional[Dict[str, Any]] = None,
    data_source_name: str = "test_data_sample.json",
    generate_from_kr3: bool = False,
    kr3_sample: Optional[int] = None,
    kr3_restaurants: Optional[int] = None,
    selected_tests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    --save-results JSON용 데이터 정보 구성.
    데이터 이름·규모, 단일/배치 구분, 배치 시 처리 레스토랑 수를 반환한다.
    """
    data_scale: Dict[str, Any] = {"restaurants": 0, "total_reviews": 0}
    if test_data and isinstance(test_data, dict) and test_data.get("restaurants"):
        data_scale["restaurants"] = len(test_data["restaurants"])
        data_scale["total_reviews"] = sum(
            len(r.get("reviews", [])) for r in test_data["restaurants"]
        )
    if generate_from_kr3:
        if kr3_sample is not None:
            data_scale["kr3_sample"] = kr3_sample
        if kr3_restaurants is not None:
            data_scale["kr3_restaurants"] = kr3_restaurants

    single_tests = {"sentiment", "summarize", "comparison", "vector"}
    batch_tests = {"sentiment_batch", "summarize_batch", "comparison_batch"}
    tests = selected_tests or []
    has_single = any(t in single_tests for t in tests)
    has_batch = any(t in batch_tests for t in tests)
    if has_batch and not has_single:
        processing_mode = "batch"
    elif has_single and not has_batch:
        processing_mode = "single"
    else:
        processing_mode = "mixed"

    restaurants_processed_in_batch: Dict[str, int] = {}
    for t in tests:
        if t in BATCH_RESTAURANTS:
            restaurants_processed_in_batch[t] = BATCH_RESTAURANTS[t]

    return {
        "data_source": "kr3.tsv" if generate_from_kr3 else data_source_name,
        "data_scale": data_scale,
        "processing_mode": processing_mode,
        "restaurants_processed_in_batch": restaurants_processed_in_batch or None,
    }


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
        url = f"{get_base_url()}{endpoint}"
    latencies = []
    success_count = 0
    error_count = 0
    error_4xx_count = 0
    error_5xx_count = 0
    status_codes = []

    # CPU/메모리 메트릭 수집 시작
    if psutil:
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory()
    else:
        cpu_before, mem_before = 0, None

    # 워밍업
    for i in range(warmup_iterations):
        try:
            requests.post(url, json=payload, timeout=timeout, headers=get_request_headers())
        except Exception:
            pass

    # 실제 측정
    measurement_start_time = time.perf_counter()
    last_successful_response = None  # 정확도 평가를 위해 마지막 성공 응답 저장
    for i in range(num_iterations):
        try:
            start_time = time.perf_counter()
            response = requests.post(url, json=payload, timeout=timeout, headers=get_request_headers())
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
    if psutil:
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory()
    else:
        cpu_after, mem_after = None, None

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
    
    return True, stats


def load_test(
    endpoint: str,
    payload: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]],
    total_requests: int = 100,
    concurrent_users: int = 10,
    timeout: int = 60,
    ramp_up_seconds: int = 0,
    consecutive_connection_errors_abort: int = 10,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    부하테스트: payload를 반복 전송하여 처리량(req/s)과 지연(P50/P95/P99)을 측정합니다.
    서버 다운 시 연속 연결 실패가 consecutive_connection_errors_abort회 이상이면 조기 종료하고
    그때까지의 결과를 부분 통계로 반환합니다.

    Args:
        endpoint: API 엔드포인트 경로 (또는 전체 URL)
        payload: 요청 페이로드 (고정 dict 또는 (request_index) -> dict callable)
        total_requests: 총 요청 수
        concurrent_users: 동시 사용자 수
        timeout: 요청 타임아웃 (초)
        ramp_up_seconds: 점진적 부하 증가 시간 (초)
        consecutive_connection_errors_abort: 연속 연결 실패 N회 시 조기 종료 (0이면 비활성화)

    Returns:
        (성공 여부, 부하테스트 통계 딕셔너리. 조기 종료 시에도 attempted 수 기준 부분 통계 반환)
    """
    if endpoint.startswith(("http://", "https://")):
        url = endpoint
    else:
        url = f"{get_base_url()}{endpoint}"

    latencies: List[float] = []
    success_count = 0
    error_count = 0
    error_4xx_count = 0
    error_5xx_count = 0
    status_codes: List[int] = []
    request_timestamps: List[float] = []

    def make_request(request_id: int, req_payload: Dict[str, Any]) -> Tuple[int, float, int, Optional[Dict[str, Any]], bool]:
        """단일 요청 실행. 반환: (request_id, latency, status_code, result, is_connection_error)."""
        try:
            start_time = time.perf_counter()
            response = requests.post(url, json=req_payload, timeout=timeout, headers=get_request_headers())
            end_time = time.perf_counter()
            latency = end_time - start_time
            status_code = response.status_code
            result = None
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    pass
            return request_id, latency, status_code, result, False
        except Exception as e:
            is_conn = isinstance(e, (
                requests.exceptions.ConnectionError,
                getattr(requests.exceptions, "ConnectTimeout", type(None)) or type(None),
                getattr(requests.exceptions, "ReadTimeout", type(None)) or type(None),
            ))
            if not is_conn and type(e).__name__ in ("ConnectTimeout", "ReadTimeout"):
                is_conn = "requests" in (getattr(type(e), "__module__", "") or "")
            return request_id, -1, 0, None, bool(is_conn)

    if psutil:
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory()
    else:
        cpu_before, mem_before = 0, None

    print_info(f"부하테스트 시작: 총 {total_requests}개 요청, 동시 사용자 {concurrent_users}명")
    if ramp_up_seconds > 0:
        print_info(f"점진적 부하 증가: {ramp_up_seconds}초 동안 부하 증가")
    if consecutive_connection_errors_abort > 0:
        print_info(f"서버 다운 감지 시 연속 {consecutive_connection_errors_abort}회 연결 실패 후 조기 종료")

    test_start_time = time.perf_counter()
    abort = False
    consecutive_connection_errors = 0
    attempted_count = 0

    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures: Dict[Any, int] = {}
        next_i = 0

        while next_i < total_requests and not abort:
            while len(futures) < concurrent_users and next_i < total_requests:
                if ramp_up_seconds > 0:
                    delay = (ramp_up_seconds / total_requests) * next_i
                    if delay > 0:
                        time.sleep(delay)
                req_payload = payload(next_i) if callable(payload) else payload
                fut = executor.submit(make_request, next_i, req_payload)
                futures[fut] = next_i
                next_i += 1
            if not futures:
                break
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for f in done:
                i = futures.pop(f)
                attempted_count += 1
                try:
                    request_id, latency, status_code, result, is_connection_error = f.result()
                    request_timestamps.append(time.perf_counter())
                    if is_connection_error:
                        consecutive_connection_errors += 1
                    else:
                        consecutive_connection_errors = 0

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
                except Exception:
                    error_count += 1
                    consecutive_connection_errors += 1

                if consecutive_connection_errors_abort > 0 and consecutive_connection_errors >= consecutive_connection_errors_abort:
                    abort = True
                    print_warning(f"서버 다운 감지(연속 연결 실패 {consecutive_connection_errors}회). 조기 종료. 시도한 요청: {attempted_count}/{total_requests}")
                    break
            if abort:
                break

        if abort and futures:
            for f in as_completed(futures):
                attempted_count += 1
                try:
                    request_id, latency, status_code, result, is_connection_error = f.result()
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
                except Exception:
                    error_count += 1

    test_end_time = time.perf_counter()

    # CPU/메모리 메트릭 수집 종료
    if psutil:
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory()
    else:
        cpu_after, mem_after = None, None

    total_time = test_end_time - test_start_time
    effective_total = attempted_count if attempted_count > 0 else total_requests

    if attempted_count == 0:
        print_error("부하테스트 실패: 시도한 요청이 없습니다.")
        return False, None

    if not latencies:
        print_error("부하테스트: 성공한 요청이 없습니다 (모두 실패 또는 조기 종료).")
        if status_codes:
            print_info(f"  상태 코드 분포: {status_codes}")
        print_info(f"  시도/실패: {attempted_count}, 4xx: {error_4xx_count}, 5xx: {error_5xx_count}")
        if abort:
            print_info("  부분 결과만 저장됩니다.")

    throughput_req_per_sec = len(latencies) / total_time if total_time > 0 else 0
    if len(request_timestamps) > 1:
        intervals = [request_timestamps[i] - request_timestamps[i-1] for i in range(1, len(request_timestamps))]
        avg_interval = statistics.mean(intervals) if intervals else 0
        actual_rps = 1.0 / avg_interval if avg_interval > 0 else 0
    else:
        actual_rps = throughput_req_per_sec

    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50_latency = calculate_percentile(latencies, 50)
        p95_latency = calculate_percentile(latencies, 95)
        p99_latency = calculate_percentile(latencies, 99)
    else:
        avg_latency = min_latency = max_latency = p50_latency = p95_latency = p99_latency = 0.0

    if len(request_timestamps) > 1:
        time_window = 1.0
        max_concurrent = 0
        for ts in request_timestamps:
            window_end = ts + time_window
            concurrent_count = sum(1 for t in request_timestamps if ts <= t < window_end)
            max_concurrent = max(max_concurrent, concurrent_count)
    else:
        max_concurrent = 1

    stats = {
        "total_requests": effective_total,
        "attempted_requests": attempted_count,
        "concurrent_users": concurrent_users,
        "success_count": success_count,
        "error_count": error_count,
        "error_4xx_count": error_4xx_count,
        "error_5xx_count": error_5xx_count,
        "success_rate": (success_count / effective_total) * 100 if effective_total > 0 else 0,
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
        "aborted_early": abort,
    }

    if cpu_after is not None:
        stats["cpu_usage_percent"] = cpu_after
    if mem_after is not None:
        stats["memory_usage_percent"] = mem_after.percent
        stats["memory_used_mb"] = mem_after.used / (1024 ** 2)
        stats["memory_total_mb"] = mem_after.total / (1024 ** 2)

    return True, stats


def fetch_prometheus_metrics(
    prometheus_url: str,
    start_ts: float,
    end_ts: float,
    step: str = "15s",
) -> Dict[str, Any]:
    """
    부하테스트 구간(start_ts ~ end_ts)에 해당하는 Prometheus 메트릭을 query_range로 조회해
    JSON 저장용 딕셔너리로 반환합니다. 실패 시 빈 dict 또는 부분 결과 반환.
    """
    base = prometheus_url.rstrip("/")
    api = f"{base}/api/v1/query_range"
    out = {"time_range": {"start": start_ts, "end": end_ts}, "queries": {}}
    queries = [
        ("ttfur_p95", "histogram_quantile(0.95, sum by (le, pipeline, analysis_type) (rate(llm_ttft_seconds_bucket{job=\"fastapi\"}[5m])))"),
        ("analysis_success_rate", "sum by (pipeline) (rate(analysis_requests_total{job=\"fastapi\",status=\"success\"}[5m])) / (sum by (pipeline) (rate(analysis_requests_total{job=\"fastapi\"}[5m])) + 1e-9)"),
        ("completion_time_p95", "histogram_quantile(0.95, sum by (le, pipeline, analysis_type) (rate(analysis_processing_time_seconds_bucket{job=\"fastapi\"}[5m])))"),
        ("completed_jobs_per_sec", "sum by (pipeline, analysis_type) (rate(analysis_requests_total{job=\"fastapi\",status=\"success\"}[5m]))"),
        ("queue_depth", "app_queue_depth{job=\"fastapi\"}"),
        ("worker_utilization", "sum by (pipeline) (app_worker_busy{job=\"fastapi\"}) / (count by (pipeline) (app_worker_busy{job=\"fastapi\"}) + 1e-9)"),
    ]
    for name, query in queries:
        try:
            r = requests.get(api, params={"query": query, "start": start_ts, "end": end_ts, "step": step}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "success" and "data" in data:
                    out["queries"][name] = {"query": query, "data": data["data"]}
                else:
                    out["queries"][name] = {"query": query, "error": data.get("error", "unknown")}
            else:
                out["queries"][name] = {"query": query, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            out["queries"][name] = {"query": query, "error": str(e)}
    return out


def save_container_logs_on_abort(
    results_by_port: Dict[str, Dict[str, Any]],
    log_dir: Union[str, Path],
    container_prefix: str = "tasteam-new-async",
) -> None:
    """
    results_by_port 중 하나라도 aborted_early이면, old_sync/new_sync/new_async 컨테이너의
    docker logs를 log_dir 아래 logs_rampup_old_sync.txt 등으로 저장합니다.
    """
    any_aborted = False
    for port_data in (results_by_port or {}).values():
        for st in (port_data or {}).values():
            if isinstance(st, dict) and st.get("aborted_early"):
                any_aborted = True
                break
        if any_aborted:
            break
    if not any_aborted:
        return
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    containers = [
        ("old_sync", f"{container_prefix}-old_sync-1"),
        ("new_sync", f"{container_prefix}-new_sync-1"),
        ("new_async", f"{container_prefix}-new_async-1"),
    ]
    for name, cid in containers:
        out_file = log_path / f"logs_rampup_{name}.txt"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                subprocess.run(
                    ["docker", "logs", cid],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=False,
                )
            print_info(f"컨테이너 로그 저장: {out_file}")
        except FileNotFoundError:
            print_warning(f"docker 미설치 또는 PATH 없음. {cid} 로그 저장 생략.")
            break
        except subprocess.TimeoutExpired:
            print_warning(f"docker logs {cid} 타임아웃. {out_file} (부분 저장됨)")
        except Exception as e:
            print_warning(f"docker logs {cid} 실패: {e}. {out_file} 생략.")


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
        analysis_type: 분석 타입 ('sentiment', 'summary', 'comparison')
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
                base_url=get_base_url(),
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
                base_url=get_base_url(),
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
            # 새로운 파이프라인: categories 필드 지원
            predicted_summary = api_result.get("overall_summary", "")
            if not predicted_summary and api_result.get("categories"):
                # categories에서 overall_summary 생성 시도
                categories = api_result.get("categories", {})
                summaries = []
                for cat_data in categories.values():
                    if isinstance(cat_data, dict) and cat_data.get("summary"):
                        summaries.append(cat_data["summary"])
                predicted_summary = " ".join(summaries) if summaries else ""
            
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
        
        elif analysis_type == "comparison":
            evaluator = StrengthExtractionEvaluator(
                base_url=get_base_url(),
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
            predicted_strengths = api_result.get("comparisons", api_result.get("strengths", []))
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
    """
    감성 분석 테스트 (새 파이프라인: HuggingFace 1차 분류 + LLM 재판정)
    
    입력 예시:
        {
            "restaurant_id": 1,
            "reviews": [
                {"restaurant_id": 1, "content": "맛이 정말 좋아요", ...},
                {"restaurant_id": 1, "content": "서비스가 별로였어요", ...}
            ]
        }
    
    출력 예시:
        ✓ 감성 분석 성공 (소요 시간: 2.34초)
          - 긍정 비율: 60%
          - 부정 비율: 30%
          - 중립 비율: 10%
          - 긍정 개수: 6
          - 부정 개수: 3
          - 중립 개수: 1
          - 전체 개수: 10
    """
    print_header("1. 감성 분석 테스트")
    
    url = f"{get_base_url()}{API_PREFIX}/sentiment/analyze"
    # 리뷰는 벡터 DB에서 조회 (restaurant_id만 전달)
    payload = {"restaurant_id": SAMPLE_RESTAURANT_ID}
    
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
            response = requests.post(url, json=payload, timeout=120, headers=get_request_headers())
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"감성 분석 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 긍정 비율: {data.get('positive_ratio', 'N/A')}%")
                print(f"  - 부정 비율: {data.get('negative_ratio', 'N/A')}%")
                print(f"  - 중립 비율: {data.get('neutral_ratio', 'N/A')}%")  # 새로 추가된 필드
                print(f"  - 긍정 개수: {data.get('positive_count', 'N/A')}")
                print(f"  - 부정 개수: {data.get('negative_count', 'N/A')}")
                print(f"  - 중립 개수: {data.get('neutral_count', 'N/A')}")  # 새로 추가된 필드
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
    """
    배치 감성 분석 테스트 (새 파이프라인: HuggingFace 1차 분류 + LLM 재판정)
    
    입력 예시:
        { "restaurants": [ {"restaurant_id": 1}, {"restaurant_id": 2} ] }
    (리뷰는 벡터 DB에서 조회)

    출력 예시:
        ✓ 배치 감성 분석 성공 (소요 시간: 5.67초)
          - 처리된 레스토랑 수: 2
            레스토랑 1: 긍정 60%, 부정 30%
            레스토랑 2: 긍정 70%, 부정 20%
    """
    print_header("2. 배치 감성 분석 테스트")
    
    url = f"{get_base_url()}{API_PREFIX}/sentiment/analyze/batch"
    # 로드된 테스트 데이터의 레스토랑 최대 10개 사용, 없으면 SAMPLE_RESTAURANT_ID + 0..9
    if BATCH_RESTAURANT_IDS:
        rids = BATCH_RESTAURANT_IDS[:10]
    else:
        rids = [SAMPLE_RESTAURANT_ID + i for i in range(10)]
    restaurants_payload = [{"restaurant_id": rid} for rid in rids]
    payload = {"restaurants": restaurants_payload}
    
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
            response = requests.post(url, json=payload, timeout=120, headers=get_request_headers())
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
    """
    리뷰 요약 테스트 (새 파이프라인: 카테고리별 하이브리드 검색 + 요약)
    
    입력:
        {
            "restaurant_id": 1,
            "limit": 10
        }
    
    출력:
        전체 요약 + 카테고리별 요약 (service, price, food) + 포인트/근거 개수
        
        예시:
        ✓ 리뷰 요약 성공 (소요 시간: 9.17초)
          레스토랑 1:
            * 전체 요약: 이 음식점은 분위기와 음식의 맛이 뛰어나며...
            * service: 서비스가 친절하고 응대가 빠릅니다... (포인트 3개, 근거 5개)
            * price: 가격 대비 만족스러운 경험을 제공합니다... (포인트 2개, 근거 4개)
            * food: 음식의 맛과 품질이 우수합니다... (포인트 4개, 근거 6개)
    """
    print_header("3. 리뷰 요약 테스트")
    
    url = f"{get_base_url()}{API_PREFIX}/llm/summarize"
    payload = {
        "restaurant_id": SAMPLE_RESTAURANT_ID,
        "limit": 10,
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=1000)
            
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
            # 기본 테스트 모드 (요약은 recall_seeds·하이브리드·LLM으로 120초 초과 가능)
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=300, headers=get_request_headers())
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"리뷰 요약 성공 (소요 시간: {elapsed_time:.2f}초)")
                restaurant_id = data.get('restaurant_id', 'N/A')
                overall_summary = data.get('overall_summary', 'N/A')[:60]
                categories = data.get('categories')
                
                print(f"    레스토랑 {restaurant_id}:")
                print(f"      * 전체 요약: {overall_summary}...")
                
                if categories:
                    for cat_name, cat_data in categories.items():
                        if isinstance(cat_data, dict):
                            summary = cat_data.get('summary', '')[:60] if cat_data.get('summary') else 'N/A'
                            bullets_count = len(cat_data.get('bullets', []))
                            evidence_count = len(cat_data.get('evidence', []))
                            print(f"      * {cat_name}: {summary}... (포인트 {bullets_count}개, 근거 {evidence_count}개)")
                        elif isinstance(cat_data, str):
                            # 문자열로 저장된 경우 (직렬화된 JSON)
                            print(f"      * {cat_name}: {cat_data[:50]}...")
                else:
                    print(f"      * 카테고리별 요약: 없음")
                
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
    """
    배치 리뷰 요약 테스트 (새 파이프라인: 카테고리별 하이브리드 검색 + 요약)
    
    입력:
        {
            "restaurants": [
                {"restaurant_id": 1, "limit": 10},
                {"restaurant_id": 2, "limit": 10}
            ]
        }
    
    출력:
        전체 요약 + 카테고리별 요약 (service, price, food) + 포인트/근거 개수
        
        예시:
        ✓ 배치 리뷰 요약 성공 (소요 시간: 27.56초)
          - 처리된 레스토랑 수: 2
            레스토랑 1:
              * 전체 요약: 이 음식점은 분위기와 음식의 맛이 뛰어나며...
              * service: 서비스가 친절하고 응대가 빠릅니다... (포인트 3개, 근거 5개)
              * price: 가격 대비 만족스러운 경험을 제공합니다... (포인트 2개, 근거 4개)
            레스토랑 2:
              * 전체 요약: 음식의 품질과 서비스가 우수합니다...
              * service: 직원들이 친절하고 주문 처리가 빠릅니다... (포인트 4개, 근거 6개)
    """
    print_header("4. 배치 리뷰 요약 테스트")
    
    url = f"{get_base_url()}{API_PREFIX}/llm/summarize/batch"
    # sentiment_batch와 동일: 로드된 테스트 데이터의 레스토랑 최대 10개 사용, 없으면 SAMPLE_RESTAURANT_ID + 0..9
    if BATCH_RESTAURANT_IDS:
        rids = BATCH_RESTAURANT_IDS[:10]
    else:
        rids = [SAMPLE_RESTAURANT_ID + i for i in range(10)]
    payload = {
        "restaurants": [{"restaurant_id": rid} for rid in rids],
        "limit": 10,
        "min_score": 0.0,
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=1000)
            
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
            response = requests.post(url, json=payload, timeout=400, headers=get_request_headers())
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"배치 리뷰 요약 성공 (소요 시간: {elapsed_time:.2f}초)")
                print(f"  - 처리된 레스토랑 수: {len(data.get('results', []))}")
                for result in data.get('results', []):
                    restaurant_id = result.get('restaurant_id')
                    overall_summary = result.get('overall_summary', 'N/A')[:60]
                    categories = result.get('categories')
                    
                    print(f"    레스토랑 {restaurant_id}:")
                    print(f"      * 전체 요약: {overall_summary}...")
                    
                    if categories:
                        for cat_name, cat_data in categories.items():
                            if isinstance(cat_data, dict):
                                summary = cat_data.get('summary', '')[:60] if cat_data.get('summary') else 'N/A'
                                bullets_count = len(cat_data.get('bullets', []))
                                evidence_count = len(cat_data.get('evidence', []))
                                print(f"      * {cat_name}: {summary}... (포인트 {bullets_count}개, 근거 {evidence_count}개)")
                    else:
                        print(f"      * 카테고리별 요약: 없음")
                return True
            else:
                print_error(f"배치 리뷰 요약 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"배치 리뷰 요약 중 오류: {str(e)}")
        return False


def test_comparison(enable_benchmark: bool = False, num_iterations: int = 5):
    """
    다른 음식점과의 비교 테스트 (src 파이프라인: Kiwi + service/price 비율 + lift_percentage)
    
    입력:
        {
            "restaurant_id": 1
        }
    
    출력:
        comparisons: [{ category, lift_percentage }], category_lift (카테고리별 lift %)
        예시:
        ✓ 비교 성공 (소요 시간: 3.45초)
          레스토랑 1:
            * 비교 항목 수: 2개
            * service: lift 20%
            * price: lift 15%
    """
    print_header("5. 비교 테스트 (다른 음식점들과의 비교)")
    
    url = f"{get_base_url()}{API_PREFIX}/llm/comparison"
    # comparison_in_aspect와 맞추기: test_data에 4가 있으면 4, 없으면 SAMPLE_RESTAURANT_ID
    rid = STRENGTH_TARGET_RESTAURANT_ID if STRENGTH_TARGET_RESTAURANT_ID is not None else SAMPLE_RESTAURANT_ID
    payload = {
        "restaurant_id": rid,
    }
    
    try:
        if enable_benchmark:
            # 성능 측정 모드
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=180)
            
            if success and stats:
                print_success(f"비교 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info("처리 시간 통계:")
                print(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print(f"  - P95: {stats['p95_latency_sec']:.3f}초")
                print(f"  - P99: {stats['p99_latency_sec']:.3f}초")
                if stats.get("throughput_req_per_sec"):
                    print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
                print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_iterations']})")
                
                # SQLite에서 메트릭 조회
                db_metrics = query_metrics_from_db("comparison", limit=5)
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
                    analysis_type="comparison",
                    restaurant_id=rid,
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
                test_metrics["비교"] = {
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
            response = requests.post(url, json=payload, timeout=1000, headers=get_request_headers())
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"비교 성공 (소요 시간: {elapsed_time:.2f}초)")
                restaurant_id = data.get('restaurant_id', 'N/A')
                comparisons = data.get('comparisons', data.get('strengths', []))
                
                print(f"    레스토랑 {restaurant_id}:")
                category_lift = data.get('category_lift') or {}
                print(f"      * 비교 항목 수: {len(comparisons)}개")
                if category_lift:
                    parts = [f"{k}: {v}%" for k, v in category_lift.items()]
                    print(f"      * 카테고리별 lift: {', '.join(parts)}")
                comparison_display = data.get('comparison_display', data.get('strength_display', []))
                if comparison_display:
                    print(f"      * comparison_display (전체 평균 대비):")
                    for s in comparison_display:
                        print(f"        - {s}")
                
                if comparisons:
                    for comp in comparisons:
                        category = comp.get('category', comp.get('aspect', 'N/A'))
                        lift_percentage = comp.get('lift_percentage')
                        if lift_percentage is not None:
                            print(f"      * {category}: lift {lift_percentage}%")
                        else:
                            print(f"      * {category}")
                else:
                    if category_lift:
                        print(f"      * 비교 항목(양수 lift): 없음 — 전체 평균 대비 양수인 카테고리 없음")
                    else:
                        print(f"      * 비교 항목: 없음")
                
                # 정확도 평가 (Ground Truth 비교, 기본 모드에서도 수행)
                ground_truth_path = str(project_root / "scripts" / "Ground_truth_strength.json")
                accuracy_metrics = evaluate_accuracy(
                    analysis_type="comparison",
                    restaurant_id=rid,
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
                print_error(f"비교 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"비교 중 오류: {str(e)}")
        return False


def test_comparison_batch(enable_benchmark: bool = False, num_iterations: int = 5):
    """
    다수 음식점에 대한 비교 배치 테스트 (Kiwi + lift).
    COMPARISON_BATCH_ASYNC=true면 음식점 간 병렬, false(기본값)면 순차.
    """
    print_header("5-2. 배치 비교 테스트")
    
    url = f"{get_base_url()}{API_PREFIX}/llm/comparison/batch"
    # sentiment_batch와 동일: 로드된 테스트 데이터의 레스토랑 최대 10개 사용.
    # (데이터가 없으면 STRENGTH_TARGET/SAMPLE_RESTAURANT_ID 기준으로 10개 구성)
    if BATCH_RESTAURANT_IDS:
        rids = BATCH_RESTAURANT_IDS[:10]
    else:
        base_rid = STRENGTH_TARGET_RESTAURANT_ID if STRENGTH_TARGET_RESTAURANT_ID is not None else SAMPLE_RESTAURANT_ID
        rids = [base_rid + i for i in range(10)]
    payload = {
        "restaurants": [{"restaurant_id": rid} for rid in rids],
    }
    
    try:
        if enable_benchmark:
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=600)
            if success and stats:
                print_success(f"배치 비교 성공 (평균 처리 시간: {stats['avg_latency_sec']:.2f}초)")
                print_info(f"  - 평균: {stats['avg_latency_sec']:.3f}초")
                print_info(f"  - 성공률: {stats['success_rate']:.1f}%")
                return True
            else:
                print_error("배치 비교 성능 측정 실패")
                return False
        else:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=600, headers=get_request_headers())
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print_success(f"배치 비교 성공 (소요 시간: {elapsed_time:.2f}초, {len(results)}개 레스토랑)")
                for r in results:
                    rid_out = r.get("restaurant_id", "N/A")
                    comparisons = r.get("comparisons", [])
                    category_lift = r.get("category_lift") or {}
                    print(f"    레스토랑 {rid_out}: 비교 항목 {len(comparisons)}개, {category_lift}")
                return True
            else:
                print_error(f"배치 비교 실패: {response.status_code}")
                print(f"  응답: {response.text[:200]}")
                return False
    except Exception as e:
        print_error(f"배치 비교 중 오류: {str(e)}")
        return False


def test_vector_upload(enable_benchmark: bool = False, num_iterations: int = 5):
    """벡터 업로드 테스트 (vector/search/similar API는 제거되어 업로드만 테스트)"""
    print_header("6. 벡터 업로드 테스트")
    url = f"{get_base_url()}{API_PREFIX}/vector/upload"
    payload = {
        "reviews": [
            {"restaurant_id": SAMPLE_RESTAURANT_ID, "content": "테스트 리뷰입니다.", "id": "upload-test-1"}
        ],
        "restaurants": [
            {"id": SAMPLE_RESTAURANT_ID, "name": "Test Restaurant", "full_address": None, "location": None, "created_at": None}
        ],
    }
    try:
        if enable_benchmark:
            print_info(f"성능 측정 모드: {num_iterations}회 반복 실행 중...")
            success, stats = measure_performance(url, payload, num_iterations=num_iterations, warmup_iterations=1, timeout=60)
            if success and stats:
                print_success(f"벡터 업로드 성공 (평균: {stats.get('avg_latency_sec', 0):.2f}초)")
                test_metrics["벡터 업로드"] = {"performance": stats, "sqlite_metrics": None, "accuracy": None}
                return True
            return False
        response = requests.post(url, json=payload, timeout=60, headers=get_request_headers())
        if response.status_code == 200:
            data = response.json()
            points = data.get("points_count", 0)
            print_success(f"벡터 업로드 성공 (points_count={points})")
            test_metrics["벡터 업로드"] = {"performance": {"elapsed_sec": response.elapsed.total_seconds()}, "sqlite_metrics": None, "accuracy": None}
            return True
        print_error(f"벡터 업로드 실패: {response.status_code} — {response.text[:200]}")
        return False
    except Exception as e:
        print_error(f"벡터 업로드 중 오류: {str(e)}")
        return False


def test_vector_search(enable_benchmark: bool = False, num_iterations: int = 5):
    """벡터 유사 검색 테스트 (API 제거됨. 레거시 서버용. 기본 테스트에서는 vector=업로드만 사용)"""
    print_header("6. 벡터 검색 테스트 (레거시: search/similar)")
    url = f"{get_base_url()}{API_PREFIX}/vector/search/similar"
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
                                base_url=get_base_url(),
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
            response = requests.post(url, json=payload, timeout=30, headers=get_request_headers())
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


def run_tests_for_model(
    model_name: str,
    provider: str,
    enable_benchmark: bool = False,
    iterations: int = 5,
    tests: Optional[List[str]] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    특정 모델에 대한 테스트 실행
    
    Args:
        model_name: 테스트할 모델명
        provider: LLM 제공자 ("openai", "local", "runpod")
        enable_benchmark: 성능 측정 모드 활성화 여부
        iterations: 성능 측정 반복 횟수
        base_url: 이 스레드에서 사용할 서버 URL (None이면 전역/스레드별 BASE_URL 사용)
        
    Returns:
        테스트 결과 딕셔너리
    """
    if base_url is not None:
        _thread_local.base_url = base_url
    original_provider = os.getenv("LLM_PROVIDER")
    original_model = os.getenv("OPENAI_MODEL") if provider == "openai" else os.getenv("LLM_MODEL")
    try:
        os.environ["LLM_PROVIDER"] = provider
        if provider == "openai":
            os.environ["OPENAI_MODEL"] = model_name
        else:
            os.environ["LLM_MODEL"] = model_name
        
        print_header(f"모델 테스트: {model_name} ({provider})")
        print_info(f"서버 URL: {get_base_url()}")
        
        # test_metrics 초기화 (모델별로 독립적으로 관리)
        global test_metrics
        original_test_metrics = test_metrics.copy()
        test_metrics.clear()
        
        # 테스트 실행
        selected_tests = tests or ["summarize", "summarize_batch"]
        if "all" in selected_tests:
            selected_tests = ["sentiment", "sentiment_batch", "summarize", "summarize_batch", "comparison", "comparison_batch", "vector"]

        test_registry = {
            "sentiment": ("감성 분석", lambda: test_sentiment_analysis(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "sentiment_batch": ("배치 감성 분석", lambda: test_sentiment_analysis_batch(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "summarize": ("리뷰 요약", lambda: test_summarize(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "summarize_batch": ("배치 리뷰 요약", lambda: test_summarize_batch(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "comparison": ("비교", lambda: test_comparison(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "comparison_batch": ("배치 비교", lambda: test_comparison_batch(enable_benchmark=enable_benchmark, num_iterations=iterations)),
            "vector": ("벡터 업로드", lambda: test_vector_upload(enable_benchmark=enable_benchmark, num_iterations=iterations)),
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
        # 스레드별 base_url 해제 (병렬 compare_models용)
        if base_url is not None:
            setattr(_thread_local, "base_url", None)
        # 환경 변수 복원
        if original_provider is not None:
            os.environ["LLM_PROVIDER"] = original_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)
        
        if provider == "openai":
            if original_model is not None:
                os.environ["OPENAI_MODEL"] = original_model
            else:
                os.environ.pop("OPENAI_MODEL", None)
        else:
            if original_model is not None:
                os.environ["LLM_MODEL"] = original_model
            else:
                os.environ.pop("LLM_MODEL", None)


def _run_one_model_worker(
    model_name: str,
    port: int,
    provider: str,
    enable_benchmark: bool,
    iterations: int,
    tests: Optional[List[str]],
    test_data: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """단일 모델 테스트를 스레드에서 실행 (compare_models 병렬용). (model_name, result) 반환."""
    url = f"http://localhost:{port}"
    _thread_local.base_url = url
    try:
        try:
            response = requests.get(f"{url}/health", timeout=5, headers=get_request_headers())
            if response.status_code != 200:
                return (model_name, {
                    "model_name": model_name,
                    "provider": provider,
                    "success_count": 0,
                    "total_count": 0,
                    "success_rate": 0,
                    "results": [],
                    "error": f"서버 응답 실패 (포트 {port}, status={response.status_code})",
                })
        except Exception as e:
            return (model_name, {
                "model_name": model_name,
                "provider": provider,
                "success_count": 0,
                "total_count": 0,
                "success_rate": 0,
                "results": [],
                "error": f"서버 연결 실패 (포트 {port}): {e}",
            })
        if test_data:
            upload_data_to_qdrant(test_data)
        result = run_tests_for_model(
            model_name=model_name,
            provider=provider,
            enable_benchmark=enable_benchmark,
            iterations=iterations,
            tests=tests,
            base_url=url,
        )
        return (model_name, result)
    finally:
        setattr(_thread_local, "base_url", None)


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
    data_info: Optional[Dict[str, Any]] = None,
    use_pipeline_labels: bool = False,
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
    # 포트 자동 할당 (지정되지 않은 경우)
    if base_ports is None:
        base_ports = [8001 + i for i in range(len(models))]
    
    if len(base_ports) != len(models):
        unit = "파이프라인" if use_pipeline_labels else "모델"
        print_error(f"포트 개수({len(base_ports)})와 {unit} 개수({len(models)})가 일치하지 않습니다.")
        sys.exit(1)
    
    unit = "파이프라인" if use_pipeline_labels else "모델"
    print_header(f"여러 {unit} 비교 테스트 ({len(models)}개 {unit})")
    print_info(f"제공자: {provider}")
    print_info(f"{unit} 목록: {', '.join(models)}")
    print_info(f"\n각 {unit}은 별도 포트에서 실행 중이어야 합니다:")
    for model, port in zip(models, base_ports):
        print_info(f"  - {model}: http://localhost:{port}")
    
    all_results = {}
    # 8001, 8002, 8003 등 여러 포트에 동시에 요청 (스레드 풀)
    print_info(f"동시 실행: {len(models)}개 {unit}에 병렬로 요청을 보냅니다.")
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                _run_one_model_worker,
                model_name,
                port,
                provider,
                enable_benchmark,
                iterations,
                tests,
                test_data,
            ): (model_name, port)
            for model_name, port in zip(models, base_ports)
        }
        for future in as_completed(futures):
            model_name, port = futures[future]
            try:
                name, result = future.result()
                all_results[name] = result
                err = result.get("error")
                lbl = "파이프라인" if use_pipeline_labels else "모델"
                if err:
                    print_warning(f"{lbl} {name} (포트 {port}): {err}")
                else:
                    print_success(f"{lbl} {name} (포트 {port}) 테스트 완료: {result.get('success_count', 0)}/{result.get('total_count', 0)} 성공")
            except Exception as e:
                lbl = "파이프라인" if use_pipeline_labels else "모델"
                print_error(f"{lbl} {model_name} (포트 {port}) 예외: {e}")
                all_results[model_name] = {
                    "model_name": model_name,
                    "provider": provider,
                    "success_count": 0,
                    "total_count": 0,
                    "success_rate": 0,
                    "results": [],
                    "error": str(e),
                }
    
    # 비교 리포트 생성
    if generate_report:
        report_title = "파이프라인 비교 리포트" if use_pipeline_labels else "모델 비교 리포트"
        print_header(report_title)
        print("\n성공률 비교:")
        for model_name, result in all_results.items():
            success_rate = result.get("success_rate", 0)
            status = "✓" if success_rate == 100 else "⚠" if success_rate >= 50 else "✗"
            print(f"  {status} {model_name}: {success_rate:.1f}% ({result['success_count']}/{result['total_count']})")
    
    # 결과 저장 (model_info, data_info 포함 래핑)
    if save_results:
        model_info = _effective_model_info()
        model_info["llm_models_compared"] = list(all_results.keys())
        out = {
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
            "results": all_results,
        }
        if data_info is not None:
            out["data_info"] = data_info
        with open(save_results, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
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
  python test_all_task.py --model "gpt-4o-mini" --provider openai

  # 벤치마크 (메트릭·CPU 모니터링 활성화)
  python test_all_task.py --benchmark --save-results result.json

  # 여러 모델 비교
  python test_all_task.py --compare-models --models "gpt-4o-mini" "gpt-3.5-turbo" \\
      --provider openai --benchmark --save-results results.json

  # 부하테스트 (baseline)
  python test_all_task.py --load-test --total-requests 500 --concurrent-users 5 \\
      --ramp-up 20 --save-results load_test_baseline_results.json

  # 부하테스트 (stress)
  python test_all_task.py --load-test --total-requests 1000 --concurrent-users 15 \\
      --ramp-up 30 --save-results load_test_stress_results.json
        """
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="성능 측정 모드 전체 활성화 (메트릭 + CPU 모니터 + GPU 수집, 기존 동작)"
    )
    parser.add_argument(
        "--benchmark-metrics",
        action="store_true",
        help="서버 요청 메트릭만 수집 (X-Benchmark → logs + metrics.db)"
    )
    parser.add_argument(
        "--benchmark-cpu",
        action="store_true",
        help="서버 CPU 모니터만 수집 (X-Enable-CPU-Monitor → logs/cpu_usage.log)"
    )
    parser.add_argument(
        "--benchmark-gpu",
        action="store_true",
        help="서버 GPU 모니터만 수집 (X-Enable-GPU-Monitor → logs/gpu_usage.log)"
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
        "--pipelines",
        action="store_true",
        help="출력 문구를 파이프라인 기준으로 표시 (--compare-models와 함께 사용, 예: old_sync/new_sync/new_async)"
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
        choices=["all", "sentiment", "sentiment_batch", "summarize", "summarize_batch", "comparison", "comparison_batch", "vector"],
        help="실행할 테스트 선택 (기본값: all). src 기반 API 테스트만 포함.",
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
        "--load-test-data",
        type=str,
        default=None,
        help="부하테스트용 데이터 JSON 경로 (예: real_service_simul_review_data_640k.json). 미지정 시 기본 테스트 데이터 사용"
    )
    parser.add_argument(
        "--load-test-max-reviews-per-restaurant",
        type=int,
        default=100,
        help="부하테스트 감성 배치 시 레스토랑당 최대 리뷰 수 (--load-test-data 사용 시, 기본값: 100)"
    )
    parser.add_argument(
        "--load-test-ports",
        type=int,
        nargs="+",
        default=None,
        help="부하테스트를 여러 포트에 동시 전송. 예: --load-test-ports 8001 8002 8003 (포트별 처리량/지연 수집)"
    )
    parser.add_argument(
        "--load-test-scenario",
        type=str,
        default=None,
        help="부하테스트 요청 순서 시나리오 파일 (한 줄에 restaurant_id 하나, convert_kr3_tsv.py --output-scenario로 생성). --load-test-data와 함께 사용"
    )
    parser.add_argument(
        "--no-load-test-upload",
        action="store_true",
        help="--load-test-data 지정 시에도 벡터 업로드 생략 (이미 업로드된 경우). 기본값: false (업로드 수행)"
    )
    parser.add_argument(
        "--load-test-upload-timeout",
        type=int,
        default=3600,
        help="부하테스트 벡터 업로드 요청 타임아웃(초). 대용량 JSON 시 조정 (기본값: 3600)"
    )
    parser.add_argument(
        "--load-test-max-reviews",
        type=int,
        default=None,
        metavar="N",
        help="부하테스트 입력 데이터를 최대 N건 리뷰로 제한 (예: 5000). 업로드·감성 배치·요약/비교 모두 이 데이터만 사용. 64만 건 JSON을 5000건만 쓸 때 사용"
    )
    parser.add_argument(
        "--load-test-max-upload-reviews",
        type=int,
        default=None,
        metavar="N",
        help="(--load-test-max-reviews 미사용 시만) 벡터 업로드만 최대 N건으로 제한. 입력 데이터는 그대로 두고 업로드량만 줄일 때 사용"
    )
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default="",
        help="부하테스트 결과 저장 시 해당 구간 Prometheus 메트릭을 조회해 JSON에 포함 (예: http://localhost:9090). 비우면 생략"
    )
    parser.add_argument(
        "--abort-after-connection-errors",
        type=int,
        default=10,
        metavar="N",
        help="부하테스트 시 연속 N회 연결 실패 시 조기 종료 (기본값: 10, 0이면 비활성화)"
    )
    parser.add_argument(
        "--save-container-logs",
        type=str,
        default="",
        metavar="DIR",
        help="부하테스트 중 앱 다운(조기 종료) 시 docker logs로 old_sync/new_sync/new_async 컨테이너 로그를 DIR에 저장 (예: . 또는 ./logs). 비우면 미실행"
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
        "--test-data-max-reviews",
        type=int,
        default=None,
        metavar="N",
        help="일반 테스트 데이터(data/test_data_sample.json) 로드 시 사용할 최대 리뷰 수 (예: 2000). 미지정 시 전체 사용"
    )
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        help="각 모델별 서버 포트 리스트 (--compare-models와 함께 사용). 예: --ports 8001 8002 8003. 지정하지 않으면 8001부터 자동 할당"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API 서버 베이스 URL (모든 테스트에서 사용). 미지정 시 http://localhost:8001"
    )
    args = parser.parse_args()

    # --base-url: API 서버 베이스 URL을 CLI로 지정
    if args.base_url:
        _thread_local.base_url = args.base_url.rstrip("/")

    # 벤치마크 옵션: --benchmark이면 메트릭+CPU+GPU 모두, 개별 플래그로 분리 가능
    benchmark_metrics = args.benchmark or getattr(args, "benchmark_metrics", False)
    benchmark_cpu = args.benchmark or getattr(args, "benchmark_cpu", False)
    benchmark_gpu = args.benchmark or getattr(args, "benchmark_gpu", False)

    if benchmark_metrics:
        BENCHMARK_HEADERS["X-Benchmark"] = "true"
    if benchmark_cpu:
        BENCHMARK_HEADERS["X-Enable-CPU-Monitor"] = "true"
    if benchmark_gpu:
        BENCHMARK_HEADERS["X-Enable-GPU-Monitor"] = "true"  # 서버 GPU 로그 (logs/gpu_usage.log)

    # 메트릭/CPU/GPU 중 하나라도 켜져 있으면 벤치마크 모드 (반복·measure_performance 사용)
    enable_benchmark_mode = args.benchmark or benchmark_metrics or benchmark_cpu or benchmark_gpu

    print_header("API 전체 기능 통합 테스트")
    print_info(f"서버 URL: {get_base_url()}")
    print_info("FastAPI 서버를 테스트합니다")

    # 현재 테스트 모델 (환경 변수 기준, 서버와 동일한 env 사용 시 일치)
    mi = _effective_model_info()
    if mi["llm_provider"] == "openai" and mi["llm_model"]:
        print_info(f"현재 테스트 모델 — LLM: {mi['llm_model']} (provider=openai)")
    elif mi["llm_model"]:
        print_info(f"현재 테스트 모델 — LLM: {mi['llm_model']} (provider={mi['llm_provider']})")
    else:
        print_info(f"현재 테스트 모델 — LLM: (미설정, 서버 기본값 사용) provider={mi['llm_provider']}")
    if mi["sentiment_model"]:
        print_info(f"  — Sentiment: {mi['sentiment_model']}")
    print_info(f"  — Dense embedding: {mi['dense_embedding_model']}")
    print_info(f"  — Sparse embedding: {mi['sparse_embedding_model']}")

    if args.benchmark or benchmark_metrics or benchmark_cpu or benchmark_gpu:
        print_info("성능 측정 모드 (QUANTITATIVE_METRICS.md 지표 측정)")
        if benchmark_metrics:
            print_info("  - 서버 요청 메트릭: X-Benchmark (logs + metrics.db)")
        if benchmark_cpu:
            print_info("  - 서버 CPU 모니터: X-Enable-CPU-Monitor (logs/cpu_usage.log)")
        if benchmark_gpu:
            print_info("  - 서버 GPU 모니터: X-Enable-GPU-Monitor (logs/gpu_usage.log)")
        if args.benchmark or benchmark_metrics:
            print_info(f"  - 반복 횟수: {args.iterations}")
    
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

    # 서버 헬스 체크 (모든 테스트가 src 기반 API 호출)
    if not check_server_health():
        sys.exit(1)

    # 테스트 데이터 생성 (--load-test + --load-test-data 사용 시 부하테스트용 JSON만 쓰므로 생략)
    data_result = None
    if not (getattr(args, "load_test", False) and getattr(args, "load_test_data", None)):
        data_result = generate_test_data(
            generate_from_kr3=args.generate_from_kr3,
            kr3_sample=args.kr3_sample,
            kr3_restaurants=args.kr3_restaurants,
            max_reviews=getattr(args, "test_data_max_reviews", None),
        )
    temp_json_path = None
    test_data = None

    if data_result:
        data, temp_json_path = data_result
        test_data = data  # compare_models에 전달할 데이터 저장
        
        # SAMPLE_RESTAURANT_ID와 SAMPLE_REVIEWS를 실제 데이터로 업데이트
        if data.get("restaurants"):
            global SAMPLE_RESTAURANT_ID, SAMPLE_REVIEWS, STRENGTH_TARGET_RESTAURANT_ID
            first_restaurant = data["restaurants"][0]
            SAMPLE_RESTAURANT_ID = first_restaurant.get("restaurant_id", 1)
            # comparison_in_aspect와 맞추기: restaurant_id=4가 있으면 비교에서 4 사용
            STRENGTH_TARGET_RESTAURANT_ID = 4 if any((r.get("restaurant_id") or 0) == 4 for r in data.get("restaurants", [])) else None
            # 리뷰 객체를 ReviewModel 형식으로 저장 (API가 ReviewModel 리스트를 기대)
            SAMPLE_REVIEWS = []
            for idx, review in enumerate(first_restaurant.get("reviews", [])):
                if isinstance(review, dict) and review.get('content'):
                    # SentimentReviewInput 형식 (id, restaurant_id, content, created_at)
                    review_obj = {
                        'id': review.get('id') or (idx + 1),
                        'restaurant_id': review.get('restaurant_id', SAMPLE_RESTAURANT_ID),
                        'content': review.get('content', ''),
                        'created_at': review.get('created_at') or datetime.now().isoformat(),
                    }
                    SAMPLE_REVIEWS.append(review_obj)
            print_info(f"테스트 레스토랑 ID: {SAMPLE_RESTAURANT_ID}")
            print_info(f"테스트 리뷰 수: {len(SAMPLE_REVIEWS)}개")
            # 배치 테스트에서 로드된 레스토랑 목록 사용
            global BATCH_RESTAURANT_IDS
            BATCH_RESTAURANT_IDS = [r.get("restaurant_id") for r in data["restaurants"] if r.get("restaurant_id") is not None]
    
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
        
        # data_info 구성 (--save-results용)
        compare_tests = args.tests or ["summarize", "summarize_batch"]
        if "all" in compare_tests:
            compare_tests = ["sentiment", "sentiment_batch", "summarize", "summarize_batch", "comparison", "comparison_batch", "vector"]
        compare_data_info = build_data_info(
            test_data=test_data,
            data_source_name="test_data_sample.json",
            generate_from_kr3=args.generate_from_kr3,
            kr3_sample=args.kr3_sample,
            kr3_restaurants=args.kr3_restaurants,
            selected_tests=compare_tests,
        )
        # compare_models() 함수 호출
        comparison_results = compare_models(
            models=args.models,
            provider=args.provider,
            enable_benchmark=enable_benchmark_mode,
            iterations=args.iterations,
            save_results=args.save_results,
            generate_report=args.generate_report,
            tests=args.tests,
            base_ports=args.ports,
            test_data=test_data,
            data_info=compare_data_info,
            use_pipeline_labels=getattr(args, "pipelines", False),
        )
        
        # 결과 요약 출력
        done_title = "파이프라인 비교 테스트 완료" if getattr(args, "pipelines", False) else "모델 비교 테스트 완료"
        print_header(done_title)
        if args.save_results:
            print_success(f"결과가 저장되었습니다: {args.save_results}")
        
        # 임시 파일 정리
        if temp_json_path and os.path.exists(temp_json_path):
            try:
                os.unlink(temp_json_path)
            except Exception:
                pass
        
        sys.exit(0)
    
    # 일반 모드: 데이터 업로드 (compare_models 모드가 아닐 때)
    # --load-test 모드에서는 upload_load_test_data_to_ports로만 업로드하므로 test_data_sample 업로드 생략
    if not getattr(args, "load_test", False) and test_data:
        if upload_data_to_qdrant(test_data):
            print_success("테스트 데이터 준비 완료")
        else:
            print_warning("Qdrant upload 실패. 일부 테스트가 실패할 수 있습니다.")
    elif not getattr(args, "load_test", False) and not test_data:
        print_warning("테스트 데이터 생성 실패. 일부 테스트가 실패할 수 있습니다.")
    
    # 임시 파일 정리
    if temp_json_path and os.path.exists(temp_json_path):
        try:
            os.unlink(temp_json_path)
        except Exception:
            pass
    
    # -------------------------------------------------------------------------
    # 부하테스트 모드 (--load-test)
    # -------------------------------------------------------------------------
    def _run_load_test_one_port(
        port: int,
        sentiment_pl: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]],
        summary_pl: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]],
        comparison_pl: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]],
        total_requests: int,
        concurrent_users: int,
        ramp_up: int,
        abort_threshold: int = 10,
    ) -> Tuple[int, Dict[str, Any]]:
        """한 포트에 대해 감성/요약/비교 부하테스트 실행. (port, load_test_results) 반환."""
        _thread_local.base_url = f"http://localhost:{port}"
        try:
            results = {}
            kw = dict(total_requests=total_requests, concurrent_users=concurrent_users, ramp_up_seconds=ramp_up, consecutive_connection_errors_abort=abort_threshold)
            success, stats = load_test(endpoint=f"{get_base_url()}{API_PREFIX}/sentiment/analyze/batch", payload=sentiment_pl, timeout=120, **kw)
            if success and stats:
                results["배치 감성 분석"] = stats
            success, stats = load_test(endpoint=f"{get_base_url()}{API_PREFIX}/llm/summarize/batch", payload=summary_pl, timeout=180, **kw)
            if success and stats:
                results["배치 리뷰 요약"] = stats
            success, stats = load_test(endpoint=f"{get_base_url()}{API_PREFIX}/llm/comparison/batch", payload=comparison_pl, timeout=300, **kw)
            if success and stats:
                results["배치 비교"] = stats
            return (port, results)
        finally:
            setattr(_thread_local, "base_url", None)
    # 목적: API 처리량(req/s) 및 지연(P50/P95/P99) 측정.
    #
    # 사용 시나리오별 목적:
    #   1) run_all_restaurants_api.py + 640k JSON
    #      → 대규모 데이터를 한 번 끝까지 처리할 수 있는지 검증 (E2E/배치 검증).
    #   2) 기존 load_test (--load-test만, --load-test-data 없음)
    #      → 작은 payload로 처리량/지연 측정 (소형 요청 기준).
    #   3) load_test + --load-test-data real_service_simul_review_data_640k.json
    #      → 큰 payload로 처리량/지연 측정 (대형 요청 기준). 2번과 대비 시,
    #        감성 배치는 레스토랑당 리뷰를 --load-test-max-reviews-per-restaurant(기본 100)로
    #        통일해 두면 요청 크기가 고정되어 비교가 명확해짐.
    #
    # payload 구성:
    #   - 감성 배치: 10개 레스토랑, 각 레스토랑당 리뷰(기본 최대 100개). 동일 payload 반복 전송.
    #   - 요약/비교 배치: restaurant_id 목록만 전송(서버가 Qdrant 등에서 조회).
    #     현재는 파일/기본 데이터에서 첫 번째·두 번째 restaurant_id 2개만 사용(기본값).
    #     "다수 음식점 배치 처리"를 보려면 배치 크기(레스토랑 수)를 늘리는 옵션 확장을 고려.
    # -------------------------------------------------------------------------
    if args.load_test:
        print_header("부하테스트 모드")
        print_info(f"총 요청 수: {args.total_requests}")
        print_info(f"동시 사용자 수: {args.concurrent_users}")
        if args.ramp_up > 0:
            print_info(f"점진적 부하 증가: {args.ramp_up}초")
        
        # 부하테스트용 payload: --load-test-data 지정 시 해당 JSON에서 생성
        load_test_sentiment_payload = None
        load_test_rid1 = SAMPLE_RESTAURANT_ID
        load_test_rid2 = SAMPLE_RESTAURANT_ID + 1
        load_test_rids_10: List[int] = [SAMPLE_RESTAURANT_ID + i for i in range(10)]  # 시나리오 없을 때 요약/비교 배치용 10개
        restaurants_list: List[Dict[str, Any]] = []
        load_test_data: Optional[Dict[str, Any]] = None
        if getattr(args, "load_test_data", None):
            load_test_data_path = Path(args.load_test_data)
            if not load_test_data_path.is_absolute():
                load_test_data_path = project_root / load_test_data_path
            if load_test_data_path.exists():
                try:
                    with open(load_test_data_path, "r", encoding="utf-8") as f:
                        load_test_data = json.load(f)
                    restaurants_list = load_test_data.get("restaurants") or []
                    # 입력 데이터 자체를 N건으로 제한 (업로드·감성 배치·요약/비교 공통)
                    max_input_reviews = getattr(args, "load_test_max_reviews", None)
                    if max_input_reviews is not None:
                        load_test_data = _trim_load_test_data_to_max_reviews(load_test_data, max_input_reviews)
                        restaurants_list = load_test_data.get("restaurants") or []
                        total_rev = sum(len(r.get("reviews") or []) for r in restaurants_list)
                        print_info(f"입력 데이터 제한: 최대 {max_input_reviews}건 리뷰 사용 (실제 {total_rev}건, 레스토랑 {len(restaurants_list)}개)")
                    if restaurants_list:
                        # 감성 배치: 상위 10개 레스토랑 (리뷰는 벡터 DB에서 조회)
                        restaurants_payload = [{"restaurant_id": r.get("restaurant_id", 0)} for r in restaurants_list[:10]]
                        if restaurants_payload:
                            load_test_sentiment_payload = {"restaurants": restaurants_payload}
                        if len(restaurants_list) >= 2:
                            load_test_rid1 = restaurants_list[0].get("restaurant_id", SAMPLE_RESTAURANT_ID)
                            load_test_rid2 = restaurants_list[1].get("restaurant_id", SAMPLE_RESTAURANT_ID + 1)
                        load_test_rids_10 = [r.get("restaurant_id", SAMPLE_RESTAURANT_ID + i) for i, r in enumerate(restaurants_list[:10])]
                        print_info(f"부하테스트 데이터: {load_test_data_path.name} (레스토랑 {len(restaurants_list)}개)")
                except Exception as e:
                    print_warning(f"부하테스트 데이터 로드 실패 ({args.load_test_data}): {e}. 기본 데이터 사용.")
            else:
                print_warning(f"부하테스트 데이터 파일 없음: {load_test_data_path}. 기본 데이터 사용.")
        
        # 감성 배치 payload (단일/다중 포트 공통)
        if load_test_sentiment_payload is not None:
            sentiment_payload = load_test_sentiment_payload
        else:
            restaurants_payload = [{"restaurant_id": SAMPLE_RESTAURANT_ID + i} for i in range(10)]
            sentiment_payload = {"restaurants": restaurants_payload}
        
        # 시나리오 없을 때: 배치 요약·비교는 상위 10개 레스토랑 전체 사용. 시나리오 있을 때는 요청별 2개만 사용
        sentiment_payload_or_fn: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]] = sentiment_payload
        rids_10 = load_test_rids_10
        summary_payload_or_fn: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]] = (lambda rids=rids_10: lambda i: {"restaurants": [{"restaurant_id": rid} for rid in rids], "limit": 10, "min_score": 0.0})()
        comparison_payload_or_fn: Union[Dict[str, Any], Callable[[int], Dict[str, Any]]] = (lambda rids=rids_10: lambda i: {"restaurants": [{"restaurant_id": rid} for rid in rids]})()
        if getattr(args, "load_test_scenario", None) and restaurants_list:
            scenario_path = Path(args.load_test_scenario)
            if not scenario_path.is_absolute():
                scenario_path = project_root / scenario_path
            if scenario_path.exists():
                try:
                    with open(scenario_path, "r", encoding="utf-8") as f:
                        scenario_rids = [int(line.strip()) for line in f if line.strip()]
                    if scenario_rids:
                        max_reviews_scenario = getattr(args, "load_test_max_reviews_per_restaurant", 100) or 100
                        rest_by_id = {r.get("restaurant_id"): r for r in restaurants_list}
                        default_sentiment = sentiment_payload

                        def _sentiment_fn(i: int) -> Dict[str, Any]:
                            rid = scenario_rids[i % len(scenario_rids)]
                            if not rest_by_id.get(rid):
                                return default_sentiment
                            return {"restaurants": [{"restaurant_id": rid}]}

                        batch_size = 10

                        def _summary_fn(i: int) -> Dict[str, Any]:
                            L = len(scenario_rids)
                            n = min(batch_size, L)
                            rids = [scenario_rids[(i + k) % L] for k in range(n)]
                            return {"restaurants": [{"restaurant_id": rid} for rid in rids], "limit": 10, "min_score": 0.0}

                        def _comparison_fn(i: int) -> Dict[str, Any]:
                            L = len(scenario_rids)
                            n = min(batch_size, L)
                            rids = [scenario_rids[(i + k) % L] for k in range(n)]
                            return {"restaurants": [{"restaurant_id": rid} for rid in rids]}

                        sentiment_payload_or_fn = _sentiment_fn
                        summary_payload_or_fn = _summary_fn
                        comparison_payload_or_fn = _comparison_fn
                        print_info(f"부하테스트 시나리오: {scenario_path.name} ({len(scenario_rids)} 요청 순서)")
                except Exception as e:
                    print_warning(f"시나리오 파일 로드 실패 ({args.load_test_scenario}): {e}. 고정 payload 사용.")
            else:
                print_warning(f"시나리오 파일 없음: {scenario_path}. 고정 payload 사용.")
        
        # --load-test-data 지정 시 벡터 DB 업로드 (요약/비교 API에서 검색 가능하도록)
        if (
            load_test_data is not None
            and restaurants_list
            and not getattr(args, "no_load_test_upload", False)
        ):
            ports_for_upload = getattr(args, "load_test_ports", None)
            upload_timeout = getattr(args, "load_test_upload_timeout", 3600) or 3600
            # 입력 데이터를 이미 --load-test-max-reviews로 잘랐으면 업로드도 그대로; 아니면 --load-test-max-upload-reviews만 적용
            max_upload_reviews = None if getattr(args, "load_test_max_reviews", None) is not None else getattr(args, "load_test_max_upload_reviews", None)
            if max_upload_reviews is not None:
                print_info(f"벡터 업로드만 제한: 최대 {max_upload_reviews}건 (--load-test-max-upload-reviews)")
            if not upload_load_test_data_to_ports(load_test_data, ports_for_upload, timeout=upload_timeout, max_reviews=max_upload_reviews):
                print_warning("벡터 업로드 실패. 요약/비교 API가 빈 결과를 반환할 수 있습니다.")
        elif load_test_data is not None and getattr(args, "no_load_test_upload", False):
            print_info("--no-load-test-upload: 벡터 업로드 생략")
        
        load_test_ports = getattr(args, "load_test_ports", None)
        abort_threshold = getattr(args, "abort_after_connection_errors", 10)
        if load_test_ports:
            # 여러 포트에 동시 부하테스트
            print_info(f"동시 실행: 포트 {load_test_ports}에 각각 부하테스트 전송")
            results_by_port = {}
            load_test_global_start = time.time()
            with ThreadPoolExecutor(max_workers=len(load_test_ports)) as executor:
                futures = {
                    executor.submit(
                        _run_load_test_one_port,
                        port,
                        sentiment_payload_or_fn,
                        summary_payload_or_fn,
                        comparison_payload_or_fn,
                        args.total_requests,
                        args.concurrent_users,
                        args.ramp_up,
                        abort_threshold,
                    ): port
                    for port in load_test_ports
                }
                for future in as_completed(futures):
                    port = futures[future]
                    try:
                        p, res = future.result()
                        results_by_port[str(p)] = res
                        for name, st in (res or {}).items():
                            thr = st.get("throughput_req_per_sec") or 0
                            print_success(f"포트 {p} {name}: {thr:.2f} req/s")
                    except Exception as e:
                        print_error(f"포트 {port} 부하테스트 예외: {e}")
                        results_by_port[str(port)] = {}
            load_test_global_end = time.time()
            save_container_logs_dir = getattr(args, "save_container_logs", "") or ""
            if save_container_logs_dir.strip():
                save_container_logs_on_abort(
                    results_by_port,
                    save_container_logs_dir.strip(),
                    container_prefix=os.getenv("DOCKER_COMPOSE_PROJECT", "tasteam-new-async"),
                )
            if args.save_results:
                load_test_data_info = build_data_info(
                    test_data=test_data,
                    data_source_name="test_data_sample.json",
                    generate_from_kr3=args.generate_from_kr3,
                    kr3_sample=args.kr3_sample,
                    kr3_restaurants=args.kr3_restaurants,
                    selected_tests=[],
                )
                load_test_output = {
                    "timestamp": datetime.now().isoformat(),
                    "load_test_mode": True,
                    "load_test_ports": load_test_ports,
                    "model_info": _effective_model_info(),
                    "data_info": load_test_data_info,
                    "total_requests": args.total_requests,
                    "concurrent_users": args.concurrent_users,
                    "ramp_up_seconds": args.ramp_up,
                    "results_by_port": results_by_port,
                }
                prometheus_url = getattr(args, "prometheus_url", "") or ""
                if prometheus_url.strip():
                    try:
                        load_test_output["prometheus_metrics"] = fetch_prometheus_metrics(
                            prometheus_url.strip(), load_test_global_start, load_test_global_end
                        )
                        print_info("Prometheus 메트릭을 결과 JSON에 포함했습니다.")
                    except Exception as e:
                        load_test_output["prometheus_metrics"] = {"error": str(e)}
                with open(args.save_results, "w", encoding="utf-8") as f:
                    json.dump(load_test_output, f, ensure_ascii=False, indent=2)
                print_success(f"\n부하테스트 결과가 저장되었습니다: {args.save_results}")
            sys.exit(0)
        
        # 단일 포트: 기존 순차 실행
        load_test_results = {}
        load_test_global_start = time.time()
        print_header("1. 배치 감성 분석 부하테스트")
        url = f"{get_base_url()}{API_PREFIX}/sentiment/analyze/batch"
        success, stats = load_test(
            endpoint=url,
            payload=sentiment_payload_or_fn,
            total_requests=args.total_requests,
            concurrent_users=args.concurrent_users,
            timeout=120,
            ramp_up_seconds=args.ramp_up,
            consecutive_connection_errors_abort=abort_threshold,
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
        
        # 2. 배치 리뷰 요약 부하테스트 (새 파이프라인)
        print_header("2. 배치 리뷰 요약 부하테스트")
        url = f"{get_base_url()}{API_PREFIX}/llm/summarize/batch"
        success, stats = load_test(
            endpoint=url,
            payload=summary_payload_or_fn,
            total_requests=args.total_requests,
            concurrent_users=args.concurrent_users,
            timeout=180,
            ramp_up_seconds=args.ramp_up,
            consecutive_connection_errors_abort=abort_threshold,
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
        
        # 3. 배치 비교 부하테스트
        print_header("3. 배치 비교 부하테스트")
        url = f"{get_base_url()}{API_PREFIX}/llm/comparison/batch"
        success, stats = load_test(
            endpoint=url,
            payload=comparison_payload_or_fn,
            total_requests=args.total_requests,
            concurrent_users=args.concurrent_users,
            timeout=300,
            ramp_up_seconds=args.ramp_up,
            consecutive_connection_errors_abort=abort_threshold,
        )
        load_test_global_end = time.time()
        if success and stats:
            print_success("배치 비교 부하테스트 완료")
            print(f"  - 평균 응답 시간: {stats['avg_latency_sec']:.3f}초")
            print(f"  - P50 응답 시간: {stats.get('p50_latency_sec', 'N/A'):.3f}초" if stats.get('p50_latency_sec') else "  - P50 응답 시간: N/A")
            print(f"  - P95 응답 시간: {stats['p95_latency_sec']:.3f}초")
            print(f"  - P99 응답 시간: {stats['p99_latency_sec']:.3f}초")
            print(f"  - 처리량: {stats['throughput_req_per_sec']:.2f} req/s")
            print(f"  - 성공률: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['total_requests']})")
            print(f"  - 최대 동시 요청 수: {stats.get('max_concurrent_requests', 'N/A')}")
            load_test_results["배치 비교"] = stats

        save_container_logs_dir = getattr(args, "save_container_logs", "") or ""
        if save_container_logs_dir.strip() and any(
            (st or {}).get("aborted_early") for st in load_test_results.values()
        ):
            save_container_logs_on_abort(
                {"1": load_test_results},
                save_container_logs_dir.strip(),
                container_prefix=os.getenv("DOCKER_COMPOSE_PROJECT", "tasteam-new-async"),
            )

        # 결과 저장
        if args.save_results:
            load_test_data_info = build_data_info(
                test_data=test_data,
                data_source_name="test_data_sample.json",
                generate_from_kr3=args.generate_from_kr3,
                kr3_sample=args.kr3_sample,
                kr3_restaurants=args.kr3_restaurants,
                selected_tests=[],
            )
            load_test_output = {
                "timestamp": datetime.now().isoformat(),
                "server_url": get_base_url(),
                "load_test_mode": True,
                "model_info": _effective_model_info(),
                "data_info": load_test_data_info,
                "total_requests": args.total_requests,
                "concurrent_users": args.concurrent_users,
                "ramp_up_seconds": args.ramp_up,
                "test_results": load_test_results,
            }
            prometheus_url = getattr(args, "prometheus_url", "") or ""
            if prometheus_url.strip():
                try:
                    load_test_output["prometheus_metrics"] = fetch_prometheus_metrics(
                        prometheus_url.strip(), load_test_global_start, load_test_global_end
                    )
                    print_info("Prometheus 메트릭을 결과 JSON에 포함했습니다.")
                except Exception as e:
                    load_test_output["prometheus_metrics"] = {"error": str(e)}
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
        selected_tests = ["sentiment", "sentiment_batch", "summarize", "summarize_batch", "comparison", "comparison_batch", "vector"]

    test_registry = {
        "sentiment": ("감성 분석", lambda: test_sentiment_analysis(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "sentiment_batch": ("배치 감성 분석", lambda: test_sentiment_analysis_batch(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "summarize": ("리뷰 요약", lambda: test_summarize(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "summarize_batch": ("배치 리뷰 요약", lambda: test_summarize_batch(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "comparison": ("비교", lambda: test_comparison(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "comparison_batch": ("배치 비교", lambda: test_comparison_batch(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
        "vector": ("벡터 업로드", lambda: test_vector_upload(enable_benchmark=enable_benchmark_mode, num_iterations=args.iterations)),
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
        data_info = build_data_info(
            test_data=test_data,
            data_source_name="test_data_sample.json",
            generate_from_kr3=args.generate_from_kr3,
            kr3_sample=args.kr3_sample,
            kr3_restaurants=args.kr3_restaurants,
            selected_tests=selected_tests,
        )
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "server_url": get_base_url(),
            "benchmark_mode": enable_benchmark_mode,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_cpu": benchmark_cpu,
            "benchmark_gpu": benchmark_gpu,
            "iterations": args.iterations if enable_benchmark_mode else None,
            "model_info": _effective_model_info(),
            "data_info": data_info,
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
    
    if enable_benchmark_mode:
        print_info("\n성능 측정 모드로 실행되었습니다.")
        if benchmark_metrics:
            print_info("더 자세한 메트릭은 SQLite 데이터베이스를 확인하세요:")
            print_info(f"  sqlite3 {METRICS_DB_PATH}")
        print_info("QUANTITATIVE_METRICS.md의 SQL 쿼리를 사용하여 추가 분석이 가능합니다.")
    
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
