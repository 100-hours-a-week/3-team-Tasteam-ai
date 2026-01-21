#!/usr/bin/env python3
"""
성능 벤치마크 스크립트

기존 모델 추론과 최적화 적용 후의 성능 지표를 수집하고 비교합니다.
- 지연 시간 (Latency): 평균, 최소, 최대, 중앙값, P95, P99
- 처리량 (Throughput): 초당 처리 리뷰 수
- GPU 사용률 (vLLM 모드)
- 병목 분석
"""

import os
import sys
import json
import time
import argparse
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import requests

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.gpu_monitor import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """성능 벤치마크 클래스"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        test_data_path: Optional[str] = None,
    ):
        """
        Args:
            base_url: API 서버 URL
            test_data_path: 테스트 데이터 JSON 파일 경로
        """
        self.base_url = base_url
        self.test_data_path = test_data_path
        
        # GPUMonitor 초기화
        self.gpu_monitor = GPUMonitor(device_index=0)
        
        # 테스트 데이터 로드
        if test_data_path and Path(test_data_path).exists():
            with open(test_data_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
        else:
            self.test_data = None
            logger.warning(f"테스트 데이터 파일을 찾을 수 없습니다: {test_data_path}")
    
    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """GPU 사용률 및 메모리 사용량 조회 (GPUMonitor 사용)"""
        try:
            metrics = self.gpu_monitor.get_metrics()
            if metrics:
                return {
                    "gpu_utilization": metrics.get("gpu_util_percent", 0.0),
                    "memory_used_mb": metrics.get("memory_used_mb", 0.0),
                    "memory_total_mb": metrics.get("memory_total_mb", 0.0),
                    "memory_usage_percent": metrics.get("memory_util_percent", 0.0)
                }
        except Exception as e:
            logger.error(f"GPU 메트릭 조회 실패: {e}")
        return None
    
    def measure_single_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        num_iterations: int = 10,
        warmup_iterations: int = 2,
    ) -> Dict[str, Any]:
        """
        단일 요청 성능 측정
        
        Args:
            endpoint: API 엔드포인트 경로
            payload: 요청 페이로드
            num_iterations: 측정 반복 횟수
            warmup_iterations: 워밍업 반복 횟수
            
        Returns:
            성능 지표 딕셔너리
        """
        url = f"{self.base_url}{endpoint}"
        latencies = []
        success_count = 0
        error_count = 0
        
        # 워밍업
        for i in range(warmup_iterations):
            try:
                response = requests.post(url, json=payload, timeout=600)
                logger.debug(f"워밍업 {i+1}/{warmup_iterations}: {response.status_code}")
            except Exception as e:
                logger.warning(f"워밍업 실패: {e}")
        
        # GPU 메트릭 수집 시작 (vLLM 모드일 경우)
        gpu_metrics_before = self.get_gpu_usage()
        
        # 실제 측정
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            try:
                response = requests.post(url, json=payload, timeout=600)
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                
                if response.status_code == 200:
                    latencies.append(latency)
                    success_count += 1
                else:
                    logger.warning(f"요청 {i+1}: HTTP {response.status_code}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"요청 {i+1} 실패: {e}")
                error_count += 1
        
        # GPU 메트릭 수집 종료
        gpu_metrics_after = self.get_gpu_usage()
        
        if not latencies:
            return {
                "error": "모든 요청이 실패했습니다.",
                "success_count": success_count,
                "error_count": error_count
            }
        
        # 통계 계산
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        results = {
            "num_iterations": num_iterations,
            "success_count": success_count,
            "error_count": error_count,
            "latency": {
                "avg": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "median": statistics.median(latencies),
                "p95": latencies_sorted[int(n * 0.95)] if n > 0 else None,
                "p99": latencies_sorted[int(n * 0.99)] if n > 0 else None,
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "all_values": latencies
            },
            "gpu_metrics": {
                "before": gpu_metrics_before,
                "after": gpu_metrics_after
            }
        }
        
        return results
    
    def measure_batch_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        num_iterations: int = 5,
        warmup_iterations: int = 1,
    ) -> Dict[str, Any]:
        """
        배치 요청 성능 측정
        
        Args:
            endpoint: API 엔드포인트 경로
            payload: 요청 페이로드 (배치 데이터 포함)
            num_iterations: 측정 반복 횟수
            warmup_iterations: 워밍업 반복 횟수
            
        Returns:
            성능 지표 딕셔너리 (처리량 포함)
        """
        url = f"{self.base_url}{endpoint}"
        latencies = []
        throughputs = []
        success_count = 0
        error_count = 0
        
        # 총 리뷰 수 계산
        total_reviews = 0
        if "restaurants" in payload:
            total_reviews = sum(len(r.get("reviews", [])) for r in payload["restaurants"])
        
        # 워밍업
        for i in range(warmup_iterations):
            try:
                response = requests.post(url, json=payload, timeout=600)
                logger.debug(f"워밍업 {i+1}/{warmup_iterations}: {response.status_code}")
            except Exception as e:
                logger.warning(f"워밍업 실패: {e}")
        
        # GPU 메트릭 수집 시작
        gpu_metrics_before = self.get_gpu_usage()
        
        # 실제 측정
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            try:
                response = requests.post(url, json=payload, timeout=600)
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                
                if response.status_code == 200:
                    latencies.append(latency)
                    
                    # 처리량 계산 (초당 처리 리뷰 수)
                    if total_reviews > 0:
                        throughput = total_reviews / latency
                        throughputs.append(throughput)
                    
                    success_count += 1
                    logger.info(f"배치 요청 {i+1}/{num_iterations}: {latency:.4f}s, 처리량: {throughput:.2f} reviews/s")
                else:
                    logger.warning(f"요청 {i+1}: HTTP {response.status_code}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"요청 {i+1} 실패: {e}")
                error_count += 1
        
        # GPU 메트릭 수집 종료
        gpu_metrics_after = self.get_gpu_usage()
        
        if not latencies:
            return {
                "error": "모든 요청이 실패했습니다.",
                "success_count": success_count,
                "error_count": error_count
            }
        
        # 통계 계산
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        results = {
            "num_iterations": num_iterations,
            "success_count": success_count,
            "error_count": error_count,
            "total_reviews": total_reviews,
            "latency": {
                "avg": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "median": statistics.median(latencies),
                "p95": latencies_sorted[int(n * 0.95)] if n > 0 else None,
                "p99": latencies_sorted[int(n * 0.99)] if n > 0 else None,
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "all_values": latencies
            },
            "throughput": {
                "avg": statistics.mean(throughputs) if throughputs else None,
                "min": min(throughputs) if throughputs else None,
                "max": max(throughputs) if throughputs else None,
                "median": statistics.median(throughputs) if throughputs else None,
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 and throughputs else 0.0,
                "all_values": throughputs
            },
            "gpu_metrics": {
                "before": gpu_metrics_before,
                "after": gpu_metrics_after
            }
        }
        
        return results
    
    def benchmark_sentiment_analysis(
        self,
        mode: str = "single",  # "single" or "batch"
        num_iterations: int = 10,
        warmup_iterations: int = 2,
    ) -> Dict[str, Any]:
        """
        감성 분석 성능 벤치마크
        
        Args:
            mode: 측정 모드 ("single" 또는 "batch")
            num_iterations: 측정 반복 횟수
            warmup_iterations: 워밍업 반복 횟수
            
        Returns:
            성능 지표 딕셔너리
        """
        if not self.test_data:
            raise ValueError("테스트 데이터가 로드되지 않았습니다.")
        
        if mode == "single":
            # 단일 레스토랑 감성 분석
            first_restaurant = self.test_data["restaurants"][0]
            payload = {
                "restaurant_id": first_restaurant["restaurant_id"],
                "reviews": first_restaurant["reviews"]
            }
            
            logger.info(f"단일 레스토랑 감성 분석 벤치마크 시작 (리뷰 수: {len(first_restaurant['reviews'])})")
            return self.measure_single_request(
                endpoint="/api/v1/sentiment/analyze",
                payload=payload,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )
        
        else:  # batch
            # 배치 감성 분석
            payload = {
                "restaurants": self.test_data["restaurants"],
                "max_tokens_per_batch": self.test_data.get("max_tokens_per_batch")
            }
            
            total_reviews = sum(len(r.get("reviews", [])) for r in self.test_data["restaurants"])
            logger.info(f"배치 감성 분석 벤치마크 시작 (레스토랑 수: {len(self.test_data['restaurants'])}, 총 리뷰 수: {total_reviews})")
            
            return self.measure_batch_request(
                endpoint="/api/v1/sentiment/analyze/batch",
                payload=payload,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )
    
    def compare_modes(
        self,
        mode_configs: List[Dict[str, Any]],
        test_mode: str = "batch",
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        여러 실행 모드 성능 비교
        
        Args:
            mode_configs: 실행 모드 설정 리스트
                [{"name": "로컬 Transformers", "use_pod_vllm": False, "use_runpod": False}, ...]
            test_mode: 측정 모드 ("single" 또는 "batch")
            num_iterations: 측정 반복 횟수
            
        Returns:
            비교 결과 딕셔너리
        """
        results = {}
        base_result = None
        
        for config in mode_configs:
            mode_name = config["name"]
            logger.info(f"\n{'='*60}")
            logger.info(f"모드: {mode_name}")
            logger.info(f"{'='*60}")
            
            # 환경 변수 설정 안내
            logger.info(f"환경 변수 설정 필요:")
            logger.info(f"  export USE_POD_VLLM=\"{str(config.get('use_pod_vllm', False)).lower()}\"")
            logger.info(f"  export USE_RUNPOD=\"{str(config.get('use_runpod', False)).lower()}\"")
            logger.info(f"\n서버를 재시작한 후 Enter를 눌러주세요...")
            input()
            
            # 벤치마크 실행
            try:
                result = self.benchmark_sentiment_analysis(
                    mode=test_mode,
                    num_iterations=num_iterations,
                    warmup_iterations=2 if test_mode == "single" else 1,
                )
                results[mode_name] = result
                
                # 첫 번째 모드를 기준으로 설정
                if base_result is None:
                    base_result = result
                
            except Exception as e:
                logger.error(f"벤치마크 실행 실패 ({mode_name}): {e}")
                results[mode_name] = {"error": str(e)}
        
        # 성능 비교 분석
        comparison = {}
        if base_result and len(results) > 1:
            base_latency = base_result.get("latency", {}).get("avg")
            base_throughput = base_result.get("throughput", {}).get("avg") if "throughput" in base_result else None
            
            if base_latency:
                for mode_name, result in results.items():
                    if mode_name in comparison or "error" in result:
                        continue
                    
                    latency = result.get("latency", {}).get("avg")
                    throughput = result.get("throughput", {}).get("avg") if "throughput" in result else None
                    
                    if latency:
                        latency_improvement = ((base_latency - latency) / base_latency) * 100 if base_latency > 0 else 0
                    else:
                        latency_improvement = None
                    
                    if base_throughput and throughput:
                        throughput_improvement = ((throughput - base_throughput) / base_throughput) * 100 if base_throughput > 0 else 0
                    else:
                        throughput_improvement = None
                    
                    comparison[mode_name] = {
                        "latency_improvement_percent": latency_improvement,
                        "throughput_improvement_percent": throughput_improvement,
                        "latency": latency,
                        "throughput": throughput
                    }
        
        return {
            "results": results,
            "comparison": comparison,
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"결과 저장 완료: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        if "error" in results:
            logger.error(f"벤치마크 실패: {results['error']}")
            return
        
        logger.info("\n" + "="*60)
        logger.info("성능 벤치마크 결과 요약")
        logger.info("="*60)
        
        latency = results.get("latency", {})
        if latency:
            logger.info(f"\n지연 시간 (Latency):")
            logger.info(f"  평균: {latency.get('avg', 0):.4f}s")
            logger.info(f"  최소: {latency.get('min', 0):.4f}s")
            logger.info(f"  최대: {latency.get('max', 0):.4f}s")
            logger.info(f"  중앙값: {latency.get('median', 0):.4f}s")
            logger.info(f"  P95: {latency.get('p95', 0):.4f}s")
            logger.info(f"  P99: {latency.get('p99', 0):.4f}s")
            logger.info(f"  표준편차: {latency.get('std', 0):.4f}s")
        
        throughput = results.get("throughput", {})
        if throughput and throughput.get("avg"):
            logger.info(f"\n처리량 (Throughput):")
            logger.info(f"  평균: {throughput.get('avg', 0):.2f} reviews/s")
            logger.info(f"  최소: {throughput.get('min', 0):.2f} reviews/s")
            logger.info(f"  최대: {throughput.get('max', 0):.2f} reviews/s")
            logger.info(f"  중앙값: {throughput.get('median', 0):.2f} reviews/s")
        
        gpu_metrics = results.get("gpu_metrics", {})
        if gpu_metrics.get("before") or gpu_metrics.get("after"):
            logger.info(f"\nGPU 메트릭:")
            if gpu_metrics.get("before"):
                before = gpu_metrics["before"]
                logger.info(f"  시작 시 GPU 사용률: {before.get('gpu_utilization', 0):.1f}%")
                logger.info(f"  시작 시 메모리 사용: {before.get('memory_used_mb', 0):.0f}MB / {before.get('memory_total_mb', 0):.0f}MB ({before.get('memory_usage_percent', 0):.1f}%)")
            if gpu_metrics.get("after"):
                after = gpu_metrics["after"]
                logger.info(f"  종료 시 GPU 사용률: {after.get('gpu_utilization', 0):.1f}%")
                logger.info(f"  종료 시 메모리 사용: {after.get('memory_used_mb', 0):.0f}MB / {after.get('memory_total_mb', 0):.0f}MB ({after.get('memory_usage_percent', 0):.1f}%)")
        
        logger.info(f"\n성공: {results.get('success_count', 0)}/{results.get('num_iterations', 0)}")
        logger.info(f"실패: {results.get('error_count', 0)}/{results.get('num_iterations', 0)}")
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="성능 벤치마크 스크립트")
    parser.add_argument(
        "--test-data",
        type=str,
        default="test_data.json",
        help="테스트 데이터 JSON 파일 경로 (기본값: test_data.json)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8001",
        help="API 서버 URL (기본값: http://localhost:8001)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="batch",
        help="측정 모드: single (단일 레스토랑) 또는 batch (배치 처리) (기본값: batch)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="측정 반복 횟수 (기본값: 10)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="워밍업 반복 횟수 (기본값: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (기본값: benchmark_results_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="여러 실행 모드 비교 모드 (로컬 Transformers, RunPod Serverless vLLM, RunPod Pod + vLLM)"
    )
    
    args = parser.parse_args()
    
    # 벤치마크 객체 생성
    benchmark = PerformanceBenchmark(
        base_url=args.base_url,
        test_data_path=args.test_data
    )
    
    if args.compare:
        # 여러 모드 비교
        mode_configs = [
            {"name": "로컬 Transformers", "use_pod_vllm": False, "use_runpod": False},
            {"name": "RunPod Serverless vLLM", "use_pod_vllm": False, "use_runpod": True},
            {"name": "RunPod Pod + vLLM", "use_pod_vllm": True, "use_runpod": False},
        ]
        
        logger.info("여러 실행 모드 성능 비교 모드")
        logger.info("각 모드마다 서버를 재시작하고 환경 변수를 설정해야 합니다.")
        
        comparison_results = benchmark.compare_modes(
            mode_configs=mode_configs,
            test_mode=args.mode,
            num_iterations=args.iterations,
        )
        
        # 결과 출력
        logger.info("\n" + "="*60)
        logger.info("모드별 성능 비교 결과")
        logger.info("="*60)
        
        for mode_name, result in comparison_results["results"].items():
            logger.info(f"\n{mode_name}:")
            if "error" in result:
                logger.error(f"  오류: {result['error']}")
            else:
                latency = result.get("latency", {}).get("avg")
                throughput = result.get("throughput", {}).get("avg") if "throughput" in result else None
                logger.info(f"  평균 지연 시간: {latency:.4f}s" if latency else "  지연 시간: 측정 불가")
                logger.info(f"  평균 처리량: {throughput:.2f} reviews/s" if throughput else "")
        
        # 개선 사항 출력
        if comparison_results.get("comparison"):
            logger.info("\n" + "="*60)
            logger.info("성능 개선 분석")
            logger.info("="*60)
            base_mode = list(comparison_results["results"].keys())[0]
            logger.info(f"\n기준 모드: {base_mode}")
            
            for mode_name, comp in comparison_results["comparison"].items():
                if mode_name == base_mode:
                    continue
                logger.info(f"\n{mode_name} vs {base_mode}:")
                if comp.get("latency_improvement_percent") is not None:
                    improvement = comp["latency_improvement_percent"]
                    if improvement > 0:
                        logger.info(f"  ✅ 지연 시간 감소: {improvement:.2f}% (빠름)")
                    elif improvement < 0:
                        logger.info(f"  ❌ 지연 시간 증가: {abs(improvement):.2f}% (느림)")
                    else:
                        logger.info(f"  ➖ 지연 시간 변화 없음")
                
                if comp.get("throughput_improvement_percent") is not None:
                    improvement = comp["throughput_improvement_percent"]
                    if improvement > 0:
                        logger.info(f"  ✅ 처리량 향상: {improvement:.2f}% (향상)")
                    elif improvement < 0:
                        logger.info(f"  ❌ 처리량 감소: {abs(improvement):.2f}% (저하)")
                    else:
                        logger.info(f"  ➖ 처리량 변화 없음")
        
        # 결과 저장
        output_path = args.output or f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        benchmark.save_results(comparison_results, output_path)
        
    else:
        # 단일 모드 측정
        logger.info(f"{args.mode} 모드 성능 벤치마크 시작")
        
        results = benchmark.benchmark_sentiment_analysis(
            mode=args.mode,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
        )
        
        # 결과 출력
        benchmark.print_summary(results)
        
        # 결과 저장
        output_path = args.output or f"benchmark_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        benchmark.save_results(results, output_path)


if __name__ == "__main__":
    main()

