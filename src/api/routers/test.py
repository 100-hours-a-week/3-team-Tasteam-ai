"""
테스트 데이터 생성 라우터
"""
import logging
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가 (scripts 모듈 import를 위해)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from scripts.convert_kr3_tsv import (
        read_tsv_file,
        group_reviews_by_restaurant,
        convert_to_api_format
    )
except ImportError as e:
    logging.warning(f"scripts.convert_kr3_tsv 모듈을 import할 수 없습니다: {e}")
    read_tsv_file = None
    group_reviews_by_restaurant = None
    convert_to_api_format = None

from ...models import SentimentAnalysisBatchRequest
from ...config import Config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=SentimentAnalysisBatchRequest)
async def generate_test_data(
    sample: Optional[int] = Query(None, ge=1, le=1000000, description="샘플링할 리뷰 수 (None이면 전체, 권장: 100-10000)"),
    restaurants: Optional[int] = Query(None, ge=1, le=1000, description="생성할 레스토랑 수 (None이면 자동 결정)"),
    reviews_per_restaurant: Optional[int] = Query(None, ge=1, le=10000, description="레스토랑당 리뷰 수 (None이면 균등 분배)"),
    single_restaurant: bool = Query(False, description="모든 리뷰를 단일 레스토랑으로 그룹화"),
    seed: int = Query(42, ge=0, description="랜덤 시드 (재현 가능한 결과, 기본값: 42)"),
    tsv_path: Optional[str] = Query(None, description="TSV 파일 경로 (기본값: 프로젝트 루트의 kr3.tsv)"),
):
    """
    kr3.tsv 파일에서 테스트 데이터를 샘플링하여 생성합니다.
    
    클라이언트가 요청한 샘플링 옵션에 따라 테스트 데이터를 생성하고,
    배치 감성 분석 요청 형식(`SentimentAnalysisBatchRequest`)으로 반환합니다.
    
    **주요 파라미터:**
    - `sample`: 랜덤 샘플링할 리뷰 수 (None이면 전체 변환, 권장: 100-10000, 최대: 1000000)
    - `restaurants`: 생성할 레스토랑 수 (None이면 자동 결정, 최대: 1000)
    - `reviews_per_restaurant`: 레스토랑당 리뷰 수 (None이면 균등 분배, 최대: 10000)
    - `single_restaurant`: 모든 리뷰를 단일 레스토랑으로 그룹화 (기본값: False)
    - `seed`: 랜덤 시드 (재현 가능한 결과, 기본값: 42)
    - `tsv_path`: TSV 파일 경로 (기본값: 프로젝트 루트의 kr3.tsv)
    
    **사용 예시:**
    
    작은 샘플 생성 (100개 리뷰, 5개 레스토랑):
    ```bash
    curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&restaurants=5"
    ```
    
    중간 샘플 생성 (1000개 리뷰, 20개 레스토랑):
    ```bash
    curl -X POST "http://localhost:8000/api/v1/test/generate?sample=1000&restaurants=20"
    ```
    
    단일 레스토랑으로 그룹화:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&single_restaurant=true"
    ```
    
    재현 가능한 결과 (시드 지정):
    ```bash
    curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&seed=123"
    ```
    
    **응답 형식:**
    - `SentimentAnalysisBatchRequest` 형식
    - `restaurants`: 레스토랑별 리뷰 데이터 리스트
    - `max_tokens_per_batch`: 기본값 4000
    
    **참고:**
    - 생성된 데이터는 바로 `/api/v1/sentiment/analyze/batch` 엔드포인트에 사용할 수 있습니다.
    - 대용량 샘플링 시 서버 리소스 사용이 증가할 수 있습니다.
    - 프로덕션 환경에서는 사용을 제한하는 것을 권장합니다.
    """
    # scripts 모듈 import 실패 시 에러
    if read_tsv_file is None or group_reviews_by_restaurant is None or convert_to_api_format is None:
        raise HTTPException(
            status_code=503,
            detail="테스트 데이터 생성 기능을 사용할 수 없습니다. scripts 모듈을 확인하세요."
        )
    
    try:
        # TSV 파일 경로 결정
        if tsv_path is None:
            # 프로젝트 루트에서 kr3.tsv 찾기
            tsv_file = project_root / "kr3.tsv"
        else:
            tsv_file = Path(tsv_path)
        
        # 파일 존재 확인
        if not tsv_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"TSV 파일을 찾을 수 없습니다: {tsv_file}. 프로젝트 루트에 kr3.tsv 파일이 있는지 확인하세요."
            )
        
        logger.info(
            f"테스트 데이터 생성 시작: sample={sample}, restaurants={restaurants}, "
            f"single_restaurant={single_restaurant}, seed={seed}"
        )
        
        # 1. TSV 파일 읽기
        reviews = read_tsv_file(
            file_path=str(tsv_file),
            sample_size=sample,
            seed=seed
        )
        
        if not reviews:
            raise HTTPException(
                status_code=400,
                detail="읽은 리뷰가 없습니다. 샘플 크기를 조정하거나 파일을 확인하세요."
            )
        
        logger.info(f"TSV 파일에서 {len(reviews)}개 리뷰 읽기 완료")
        
        # 2. 레스토랑별로 그룹화
        restaurants_map = group_reviews_by_restaurant(
            reviews=reviews,
            num_restaurants=restaurants,
            reviews_per_restaurant=reviews_per_restaurant,
            single_restaurant=single_restaurant,
            seed=seed
        )
        
        logger.info(f"레스토랑별 그룹화 완료: {len(restaurants_map)}개 레스토랑")
        
        # 3. API 형식으로 변환
        api_format = convert_to_api_format(
            restaurants=restaurants_map,
            seed=seed
        )
        
        total_reviews = sum(len(r['reviews']) for r in api_format['restaurants'])
        
        logger.info(
            f"테스트 데이터 생성 완료: "
            f"레스토랑 {len(api_format['restaurants'])}개, "
            f"총 리뷰 {total_reviews}개"
        )
        
        return api_format
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"잘못된 요청 파라미터: {str(e)}")
    except Exception as e:
        logger.error(f"테스트 데이터 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"테스트 데이터 생성 중 오류 발생: {str(e)}"
        )

