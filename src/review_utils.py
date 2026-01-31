"""
리뷰 데이터 처리 유틸리티 모듈 ( 기반)
"""

import re
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime


def extract_content_list(reviews: List[Union[Dict, Any]]) -> List[str]:
    """
    리뷰 리스트에서 content 필드만 추출합니다 ( 기반).
    
    Args:
        reviews: 리뷰 딕셔너리 리스트 또는 Pydantic 모델 리스트 (REVIEW TABLE)
        
    Returns:
        content 리스트
    """
    content_list = []
    for r in reviews:
        # Pydantic 모델인 경우
        if hasattr(r, 'content'):
            content = r.content
        # 딕셔너리인 경우
        elif isinstance(r, dict):
            content = r.get("content", "")
        else:
            content = ""
        
        if content:
            content_list.append(content)
    
    return content_list


def estimate_tokens(text: str) -> int:
    """
    텍스트의 토큰 수를 추정합니다.
    
    한국어의 경우 단어 수 × 1.3으로 추정합니다.
    (실제 토크나이저를 사용하지 않고 빠른 추정)
    
    Args:
        text: 텍스트 문자열
        
    Returns:
        추정된 토큰 수
    """
    if not text:
        return 0
    # 단어 수 계산 (공백 기준)
    word_count = len(text.split())
    # 한국어는 단어 수 × 1.3으로 추정
    return int(word_count * 1.3)


def estimate_reviews_tokens(reviews: List[Dict]) -> int:
    """
    리뷰 리스트의 총 토큰 수를 추정합니다.
    
    Args:
        reviews: 리뷰 딕셔너리 리스트 (content 또는 review 필드 포함)
        
    Returns:
        추정된 총 토큰 수
    """
    total_tokens = 0
    for review in reviews:
        # content 또는 review 필드에서 텍스트 추출
        text = review.get("content", "") or review.get("review", "")
        total_tokens += estimate_tokens(text)
    return total_tokens


def extract_reviews_from_payloads(payloads: List[Dict]) -> List[str]:
    """
    payload 리스트에서 리뷰 텍스트만 추출합니다.
의 'content' 필드를 사용합니다.
    
    Args:
        payloads: payload 딕셔너리 리스트
        
    Returns:
        리뷰 텍스트 리스트 (content 필드)
    """
    return [p.get("content", "") for p in payloads if p.get("content")]


def extract_image_urls(images: Union[Dict, List, None]) -> List[str]:
    """
    images 필드에서 URL 리스트를 안전하게 추출합니다.
    REVIEW_IMAGE TABLE의 image_url을 추출합니다.
    
    Args:
        images: dict, list, 또는 None
        
    Returns:
        이미지 URL 리스트
    """
    image_urls = []
    
    if isinstance(images, dict):
        url = images.get("url") or images.get("image_url")
        if url:
            image_urls.append(url)
    elif isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                url = img.get("url") or img.get("image_url")
                if url:
                    image_urls.append(url)
            elif isinstance(img, str):
                image_urls.append(img)
    
    return image_urls


def validate_review_data(review: Union[Dict, Any]) -> bool:
    """
    리뷰 데이터의 유효성을 검증합니다 ( 기반).
    
    Args:
        review: 리뷰 딕셔너리 또는 Pydantic 모델 (REVIEW TABLE)
        
    Returns:
        유효성 여부
    """
    # content는 필수, id는 필수 (API 스키마와 일치)
    # Pydantic 모델인 경우
    if hasattr(review, 'content'):
        if not review.content:
            return False
        if hasattr(review, 'id'):
            return review.id is not None
        return True
    # 딕셔너리인 경우
    elif isinstance(review, dict):
        if "content" not in review or not review.get("content"):
            return False
        rid = review.get("id") or review.get("review_id")
        return rid is not None
    return False


def validate_restaurant_data(restaurant: Dict) -> bool:
    """
    레스토랑 데이터의 유효성을 검증합니다 ( 기반).
    
    Args:
        restaurant: 레스토랑 딕셔너리 (RESTAURANT TABLE)
        
    Returns:
        유효성 여부
    """
    # : name은 필수
    required_fields = ["name"]
    if not all(field in restaurant for field in required_fields):
        return False
    
    return True


def convert_review_to_schema_format(review: Dict) -> Dict:
    """
    기존 형식의 리뷰를  형식으로 변환합니다.
    
    Args:
        review: 기존 형식의 리뷰 딕셔너리
        
    Returns:
 형식의 리뷰 딕셔너리
    """
    result = {}
    
    # id 필드 (review_id를 id로 변환)
    if "review_id" in review:
        review_id = review["review_id"]
        if isinstance(review_id, (int, str)) and str(review_id).isdigit():
            result["id"] = int(review_id)
    elif "id" in review:
        result["id"] = review["id"]
    
    # content 필드 (review를 content로 변환)
    if "review" in review:
        result["content"] = review["review"]
    elif "content" in review:
        result["content"] = review["content"]
    
    # member_id 필드 (user_id를 member_id로 변환)
    if "user_id" in review:
        user_id = review["user_id"]
        if isinstance(user_id, (int, str)) and str(user_id).isdigit():
            result["member_id"] = int(user_id)
    elif "member_id" in review:
        result["member_id"] = review["member_id"]
    
    # group_id 필드 (group을 group_id로 변환)
    if "group" in review:
        group_value = review["group"]
        if isinstance(group_value, (int, str)) and str(group_value).isdigit():
            result["group_id"] = int(group_value)
    elif "group_id" in review:
        result["group_id"] = review["group_id"]
    
    # subgroup_id 필드
    if "subgroup_id" in review:
        result["subgroup_id"] = review["subgroup_id"]
    
    # created_at 필드 (datetime을 created_at으로 변환)
    if "datetime" in review:
        result["created_at"] = review["datetime"]
    elif "created_at" in review:
        result["created_at"] = review["created_at"]
    
    # 나머지 필드 복사
    for key in ["restaurant_id", "is_recommended", "updated_at", "version"]:
        if key in review:
            result[key] = review[key]
    
    # images 필드 처리 (REVIEW_IMAGE TABLE)
    if "images" in review:
        images = review["images"]
        if isinstance(images, dict) and "url" in images:
            result["images"] = [{"image_url": images["url"], "review_id": result.get("id")}]
        elif isinstance(images, list):
            result["images"] = [
                {
                    "image_url": img.get("url", img) if isinstance(img, dict) else img,
                    "review_id": result.get("id")
                }
                for img in images
            ]
    
    return result


def convert_schema_to_review(schema_review: Dict) -> Dict:
    """
 형식의 리뷰를 기존 형식으로 변환합니다 (하위 호환성).
    
    Args:
        schema_review:  형식의 리뷰 딕셔너리
    
    Returns:
        기존 형식의 리뷰 딕셔너리
    """
    result = {}
    
    # 기존 필드명으로 매핑
    if "id" in schema_review:
        result["review_id"] = str(schema_review["id"])
    
    if "content" in schema_review:
        result["review"] = schema_review["content"]
    
    if "member_id" in schema_review:
        result["user_id"] = str(schema_review["member_id"])
    
    if "group_id" in schema_review:
        result["group"] = str(schema_review["group_id"])
    
    if "created_at" in schema_review:
        result["datetime"] = schema_review["created_at"]
    
    # images 변환
    if "images" in schema_review:
        if isinstance(schema_review["images"], list) and len(schema_review["images"]) > 0:
            result["images"] = {"url": schema_review["images"][0].get("image_url", "")}
    
    # 나머지 필드 복사
    for key in ["restaurant_id", "subgroup_id", "is_recommended", "updated_at", "version"]:
        if key in schema_review:
            result[key] = schema_review[key]
    
    return result


def preprocess_review_text(text: str) -> str:
    """
    리뷰 텍스트 전처리: 언어 정규화
    
    Args:
        text: 원본 텍스트
    
    Returns:
        전처리된 텍스트
    """
    if not text:
        return ""
    
    # 이모지 제거 (선택적)
    text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
    
    # 중복 문자 제거 (예: "맛있어요요요" → "맛있어요")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def split_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리
    
    Args:
        text: 텍스트 문자열
    
    Returns:
        문장 리스트
    """
    if not text:
        return []
    
    # 간단한 문장 분리 (마침표, 느낌표, 물음표 기준)
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def preprocess_reviews(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    리뷰 전처리: 언어 정규화, 문장 분리, 메타데이터 정리
    
    Args:
        reviews: 리뷰 딕셔너리 리스트
    
    Returns:
        전처리된 리뷰 리스트
    """
    processed = []
    
    for review in reviews:
        text = review.get("content", "") or review.get("review", "")
        
        # 1. 언어 정규화
        text = preprocess_review_text(text)
        
        if not text:
            continue
        
        # 2. 문장 분리
        sentences = split_sentences(text)
        
        # 3. 메타데이터 정리
        processed.append({
            "review_id": str(review.get("id") or review.get("review_id", "")),
            "restaurant_id": review.get("restaurant_id"),
            "category": review.get("food_category_id") or review.get("category"),
            "region": review.get("region"),
            "price_band": review.get("price_band"),
            "created_at": review.get("created_at") or review.get("datetime"),
            "rating": review.get("rating") or review.get("is_recommended"),
            "text": text,
            "sentences": sentences,
            "original_review": review,  # 원본 보존
        })
    
    return processed
