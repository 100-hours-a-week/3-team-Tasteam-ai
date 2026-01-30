"""
Strength 파이프라인 모듈 (프로덕션 전용, src 자체 로직).
통계적 비율 기반 차별점 계산 및 LLM 설명 생성.
- final_pipeline / hybrid_search.final_pipeline 미의존.
- Spark 사용. 비-Spark(Kiwi만) 대체 구현 없음.
"""

import csv
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

_spark_session = None


def _get_spark() -> "SparkSession":
    """SparkSession lazy 초기화 (프로세스당 1회)."""
    global _spark_session
    if _spark_session is None and SPARK_AVAILABLE:
        _spark_session = (
            SparkSession.builder.appName("strength_pipeline")
            .master("local[6]")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
        logger.debug("SparkSession 생성 완료")
    return _spark_session


def _spark_ratios_map_partition(partition, bc_stop):
    """
    Spark mapPartitions: 텍스트 → (phrase, 1)만 emit. (strength_in_aspect와 동일)
    - Kiwi NNG/NNP, len≥2, stopwords, a!=b만 적용. 키워드(SERVICE/PRICE) 필터 없음.
    - takeOrdered(2000) 후 candidates에서 classify로 service/price 비율 계산.
    """
    import re
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    stop = set(bc_stop.value) if bc_stop.value else set()
    for text in partition:
        if not text or not isinstance(text, str):
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        try:
            tokens = [
                t.form for t in kiwi.tokenize(text)
                if t.tag in ("NNG", "NNP") and len(t.form) >= 2 and t.form not in stop
            ]
        except Exception:
            continue
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a == b or len(a) < 2 or len(b) < 2:
                continue
            yield (f"{a} {b}", 1)


def _spark_calculate_ratios(texts_rdd, stopwords: Optional[List[str]] = None) -> Dict[str, float]:
    """
    RDD of str → service/price 긍정 비율. (strength_in_aspect와 동일 파이프라인)
    (phrase,1) → reduceByKey → filter(a!=b) → takeOrdered(2000) → is_noise 제거
    → candidates에 대해 classify(service/price) → SERVICE_POSITIVE/PRICE_POSITIVE로 긍정 비율.
    대용량 TSV: mapPartitions 결과를 persist(MEMORY_AND_DISK)+count로 캐시 후 reduceByKey,
    takeOrdered 완료 후 unpersist (strength_in_aspect와 동일).
    """
    from pyspark.storagelevel import StorageLevel

    spark = _get_spark()
    sc = spark.sparkContext
    bc_stop = sc.broadcast(list(set(stopwords)) if stopwords else [])

    bigrams = (
        texts_rdd.mapPartitions(lambda p: _spark_ratios_map_partition(p, bc_stop))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    try:
        bigrams.count()  # 캐시 채우기
        bigram_counts = (
            bigrams
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda kv: len(kv[0].split()) == 2 and kv[0].split()[0] != kv[0].split()[1])
        )
        candidates = bigram_counts.takeOrdered(2000, key=lambda x: -x[1])
    finally:
        bigrams.unpersist()

    def is_noise(phrase: str) -> bool:
        sp = phrase.split()
        if len(sp) != 2:
            return True
        a, b = sp[0], sp[1]
        return len(a) < 2 or len(b) < 2

    candidates = [x for x in candidates if not is_noise(x[0])]

    def classify(phrase: str) -> List[str]:
        labels: List[str] = []
        if any(k in phrase for k in SERVICE_KW):
            labels.append("service")
        if any(k in phrase for k in PRICE_KW):
            labels.append("price")
        if not labels:
            labels.append("other")
        return labels

    category_json: Dict[str, Dict[str, int]] = {"service": {}, "price": {}}
    for phrase, count in candidates:
        for lab in classify(phrase):
            if lab in category_json:
                category_json[lab][phrase] = category_json[lab].get(phrase, 0) + count

    total_s = sum(category_json["service"].values())
    total_p = sum(category_json["price"].values())
    service_pos = sum(
        cnt for phrase, cnt in category_json["service"].items()
        if any(t in phrase.split() for t in SERVICE_POSITIVE_KW)
    )
    price_pos = sum(
        cnt for phrase, cnt in category_json["price"].items()
        if any(t in phrase for t in PRICE_POSITIVE_KW)
    )

    def safe_div(a: float, b: float) -> float:
        return round(a / b, 4) if b else 0.0

    return {
        "service": safe_div(service_pos, total_s),
        "price": safe_div(price_pos, total_p),
    }


def _classify_for_seeds(phrase: str) -> List[str]:
    """4-way 분류: service, price, food, other (recall_seeds용, strength_in_aspect와 동일)."""
    labels: List[str] = []
    if any(k in phrase for k in SERVICE_KW):
        labels.append("service")
    if any(k in phrase for k in PRICE_KW):
        labels.append("price")
    if any(k in phrase for k in FOOD_KW):
        labels.append("food")
    if not labels:
        labels.append("other")
    return labels


def _quantile_split(pairs: List[Tuple[str, int]], head_q: float = 0.02, mid_q: float = 0.20, min_head: int = 10):
    """(phrase, count) 리스트를 head/mid/tail로 분할 (strength_in_aspect와 동일)."""
    if not pairs:
        return [], [], []
    if isinstance(pairs, dict):
        pairs = list(pairs.items())
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    n = len(pairs_sorted)
    head_n = max(min_head, min(int(n * head_q), n))
    mid_end = max(head_n, min(int(n * mid_q), n))
    return pairs_sorted[:head_n], pairs_sorted[head_n:mid_end], pairs_sorted[mid_end:]


def _pick_seeds_pairs(head: List, mid: List, tail: List, mid_k: int = 5, tail_k: int = 1, seed: int = 42) -> List[Tuple[str, int]]:
    """head 전부 + mid에서 mid_k개 가중 샘플 + tail에서 tail_k개 랜덤 (strength_in_aspect와 동일)."""
    random.seed(seed)
    seeds = list(head)
    if mid:
        idxs = list(range(len(mid)))
        weights = [math.log(c + 1) for _, c in mid]
        for _ in range(min(mid_k, len(mid))):
            total = sum(weights[i] for i in idxs)
            if total <= 0:
                break
            r = random.random() * total
            acc = 0.0
            for i in idxs:
                acc += weights[i]
                if acc >= r:
                    seeds.append(mid[i])
                    idxs.remove(i)
                    break
    if tail and tail_k > 0:
        seeds.extend(random.sample(tail, k=min(tail_k, len(tail))))
    seen = set()
    out = []
    for p, c in seeds:
        if p not in seen:
            out.append((p, c))
            seen.add(p)
    return out


def _dedup_reversed_bigrams_pairs(pairs: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """(a b, c)와 (b a, c)를 하나로 묶어 카운트 합산 (strength_in_aspect와 동일)."""
    order: List[tuple] = []
    agg: Dict[tuple, Dict] = {}
    for phrase, cnt in pairs:
        parts = phrase.split()
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        key = tuple(sorted([a, b]))
        if key not in agg:
            agg[key] = {"repr": phrase, "count": cnt}
            order.append(key)
        else:
            agg[key]["count"] += cnt
    return [(agg[k]["repr"], agg[k]["count"]) for k in order]


def _spark_recall_seeds(texts_rdd, stopwords: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, int]]]:
    """
    RDD of str → recall_seeds_for_summary.
    strength_in_aspect.total_asepct_ratio의 all_seeds 생성과 동일: bigram → 4-way classify
    → quantile_split, pick_seeds_pairs, dedup_reversed_bigrams_pairs.
    """
    from pyspark.storagelevel import StorageLevel

    spark = _get_spark()
    sc = spark.sparkContext
    bc_stop = sc.broadcast(list(set(stopwords or [])))

    bigrams = (
        texts_rdd.mapPartitions(lambda p: _spark_ratios_map_partition(p, bc_stop))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    try:
        bigrams.count()
        bigram_counts = (
            bigrams
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda kv: len(kv[0].split()) == 2 and kv[0].split()[0] != kv[0].split()[1])
        )
        candidates = bigram_counts.takeOrdered(2000, key=lambda x: -x[1])
    finally:
        bigrams.unpersist()

    def is_noise(phrase: str) -> bool:
        sp = phrase.split()
        return len(sp) != 2 or len(sp[0]) < 2 or len(sp[1]) < 2

    candidates = [x for x in candidates if not is_noise(x[0])]

    category_json: Dict[str, Dict[str, int]] = {"service": {}, "price": {}, "food": {}, "other": {}}
    for phrase, count in candidates:
        for lab in _classify_for_seeds(phrase):
            if lab in category_json:
                category_json[lab][phrase] = category_json[lab].get(phrase, 0) + count

    pairs_list = [
        list(category_json["service"].items()),
        list(category_json["price"].items()),
        list(category_json["food"].items()),
        list(category_json["other"].items()),
    ]
    pairs_name = ["service", "price", "food", "other"]

    all_seeds: Dict[str, List[Tuple[str, int]]] = {"service": [], "price": [], "food": [], "other": []}
    for pairs, name in zip(pairs_list, pairs_name):
        head, mid, tail = _quantile_split(pairs)
        seeds = _pick_seeds_pairs(head, mid, tail, mid_k=5, tail_k=1)
        cleaned = _dedup_reversed_bigrams_pairs(seeds)
        all_seeds[name] = cleaned

    return all_seeds


def _load_stopwords_for_recall(project_root: Optional[str] = None) -> List[str]:
    """recall_seeds 계산용 불용어 (data/ 또는 hybrid_search/...)."""
    if project_root is None:
        project_root = str(Path(__file__).resolve().parents[1])
    for rel in ["data/stopwords-ko.txt", "hybrid_search/data_preprocessing/stopwords-ko.txt"]:
        p = Path(project_root) / rel
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return [w.strip() for w in f if w.strip()]
    return []


def compute_recall_seeds_from_file(
    path: str,
    stopwords: Optional[List[str]] = None,
    project_root: Optional[str] = None,
) -> Optional[Dict[str, List[Tuple[str, int]]]]:
    """
    strength_in_aspect와 동일: 전체 aspect 데이터(TSV/JSON)에서 recall_seeds_for_summary 계산.
    Summary 하이브리드 쿼리 시드로 사용. 파일 없음/Spark 미사용 시 None.
    """
    if not path or not isinstance(path, str):
        return None
    p = Path(path)
    if not p.is_absolute() and project_root:
        p = Path(project_root) / p
    if not p.exists():
        logger.debug("compute_recall_seeds_from_file: 파일 없음 %s", p)
        return None
    path = str(p)
    if not SPARK_AVAILABLE:
        logger.warning("pyspark 미설치. compute_recall_seeds_from_file 불가.")
        return None
    if stopwords is None:
        stopwords = _load_stopwords_for_recall(project_root)

    try:
        from pyspark.sql.functions import col, length

        spark = _get_spark()
        if path.lower().endswith(".json"):
            df = spark.read.option("multiline", "true").json(path)
            text_col = next((c for c in ("content", "text") if c in df.columns), None)
            if not text_col:
                return None
            base_df = df.select(col(text_col).alias("text")).where(
                col("text").isNotNull() & (length(col("text")) > 0)
            )
        elif path.lower().endswith(".tsv") or path.lower().endswith(".csv"):
            sep = "\t" if path.lower().endswith(".tsv") else ","
            df = spark.read.option("sep", sep).option("header", "true").csv(path)
            text_col = next((c for c in ("Review", "content", "text") if c in df.columns), None)
            if not text_col:
                return None
            base_df = df.select(col(text_col).alias("text")).where(
                col("text").isNotNull() & (length(col("text")) > 0)
            )
        else:
            return None
        texts_rdd = base_df.rdd.map(lambda r: r["text"])
        out = _spark_recall_seeds(texts_rdd, stopwords)
        logger.info("compute_recall_seeds_from_file: path=%s → service=%d, price=%d, food=%d, other=%d",
                    path, len(out.get("service", [])), len(out.get("price", [])),
                    len(out.get("food", [])), len(out.get("other", [])))
        return out
    except Exception as e:
        logger.warning("compute_recall_seeds_from_file 실패: %s — %s", path, e)
        return None


def compute_recall_seeds_from_reviews(
    reviews: List[Any],
    stopwords: Optional[List[str]] = None,
) -> Optional[Dict[str, List[Tuple[str, int]]]]:
    """전체 리뷰 리스트에서 recall_seeds_for_summary 계산 (파일 없을 때 Qdrant 등 폴백용)."""
    if not reviews or not SPARK_AVAILABLE:
        return None
    if stopwords is None:
        stopwords = _load_stopwords_for_recall(None)
    texts = [
        (r.get("content") or r.get("text") or "") if isinstance(r, dict) else str(r)
        for r in reviews
    ]
    texts = [t for t in texts if t and isinstance(t, str)]
    if not texts:
        return None
    spark = _get_spark()
    rdd = spark.sparkContext.parallelize(
        texts, numSlices=max(1, min(len(texts) // 100, 256))
    )
    return _spark_recall_seeds(rdd, stopwords)


def recall_seeds_to_seed_lists(
    recall_seeds: Optional[Dict[str, List[Tuple[str, int]]]],
) -> Tuple[Optional[List[List[str]]], Optional[List[str]]]:
    """
    recall_seeds_for_summary → [service_seeds, price_seeds, food_seeds], ["service","price","food"].
    phrase만 추출. None/비어 있으면 (None, None).
    """
    if not recall_seeds:
        return None, None
    service_seeds = [p for p, _ in recall_seeds.get("service", [])]
    price_seeds = [p for p, _ in recall_seeds.get("price", [])]
    food_seeds = [p for p, _ in recall_seeds.get("food", [])]
    if not service_seeds and not price_seeds and not food_seeds:
        return None, None
    return [service_seeds, price_seeds, food_seeds], ["service", "price", "food"]


# 카테고리·긍정 키워드 (Kiwi bigram 분류용, Spark worker에서도 사용)
SERVICE_KW = {"친절", "서비스", "응대", "직원", "사장", "불친절"}
PRICE_KW = {"가격", "가성비", "대비", "리필", "무한", "할인", "쿠폰"}
FOOD_KW = {"국수", "냉면", "버거", "치즈", "케이크", "고기", "커피", "피자", "파스타"}
SERVICE_POSITIVE_KW = {"친절"}
PRICE_POSITIVE_KW = {"가성비", "가격 합리", "가격 만족", "무한 리필", "리필 가능"}


def calculate_single_restaurant_ratios(
    reviews: List[str],
    stopwords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    단일 음식점의 카테고리별 긍정 비율 계산 (Spark + Kiwi NNG/NNP bigram). 비-Spark 대체 없음.
    """
    if not reviews:
        return {"service": 0.0, "price": 0.0}
    if not SPARK_AVAILABLE:
        logger.error("pyspark 미설치. ratio 계산 불가.")
        return {"service": 0.0, "price": 0.0}
    texts = [s for s in reviews if s and isinstance(s, str)]
    if not texts:
        return {"service": 0.0, "price": 0.0}
    spark = _get_spark()
    rdd = spark.sparkContext.parallelize(texts, numSlices=max(1, min(len(texts) // 50, 32)))
    out = _spark_calculate_ratios(rdd, stopwords)
    return {"service": round(out["service"], 2), "price": round(out["price"], 2)}


def format_strength_display(lift_service: float, lift_price: float) -> List[str]:
    """
    lift 퍼센트 → strength_display 템플릿 리스트.
    multiple = 1 + lift/100, "판교 평균의 {multiple:.2f}배 수준".
    """
    multiple_service = 1 + lift_service / 100
    multiple_price = 1 + lift_price / 100
    return [
        f"이 음식점의 서비스 만족도는 판교 평균의 {multiple_service:.2f}배 수준입니다.",
        f"이 음식점의 가격 만족도는 판교 평균의 {multiple_price:.2f}배 수준입니다.",
    ]


def calculate_all_average_ratios_from_file(
    path: str,
    stopwords: Optional[List[str]] = None,
    project_root: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    strength_in_aspect와 동일: Spark가 파일을 직접 읽어 전체 평균 비율 계산.
    - .json: content 또는 text 컬럼. option("multiline", True).
    - .tsv/.csv: Review 컬럼 (TSV는 sep=\\t). 없으면 content, text 시도.
    - 64만 건 등 대용량도 드라이버에 올리지 않고 Spark 파티션으로 처리.
    파일 없음/미지원/에러 시 None.
    """
    if not path or not isinstance(path, str):
        return None
    p = Path(path)
    if not p.is_absolute() and project_root:
        p = Path(project_root) / p
    if not p.exists():
        logger.debug(f"aspect_data 파일 없음: {p}")
        return None
    path = str(p)
    if not SPARK_AVAILABLE:
        logger.warning("pyspark 미설치. calculate_all_average_ratios_from_file 불가.")
        return None

    try:
        from pyspark.sql.functions import col, length

        spark = _get_spark()
        if path.lower().endswith(".json"):
            df = spark.read.option("multiline", "true").json(path)
            text_col = next((c for c in ("content", "text") if c in df.columns), None)
            if not text_col:
                logger.warning("calculate_all_average_ratios_from_file: JSON에 content/text 컬럼 없음")
                return None
            base_df = df.select(col(text_col).alias("text")).where(
                col("text").isNotNull() & (length(col("text")) > 0)
            )
        elif path.lower().endswith(".tsv") or path.lower().endswith(".csv"):
            sep = "\t" if path.lower().endswith(".tsv") else ","
            df = spark.read.option("sep", sep).option("header", "true").csv(path)
            text_col = next((c for c in ("Review", "content", "text") if c in df.columns), None)
            if not text_col:
                logger.warning("calculate_all_average_ratios_from_file: TSV/CSV에 Review/content/text 컬럼 없음")
                return None
            base_df = df.select(col(text_col).alias("text")).where(
                col("text").isNotNull() & (length(col("text")) > 0)
            )
        else:
            logger.debug("calculate_all_average_ratios_from_file: 미지원 확장자 %s", path)
            return None

        texts_rdd = base_df.select("text").rdd.map(lambda r: r["text"])
        return _spark_calculate_ratios(texts_rdd, stopwords)
    except Exception as e:
        logger.warning("calculate_all_average_ratios_from_file 실패: %s — %s", path, e)
        return None


def load_reviews_from_aspect_data_file(
    path: str,
    project_root: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    strength_in_aspect와 동일한 aspect_data 파일에서 리뷰 텍스트 리스트 로드.
    - .json: [{"content":"..."}] 또는 {"restaurants":[{"reviews":[...]}]} 형태. content 또는 text 사용.
    - .tsv/.csv: 'Review' 컬럼 사용 (strength_in_aspect TSV와 동일). 없으면 'content','text' 시도.
    project_root: 상대 경로일 때 기준. None이면 path 그대로 사용.
    """
    if not path or not isinstance(path, str):
        return []
    p = Path(path)
    if not p.is_absolute() and project_root:
        p = Path(project_root) / p
    if not p.exists():
        logger.debug(f"aspect_data 파일 없음: {p}")
        return []
    path = str(p)
    out: List[Dict[str, str]] = []
    try:
        if path.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        c = (r.get("content") or r.get("text") or "").strip()
                        if c:
                            out.append({"content": c})
            elif isinstance(data, dict) and "restaurants" in data:
                for rest in data["restaurants"] or []:
                    for r in (rest.get("reviews") or []):
                        if isinstance(r, dict):
                            c = (r.get("content") or r.get("text") or "").strip()
                            if c:
                                out.append({"content": c})
        elif path.lower().endswith((".tsv", ".csv")):
            with open(path, encoding="utf-8", newline="") as f:
                sep = "\t" if path.lower().endswith(".tsv") else ","
                reader = csv.DictReader(f, delimiter=sep)
                rows = list(reader)
            text_col = next((k for k in ("Review", "content", "text") if rows and k in rows[0]), None)
            if text_col:
                for row in rows:
                    c = (row.get(text_col) or "").strip()
                    if c:
                        out.append({"content": c})
        if out:
            logger.info(f"aspect_data에서 리뷰 {len(out)}건 로드: {path}")
    except Exception as e:
        logger.warning(f"aspect_data 로드 실패: {path} — {e}")
    return out


def calculate_all_average_ratios_from_reviews(
    reviews: List[Any],
    stopwords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    전체 리뷰에서 service/price 긍정 비율 계산 (Spark + Kiwi). 비-Spark 대체 없음.
    리뷰: [{"content":"..."} or {"text":"..."}, ...]
    """
    if not reviews:
        return {"service": 0.0, "price": 0.0}
    if not SPARK_AVAILABLE:
        logger.error("pyspark 미설치. 전체 평균 ratio 계산 불가.")
        return {"service": 0.0, "price": 0.0}
    texts = [
        ((r.get("content") or r.get("text") or "") if isinstance(r, dict) else "")
        for r in reviews
    ]
    texts = [t for t in texts if t and isinstance(t, str)]
    if not texts:
        return {"service": 0.0, "price": 0.0}
    spark = _get_spark()
    rdd = spark.sparkContext.parallelize(
        texts, numSlices=max(1, min(len(texts) // 100, 256))
    )
    return _spark_calculate_ratios(rdd, stopwords)


def calculate_strength_lift(
    single_restaurant_ratios: Dict[str, float],
    all_average_ratios: Dict[str, float],
) -> Dict[str, float]:
    """
    Lift 계산: (단일 - 전체) / 전체 × 100
    
    Args:
        single_restaurant_ratios: 단일 음식점의 카테고리별 긍정 비율
            {"service": 0.72, "price": 0.65}
        all_average_ratios: 전체 평균 카테고리별 긍정 비율
            {"service": 0.60, "price": 0.55}
    
    Returns:
        카테고리별 lift 퍼센트
        {"service": 20.0, "price": 18.18}
    """
    lift_dict = {}
    for category in ["service", "price"]:
        single_ratio = single_restaurant_ratios.get(category, 0.0)
        all_ratio = all_average_ratios.get(category, 0.0)
        
        if all_ratio > 0:
            lift = ((single_ratio - all_ratio) / all_ratio) * 100
            lift_dict[category] = round(lift, 2)
        else:
            lift_dict[category] = 0.0
    
    return lift_dict


def _default_description(category: str, lift: float) -> str:
    """템플릿 기반 기본 설명 (LLM 실패 시 폴백)."""
    category_name = "서비스" if category == "service" else "가격"
    if lift > 0:
        return f"이 음식점은 전체 평균 대비 {category_name} 긍정 평가 비율이 {lift}% 높습니다"
    elif lift < 0:
        return f"이 음식점은 전체 평균 대비 {category_name} 긍정 평가 비율이 {abs(lift)}% 낮습니다"
    return f"이 음식점의 {category_name} 긍정 평가 비율은 전체 평균과 유사합니다"


def generate_strength_descriptions(
    lift_dict: Dict[str, float],
    llm_utils: Optional[Any] = None,
    single_restaurant_ratios: Optional[Dict[str, float]] = None,
    all_average_ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    카테고리별 lift 수치를 **근거**로 LLM이 자연어 설명 생성 후 출력.
    
    - 항상 lift 수치를 응답에 반영하고, 해당 수치를 근거로 LLM 설명을 생성한다.
    - LLM 실패 시 템플릿 기반 기본 설명으로 폴백.
    
    Args:
        lift_dict: 카테고리별 lift 퍼센트 {"service": 20.0, "price": 18.18}
        llm_utils: LLMUtils 인스턴스
        single_restaurant_ratios: 단일 음식점 카테고리별 긍정 비율 (근거용)
        all_average_ratios: 전체 평균 카테고리별 긍정 비율 (근거용)
    
    Returns:
        카테고리별 설명 (lift 근거 반영)
    """
    import json
    import os
    import re

    descriptions = {
        cat: _default_description(cat, lift)
        for cat, lift in lift_dict.items()
    }

    if not llm_utils or not lift_dict:
        return descriptions

    try:
        lines = [
            "아래 **카테고리별 lift 수치**를 **근거**로, 각 카테고리에 대해 1문장 자연어 설명을 생성하세요.",
            "반드시 해당 lift(%) 및 비율 수치를 언급한 뒤, 이를 근거로 해석하여 설명하세요.",
            "",
            "카테고리별 근거 수치:",
        ]
        for category in ["service", "price"]:
            if category not in lift_dict:
                continue
            lift = lift_dict[category]
            single = (single_restaurant_ratios or {}).get(category)
            all_avg = (all_average_ratios or {}).get(category)
            name = "서비스" if category == "service" else "가격"
            line = f"- {name}: lift {lift}%"
            if single is not None:
                line += f", 이 음식점 긍정 비율 {single:.2f}"
            if all_avg is not None:
                line += f", 전체 평균 {all_avg:.2f}"
            lines.append(line)

        lines.append("")
        lines.append('각 카테고리별로 1문장 설명을 생성하세요. JSON 형식만 출력:')
        lines.append('{"service": "설명", "price": "설명"}')

        prompt = "\n".join(lines)

        if hasattr(llm_utils, "use_openai") and llm_utils.use_openai:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Generate natural language descriptions **based strictly on** the given lift and ratio statistics. Always cite the numbers as evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or "{}"
        else:
            raw = llm_utils._generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_new_tokens=256,
            )

        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        for k in ["service", "price"]:
            if k in data and isinstance(data[k], str) and data[k].strip():
                descriptions[k] = data[k].strip()
    except Exception as e:
        logger.warning("LLM 설명 생성 실패, 기본 설명 사용: %s", e)

    return descriptions
