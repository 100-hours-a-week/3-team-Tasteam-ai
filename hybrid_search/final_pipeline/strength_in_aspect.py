import os, sys

PY = sys.executable  # 현재 실행중인 파이썬(=driver)과 동일하게 맞춤
os.environ["PYSPARK_PYTHON"] = PY
os.environ["PYSPARK_DRIVER_PYTHON"] = PY

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
import re
from kiwipiepy import Kiwi
from itertools import islice
import random
import math
from pyspark.sql.functions import col, length

spark = (
    SparkSession.builder
    .appName("demo")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .master("local[6]")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.shuffle.partitions", "12")
    .getOrCreate()
)


def total_asepct_ratio(aspect_data, stopwords_data, target_rid=None):

    # 1) stopwords broadcast
    with open(stopwords_data, encoding="utf-8") as f:
        stopwords = set(w.strip() for w in f if w.strip())
    bc_stop = spark.sparkContext.broadcast(stopwords)

    # 2) 로딩 + 분기
    if aspect_data.endswith(".json"):
        df = spark.read.option("multiline", "true").json(aspect_data)

        # ✅ rid 사용 (JSON only)
        base_df = df.select(
            col("restaurant_id").alias("rid"),
            col("content").alias("text")
        ).where(col("text").isNotNull() & (length(col("text")) > 0))

        # ✅ JSON에서만 restaurant_id 필터
        if target_rid is not None:
            # 타입 이슈 방지: 둘 다 string으로 비교
            base_df = base_df.withColumn("rid", col("rid").cast("string")) \
                             .filter(col("rid") == str(target_rid))

        def extract_partition(rows):
            kiwi = Kiwi()
            stop = bc_stop.value

            for r in rows:
                text = r["text"]
                if not text:
                    continue

                text = re.sub(r"\s+", " ", text).strip()

                tokens = []
                for tok in kiwi.tokenize(text):
                    if tok.tag in ("NNG", "NNP"):
                        w = tok.form
                        if len(w) >= 2 and w not in stop:
                            tokens.append(w)

                for a, b in zip(tokens, tokens[1:]):
                    yield (f"{a} {b}", 1)

        bigrams = (
            base_df.select("text").rdd
                  .mapPartitions(extract_partition)
                  .persist(StorageLevel.MEMORY_AND_DISK)
        )

    elif aspect_data.endswith(".tsv") or aspect_data.endswith(".csv"):
        df = (spark.read.option("sep", "\t")
                      .option("header", "true")
                      .csv(aspect_data))

        text_df = df.select(col("Review").alias("text")) \
                    .where(col("text").isNotNull() & (length(col("text")) > 0))

        def extract_partition(rows):
            kiwi = Kiwi()
            stop = bc_stop.value

            for r in rows:
                text = r["text"]
                if not text:
                    continue

                text = re.sub(r"\s+", " ", text).strip()

                tokens = []
                for tok in kiwi.tokenize(text):
                    if tok.tag in ("NNG", "NNP"):
                        w = tok.form
                        if len(w) >= 2 and w not in stop:
                            tokens.append(w)

                for a, b in zip(tokens, tokens[1:]):
                    yield (f"{a} {b}", 1)

        bigrams = (
            text_df.rdd
                   .mapPartitions(extract_partition)
                   .persist(StorageLevel.MEMORY_AND_DISK)
        )

    else:
        raise ValueError(f"Unsupported file: {aspect_data}")

    # 3) 캐시 채우기
    bigrams.count()

    # 4) 집계 + 안전 필터
    bigram_counts = (
        bigrams
        .reduceByKey(lambda a, b: a + b)
        .filter(lambda kv: len(kv[0].split()) == 2 and kv[0].split()[0] != kv[0].split()[1])
    )

    candidates = bigram_counts.takeOrdered(2000, key=lambda x: -x[1])

    bigrams.unpersist()

    def is_noise(phrase):
        sp = phrase.split()
        if len(sp) != 2:
            return True
        a, b = sp
        return len(a) < 2 or len(b) < 2

    candidates = [x for x in candidates if not is_noise(x[0])]




    service_kw = ["친절", "서비스", "응대", "직원", "사장", "불친절"] # 전체 포함 (요약용 및 전체 개수 확보)
    price_kw   = ["가격", "가성비", "대비", "리필", "무한", "할인", "쿠폰"]
    food_kw    = ["국수", "냉면", "버거", "치즈", "케이크", "고기", "커피", "피자", "파스타"]

    def classify(phrase):
        labels = []
        if any(k in phrase for k in service_kw): labels.append("service")
        if any(k in phrase for k in price_kw):   labels.append("price")
        if any(k in phrase for k in food_kw):    labels.append("food")
        if not labels: labels.append("other")
        return labels

    category_json = {"service": {}, "price": {}, "food": {}, "other": {}}

    for phrase, count in candidates:
        for lab in classify(phrase):
            category_json[lab][phrase] = category_json[lab].get(phrase, 0) + count

    service_pairs = category_json["service"]
    price_pairs = category_json["price"]
    food_pairs = category_json["food"]
    other_pairs = category_json["other"]

    # ✅ 전체 카테고리 총량(분모) - category_json 채워진 후 계산
    total_by_category = {k: sum(v.values()) for k, v in category_json.items()}
    overall_total = sum(total_by_category.values())

    # ✅ 긍정 필터(분자)
    FRIENDLY_SERVICE_TOKENS = {"친절"}
    FRIENDLY_PRICE_TOKENS   = {"가성비", "가격 합리", "가격 만족","무한 리필","리필 가능"}

    service_pos = sum(
        cnt for phrase, cnt in category_json["service"].items()
        if any(tok in phrase.split() for tok in FRIENDLY_SERVICE_TOKENS)
    )
    price_pos = sum(
        cnt for phrase, cnt in category_json["price"].items()
        if any(tok in phrase for tok in FRIENDLY_PRICE_TOKENS)
    )

    def safe_div(a, b):
        return round(a / b, 4) if b else 0.0

    metrics = {
        "total_by_category": total_by_category,
        "overall_total": overall_total,
        "service_positive_count": service_pos,
        "price_positive_count": price_pos,
        "service_positive_ratio": safe_div(service_pos, total_by_category["service"]),
        "price_positive_ratio": safe_div(price_pos, total_by_category["price"]),
    }

    def quantile_split(pairs, head_q=0.02, mid_q=0.20, min_head=10):
        if not pairs:
            return [], [], []

        # ✅ dict이면 items로 변환
        if isinstance(pairs, dict):
            pairs = list(pairs.items())

        # ✅ 타입 검증
        if isinstance(pairs[0], str):
            raise ValueError("pairs must be [(phrase, count), ...] or dict{phrase:count}")

        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        n = len(pairs_sorted)

        head_n = max(min_head, int(n * head_q))
        head_n = min(head_n, n)

        mid_end = int(n * mid_q)
        mid_end = max(mid_end, head_n)
        mid_end = min(mid_end, n)

        head = pairs_sorted[:head_n]
        mid  = pairs_sorted[head_n:mid_end]
        tail = pairs_sorted[mid_end:]
        return head, mid, tail

    pairs_list = [service_pairs, price_pairs, food_pairs, other_pairs]
    pairs_name = ["service", "price", "food", "other"]

    def pick_seeds_pairs(head, mid, tail, mid_k=5, tail_k=1, seed=42):
        random.seed(seed)

        # ✅ head는 전부 포함 (p,c) 그대로
        seeds = list(head)

        # ✅ mid는 가중 샘플링 (log(count))
        if mid:
            idxs = list(range(len(mid)))
            weights = [math.log(c + 1) for _, c in mid]

            for _ in range(min(mid_k, len(mid))):
                total = sum(weights[i] for i in idxs)
                r = random.random() * total
                acc = 0.0
                for i in idxs:
                    acc += weights[i]
                    if acc >= r:
                        seeds.append(mid[i])   # ✅ (p,c) 그대로
                        idxs.remove(i)
                        break

        # ✅ tail은 랜덤 (탐색용)
        if tail and tail_k > 0:
            picks = random.sample(tail, k=min(tail_k, len(tail)))
            seeds.extend(picks)  # ✅ (p,c) 그대로

        # ✅ 중복 제거(phrase 기준, 순서 유지)
        seen = set()
        out = []
        for p, c in seeds:
            if p not in seen:
                out.append((p, c))
                seen.add(p)

        return out


    def dedup_reversed_bigrams_pairs(pairs):
        # pairs: [( "a b", count ), ...]
        order = []         # canonical key의 등장 순서 유지
        agg = {}           # canonical key -> {"repr":원문표현, "count":합}

        for phrase, cnt in pairs:
            a, b = phrase.split()
            key = tuple(sorted([a, b]))  # 순서 무시용 canonical key

            if key not in agg:
                agg[key] = {"repr": phrase, "count": cnt}
                order.append(key)
            else:
                agg[key]["count"] += cnt  # ✅ 합산 (원하면 max로 변경 가능)

        return [(agg[k]["repr"], agg[k]["count"]) for k in order]


    seeds_list = []
    for name, pairs in zip(pairs_name, pairs_list):
        head, mid, tail = quantile_split(pairs)
        seeds = pick_seeds_pairs(head, mid, tail, mid_k=5, tail_k=1)
        seeds_list.append(seeds)

    all_seeds = {"service": [], "price": [], "food": [], "other": []}
    for seed_pairs, name in zip(seeds_list, pairs_name):
        cleaned = dedup_reversed_bigrams_pairs(seed_pairs)
        all_seeds[name] = cleaned

    result_json = {
        "metrics": metrics,
        "recall_seeds_for_summary": all_seeds,
    }
    return result_json

def lift(a, total):
    return 0.0 if total == 0 else (a - total) / total * 100


# ------------------------------ 비-Spark 버전 (API/라이브러리용) ------------------------------
SERVICE_KW = ["친절", "서비스", "응대", "직원", "사장", "불친절"]
PRICE_KW   = ["가격", "가성비", "대비", "리필", "무한", "할인", "쿠폰"]
FRIENDLY_SERVICE_TOKENS = {"친절"}
FRIENDLY_PRICE_TOKENS   = {"가성비", "가격 합리", "가격 만족", "무한 리필", "리필 가능"}


def _classify(phrase):
    labels = []
    if any(k in phrase for k in SERVICE_KW): labels.append("service")
    if any(k in phrase for k in PRICE_KW):   labels.append("price")
    if not labels: labels.append("other")
    return labels


def total_asepct_ratio_from_reviews(reviews, stopwords_path=None, stopwords_list=None):
    """
    리뷰 리스트로 service/price 긍정 비율만 계산 (Spark 없이, Kiwi만).
    strength_in_aspect의 total_asepct_ratio와 동일한 classify·FRIENDLY 로직.
    reviews: [{"content": "..."}, ...] 또는 [{"content":"...", "restaurant_id": int}, ...]
    stopwords_path 또는 stopwords_list 둘 중 하나 필요.
    """
    if stopwords_path:
        with open(stopwords_path, encoding="utf-8") as f:
            stop = set(w.strip() for w in f if w.strip())
    elif stopwords_list is not None:
        stop = set(w.strip() for w in stopwords_list if w and str(w).strip())
    else:
        stop = set()

    kiwi = Kiwi()
    category = {"service": {}, "price": {}}

    for r in reviews:
        text = (r.get("content") or r.get("text") or "")
        if not text or not isinstance(text, str):
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        tokens = []
        for tok in kiwi.tokenize(text):
            if tok.tag in ("NNG", "NNP"):
                w = tok.form
                if len(w) >= 2 and w not in stop:
                    tokens.append(w)
        for a, b in zip(tokens, tokens[1:]):
            if a == b or len(a) < 2 or len(b) < 2:
                continue
            phrase = f"{a} {b}"
            for lab in _classify(phrase):
                if lab in category:
                    category[lab][phrase] = category[lab].get(phrase, 0) + 1

    total_s = sum(category["service"].values())
    total_p = sum(category["price"].values())
    service_pos = sum(
        c for phrase, c in category["service"].items()
        if any(t in phrase.split() for t in FRIENDLY_SERVICE_TOKENS)
    )
    price_pos = sum(
        c for phrase, c in category["price"].items()
        if any(t in phrase for t in FRIENDLY_PRICE_TOKENS)
    )

    def _div(a, b):
        return round(a / b, 4) if b else 0.0

    return {
        "service_positive_ratio": _div(service_pos, total_s),
        "price_positive_ratio": _div(price_pos, total_p),
    }


def format_strength_display(lift_service, lift_price):
    """lift → strength_display 템플릿 리스트 (최신 파이프라인과 동일)."""
    return [
        f"이 음식점의 서비스 만족도는 {lift_service:.2f}% 높습니다.",
        f"이 음식점의 가격 만족도는 {lift_price:.2f}% 높습니다.",
    ]


# ---------------------------------------------------------------------------------------------


def run_strength_in_aspect(
    aspect_data,
    stopwords_data,
    target_data,
    target_rid=4,
):
    """
    강점 추출: 전체 vs 타겟 lift, recall_seeds_for_summary, strength_display 반환.
    - recall_seeds_for_summary: summary 하이브리드 서치 쿼리로 전달
    - strength_display: lift_service, lift_price 템플릿 문장
    """
    spark.read.option("multiline", "true").json(target_data).printSchema()

    total_aspect_metric = total_asepct_ratio(aspect_data, stopwords_data)
    target_aspect_metric = total_asepct_ratio(target_data, stopwords_data, target_rid=target_rid)

    total_service = total_aspect_metric["metrics"]["service_positive_ratio"]
    total_price   = total_aspect_metric["metrics"]["price_positive_ratio"]
    all_seeds_for_summary = total_aspect_metric["recall_seeds_for_summary"]

    a_service = target_aspect_metric["metrics"]["service_positive_ratio"]
    a_price   = target_aspect_metric["metrics"]["price_positive_ratio"]

    lift_service = lift(a_service, total_service)
    lift_price   = lift(a_price, total_price)
    multiple_service = 1 + lift_service / 100
    multiple_price = 1 + lift_price / 100
    
    STRENGTH_TEMPLATE_SERVICE = "이 음식점의 서비스 만족도는 판교 평균의 {multiple_service:.2f}배 수준입니다."
    STRENGTH_TEMPLATE_PRICE   = "이 음식점의 가격 만족도는 판교 평균의 {multiple_price:.2f}배 수준입니다."
    strength_display = [
        STRENGTH_TEMPLATE_SERVICE.format(multiple_service=multiple_service),
        STRENGTH_TEMPLATE_PRICE.format(multiple_price=multiple_price),
    ]

    return {
        "recall_seeds_for_summary": all_seeds_for_summary,
        "lift_service": lift_service,
        "lift_price": lift_price,
        "strength_display": strength_display,
    }


# --- 실행 & recall_seeds를 summary 하이브리드 서치 쿼리로 전달 ---
if __name__ == "__main__":
    _aspect_data = "/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/data/kr3.tsv"
    _stopwords_data = "/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/hybrid_search/data_preprocessing/stopwords-ko.txt"
    _target_data = "/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/data/test_data_sample.json"

    final_result_json = run_strength_in_aspect(_aspect_data, _stopwords_data, _target_data, target_rid=4)

    # 강점 추출: 템플릿 표시
    for s in final_result_json.get("strength_display", []):
        print(s)

    # recall_seeds_for_summary → summary 로직의 하이브리드 서치 쿼리로 전달
    try:
        from final_summary_pipeline import main as summary_main
        summary_main(recall_seeds_for_summary=final_result_json["recall_seeds_for_summary"])
    except Exception as e:
        print("Summary pipeline (recall_seeds 전달) skip:", e)



