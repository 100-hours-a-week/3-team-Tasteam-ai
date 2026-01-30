from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("demo").getOrCreate()

sc = spark.sparkContext

stopwords = (
    spark.sparkContext
         .textFile("/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/hybrid_search/data_preprocessing/stopwords-ko.txt")
         .collect()
)

stopwords_set = set(stopwords)

bc_stopwords = spark.sparkContext.broadcast(stopwords_set)

import re

df = spark.read.option("sep", "\t").option("header", "true").csv("/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/data/kr3.tsv")

rdd = (
    df.select("Review")
      .rdd
      .flatMap(lambda r: re.findall(r"[가-힣A-Za-z0-9]+", r[0]))
      .filter(lambda w: w not in bc_stopwords.value)
      .filter(lambda w: len(w) >= 2)          # 한 글자 제거 (선택)
      .map(lambda w: (w, 1))
      .reduceByKey(lambda a, b: a + b)
)
rdd.collect()

sorted_rdd = rdd.sortBy(lambda x: x[1], ascending=False)
sorted_rdd.take(30)

import re
from itertools import islice

# 1) 불용어 로드 + broadcast (명사 불용어 중심으로)
with open("stopwords-ko.txt", encoding="utf-8") as f:
    stopwords = set(w.strip() for w in f if w.strip())
bc_stop = spark.sparkContext.broadcast(stopwords)

# 2) 파티션 단위로 Kiwi 1번만 초기화해서 명사/NNG/NNP만 추출
def extract_noun_bigrams_partition(rows):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()  # ✅ 파티션당 1번
    stop = bc_stop.value

    for row in rows:
        text = row[0]
        if not text:
            continue

        # (선택) 너무 긴/이상한 문자 정리
        text = re.sub(r"\s+", " ", text).strip()

        tokens = []
        for tok in kiwi.tokenize(text):
            # Kiwi 품사 예: NNG(일반명사), NNP(고유명사)
            if tok.tag in ("NNG", "NNP"):
                w = tok.form
                if len(w) >= 2 and w not in stop:
                    tokens.append(w)

        # 명사 bigram 생성
        for a, b in zip(tokens, tokens[1:]):
            yield (f"{a} {b}", 1)

# 3) 실행 파이프라인
bigram_counts = (
    df.select("Review")
      .rdd
      .mapPartitions(extract_noun_bigrams_partition)
      .reduceByKey(lambda a, b: a + b)
      .filter(lambda kv: kv[0].split()[0] != kv[0].split()[1])
)

top_bigrams = bigram_counts.takeOrdered(50, key=lambda x: -x[1])
top_bigrams[:10]

for i in top_bigrams[:10]:
    print(i[0])

service = ["직원 친절", "사장 친절", "방문 의사"]
price = ["가격 대비", "무한 리필", "방문 의사"]
food = ["가락 국수", "수제 버거", "크림 치즈", "치즈 케이크"]

print({k: v for k, v in top_bigrams[:10]})

categories = {
    "service": set(service),
    "price": set(price),
    "food": set(food),
}

category_json = {k: {} for k in categories}

for phrase, count in top_bigrams:   # ← 슬라이스 제거 권장
    for cat, vocab in categories.items():
        if phrase in vocab:
            category_json[cat][phrase] = count
            
category_json

candidates = bigram_counts.takeOrdered(2000, key=lambda x: -x[1])

def is_noise(phrase):
    a, b = phrase.split()
    if len(a) < 2 or len(b) < 2:
        return True
    return False

candidates = [x for x in candidates if not is_noise(x[0])]

service_kw = ["친절", "서비스", "응대", "직원", "사장", "불친절"]
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
        category_json[lab][phrase] = count
category_json["service"]

service_pairs = category_json["service"]
price_pairs = category_json["price"]
food_pairs = category_json["food"]
other_pairs = category_json["other"]

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

for name, pairs in zip(pairs_name,pairs_list):
    head, mid, tail = quantile_split(pairs, head_q=0.02, mid_q=0.20, min_head=10)

    print(f"\n===== {name} =====")
    print(f"n={len(pairs)} | head={len(head)} | mid={len(mid)} | tail={len(tail)}")

    print("\n[HEAD]")
    print(head[:20])   # 너무 길면 일부만

    print("\n[MID]")
    print(mid[:20])

    print("\n[TAIL]")
    print(tail[:20])

import random, math

def pick_seeds(head, mid, tail, mid_k=5, tail_k=1, seed=42):
    random.seed(seed)

    # head는 전부 포함
    seeds = [p for p, _ in head]

    # mid는 가중 샘플링 (log(count))
    if mid:
        weights = [math.log(c + 1) for _, c in mid]
        idxs = list(range(len(mid)))
        for _ in range(min(mid_k, len(mid))):
            total = sum(weights[i] for i in idxs)
            r = random.random() * total
            acc = 0.0
            for i in idxs:
                acc += weights[i]
                if acc >= r:
                    seeds.append(mid[i][0])
                    idxs.remove(i)
                    break

    # tail은 랜덤 (탐색용)
    if tail and tail_k > 0:
        picks = random.sample(tail, k=min(tail_k, len(tail)))
        seeds.extend([p for p, _ in picks])

    # 중복 제거(순서 유지)
    seen = set()
    seeds = [s for s in seeds if not (s in seen or seen.add(s))]

    return seeds


for name, pairs in zip(pairs_name, pairs_list):
    head, mid, tail = quantile_split(pairs)
    seeds = pick_seeds(head, mid, tail, mid_k=5, tail_k=1)

    print(f"\n{name} seeds ({len(seeds)}):")
    print(seeds)


