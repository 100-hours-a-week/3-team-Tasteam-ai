# data
from datasets import load_dataset
import json
import pandas as pd

dataset = json.load(open("/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/data/test_data_sample.json","r"))
df = pd.DataFrame(dataset)


# embedding
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model = TextEmbedding('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
dense_embeddings = list(dense_model.embed(df["content"]))

sparse_model = SparseTextEmbedding('Qdrant/bm25')
sparse_embeddings = list(sparse_model.embed(df["content"]))


# qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector

def setup_qdrant_hybrid(passages, dense_embeddings, sparse_embeddings):
    client = QdrantClient('http://localhost:6333')
    collection_name = "hybrid_test"
    dense_dim = len(dense_embeddings[0])
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name = collection_name,
        vectors_config = {
            "dense" : VectorParams(
                size = dense_dim,
                distance = Distance.COSINE
            )
        },
        sparse_vectors_config = {
            "sparse" : SparseVectorParams()
        }
    )
    
    points = []
    for i, (passage, dense_emb, sparse_emb) in enumerate(zip(passages, dense_embeddings, sparse_embeddings)):
        
        dense_vector = list(dense_emb)
    
        sparse_vector = SparseVector(
            indices = list(sparse_emb.indices),
            values = list(sparse_emb.values)
        )
        
        point = PointStruct(
            id = i,
            vector = {
                "dense" : dense_vector,
                "sparse" : sparse_vector
            },
            payload = passages[i] # 딕서너리 -> qdrant 가능, 딕셔너리 <-> 판다스 df, 판다스 df -> qdrant payload 가능
        )
        points.append(point)
        
    client.upsert(collection_name=collection_name, points=points)
    return client

client = setup_qdrant_hybrid(dataset, dense_embeddings, sparse_embeddings)



# query
from qdrant_client import models

def query_qdrant_hybrid(
    client,
    query_dense_emb,    # list[float] or np.ndarray
    query_sparse_emb,   # fastembed SparseEmbedding (indices, values)
    limit=5,
):
    dense_vec = list(query_dense_emb)
    sparse_vec = models.SparseVector(
        indices=list(query_sparse_emb.indices),
        values=list(query_sparse_emb.values),
    )

    res = client.query_points(
        collection_name="hybrid_test",
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
            ),
        ],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="food_category_name",
                    match=models.MatchValue(value="일식")
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    return res.points


dense_text = "이 음식점의 음식 맛, 가격/가성비, 서비스/응대에 대한 전반적인 리뷰"

# seeds (기본값; recall_seeds_for_summary가 있으면 대체)
_DEFAULT_service_seeds = ['직원 친절', '사장 친절', '친절 기분', '서비스 친절', '사장 직원', '직원 서비스', '아주머니 친절']
_DEFAULT_price_seeds = ['가격 대비', '무한 리필', '가격 생각', '음식 가격', '합리 가격', '메뉴 가격', '가격 만족', '가격 퀄리티', '리필 가능', '커피 가격', '가격 사악', '런치 가격']
_DEFAULT_food_seeds = ['가락 국수', '평양 냉면', '수제 버거', '크림 치즈', '치즈 케이크', '크림 파스타', '당근 케이크', '오일 파스타', '카페 커피', '비빔 냉면', '커피 원두', '리코타 치즈', '비빔 막국수', '치즈 돈가스', '커피 산미', '치즈 파스타']


def _recall_seeds_to_seed_lists(recall_seeds_for_summary):
    """
    strength_in_aspect.recall_seeds_for_summary → [service_seeds, price_seeds, food_seeds], name_list
    recall_seeds_for_summary: {"service": [(phrase, count), ...], "price": [...], "food": [...], "other": [...]}
    """
    if not recall_seeds_for_summary:
        return None, None
    service_seeds = [p for p, c in recall_seeds_for_summary.get("service", [])]
    price_seeds = [p for p, c in recall_seeds_for_summary.get("price", [])]
    food_seeds = [p for p, c in recall_seeds_for_summary.get("food", [])]
    return [service_seeds, price_seeds, food_seeds], ["service", "price", "food"]


def run_hybrid_search(recall_seeds_for_summary=None):
    """
    하이브리드 서치: recall_seeds_for_summary가 주어지면 그걸 쿼리 시드로 사용.
    strength_in_aspect.run_strength_in_aspect()의 recall_seeds_for_summary를 넘기면 됨.
    """
    if recall_seeds_for_summary:
        seed_list, hits_name = _recall_seeds_to_seed_lists(recall_seeds_for_summary)
    else:
        seed_list = [_DEFAULT_service_seeds, _DEFAULT_price_seeds, _DEFAULT_food_seeds]
        hits_name = ["service", "price", "food"]

    hits_dict = {}
    for seeds, name in zip(seed_list, hits_name):
        print(f"\n{name} seeds ({len(seeds)}):")
        query_text = " ".join(seeds[:-1]) if len(seeds) > 1 else (" ".join(seeds) if seeds else name)
        dense_q = next(dense_model.embed(query_text))
        sparse_q = next(sparse_model.embed(query_text))
        hits = query_qdrant_hybrid(client, dense_q, sparse_q, limit=5)
        hits_dict[name] = []
        for rank, h in enumerate(hits):
            print(h.payload["content"], "score:", h.score)
            review_id = h.payload.get("review_id") or h.payload.get("id") or str(h.id)
            content = h.payload.get("content", "")
            hits_dict[name].append({
                "review_id": str(review_id),
                "snippet": content,
                "rank": rank,
                "content": content,
            })

    final_seeds_dict = {k: [item["content"] for item in v if "content" in item] for k, v in hits_dict.items()}
    hits_data_dict = {
        k: [{"review_id": item["review_id"], "snippet": item["snippet"], "rank": item["rank"]} for item in v]
        for k, v in hits_dict.items()
    }
    return hits_dict, hits_data_dict, final_seeds_dict


# 하위 호환: main() 호출 전 참조 가능 (main에서 채움)
hits_dict = {}
hits_data_dict = {}
final_seeds_dict = {}


def main(recall_seeds_for_summary=None):
    """
    recall_seeds_for_summary: strength_in_aspect의 recall_seeds_for_summary를 넘기면
    summary 하이브리드 서치의 쿼리(시드)로 사용. None이면 기존 기본 시드 사용.
    """
    global hits_dict, hits_data_dict, final_seeds_dict
    hits_dict, hits_data_dict, final_seeds_dict = run_hybrid_search(recall_seeds_for_summary=recall_seeds_for_summary)

    service_evidence = hits_data_dict.get("service", [])
    price_evidence = hits_data_dict.get("price", [])
    food_evidence = hits_data_dict.get("food", [])

    out = summarize_aspects(
        final_seeds_dict["service"],
        final_seeds_dict["price"],
        final_seeds_dict["food"],
        service_evidence_data=service_evidence,
        price_evidence_data=price_evidence,
        food_evidence_data=food_evidence,
    )
    return out


# summary_result
import os, json, re
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PRICE_HINTS = [
    "가격", "가성비", "저렴", "비싸", "비쌈", "가격대", "합리", "구성", "구성비",
    "양", "푸짐", "리필", "무한", "만족", "혜자"
]

def _clip(xs, n=8):
    xs = [x.strip() for x in xs if x and str(x).strip()]
    return xs[:n]

def _has_price_signal(text: str) -> bool:
    t = text.replace(" ", "")
    return any(k.replace(" ", "") in t for k in PRICE_HINTS)

def summarize_aspects(
    service_reviews: list[str],
    price_reviews: list[str],
    food_reviews: list[str],
    service_evidence_data: list[dict] = None,
    price_evidence_data: list[dict] = None,
    food_evidence_data: list[dict] = None,
    model: str = "gpt-5",
    per_category_max: int = 8,
    llm_utils=None,
) -> dict:

    payload = {
        "service": _clip(service_reviews, per_category_max),
        "price": _clip(price_reviews, per_category_max),
        "food": _clip(food_reviews, per_category_max),
    }

    instructions = """
너는 음식점 리뷰 분석가다.
입력으로 카테고리별 '근거 리뷰 목록'이 주어진다.
아래 JSON 스키마로만 출력하라(추가 텍스트 금지).

스키마:
{
  "service": {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "price":   {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "food":    {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "overall_summary": {"summary": string}
}

규칙:
- 각 카테고리 summary: 1문장, 과장 금지
- bullets: 3~5개, 중복 제거, 구체적으로
- evidence: 근거로 쓴 리뷰의 인덱스(각 카테고리 리스트에서 0-based)
- price는 '가격 숫자'가 없으면 '가성비/양/구성/만족감' 같은 우회표현을 근거로 요약하라.
- overall_summary는 2~3문장으로 종합 요약하라.
- 근거(입력 리뷰)에 없는 내용은 추측하지 말고 "언급이 적다"라고 표현하라.
"""

    def call_openai_responses():
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
        return resp.output_text.strip()

    # 1) 1차 호출: llm_utils가 있으면 _generate_response, 없으면 OpenAI Responses API
    if llm_utils and hasattr(llm_utils, "_generate_response"):
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        text = llm_utils._generate_response(
            messages=messages,
            temperature=0.1,
            max_new_tokens=1000,
        ).strip()
    else:
        text = call_openai_responses()

    # 2) JSON 파싱 + 실패 시 1회 재시도
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        if llm_utils and hasattr(llm_utils, "_generate_response"):
            fix = llm_utils._generate_response(
                messages=[{"role": "user", "content": f"다음 텍스트를 유효한 JSON으로만 변환: {text}"}],
                temperature=0.1,
                max_new_tokens=500,
            ).strip()
        else:
            fix = client.responses.create(
                model=model,
                reasoning={"effort": "low"},
                instructions="다음 텍스트를 '유효한 JSON'으로만 변환해. 다른 말 금지.",
                input=text,
            ).output_text.strip()
        out = json.loads(fix)

    # 3) evidence 인덱스를 실제 evidence 객체로 변환
    evidence_data_map = {
        "service": service_evidence_data or [],
        "price": price_evidence_data or [],
        "food": food_evidence_data or [],
    }
    
    for cat in ("service", "price", "food"):
        n = len(payload[cat])
        ev_indices = out.get(cat, {}).get("evidence", [])
        if isinstance(ev_indices, list):
            # 인덱스 검증 및 변환
            valid_indices = [i for i in ev_indices if isinstance(i, int) and 0 <= i < n]
            # 인덱스를 evidence 객체로 변환
            evidence_data = evidence_data_map[cat]
            out[cat]["evidence"] = [
                {
                    "review_id": evidence_data[i]["review_id"],
                    "snippet": evidence_data[i]["snippet"],
                    "rank": evidence_data[i]["rank"]
                }
                for i in valid_indices
                if i < len(evidence_data)
            ]
        else:
            out[cat]["evidence"] = []

    # 4) price 요약 게이트: evidence 리뷰들에 price 신호가 전혀 없으면 안전하게 다운그레이드
    price_ev = out.get("price", {}).get("evidence", [])
    # evidence는 이제 객체 리스트이므로 snippet을 직접 가져옴
    ev_texts = [ev.get("snippet", "") for ev in price_ev if isinstance(ev, dict)] if price_ev else []
    if ev_texts and not any(_has_price_signal(t) for t in ev_texts):
        out["price"]["summary"] = "가격 관련 언급이 많지 않아, 전반적인 만족감/구성(양 등) 중심으로만 해석 가능합니다."
        # bullets도 너무 단정적으로 쓰지 않게 최소화
        out["price"]["bullets"] = [
            "가격을 직접 언급한 리뷰가 많지 않습니다.",
            "대신 만족/구성/양(푸짐함) 관련 표현이 간접적으로 나타납니다."
        ]

    return out


if __name__ == "__main__":
    # recall_seeds_for_summary=None → 기본 시드 사용.
    # strength_in_aspect에서 호출할 때: main(recall_seeds_for_summary=res["recall_seeds_for_summary"])
    out = main(recall_seeds_for_summary=None)
    print(json.dumps(out, ensure_ascii=False, indent=2))


