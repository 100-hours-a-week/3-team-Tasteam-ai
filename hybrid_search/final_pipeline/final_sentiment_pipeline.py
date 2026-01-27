#data
import json

data = json.load(open("/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/data/test_data_sample.json","r"))


# sentiment model
from transformers import pipeline
classifier = pipeline("text-classification", model="Dilwolf/Kakao_app-kr_sentiment")

negative_reviews = []

for review in data:
    if review["restaurant_id"] == 4:
        preds = classifier(review['content'], return_all_scores=True)
        score = preds[0][1]['score']
        review['positive_score'] = score
        review['is_positive'] = score > 0.8
        if review['is_positive']:
            review["sentiment"] = "positive"
        else:
            negative_reviews.append(review)
            review["sentiment"] = "negative"
        #print(review["sentiment"])
        
 # 1) 식당 필터
filtered = [r for r in data if r.get("restaurant_id") == 4]

# 2) LLM에 보낼 대상(기존 negative만 재판정)
targets = [{"id": r["id"], "content": r["content"]} for r in filtered if r.get("sentiment") == "negative"]

# 3) LLM 입력 문자열
llm_input = "\n".join([f'{t["id"]}\t{t["content"]}' for t in targets])

import openai, os, json
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 4) LLM 호출
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a sentiment classification engine.\n"
                "Each input line is formatted as: id<TAB>review.\n"
                "Classify sentiment as one of: positive, negative, neutral.\n"
                "Return ONLY a valid JSON array like:\n"
                '[{"id":105,"sentiment":"positive"}]'
            )
        },
        {"role": "user", "content": llm_input}
    ],
    temperature=0
)

raw = response.choices[0].message.content.strip()
raw = raw[raw.find("["): raw.rfind("]")+1]  # JSON 방어
results = json.loads(raw)

# 5) id -> sentiment 맵
sentiment_map = {x["id"]: x["sentiment"] for x in results}

# 6) 원본(filtered)에 덮어쓰기 (핵심)
for r in filtered:
    if r.get("id") in sentiment_map:
        r["sentiment"] = sentiment_map[r["id"]]

filtered  # 원본 dict에 sentiment가 업데이트된 상태

# 비율 산출
positive_count = 0
negative_count = 0
neutral_count = 0
total_count = 0
for i in filtered:
    if i.get("sentiment") == "positive":
        positive_count += 1
    elif i.get("sentiment") == "negative":
        negative_count += 1
    elif i.get("sentiment") == "neutral":
        neutral_count += 1
    total_count += 1
    

positive_rate = positive_count / (positive_count + negative_count)
neutral_rate = neutral_count / (positive_count + negative_count + neutral_count)
negative_rate = 1 - positive_rate

positive_rate = round(positive_rate, 2)
neutral_rate = round(neutral_rate, 2)
negative_rate = round(negative_rate, 2)

print(round(positive_rate, 2))
print(round(neutral_rate, 2))
print(round(negative_rate, 2))

print({"restaurant id": 4})
print({"positive count": positive_count})
print({"negative count": negative_count})
print({"neutral count": neutral_count})
print({"total count": total_count})
print({"positive rate": positive_rate})
print({"neutral rate": neutral_rate})
print({"negative rate": negative_rate})


result_json = {
    "restaurant id": 4,
    "positive count": positive_count,
    "negative count": negative_count,
    "neutral count": neutral_count,
    "total count": total_count,
    "positive rate": positive_rate,
    "neutral rate": neutral_rate,
    "negative rate": negative_rate

}

print(result_json)