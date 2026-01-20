250117

기존

ratio = len(review_list) / total_count positive_count = int(round(positive_count * ratio)) negative_count = len(review_list) - positive_count

현재

total_judged = positive_count + negative_count if total_judged > 0: scale = len(review_list) / total_judged positive_count = round(positive_count * scale) negative_count = round(negative_count * scale)

변경 이유

기존

전체 리뷰에서 긍/부정 합산
전체 리뷰에서 긍정 개수를 보정
나머지 전부 부정
--> 중립 리뷰도 존재하는데, 전체 리뷰 기준으로 보정한걸 전체 리뷰에서 빼버리면 이건 전체는 긍/부정 둘중 하나다라는 소리고, 중립을 무시해서 결국 진짜 실제 리뷰에서 나타나는 정도의 비율인 긍/부정의 비율을 나타내지 않음.

현재

긍/부정/중립(정확히는 Llm이 판단 불가로 처리) 전부 합산.
긍/부정 전체 합에서 긍/부정 비율 반환 하되,
긍/부정 개수를 전체 리뷰 기준으로 보정해서 긍/부정 비율 반환.
--> 즉, 긍/부정으로 판단한 기준에서 비율을 구한거다. 비율 자체의 보정이 이루어졌을 뿐.