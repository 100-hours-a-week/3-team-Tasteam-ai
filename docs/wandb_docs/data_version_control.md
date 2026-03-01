
**굳이 안 올려도 됩니다.**

- **labeled** (train_labeled.json, val_labeled.json 등): 학습·평가에 쓰는 **실제 데이터** → 버전 관리하는 게 의미 있음. 지금처럼 labeled만 artifact로 올리면 됨.
- **dataset** (build_dataset 결과: train.json, val.json, test.json): 라벨링 **입력**용 중간 산출물.  
  - “어떤 데이터로 학습했는지” 추적은 labeled만 있어도 가능하고,  
  - 같은 split을 나중에 다시 쓰거나, “데이터 생성 → 라벨링” 전체를 재현하고 싶을 때만 dataset까지 artifact로 두면 됨.

정리하면, **데이터셋(dataset)은 필수로 올릴 필요 없고**, labeled만 올려도 충분합니다.  
전체 파이프라인 재현까지 신경 쓰고 싶다면 그때 dataset artifact를 추가하면 됩니다.

---


분할 전략을 바꿀 때를 생각하면 **데이터셋도 artifact로 두는 게 좋습니다.**

---

## 이유

1. **추적성**  
   labeled는 “어떤 dataset(어떤 split)에서 나왔는지”와 연결돼야 합니다.  
   dataset을 artifact로 올려두면, 나중에 “이 labeled 버전은 이 dataset 버전에서 나왔구나”를 바로 확인할 수 있습니다.

2. **전략 변경 전/후 비교**  
   train/val/test 비율이나 식당 단위 split 방식을 바꾸면,  
   - 예전 dataset artifact: 옛 전략의 split 결과  
   - 새 build_dataset 결과: 새 전략의 split 결과  
   로 두 버전을 남겨두고 비교·분석하기 좋습니다.

3. **재현성**  
   “그때 썼던 split 그대로 다시 쓰고 싶다”면,  
   raw 입력 + 옛 dataset artifact(또는 build_dataset 설정)가 있으면 재현이 쉽습니다.  
   labeled만 있으면 “그 labeled가 어떤 split에서 나왔는지”는 알기 어렵습니다.

---

## 정리

- **학습/평가만** 신경 쓰면: labeled artifact만 있어도 됨.  
- **분할 전략 변경**, **split 추적·비교·재현**까지 신경 쓰면:  
  **dataset도 artifact로 올려두는 게 맞고**, 그때는 dataset이 필요합니다.