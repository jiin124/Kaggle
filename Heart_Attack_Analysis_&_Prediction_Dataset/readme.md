# 필사를 통해 새롭게 알게 된 것 정리



# *파이썬 query*

[[Pandas] Query 함수 사용법 (Pandas의 꽃)](https://m.blog.naver.com/wideeyed/221867273249)

Pandas(판더스 or 팬더스)에서 조건에 부합하는 데이터를 추출할 때

가장 많이 사용하는 Query 함수에 대해 알아보겠습니다.

**장점은 가독성과 편의성이 최대 장점입니다.**

**단점은 .loc[ ] 로 구현한 것보다 속도가 느립니다.**

## Query 함수의 기능

1. 비교연산자(==,>,≥,<,≤,≠)
2. in 연산자
3. 논리연산자(and or not)
4. 외부 변수(또는 함수) 참조 연산
5. 인덱스 검색
6. 문자열 부분검색(str,contarins,str.startswith,str.endswith)

- 실습할 데이터 셋

```python
import pandas as pd
print(pd.__version__)    # 실습할 판다스 버전: 1.0.1
data = {"age": [10, 10, 21, 22], "weight": [20, 30, 60, 70]}
df = pd.DataFrame(data)
display(df)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3727e606-e945-4938-a24f-3d51d6b4b61b/Untitled.png)

```python
str_expr="age==10"#나이가 10이다.
df_q=df.query(str_expr)
df_q
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/137e721b-d36d-4535-90dd-07936896454e/Untitled.png)

# *cross_val_score으로 데이터 사이의 성능 평가하기*

<aside>
🧐 캐글이나 데이콘 분석을 하다보면 파생변수를 더 만들거나 스케일링을 해서 원본데이터와 달라지는 경우가 생긴다. 하지만 파생변수를 만들고 스케일링을 한다고 해서 무조건 성능이 좋아지는 법은 없다는 거.. 그래서 비교를 위해 이런 방법을 사용했다.

</aside>

```python
cross_val_score(
    estimator=RandomForestClassifier(max_depth=8,
                                     n_estimators=100,
                                     random_state=42),
                
    X=df.drop('output',axis=1),
    y=df['output'],
    scoring='f1',
    cv=5
).mean()

0.8506528472990282
```

```python
cross_val_score(
    estimator=RandomForestClassifier(max_depth=8,
                                     n_estimators=100,
                                     random_state=42),
                
    X=df_new.drop('output',axis=1),
    y=df_new['output'],
    scoring='f1',
    cv=5
).mean()

0.829961072815524
```

5번의 교차검증 후 평균을 내서 비교했다. 

# 원핫인코딩

```python
features_ohe=pd.get_dummies(features,drop_first=True,columns=cat_features)
```

- features : 원본데이터
- columns : 카테고리형 데이터
    - cat_features = ['sex', 'cp', 'restecg', 'exng', 'slp', 'thall', 'blood_pres_cat', 'cholesterol_cat', 'pres_chol_sum_cat', 'cat_sum']

# ✨RandomizedSearchCV

[Machine Learning - RandomizedSearchCV, GridSearchCV 정리, 실습, 최적의 하이퍼 파라미터 구하기(Optimal hyper parameters)](https://velog.io/@dlskawns/Machine-Learning-RandomizedSearchCV-GridSearchCV-%EC%A0%95%EB%A6%AC-%EC%8B%A4%EC%8A%B5)

해당 분류기의 최적의 하이퍼 파라미터를 찾기 위한 방법 중 하나.주어진 문제에 대한 분류기들로 모델을 작성한 뒤, 성능 개선을 위한 Tuning을 하는데 일일히 모든 파라미터를 다 조율해보고, 그에 맞는 최적의 조합을 찾아보긴 힘들기 때문에, 오차값이 가장 적은 하이퍼파라미터를 찾아주는 좋은 라이브러리이다.

튜닝하고싶은 파라미터를 지정하여 파라미터 값의 범위를 정하고,`n_iter`값을 지정하여 해당 수 만큼 Random하게 조합하여 반복하는 과정을 거쳐 최종적인 최적 파라미터 값을 가진다.

```python
parameters_rf=dict(
    n_estimators=range(5,1000),
    max_depth=range(4,30),
    min_samples_split=range(2,10),
    min_samples_leaf=range(1,10),
    max_features=range(2,features_train.shape[1])
)
```

```python
%%time
random_search_rf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=parameters_rf,#파라미터 입력
    n_iter=75,#random_search 탐색 횟수
    scoring='f1',#오차 평가 방법
    cv=5,#cv 검증을 위한 분할 검증 횟수
    random_state=42,
    verbose=1
)
random_search_rf.fit(features_train, target_train)
```

`Fitting 5 folds for each of 75 candidates, totalling 375 fits
CPU times: user 4min 54s, sys: 1.79 s, total: 4min 56s
Wall time: 4min 57s`

---
