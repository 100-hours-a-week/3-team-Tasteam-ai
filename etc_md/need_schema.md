REVIEW TABLE

id BIGINT PK

restaurant_id BIGINT FK

member_id BIGINT # 각 회원에게 부여되는 고유 id FK

group_id BIGINT # 예시 “10234”, “12034” FK

subgroup_id BIGINT # ex) “10234”, “12034” FK

content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 

is_recommended BOOLEAN # 0, 1

created_at TIMESTAMP

updated_at TIMESTAMP

deleted_at TIMESTAMP

REVIEW_IMAGE TABLE

id bigint PK

review_id bigint FK

image_url varchar(500)

created_at Timestamp

RESTAURANT TABLE

id BIGINT

name VARCHAR(100)

full_address VARCHAR(255)

location geometry(Point,4326)

created_at TIMESTAMP

deleted_at TIMESTAMP

RESTAURANT_FOOD_CATEGORY

id BIGINT PK

restaurant_id BIGINT FK

food_category_id BIGINT FK

FOOD CATEGORY TABLE

id BIGINT

name VARCHAR(20)

---

1. sentiment
    1. 로직
        1. sentiment에선 llm이 리뷰 단위를 반환하는게 아님. 총 리뷰(입력 토큰이 많으면 context middle lost 문제 때문에 배치로 할수도 있음)에서 긍/부정 개수를 세고, 그리고 positive_ratio = positive_count / total_count, negative_ratio = negative_count / total_count 등의 계산으로 비율까지 추가해서 반환
    2. 입력
        1. id BIGINT PK
        
        restaurant_id BIGINT FK
        
        member_id BIGINT # 각 회원에게 부여되는 고유 id FK
        
        group_id BIGINT # 예시 “1234”, “2345” FK
        
        subgroup_id BIGINT # ex) “10234”, “20345” FK
        
        content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
        
        is_recommended BOOLEAN # 0, 1
        
        created_at TIMESTAMP
        
        updated_at TIMESTAMP
        
        deleted_at TIMESTAMP
        
    3. llm 입력
        1. restaurant_id
        2. content_list = [content1, content2, content3, …]
    4. 출력
        1. restaurant_id = 
        2. positive_count = 
        3. negative_count = 
        4. total_count = 
    5. 최종 출력 ( positive_ratio = positive_count / total_count, negative_ratio = negative_count / total_count )
        1. restaurant_id = 
        2. positive_count = 
        3. negative_count = 
        4. total_count = 
        5. positive_ratio = 
        6. negative_ratio = 
2. summary
    1. vector upload
        1. 벡터 데이터
            1. id BIGINT PK
            
            restaurant_id BIGINT FK
            
            member_id BIGINT # 각 회원에게 부여되는 고유 id FK
            
            group_id BIGINT # 예시 “1234”, “2234” FK
            
            subgroup_id BIGINT # ex) “10234”, “12034” FK
            
            content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
            
            is_recommended BOOLEAN # 0, 1
            
            created_at TIMESTAMP
            
            updated_at TIMESTAMP
            
            deleted_at TIMESTAMP
            
    2. vector_search
        1. 벡터 데이터
            1. id BIGINT PK
            
            restaurant_id BIGINT FK
            
            member_id BIGINT # 각 회원에게 부여되는 고유 id FK
            
            group_id BIGINT # 예시 “kakao_1234”, “naver_1234” FK
            
            subgroup_id BIGINT # ex) “kakao_algorithm_1234”, “naver_socer_1234” FK
            
            content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
            
            is_recommended BOOLEAN # 0, 1
            
            created_at TIMESTAMP
            
            updated_at TIMESTAMP
            
            deleted_at TIMESTAMP
            
            id BIGINT PK
            
            name VARCHAR(20)
            
        2. 입력
            1. query = [”맛있다 좋다 친절하다”]
                1. filter( must ( restaurant id = summary 하고자하는 id (단일 id)))
        3. positive_review
            1. [음식이 맛있다, 주차가 되서 좋다, 직원이 친절하다]
        4. 입력
            1. query = [”맛없다 싫다 불친절하다”]
        5. negative_reivew
            1. [음식이 맛없다, 주차가 안되서 싫다, 직원이 불친절하다]
        6. llm
            1. 입력
                1. context = positive_review
                2. query = “긍/부정을 보고 전반적인 요약 수행”
            2. 출력
                1. restaurant_id = 
                2. positive_reivew = 
                3. negative_review = 
                4. overall_review = 
3. strength
    1. vector upload
        1. 벡터 데이터
            1. REVIEW TABLE
            
            id BIGINT PK
            
            restaurant_id BIGINT FK
            
            member_id BIGINT # 각 회원에게 부여되는 고유 id FK
            
            group_id BIGINT # 예시 “1234”, “2345” FK
            
            subgroup_id BIGINT # ex) “10234”, “20345” FK
            
            content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
            
            is_recommended BOOLEAN # 0, 1
            
            created_at TIMESTAMP
            
            updated_at TIMESTAMP
            
            deleted_at TIMESTAMP
            
            1. RESTAURANT_FOOD_CATEGORY
            
            id BIGINT PK
            
            restaurant_id BIGINT FK
            
            food_category_id BIGINT FK
            
            1. FOOD CATEGORY TABLE
                
                
                id BIGINT PK
                
                name VARCHAR(20)
                
    2. vector search
        1. 데이터
            1. 벡터 데이터
                1. REVIEW TABLE
                
                id BIGINT PK
                
                restaurant_id BIGINT FK
                
                member_id BIGINT # 각 회원에게 부여되는 고유 id FK
                
                group_id BIGINT # 예시 “1234”, “2345” FK
                
                subgroup_id BIGINT # ex) “10234”, “20345” FK
                
                content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
                
                is_recommended BOOLEAN # 0, 1
                
                created_at TIMESTAMP
                
                updated_at TIMESTAMP
                
                deleted_at TIMESTAMP
                
                1. RESTAURANT_FOOD_CATEGORY
                
                id BIGINT PK
                
                restaurant_id BIGINT FK
                
                food_category_id BIGINT FK
                
                1. FOOD CATEGORY TABLE
                    
                    
                    id BIGINT PK
                    
                    name VARCHAR(20)
                    
        2. 입력
            1. query = [”음식이 맛있다, 주차가 되서 좋다, 직원이 친절하다.]
            2. target_positive_review
                1. positive_review = []
                    1. filter ( must = restaurant_id == res_1234 and  == “)
            3. comparison_positive_review
                1. filter ( not must = restaurant_id == res_1234 and must 카테고리 == “) # 각각의 음식점에 대해 벡터 검색을 진행해 limit=1의 장점을 가져와 하나로 취합
        3. llm 입력
            1. context
                1. comparison_postive_review
            2. query
                1. 요약을 수행하라
            3. 출력
                1. comparison_summary
        4. llm 입력
            1. context
                1. target_positive_review
                2. comparison_positive_review
                3. 모든 음식점은 같은 음식 카테고리에 속한다.
            2. query
                1. comparsion_positive_review에 대해 요약을 수행하라
                2. 그 요약된 값을 비교해 target_restaurant이 다른 음식점들에 비해 가지는 장점이 뭔지 도출하라.
            3. 출력
                1. restaurant_id =
                2. comparision_summary = comparision_positive_reiview 요약된 값
                3. target_restaurant_strength = 도출한 target 음식점의 강점
4. 리뷰 이미지 반환
    1. 입력
        1. 쿼리
    2. 벡터 데이터
        1. REVIEW TABLE
        
        id BIGINT PK
        
        restaurant_id BIGINT FK
        
        member_id BIGINT # 각 회원에게 부여되는 고유 id FK
        
        group_id BIGINT # 예시 “1234”, “1234” FK
        
        subgroup_id BIGINT # ex) “10234”, “12034” FK
        
        content VARCHAR(1000) # ex) “음식이 맛있네요! 또 가고싶어요!” 
        
        is_recommended BOOLEAN # 0, 1
        
        created_at TIMESTAMP
        
        updated_at TIMESTAMP
        
        deleted_at TIMESTAMP
        
        1. REVIEW_IMAGE TABLE
        
        id bigint PK
        
        review_id bigint FK
        
        image_url varchar(500)
        
        created_at Timestamp
        
    3. 벡터 검색
        1. [”분위기 좋다”]
    4. 출력
        1. restaurant_id
        2. 리뷰 id
        3. 리뷰 image_url