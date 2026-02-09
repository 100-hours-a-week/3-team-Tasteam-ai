restaurant

id	BIGINT	N	PK	Y	IDENTITY	
name	VARCHAR(100)	N	-	N		빈 문자열 불가
full_address	VARCHAR(255)	N	-	N		
location	geometry(Point,4326)	N	-	N		WGS84
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N		
deleted_at	TIMESTAMP	Y	-	N		소프트 삭제

menu_categories

id	BIGINT	N	PK	Y	IDENTITY	
restaurant_id	BIGINT	N	FK	N		restaurant.id 참조
name	VARCHAR(100)	N	-	N		카테고리명 (메인, 음료 등)
display_order	INT	N	-	N	0	노출 순서
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N	CURRENT_TIMESTAMP	

menus

id	BIGINT	N	PK	Y	IDENTITY	
category_id	BIGINT	N	FK	N		menu_categories.id 참조
name	VARCHAR(150)	N	-	N		메뉴명
description	VARCHAR(500)	Y	-	N		메뉴 설명
price	INT	N	-	N		표시 가격 (원 단위)
image_url	VARCHAR(500)	Y	-	N		메뉴 이미지 URL
is_recommended	BOOLEAN	N	-	N	FALSE	추천 메뉴 여부
display_order	INT	N	-	N	0	노출 순서
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N	CURRENT_TIMESTAMP	

restaurant_schedule_overrides

id	BIGINT	N	PK	Y	IDENTITY	
restaurant_id	BIGINT	N	FK	N		restaurant.id 참조
date	DATE	N	-	N		
open_time	TIME	Y	-	N		
close_time	TIME	Y	-	N		
is_closed	BOOLEAN	N	-	N		휴무 여부
reason	VARCHAR(255)	Y	-	N		휴무 사유
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N		



id	BIGINT	N	PK	Y	IDENTITY	
restaurant_id	BIGINT	N	FK	N		restaurant.id 참조
day_of_week	SMALLINT	N	-	N		요일 (1=월 ~ 7=일)
open_time	TIME	Y	-	N		
close_time	TIME	Y	-	N		
is_closed	BOOLEAN	N	-	N		휴무 여부
effective_from	DATE	Y	-	N		정책 시작/종료 시점
effective_to	DATE	Y	-	N		
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N	CURRENT_TIMESTAMP	


id	BIGINT	N	PK	Y	IDENTITY	
restaurant_id	BIGINT	N	FK	N		restaurant.id
food_category_id	BIGINT	N	FK	N		food_category.id


id	BIGINT	N	PK	Y	IDENTITY	SEQUENCE
name	VARCHAR(20)	N	-	N		빈 문자열 불가