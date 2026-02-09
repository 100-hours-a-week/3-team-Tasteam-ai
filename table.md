restaurant

테이블명 : restaurant
테이블 설명 : 음식점의 기본 정보를 저장하는 마스터 테이블
책임 : 음식점 식별, 위치 정보 관리, 소프트 삭제 상태 관리
생명주기 : 생성 → 유지 → 소프트 삭제
삭제 정책 : deleted_at 기반 소프트 삭제
주요 조회 패턴 :
id 기반 상세 조회
위치 기반 조회 (PostGIS)
제약조건 :
name 빈 문자열 불가
인덱스 :
PARTICAL INDEX (GIST(location), deleted_at = NULL)
설계 근거 및 향후 확장성 :
위치 정보는 PostGIS Point(4326)로 관리하여 거리/반경 검색 확장 가능
활성 데이터만을 대상으로 Partial GIST Index를 적용, 모든 위치 기반 조회는 해당 인덱스 사용을 전제로 한다
주소 세부 정보는 restaurant_address 테이블로 분리

컬럼명	데이터 타입	NULL 허용	Key (PK/FK/-)	UNIQUE	기본값 / IDENTITY	ENUM / 제약 / 비고
id	BIGINT	N	PK	Y	IDENTITY	
name	VARCHAR(100)	N	-	N		빈 문자열 불가
full_address	VARCHAR(255)	N	-	N		
location	geometry(Point,4326)	N	-	N		WGS84
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N		
deleted_at	TIMESTAMP	Y	-	N		소프트 삭제

review

테이블명 : review
테이블 설명 : 그룹/하위 그룹 단위로 작성되는 음식점 리뷰
책임 : 리뷰 내용 및 음식점 추천 여부 관리
생명주기 : 생성 → 수정 → 삭제
삭제 정책 : deleted_at 기반 소프트 삭제
주요 조회 패턴 :
restaurant_id/group_id/subgroup_id 기준 리뷰 조회
제약 조건: -
인덱스 :
PARTICAL INDEX(restaurant_id, created_at DESC, deleted_at = NULL)
PARTICAL INDEX(group_id, created_at DESC, deleted_at = NULL)
PARTICAL INDEX(subgroup_id, created_at DESC, deleted_at = NULL)
설계 근거 :
내용 없이 키워드만 있는 리뷰를 허용하므로 내용(content) 필드 NULL 허용
리뷰는 항상 그룹에 속하지만 하위 그룹에는 속하지 않는 경우 있으므로 하위 그룹(subgroup_id) 필드 NULL 허용

[테이블 정의]
컬럼명	데이터 타입	NULL 허용	Key (PK/FK/-)	UNIQUE	기본값 / IDENTITY	ENUM / 제약 / 비고
id	BIGINT	N	PK	Y	IDENTITY	
restaurant_id	BIGINT	N	FK	N		restaurant.id
member_id	BIGINT	N	FK	N		member.id
group_id	BIGINT	N	FK	N		group.id
subgroup_id	BIGINT	Y	FK	N		subgroup.id
content	VARCHAR(1000)	Y	-	N		
is_recommended	BOOLEAN	N	-	N		
created_at	TIMESTAMP	N	-	N		
updated_at	TIMESTAMP	N	-	N		
deleted_at	TIMESTAMP	Y	-	N		

