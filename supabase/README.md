# Supabase 초기화와 마이그레이션

`migrations/`의 `20261028140813`, `20261113013319`는 운영 프로젝트의 기존 이력이다.
`20261116024530`은 삭제 후 파일 정리 대기열을 추가하는 새 마이그레이션이며 아직 적용되지 않았다.
`bootstrap/initial_secure_schema.sql`은 애플리케이션 테이블이 하나도 없는 **새 Supabase DB 전용**이다.
기존 테이블이 하나라도 있으면 트랜잭션 전체가 실패한다.

## 새 프로젝트

백엔드를 시작하기 전에, 연결 대상이 새 프로젝트인지 확인하고 다음 순서로 실행한다.
`siot_new`는 비밀번호를 명령줄에 노출하지 않도록 `pg_service.conf`와 `.pgpass`에 별도로 설정한 PostgreSQL 서비스 이름이다.

```bash
PGSERVICE=siot_new psql -X -v ON_ERROR_STOP=1 -f supabase/bootstrap/initial_secure_schema.sql
npx --yes supabase@3.1.0 migration up --remote=<새-프로젝트-ref>
```

두 명령은 반드시 **같은 새 DB**를 가리켜야 한다. CLI의 대상 표시를 확인하고 승인한다.
초기 SQL은 테이블 생성, RLS 활성화, 익명·인증 역할의 테이블 권한 회수를 하나의 트랜잭션으로 처리한다.
차량 탐지 작업 테이블도 처음부터 RLS와 권한 회수가 적용되며, 두 번째 이력에서 명시적 거부 정책이 추가된다.
완료 후 Supabase 보안 진단과 실제 백엔드 계정의 접근을 확인한다.

## 기존 프로젝트

운영 DB에는 초기 SQL을 실행하지 않는다. 먼저 `supabase migration list --remote=<프로젝트-ref>`로
원격 이력과 저장소 파일을 비교한다. 불일치가 있으면 `migration up`이나 `migration repair`를 실행하기 전에
기존 데이터·스키마·보안 정책을 별도로 확인한다. 이 작업은 원격 DB의 마이그레이션 기록을 변경하지 않았다.

백엔드가 PostgreSQL에 새 테이블을 자동 생성할 때도 같은 트랜잭션에서 RLS를 켜고 공개 역할의 권한을 회수한다.
기존 테이블의 권한은 백엔드 시작 시 자동 변경하지 않는다. MySQL에는 PostgreSQL의 RLS 구문을 적용하지 않는다.
