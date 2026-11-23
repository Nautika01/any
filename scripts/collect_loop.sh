#!/usr/bin/env bash
# 정해진 시간 동안 일정 간격으로 수집 → 커밋 → 푸시를 되풀이한다.
# GitHub의 예약 실행(cron)은 건너뛰는 일이 잦아서, 한 번 시작된 작업이
# 15분 눈금(:00 :15 :30 :45)에 맞춰 스스로 돌게 한다.
#
# 환경변수(시험용으로 바꿀 수 있다)
#   DURATION     전체 실행 시간(초). 기본 5시간 30분 — Actions 작업 한도 6시간 안쪽.
#   INTERVAL     수집 간격(초). 기본 900.
#   COLLECT_CMD  수집 명령. 기본 python src/collect.py
set -u
DURATION=${DURATION:-19800}
INTERVAL=${INTERVAL:-900}
COLLECT_CMD=${COLLECT_CMD:-python src/collect.py}

end=$(( $(date +%s) + DURATION ))
ok=0
fail=0

push_data() {
  git add data
  git diff --cached --quiet && return 0
  git commit -q -m "data: $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M')"
  # 다른 커밋과 겹쳐 거절되면 최신 내용을 받아 다시 시도한다
  for _ in 1 2 3; do
    git pull -q --rebase && git push -q && return 0
    sleep 5
  done
  return 1
}

while :; do
  git pull -q --rebase || true   # 수집기 코드가 바뀌었으면 다음 회차부터 반영
  if $COLLECT_CMD && push_data; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "::warning::수집 또는 푸시 실패 (누적 ${fail}회). 다음 회차에 다시 시도한다."
  fi
  now=$(date +%s)
  wait=$(( INTERVAL - now % INTERVAL ))   # 다음 눈금까지 남은 시간
  [ $(( now + wait )) -ge "$end" ] && break
  sleep "$wait"
done

echo "수집 성공 ${ok}회, 실패 ${fail}회"
[ "$ok" -gt 0 ]   # 한 번도 성공 못 했으면 작업을 실패로 표시
