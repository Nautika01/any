#!/usr/bin/env bash
# 정해진 시간 동안 일정 간격으로 수집 → 커밋 → 푸시를 되풀이한다.
# GitHub의 예약 실행(cron)은 건너뛰는 일이 잦아서, 한 번 시작된 작업이
# 15분 눈금(:00 :15 :30 :45)에 맞춰 스스로 돌게 한다.
#
# 연속으로 실패하면 일찍 끝낸다. 2026-11-25에 러너 한 대에서 따릉이 API 접속이
# 5시간 30분 내내 시간 초과였고, 다음 러너에서는 바로 됐다. 같은 러너에서 버티면
# 그 시간을 통째로 잃으므로, 몇 번 안 되면 물러나 다음 실행(다른 러너)에 넘긴다.
#
# 환경변수(시험용으로 바꿀 수 있다)
#   DURATION      전체 실행 시간(초). 기본 5시간 30분 — Actions 작업 한도 6시간 안쪽.
#   INTERVAL      수집 간격(초). 기본 900.
#   COLLECT_CMD   수집 명령. 기본 python src/collect.py
#   MAX_FAILS     연속 실패가 이만큼 쌓이면 일찍 끝낸다. 기본 3. 0이면 끝까지 버틴다.
#   RETRY         실패한 뒤 다시 시도할 때까지 기다리는 시간(초). 기본 120.
#   GITHUB_OUTPUT 있으면 healthy=true|false 를 적는다(한 번이라도 수집했는지).
set -u

# 실행 중에 git pull이 이 파일을 바꿔도 안전하도록 전체를 함수로 읽어 둔 뒤 실행한다.
main() {
  local duration=${DURATION:-19800} interval=${INTERVAL:-900} retry=${RETRY:-120}
  local max_fails=${MAX_FAILS:-3} collect_cmd=${COLLECT_CMD:-python src/collect.py}
  local end=$(( $(date +%s) + duration )) ok=0 fail=0 streak=0 gave_up=false now wait

  while :; do
    git pull -q --rebase || true   # 수집기 코드가 바뀌었으면 다음 회차부터 반영
    if $collect_cmd && push_data; then
      ok=$((ok + 1))
      streak=0
    else
      fail=$((fail + 1))
      streak=$((streak + 1))
      echo "::warning::수집 또는 푸시 실패 (연속 ${streak}회, 누적 ${fail}회)."
      if [ "$max_fails" -gt 0 ] && [ "$streak" -ge "$max_fails" ]; then
        echo "::error::연속 ${streak}회 실패. 이 러너에서는 그만두고 다음 실행에 넘긴다."
        gave_up=true
        break
      fi
    fi
    now=$(date +%s)
    if [ "$streak" -gt 0 ]; then
      wait=$retry                             # 실패했으면 다음 눈금까지 기다리지 않고 곧 다시 시도
    else
      wait=$(( interval - now % interval ))   # 다음 눈금까지 남은 시간
    fi
    [ $(( now + wait )) -ge "$end" ] && break
    sleep "$wait"
  done

  echo "수집 성공 ${ok}회, 실패 ${fail}회"
  local healthy=false
  [ "$ok" -gt 0 ] && healthy=true
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "healthy=${healthy}" >> "$GITHUB_OUTPUT"
  # 한 번도 성공 못 했거나 연속 실패로 물러났으면 작업을 실패로 표시
  [ "$ok" -gt 0 ] && [ "$gave_up" = false ]
}

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

main "$@"
exit
