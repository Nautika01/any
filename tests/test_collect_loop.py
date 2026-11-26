"""scripts/collect_loop.sh 를 가짜 수집 명령과 임시 git 저장소로 돌려 본다."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "collect_loop.sh"


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def run_loop(tmp_path, collect_body: str, **env):
    """collect_body는 가짜 수집기(bash). 몇 번째 호출인지 $n 으로 알 수 있다."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(tmp_path, "clone", "-q", str(remote), str(work))
    git(work, "config", "user.name", "test")
    git(work, "config", "user.email", "test@example.com")
    (work / "README").write_text("x")
    git(work, "add", "README")
    git(work, "commit", "-q", "-m", "init")
    git(work, "push", "-q", "-u", "origin", "main")
    fake = tmp_path / "collect.sh"
    fake.write_text(
        "n=$(( $(cat calls 2>/dev/null || echo 0) + 1 )); echo $n > calls\n" + collect_body + "\n")
    output = tmp_path / "gh_output"
    full_env = {**os.environ, "COLLECT_CMD": f"bash {fake}", "INTERVAL": "1", "RETRY": "0",
                "DURATION": "4", "GITHUB_OUTPUT": str(output), **{k: str(v) for k, v in env.items()}}
    proc = subprocess.run(["bash", str(SCRIPT)], cwd=work, env=full_env, capture_output=True, text=True, timeout=60)
    calls = int((work / "calls").read_text())
    commits = subprocess.run(["git", "log", "--oneline", "origin/main"], cwd=work, capture_output=True,
                             text=True).stdout.count("data:")
    return proc, calls, commits, output.read_text().strip()


WRITE = 'mkdir -p data && echo $n > data/$n.txt'


def test_collects_on_every_tick_and_pushes(tmp_path):
    proc, calls, commits, out = run_loop(tmp_path, WRITE)
    assert proc.returncode == 0
    assert calls >= 3 and commits == calls
    assert out == "healthy=true"


def test_gives_up_after_consecutive_failures(tmp_path):
    """계속 실패하면 끝까지 버티지 않고 MAX_FAILS번 만에 물러난다(다음 러너에 넘기기 위해)."""
    proc, calls, commits, out = run_loop(tmp_path, "exit 1", DURATION=600, INTERVAL=600, MAX_FAILS=3)
    assert proc.returncode == 1
    assert calls == 3 and commits == 0
    assert out == "healthy=false"
    assert "다음 실행에 넘긴다" in proc.stdout


def test_retries_soon_after_a_failure_instead_of_waiting_for_the_tick(tmp_path):
    """한 번 실패하고 바로 성공하면 다음 눈금(여기서는 10분 뒤)까지 기다리지 않고 수집한다."""
    proc, calls, commits, out = run_loop(tmp_path, f"[ $n -eq 1 ] && exit 1\n{WRITE}",
                                         DURATION=3, INTERVAL=600, MAX_FAILS=3)
    assert proc.returncode == 0
    assert calls == 2 and commits == 1
    assert out == "healthy=true"


def test_success_resets_the_failure_streak(tmp_path):
    """실패가 띄엄띄엄 나오면 물러나지 않는다."""
    proc, calls, commits, out = run_loop(tmp_path, f"[ $(( n % 2 )) -eq 1 ] && exit 1\n{WRITE}", MAX_FAILS=2)
    assert "다음 실행에 넘긴다" not in proc.stdout
    assert calls >= 4 and commits == calls // 2
    assert out == "healthy=true"


def test_max_fails_zero_keeps_trying_to_the_end(tmp_path):
    """MAX_FAILS=0이면 예전처럼 끝까지 버티고, 한 번도 못 했으면 실패로 끝난다."""
    proc, calls, commits, out = run_loop(tmp_path, "exit 1", MAX_FAILS=0, DURATION=3, RETRY=1)
    assert proc.returncode == 1
    assert calls >= 3
    assert "다음 실행에 넘긴다" not in proc.stdout
    assert out == "healthy=false"
