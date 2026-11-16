
# Root Error 탐색 알고리즘 (규칙 기반) - "후보 전체 순위 매기기" 버전
#
# 더 이상 "딱 하나만 찍기"가 아니라, incident 안의 모든 후보(ERROR/CRITICAL 로그)를
# 전부 순위를 매겨서 나열한다. 그중 진짜 Root가 몇 위에 와 있는지로 평가한다.
#
# ⚠️ 데이터 누수(data leakage) 금지
#    causal_role / root_cause_type / is_root_error 는 정답 라벨 -> 입력으로 쓰지 않는다.
#    평가(채점)할 때만 사용한다. 규칙이 실제로 보는 입력은
#    timestamp, module, error_code, severity, message 뿐이다.
#
# 후보 풀(candidate pool) = incident 안에서 severity가 ERROR 또는 CRITICAL인 로그 전체
#    (ROOT 1개 + CONSEQUENCE 몇 개 + DECOY 1~3개 + decoy의 가짜 후속 로그들)
#    -> incident마다 보통 2~10개 정도. "후보 20개"는 혹시 더 많아지는 경우에도 안전하게
#       다 보여주기 위한 상한선으로 쓴다 (실제로는 20개까지 가는 경우는 거의 없다).
#
# 세 가지 방식을 모두 "전체 순위" 형태로 비교한다.
#   A. naive   : 그냥 시간순 (가장 먼저 발생한 게 1위)
#   B. severity: severity 높은 순 (CRITICAL > ERROR), 동점이면 먼저 발생한 쪽이 위
#   C. cascade : "이후 window초 안에 같은 module에서 ERROR가 몇 번 더 뒤따르는가" 점수로 순위
#                (동점이면 severity 높은 쪽 우선, 그래도 동점이면 먼저 발생한 쪽)

import pandas as pd

from generate_logs import scenarios as _SCENARIOS  # 점검 가이드 텍스트용 (error_code는 라벨이 아니라 feature)

MAX_SHOW = 20  # 한 incident에서 후보를 최대 몇 개까지 보여줄지 (전체 순위, 상한선)
WINDOW = 15.0  # cascade 점수를 셀 때 "이후 몇 초까지"를 볼지

SEVERITY_RANK = {"CRITICAL": 3, "ERROR": 2, "WARNING": 1, "INFO": 0}


def get_candidates(g):
    """후보 풀: severity가 ERROR 또는 CRITICAL인 로그만. 원래 index를 보존한 채 시간순 정렬."""
    cand = g[g["severity"].isin(["ERROR", "CRITICAL"])].sort_values("timestamp")
    return cand.reset_index()  # 'index' 컬럼에 원래 df 인덱스가 보존됨


# --------------------------------------------------
# A. naive: 시간순 (가장 먼저 = 1위)
# --------------------------------------------------
def rank_naive(g):
    cand = get_candidates(g)
    return list(cand["index"])  # 이미 시간순 정렬돼 있음


# --------------------------------------------------
# B. severity: 심각도 높은 순
# --------------------------------------------------
def rank_severity(g):
    cand = get_candidates(g)
    keys = [(SEVERITY_RANK[s], -i) for i, s in enumerate(cand["severity"])]
    order = sorted(range(len(cand)), key=lambda k: keys[k], reverse=True)
    return [cand.loc[o, "index"] for o in order]


# --------------------------------------------------
# C. cascade: 뒤따르는 연쇄 에러 개수 기반 점수
# --------------------------------------------------
def rank_cascade(g, window=WINDOW):
    cand = get_candidates(g)
    times = cand["timestamp"].values
    modules = cand["module"].values
    severities = cand["severity"].values

    keys = []
    for i in range(len(cand)):
        score = 0
        for j in range(i + 1, len(cand)):
            dt = (pd.Timestamp(times[j]) - pd.Timestamp(times[i])).total_seconds()
            if dt > window:
                break
            if modules[j] == modules[i] and severities[j] == "ERROR":
                score += 1
        keys.append((score, SEVERITY_RANK[severities[i]], -i))

    order = sorted(range(len(cand)), key=lambda k: keys[k], reverse=True)
    return [cand.loc[o, "index"] for o in order]


# --------------------------------------------------
# 평가: 진짜 Root가 전체 순위에서 몇 위에 오는가
# --------------------------------------------------
def evaluate_ranking(df, rank_fn):
    ranks, reciprocal = [], []
    for inc_id, g in df.groupby("incident_id"):
        ranked = rank_fn(g)
        true_root_idx = g.index[g["is_root_error"]][0]
        pos = ranked.index(true_root_idx) + 1
        ranks.append(pos)
        reciprocal.append(1.0 / pos)

    ranks = pd.Series(ranks)
    return {
        "Top-1 정확도": round((ranks == 1).mean(), 3),
        "Top-3 정확도": round((ranks <= 3).mean(), 3),
        "Top-5 정확도": round((ranks <= 5).mean(), 3),
        "MRR": round(sum(reciprocal) / len(reciprocal), 3),
        "평균 순위": round(ranks.mean(), 2),
        "순위 분포": ranks.value_counts().sort_index().to_dict(),
    }


def print_full_ranking(inc_id, rank_fn, label):
    g = df[df["incident_id"] == inc_id]
    ranked = rank_fn(g)[:MAX_SHOW]
    print(f"\n--- {label} 방식 / incident={inc_id} / 후보 {len(ranked)}개 전체 순위 ---")
    rows = df.loc[ranked, ["timestamp", "module", "error_code", "severity", "causal_role"]].copy()
    rows.insert(0, "순위", range(1, len(rows) + 1))
    rows["실제 Root?"] = df.loc[ranked, "is_root_error"].map({True: "⭐ 이게 진짜 Root", False: ""})
    print(rows.to_string(index=False))


# ====================================================================
# Streamlit 앱(app.py)에서 쓰는 확장 기능
#   - rank_cascade()는 '순위만' 반환하지만, 화면에는 점수도 보여주고
#     경합(여러 후보가 비슷한 수준) 여부도 판단해야 해서 아래 두 함수를 추가한다.
#   - rank_naive / rank_severity / rank_cascade / evaluate_ranking / print_full_ranking은
#     manual_analysis.py, visualize_logs.py가 그대로 쓰고 있어서 전혀 건드리지 않는다.
# ====================================================================

def rank_cascade_scored(g, window=WINDOW):
    """rank_cascade와 완전히 같은 점수·동점 처리 규칙으로 순위를 매기되,
    각 후보의 cascade 점수(score)까지 함께 반환한다.
    반환: [{"index": 원본 df 인덱스, "score": 점수}, ...] (점수 높은 순으로 이미 정렬됨)"""
    cand = get_candidates(g)
    times = cand["timestamp"].values
    modules = cand["module"].values
    severities = cand["severity"].values

    scored = []
    for i in range(len(cand)):
        score = 0
        for j in range(i + 1, len(cand)):
            dt = (pd.Timestamp(times[j]) - pd.Timestamp(times[i])).total_seconds()
            if dt > window:
                break
            if modules[j] == modules[i] and severities[j] == "ERROR":
                score += 1
        key = (score, SEVERITY_RANK[severities[i]], -i)
        scored.append({"index": cand.loc[i, "index"], "score": score, "_key": key})

    scored.sort(key=lambda r: r["_key"], reverse=True)
    for r in scored:
        del r["_key"]
    return scored


def diagnose(g, window=WINDOW):
    """워크시트 12번 '모를 때' 규칙을 반영한 진단 결과.

    규칙:
      - cascade 점수가 0인 후보(뒤따르는 연쇄가 전혀 없음) = 가장 먼저 배제할 수 있는 후보 -> excluded
      - 남은 후보(score > 0) 중 1위 자리가 (점수, severity)까지 전부 동점이면 '경합' 상태로 본다.
        이 경우 남은 건 원본 index 순서라는 임의 기준뿐이라, 하나로 단정하지 않고
        순위를 확정하지 않은 채 후보만 나열한다 (ambiguous=True)
      - 점수만 같고 severity로 구분되거나, 1위 후보가 하나뿐이면 평소처럼 전체 순위를 그대로 쓴다
        (ambiguous=False)

    반환 dict:
      ambiguous   : bool
      ranked      : 전체 후보 (점수 포함, 항상 채워짐 - 참고용 전체 목록)
      candidates  : 화면에 보여줄 후보 목록 (ambiguous=True면 순위 없는 후보들, False면 ranked와 동일)
      excluded    : 명백히 배제된 후보 (score == 0)
      message     : 경합/예외 상황일 때 사용자에게 보여줄 안내 문구 (아니면 빈 문자열)
    """
    scored = rank_cascade_scored(g, window=window)
    if not scored:
        return {"ambiguous": False, "ranked": [], "candidates": [], "excluded": [],
                "message": "이 incident에는 ERROR/CRITICAL 후보가 없습니다."}

    cand = get_candidates(g)
    severity_of = dict(zip(cand["index"], cand["severity"]))

    excluded = [r for r in scored if r["score"] == 0]
    remaining = [r for r in scored if r["score"] > 0]

    if not remaining:
        return {
            "ambiguous": True, "ranked": scored, "candidates": scored, "excluded": [],
            "message": "어떤 후보도 뒤따르는 연쇄 에러가 없어 근거가 약합니다. "
                       "순위를 확정하지 않고 전체 후보를 보여드립니다 - 직접 데이터 분석이 필요합니다.",
        }

    top_score = remaining[0]["score"]
    top_sev_rank = SEVERITY_RANK[severity_of[remaining[0]["index"]]]
    top_group = [
        r for r in remaining
        if r["score"] == top_score and SEVERITY_RANK[severity_of[r["index"]]] == top_sev_rank
    ]

    if len(top_group) == 1:
        return {"ambiguous": False, "ranked": scored, "candidates": scored,
                "excluded": excluded, "message": ""}

    return {
        "ambiguous": True,
        "ranked": scored,
        "candidates": remaining,
        "excluded": excluded,
        "message": f"{len(top_group)}개의 후보가 점수·심각도까지 동일한 수준으로 의심되어 "
                   f"하나의 원인으로 단정하기 어렵습니다. "
                   f"우선순위를 매기지 않고 후보만 제시합니다 - 직접 데이터 분석이 필요합니다.",
    }


# --------------------------------------------------
# 점검 가이드: error_code(피처) -> 어떤 계통을 점검하면 되는지 안내 텍스트
#   causal_role/root_cause_type(라벨)이 아니라 scenarios에 원래부터 정의돼 있던
#   root_code/events(= 실제로 로그에 찍히는 error_code)만 키로 쓰므로 데이터 누수가 아니다.
#   (부품 추천은 이번 범위에서 제외했으므로, "어떤 계통/항목을 점검하라" 수준까지만 안내한다)
# --------------------------------------------------
INSPECTION_GUIDE = {}
for _rc_type, _sc in _SCENARIOS.items():
    _guide_text = (f"{_sc['module']} 계통 점검 필요 - '{_sc['root_message']}' 계열 이상 패턴입니다. "
                   f"{_sc['module']} 모듈의 제어/센서 캘리브레이션 상태를 우선 확인하세요.")
    for _code in [_sc["root_code"]] + [c for c, _ in _sc["events"]]:
        INSPECTION_GUIDE[_code] = _guide_text


def get_inspection_guide(module, error_code):
    return INSPECTION_GUIDE.get(
        error_code,
        f"[{module}] 등록된 점검 가이드가 없는 에러코드입니다. 현장 매뉴얼에서 '{error_code}'를 직접 확인하세요.",
    )


if __name__ == "__main__":
    df = pd.read_csv("virtual_lithography_logs.csv", parse_dates=["timestamp"])

    print("=" * 70)
    print("규칙 A (naive: 시간순)")
    print("=" * 70)
    for k, v in evaluate_ranking(df, rank_naive).items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("규칙 B (severity 높은 순)")
    print("=" * 70)
    for k, v in evaluate_ranking(df, rank_severity).items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("규칙 C (cascade: 연쇄 패턴 점수)")
    print("=" * 70)
    for k, v in evaluate_ranking(df, rank_cascade).items():
        print(f"{k}: {v}")

    # ----------------------------------------------
    # 예시: 규칙 C 기준으로 "쉬운 incident" 1개, "어려운 incident" 1개 전체 순위 출력
    # ----------------------------------------------
    easy_id, hard_id = None, None
    for inc_id, g in df.groupby("incident_id"):
        ranked = rank_cascade(g)
        true_root_idx = g.index[g["is_root_error"]][0]
        pos = ranked.index(true_root_idx) + 1
        if pos == 1 and easy_id is None:
            easy_id = inc_id
        if pos >= 3 and hard_id is None:
            hard_id = inc_id
        if easy_id and hard_id:
            break

    print("\n" + "#" * 70)
    print("# 전체 순위 예시 (규칙 C 기준)")
    print("#" * 70)
    if easy_id:
        print_full_ranking(easy_id, rank_cascade, "cascade - 쉬운 사례")
    if hard_id:
        print_full_ranking(hard_id, rank_cascade, "cascade - 어려운 사례 (Root가 밀림)")
