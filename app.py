
# AI:Go - ASML FSE 에러 원인 진단 보조 화면 (Streamlit)
#
# 워크시트 9번에서 정한 세 가지 결과물 형태를 탭 3개로 구현한다.
#   1) 실시간 진단      -> "장비 상태 모니터링 화면 / AI 고장 진단(이상 발생시 알람)"
#   2) 모델 성능 리포트  -> "최종 진단 보고서"에 들어갈 수치/그래프
#   3) 직접 분석 연습    -> manual_analysis.py를 화면으로 옮긴 것 (내 직감 vs AI 비교)
#
# 실행 방법: 이 파일과 generate_logs.py / detect_root_error.py / virtual_lithography_logs.csv를
#           같은 폴더에 두고   streamlit run app.py
#
# ⚠️ 데이터 누수 주의: 화면에서 "AI가 보는 입력"은 timestamp/module/error_code/severity/message뿐이고,
#    causal_role / root_cause_type / is_root_error는 "검증 모드"를 켰을 때만 참고용으로 보여준다
#    (실제 현장이라면 이 라벨은 존재하지 않는다 - 지금은 테스트/검증 목적).

import random

import matplotlib
matplotlib.use("Agg")  # 화면(디스플레이) 없는 서버 환경에서도 그래프를 그릴 수 있게
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from generate_logs import generate
from detect_root_error import (
    diagnose,
    evaluate_ranking,
    rank_naive,
    rank_severity,
    rank_cascade,
    get_inspection_guide,
)

# ======================================================
# 0. 기본 설정 / 공통 상수
# ======================================================

st.set_page_config(page_title="AI:Go - FSE 에러 원인 진단", layout="wide", page_icon="🛠️")

DEFAULT_CSV = "virtual_lithography_logs.csv"
TMP_CSV = "_streamlit_generated.csv"  # "새 데이터 생성" 버튼을 누르면 여기에 덮어써서 다시 읽는다
FEATURE_COLS = ["timestamp", "module", "error_code", "severity", "message"]

# visualize_logs.py와 통일한 색상 (causal_role 기준 - 검증 모드에서만 사용)
COLOR_NOISE = "#898781"
COLOR_CONSEQUENCE = "#2a78d6"
COLOR_DECOY = "#eb6834"
COLOR_ROOT = "#1baf7a"
ROLE_COLOR = {"NOISE": COLOR_NOISE, "CONSEQUENCE": COLOR_CONSEQUENCE,
              "DECOY": COLOR_DECOY, "ROOT": COLOR_ROOT}
ROLE_MARKER = {"NOISE": "o", "CONSEQUENCE": "^", "DECOY": "X", "ROOT": "*"}
ROLE_SIZE = {"NOISE": 35, "CONSEQUENCE": 70, "DECOY": 90, "ROOT": 220}
ROLE_ORDER = ["NOISE", "CONSEQUENCE", "DECOY", "ROOT"]

# severity 기준 색상 (실제 운영 화면 - 정답 라벨 없이도 쓸 수 있는 색)
SEVERITY_COLOR = {"CRITICAL": "#d6432a", "ERROR": "#eb8c34", "WARNING": "#d6b82a", "INFO": "#898781"}


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data(show_spinner=False)
def compute_report(df):
    """세 가지 규칙(naive/severity/cascade) 성능 비교.
    st.cache_data는 DataFrame을 그대로 캐시 키로 쓸 수 있어서 json 변환이 필요 없다."""
    report = {
        "naive": evaluate_ranking(df, rank_naive),
        "severity": evaluate_ranking(df, rank_severity),
        "cascade": evaluate_ranking(df, rank_cascade),
    }
    return report


def init_state():
    if "df" not in st.session_state:
        st.session_state.df = load_csv(DEFAULT_CSV)
        st.session_state.data_label = "기본 데이터 (seed=42, 200 incidents)"
    if "verify_mode" not in st.session_state:
        st.session_state.verify_mode = False
    if "practice_score" not in st.session_state:
        st.session_state.practice_score = {"me": 0, "algo": 0, "rounds": 0}
    if "practice_incident" not in st.session_state:
        st.session_state.practice_incident = None
    if "practice_revealed" not in st.session_state:
        st.session_state.practice_revealed = False


init_state()
df = st.session_state.df
incident_ids = sorted(df["incident_id"].unique().tolist())

# ======================================================
# 사이드바 - 데이터 상태 / 새 데이터 생성
# ======================================================

with st.sidebar:
    st.markdown("### ⚙️ 데이터")
    st.caption(st.session_state.data_label)
    st.write(f"incident **{df['incident_id'].nunique()}**개 · 로그 **{len(df):,}**행")

    if st.button("🔄 새로운 가상 데이터로 갈아타기", use_container_width=True):
        new_seed = random.randint(1, 999_999)
        with st.spinner(f"seed={new_seed}로 새 데이터 생성 중..."):
            new_df = generate(seed=new_seed, out_path=TMP_CSV, verbose=False)
        st.session_state.df = new_df
        st.session_state.data_label = f"새로 생성된 데이터 (seed={new_seed}, 200 incidents)"
        st.session_state.practice_incident = None
        st.session_state.practice_revealed = False
        compute_report.clear()
        st.rerun()

    st.divider()
    st.session_state.verify_mode = st.toggle(
        "🔬 검증 모드 (정답 라벨 보기)",
        value=st.session_state.verify_mode,
        help="실제 현장에는 없는 정답(causal_role 등)을 테스트용으로 같이 보여줍니다.",
    )
    st.divider()
    st.caption(
        "가상 노광장비 로그를 바탕으로 Root Cause 후보를 자동으로 좁혀주는 연습용 프로토타입입니다. "
        "실제 ASML 장비 데이터가 아니며, 최종 점검·수리 판단은 사람이 합니다."
    )

st.title("🛠️ AI:Go - 에러 원인 진단 보조")
st.caption("ASML FSE가 에러코드를 순서대로 다 확인하지 않고도, 원인 후보를 우선순위별로 빠르게 좁혀보는 화면입니다.")

tab1, tab2, tab3 = st.tabs(["🔍 실시간 진단", "📊 모델 성능 리포트", "🧪 직접 분석 연습"])

# ======================================================
# TAB 1. 실시간 진단
# ======================================================
with tab1:
    col_pick, col_info = st.columns([2, 3])
    with col_pick:
        inc_id = st.selectbox("점검할 incident 선택", incident_ids, key="tab1_inc")
    g = df[df["incident_id"] == inc_id].sort_values("timestamp")
    n_candidates = int(g["severity"].isin(["ERROR", "CRITICAL"]).sum())
    with col_info:
        st.metric("전체 로그", f"{len(g)}행", f"후보(ERROR/CRITICAL) {n_candidates}개", delta_color="off")

    st.markdown("#### 📋 장비 상태 모니터링 - 발생한 로그")
    show_cols = FEATURE_COLS + (["causal_role", "is_root_error"] if st.session_state.verify_mode else [])
    st.dataframe(g[show_cols], use_container_width=True, height=240, hide_index=True)

    st.markdown("#### 🚨 AI 고장 진단")
    result = diagnose(g)

    if result["ambiguous"]:
        st.warning("⚠️ " + result["message"])

        rows = []
        for r in result["candidates"]:
            row = df.loc[r["index"]]
            rows.append({
                "timestamp": row["timestamp"], "module": row["module"], "error_code": row["error_code"],
                "severity": row["severity"], "cascade 점수": r["score"],
                "점검 가이드": get_inspection_guide(row["module"], row["error_code"]),
            })
        cand_table = pd.DataFrame(rows)
        if st.session_state.verify_mode:
            cand_table["실제 Root?"] = [
                "⭐" if df.loc[r["index"], "is_root_error"] else "" for r in result["candidates"]
            ]
        st.caption("※ 경합 상태라 순위를 매기지 않고 후보만 나열합니다.")
        st.dataframe(cand_table, use_container_width=True, hide_index=True)

        if result["excluded"]:
            with st.expander(f"배제된 후보 보기 ({len(result['excluded'])}개 - 뒤따르는 연쇄 에러 없음)"):
                ex_rows = [df.loc[r["index"], ["timestamp", "module", "error_code", "severity"]] for r in result["excluded"]]
                st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
    else:
        rows = []
        for rank, r in enumerate(result["candidates"][:10], start=1):
            row = df.loc[r["index"]]
            rows.append({
                "순위": rank, "timestamp": row["timestamp"], "module": row["module"],
                "error_code": row["error_code"], "severity": row["severity"], "cascade 점수": r["score"],
                "점검 가이드": get_inspection_guide(row["module"], row["error_code"]) if rank <= 3 else "",
            })
        rank_table = pd.DataFrame(rows)
        if st.session_state.verify_mode:
            rank_table["실제 Root?"] = [
                "⭐ 이게 진짜 Root" if df.loc[r["index"], "is_root_error"] else ""
                for r in result["candidates"][:10]
            ]
        st.success(f"1순위 후보: **{rows[0]['module']} / {rows[0]['error_code']}**")
        st.dataframe(rank_table, use_container_width=True, hide_index=True)

    # ---- 타임라인 시각화 ----
    st.markdown("#### 📈 incident 타임라인")
    color_mode = "severity"
    if st.session_state.verify_mode:
        color_mode = st.radio("색상 기준", ["severity (실제 운영 화면)", "causal_role (검증 모드)"],
                               horizontal=True, label_visibility="collapsed")
        color_mode = "causal_role" if "causal_role" in color_mode else "severity"

    fig, ax = plt.subplots(figsize=(10, 2.6))
    t0 = g["timestamp"].min()
    offsets = (g["timestamp"] - t0).dt.total_seconds()

    if color_mode == "causal_role":
        for role in ROLE_ORDER:
            mask = g["causal_role"] == role
            if mask.sum() == 0:
                continue
            ax.scatter(offsets[mask], [0] * mask.sum(), c=ROLE_COLOR[role], marker=ROLE_MARKER[role],
                       s=ROLE_SIZE[role], edgecolors="#0b0b0b", linewidths=0.6,
                       zorder=3 if role == "ROOT" else 2, label=role)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False, fontsize=9)
    else:
        for sev in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            mask = g["severity"] == sev
            if mask.sum() == 0:
                continue
            size = 40 if sev in ("INFO", "WARNING") else (90 if sev == "ERROR" else 180)
            marker = "o" if sev in ("INFO", "WARNING") else ("^" if sev == "ERROR" else "*")
            ax.scatter(offsets[mask], [0] * mask.sum(), c=SEVERITY_COLOR[sev], marker=marker,
                       s=size, edgecolors="#0b0b0b", linewidths=0.6, label=sev)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False, fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("incident 시작 이후 경과 시간 (초)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

# ======================================================
# TAB 2. 모델 성능 리포트
# ======================================================
with tab2:
    st.markdown("#### 📊 규칙별 성능 비교 (naive vs severity vs cascade)")
    report = compute_report(df)

    c1, c2, c3, c4 = st.columns(4)
    cascade_r = report["cascade"]
    c1.metric("Top-1 정확도", f"{cascade_r['Top-1 정확도']*100:.1f}%")
    c2.metric("Top-3 정확도", f"{cascade_r['Top-3 정확도']*100:.1f}%")
    c3.metric("Top-5 정확도", f"{cascade_r['Top-5 정확도']*100:.1f}%")
    c4.metric("MRR", f"{cascade_r['MRR']:.3f}")
    st.caption("현재 채택된 규칙(cascade) 기준 - 전체 incident에 대해 진짜 Root가 전체 순위에서 몇 위로 나오는지로 계산")

    metric_names = ["Top-1 정확도", "Top-3 정확도", "Top-5 정확도", "MRR"]
    chart_df = pd.DataFrame(
        {rule: [report[rule][m] for m in metric_names] for rule in ["naive", "severity", "cascade"]},
        index=metric_names,
    )
    st.markdown("##### 규칙별 비교")
    st.bar_chart(chart_df, height=320)

    st.markdown("##### cascade 기준 - 진짜 Root가 몇 위로 나오는가 (순위 분포)")
    rank_dist = cascade_r["순위 분포"]
    rank_series = pd.Series(rank_dist).sort_index()
    rank_series.index = [f"{i}위" for i in rank_series.index]
    st.bar_chart(rank_series, height=280)

    st.markdown("##### 로그 유형 전체 분포 (causal_role)")
    role_counts = df["causal_role"].value_counts().reindex(ROLE_ORDER).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(role_counts.index, role_counts.values, color=[ROLE_COLOR[r] for r in role_counts.index], width=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylabel("로그 개수")
    for i, v in enumerate(role_counts.values):
        ax2.text(i, v + max(role_counts.values) * 0.02, str(int(v)), ha="center", fontsize=9, color="#52514e")
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("naive / severity 규칙 상세 수치"):
        st.json({"naive": report["naive"], "severity": report["severity"]})

# ======================================================
# TAB 3. 직접 분석 연습
# ======================================================
with tab3:
    st.markdown("#### 🧪 로그만 보고 Root Error를 직접 추리해보기")
    st.caption("manual_analysis.py와 같은 연습을 화면에서 합니다. causal_role 같은 정답은 가려져 있습니다.")

    score = st.session_state.practice_score
    s1, s2, s3 = st.columns(3)
    s1.metric("문제 수", score["rounds"])
    s2.metric("내 정답률", f"{(score['me']/score['rounds']*100) if score['rounds'] else 0:.0f}%")
    s3.metric("AI(cascade) 정답률", f"{(score['algo']/score['rounds']*100) if score['rounds'] else 0:.0f}%")

    if st.button("🎲 새 문제 뽑기"):
        st.session_state.practice_incident = random.choice(incident_ids)
        st.session_state.practice_revealed = False
        st.rerun()

    if st.session_state.practice_incident is None:
        st.info("위 버튼을 눌러 문제를 뽑아보세요.")
    else:
        pg = df[df["incident_id"] == st.session_state.practice_incident].sort_values("timestamp").reset_index()
        st.write(f"**incident_id = {st.session_state.practice_incident}**  (로그 {len(pg)}개)")
        display = pg[FEATURE_COLS].copy()
        display.insert(0, "번호", range(1, len(display) + 1))
        st.dataframe(display, use_container_width=True, hide_index=True, height=260)

        choice = st.radio(
            "몇 번 로그가 Root Error인 것 같나요?",
            options=list(range(1, len(pg) + 1)),
            horizontal=True,
            key=f"choice_{st.session_state.practice_incident}",
        )

        if not st.session_state.practice_revealed:
            if st.button("✅ 제출하고 정답 보기"):
                st.session_state.practice_revealed = True
                st.rerun()
        else:
            true_root_num = int(pg.index[pg["is_root_error"]][0]) + 1
            cascade_ranked = rank_cascade(df[df["incident_id"] == st.session_state.practice_incident])
            algo_top1_idx = cascade_ranked[0]
            algo_top1_num = int(pg.index[pg["index"] == algo_top1_idx][0]) + 1

            revealed = pg[["timestamp", "module", "error_code", "severity", "causal_role"]].copy()
            revealed.insert(0, "번호", range(1, len(revealed) + 1))
            revealed["실제 Root?"] = pg["is_root_error"].map({True: "⭐ Root", False: ""})
            st.dataframe(revealed, use_container_width=True, hide_index=True, height=260)

            my_correct = (choice == true_root_num)
            algo_correct = (algo_top1_num == true_root_num)

            st.write(f"내 답: **{choice}번** / 정답(진짜 Root): **{true_root_num}번** / AI(cascade) 1순위 예측: **{algo_top1_num}번**")
            if my_correct and algo_correct:
                st.success("둘 다 정답! 👍")
            elif my_correct and not algo_correct:
                st.info("내가 맞고 AI는 틀렸어요. (사람 직감이 더 나은 케이스)")
            elif not my_correct and algo_correct:
                st.warning("AI는 맞고 나는 틀렸어요.")
            else:
                st.error("둘 다 틀렸어요. 꽤 헷갈리는 incident였네요.")

            if st.button("➡️ 다음 문제"):
                score["rounds"] += 1
                score["me"] += int(my_correct)
                score["algo"] += int(algo_correct)
                st.session_state.practice_incident = random.choice(incident_ids)
                st.session_state.practice_revealed = False
                st.rerun()
