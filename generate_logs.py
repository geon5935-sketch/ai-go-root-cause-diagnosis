
# 학습 목표
# 가상의 반도체 노광장비 에러 로그 데이터 생성 (개선판 v3)
# Root Error, 후속 에러, 가짜 후보(Decoy), 일반 로그를 구분하여 CSV 파일로 저장
#
# [v1 -> v2 개선 사항]
# 1. 완전 중복 로그 발생 방지 (incident 안에서 타임스탬프를 고유하게 보장)
# 2. Root Error가 항상 시나리오의 "맨 처음"에만 나오지 않도록
#    - 전조(precursor) 노이즈 로그를 Root 이전에도 배치 (0~8개, 랜덤)
#    - Root 발생 시점 자체도 전조 로그 양에 따라 달라짐
# 3. 노이즈 로그를 Root 이전/이후 전체 구간에 자연스럽게 흩어서 배치
#    (후속 에러들 사이사이에도 끼어들 수 있음)
# 4. 같은 고장 유형(RC-00X)이라도 매번 다른 "모양"이 나오도록
#    - 전조 로그 개수, Root 발생 시점, 후속 에러 개수(2~3개), 이벤트 간 간격을
#      모두 incident마다 랜덤으로 결정
#
# [v2 -> v3 개선 사항] - "여러 후보 중 진짜 원인의 순위를 매기는" 문제로 바꾸기 위함
# 5. incident마다 가짜 후보(Decoy) 1~3개를 추가. 실제 원인이 아닌 단발성 이상 신호이며,
#    severity는 대부분 ERROR지만 일부는 CRITICAL로도 생성해서 "severity만 보고 구분"이 안 되게 함.
# 6. decoy도 40% 확률로 자기만의 "가짜 연쇄"(후속 ERROR 1~2개)를 달고 나오게 해서,
#    cascade 패턴 기준으로도 가끔은 decoy가 진짜 Root보다 더 수상해 보이도록 함
#    -> 그 결과 "진짜 Root가 항상 후보 1순위"가 아니게 되고, 순위를 매기는 문제 자체가 의미를 가짐.
#
# [v3 -> v4 개선 사항] - Streamlit 앱에서 재사용하기 위함
# 7. 전체 생성 로직을 generate(seed, out_path) 함수로 감싸서,
#    다른 스크립트(app.py)가 import해서 "새 시드로 새 데이터 생성"을 그 자리에서 호출할 수 있게 함.
#    (이전에는 모듈 최상단에서 바로 실행돼서, import만 해도 파일을 덮어썼음 - 이제는 그런 부작용이 없음)
# 8. python3 generate_logs.py --seed 123 --out other.csv 형태로 시드/출력 경로를 바꿀 수 있게 함.
#    인자를 안 주면 이전과 완전히 같게 동작 (seed=42, virtual_lithography_logs.csv).
#
# ⚠️ 데이터 누수(data leakage) 주의
#    causal_role / root_cause_type / is_root_error 는 "정답 라벨"입니다.
#    Root Error 탐색 알고리즘을 만들 때 이 컬럼들을 입력 피처로 쓰면 안 됩니다.
#    아래 LABEL_COLUMNS / FEATURE_COLUMNS 를 참고하세요.
#    causal_role은 이제 ROOT / CONSEQUENCE / DECOY / NOISE 4가지입니다.

import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --------------------------------------------------
# 1. 기본 설정 (랜덤 시드는 generate() 안에서 설정 - import 시점에는 아무것도 안 함)
# --------------------------------------------------

equipment_id = "NXE-SIM-001"
start_time = datetime(2026, 1, 1, 0, 0, 0)

# AI 모델을 만들 때 참고할 컬럼 구분
LABEL_COLUMNS = ["causal_role", "root_cause_type", "is_root_error"]   # 정답 라벨 (입력 금지)
FEATURE_COLUMNS = ["timestamp", "module", "error_code", "severity", "message"]  # 입력 피처 후보

# --------------------------------------------------
# 2. 가상 고장 시나리오 정의
# --------------------------------------------------

scenarios = {
    "RC-001": {
        "module": "Stage",
        "root_code": "VERR-001",
        "root_message": "Stage control response abnormal",
        "events": [
            ("VERR-002", "Stage position deviation detected"),
            ("VERR-003", "Stage movement timeout"),
            ("VERR-004", "Exposure operation interrupted")
        ]
    },

    "RC-002": {
        "module": "Cooling",
        "root_code": "VERR-005",
        "root_message": "Cooling performance abnormal",
        "events": [
            ("VERR-006", "Cooling flow rate decreased"),
            ("VERR-007", "Temperature warning detected"),
            ("VERR-008", "Equipment protection shutdown")
        ]
    },

    "RC-003": {
        "module": "Vacuum",
        "root_code": "VERR-009",
        "root_message": "Vacuum pressure abnormal",
        "events": [
            ("VERR-010", "Chamber pressure increased"),
            ("VERR-011", "Vacuum warning detected"),
            ("VERR-012", "Equipment operation stopped")
        ]
    },

    "RC-004": {
        "module": "EUV_Source",
        "root_code": "VERR-013",
        "root_message": "EUV source output unstable",
        "events": [
            ("VERR-014", "Source output fluctuation detected"),
            ("VERR-015", "Source stability warning"),
            ("VERR-016", "Exposure operation interrupted")
        ]
    }
}

# --------------------------------------------------
# 3. 일반 로그 정의 (Noise Log)
# --------------------------------------------------

noise_messages = [
    ("INFO-001", "System status check completed", "INFO"),
    ("INFO-002", "Module communication normal", "INFO"),
    ("WARN-001", "Minor temperature variation", "WARNING"),
    ("INFO-003", "Operation status updated", "INFO"),
    ("INFO-004", "Sensor data recorded", "INFO"),
    ("INFO-005", "Equipment monitoring completed", "INFO"),
    ("WARN-002", "Minor signal fluctuation", "WARNING"),
    ("WARN-003", "Pressure sensor fluctuation", "WARNING"),
    ("INFO-006", "Routine calibration check", "INFO"),
    ("WARN-004", "Vibration slightly elevated", "WARNING"),
]

modules_all = ["Stage", "Cooling", "Vacuum", "EUV_Source"]

# --------------------------------------------------
# 3-1. 가짜 후보 (Decoy) 정의
#      -> 실제 원인은 아니지만 ERROR/가끔 CRITICAL 수준으로 "그럴듯해 보이는" 단발성 이상 신호.
#         진짜 Root와 달리 자기 자신에 이어지는 연쇄(cascade)가 없다는 점이 핵심 차이.
# --------------------------------------------------

decoy_error_pool = [
    ("VERR-020", "Position sensor glitch detected"),
    ("VERR-021", "Unexpected retry on motion command"),
    ("VERR-022", "Momentary communication timeout"),
    ("VERR-023", "Secondary sensor reading out of range"),
    ("VERR-024", "Redundant safety check triggered"),
    ("VERR-025", "Brief calibration offset detected"),
]

decoy_critical_pool = [
    ("VERR-026", "Unexpected pressure/temperature spike (self-recovered)"),
    ("VERR-027", "Brief control loop instability"),
]

decoy_followup_pool = [
    ("VERR-030", "Related subsystem check failed"),
    ("VERR-031", "Secondary retry attempt failed"),
    ("VERR-032", "Downstream interlock flagged"),
]

DECOY_CRITICAL_PROB = 0.35      # decoy 중 일부는 severity만으로 구분 안 되게 CRITICAL로도 생성
DECOY_FAKE_CASCADE_PROB = 0.55  # decoy도 가끔 자기만의 "가짜 연쇄"를 달고 나와서 진짜 Root보다 더 수상해 보일 수 있게
# (위 두 값 + 아래 n_followup 범위는 "진짜 Root가 후보 1순위가 아닌 비율"이 약 20%가 되도록 조정한 값)

# --------------------------------------------------
# 4. 타임스탬프 고유화 헬퍼
#    (incident 안에서 완전히 같은 행이 생기지 않도록 보장)
# --------------------------------------------------

def unique_random_offset(used_offsets, low, high, tries=50):
    """[low, high] 구간에서 아직 쓰이지 않은 오프셋(초, 소수점 포함)을 뽑는다."""
    for _ in range(tries):
        offset = round(random.uniform(low, high), 3)
        if offset not in used_offsets:
            used_offsets.add(offset)
            return offset
    # 극히 드물게 계속 겹치면 살짝 밀어서 강제로 고유화
    offset = round(high + 0.001, 3)
    while offset in used_offsets:
        offset = round(offset + 0.001, 3)
    used_offsets.add(offset)
    return offset


def nudge_unique(offset, used_offsets):
    """순차적으로 증가하는 오프셋이 우연히 겹치면 아주 조금 뒤로 밀어준다."""
    offset = round(offset, 3)
    while offset in used_offsets:
        offset = round(offset + 0.001, 3)
    used_offsets.add(offset)
    return offset


# --------------------------------------------------
# 5. 로그 데이터 생성 (함수로 감싸서 재사용 가능하게)
# --------------------------------------------------

def generate(seed: int = 42, out_path: str = "virtual_lithography_logs.csv", verbose: bool = True):
    """가상 로그 데이터를 생성해서 DataFrame으로 반환하고, out_path에 CSV로도 저장한다.

    seed를 바꾸면 같은 로직으로 완전히 다른 가상 데이터를 새로 뽑을 수 있다.
    (Streamlit 앱의 "새 가상 데이터 생성" 버튼이 이 함수를 그 자리에서 호출한다.)
    """
    random.seed(seed)
    np.random.seed(seed)

    logs = []

    scenario_list = [
        (rc_type, scenario)
        for rc_type, scenario in scenarios.items()
        for _ in range(50)
    ]
    # 4개 유형 x 50개 = 총 200개 시나리오

    for scenario_id, (rc_type, scenario) in enumerate(scenario_list):

        incident_id = f"INC-{scenario_id + 1:04d}"
        incident_start = start_time + timedelta(minutes=scenario_id * 10)

        used_offsets = set()

        # ----------------------------------------------
        # 5-0. 이번 incident의 "모양"을 랜덤하게 결정
        #      -> 같은 RC 유형이라도 매번 다른 패턴이 나오게 하는 핵심 부분
        # ----------------------------------------------
        total_noise = random.randint(15, 25)
        pre_noise_n = random.randint(0, min(8, total_noise))   # Root 이전 전조 로그 개수 (0개일 수도 있음)
        post_noise_n = total_noise - pre_noise_n

        # Root 발생 시점: 전조 로그가 없으면 바로 0초, 있으면 5~15초 사이 랜덤
        root_offset = 0.0 if pre_noise_n == 0 else round(random.uniform(5, 15), 3)
        root_offset = nudge_unique(root_offset, used_offsets)

        # 후속 에러 개수: 대부분 3개지만, 가끔 2개만 발생 (캐스케이드가 덜 진행된 경우)
        n_cons = random.choice([3, 3, 3, 2])
        cons_events = scenario["events"][:n_cons]

        # ----------------------------------------------
        # 5-1. Root 이전 전조(precursor) 노이즈 로그
        # ----------------------------------------------
        pre_low = -max(30, root_offset * 2 + 5)
        for _ in range(pre_noise_n):
            offset = unique_random_offset(used_offsets, pre_low, -0.5)
            code, message, severity = random.choice(noise_messages)
            logs.append({
                "incident_id": incident_id,
                "equipment_id": equipment_id,
                "timestamp": incident_start + timedelta(seconds=offset),
                "module": random.choice(modules_all),
                "error_code": code,
                "severity": severity,
                "message": message,
                "event_type": "INFO" if severity == "INFO" else "WARNING",
                "causal_role": "NOISE",
                "root_cause_type": rc_type,
                "is_root_error": False
            })

        # ----------------------------------------------
        # 5-2. Root Error
        # ----------------------------------------------
        logs.append({
            "incident_id": incident_id,
            "equipment_id": equipment_id,
            "timestamp": incident_start + timedelta(seconds=root_offset),
            "module": scenario["module"],
            "error_code": scenario["root_code"],
            "severity": "CRITICAL",
            "message": scenario["root_message"],
            "event_type": "ERROR",
            "causal_role": "ROOT",
            "root_cause_type": rc_type,
            "is_root_error": True
        })

        # ----------------------------------------------
        # 5-3. 후속 에러 (간격도 매번 랜덤)
        # ----------------------------------------------
        last_cons_offset = root_offset
        for code, message in cons_events:
            gap = round(random.uniform(1.5, 6.0), 3)
            offset = nudge_unique(last_cons_offset + gap, used_offsets)
            last_cons_offset = offset

            logs.append({
                "incident_id": incident_id,
                "equipment_id": equipment_id,
                "timestamp": incident_start + timedelta(seconds=offset),
                "module": scenario["module"],
                "error_code": code,
                "severity": "ERROR",
                "message": message,
                "event_type": "ERROR",
                "causal_role": "CONSEQUENCE",
                "root_cause_type": rc_type,
                "is_root_error": False
            })

        # ----------------------------------------------
        # 5-4. Root 이후 노이즈 로그
        #      (후속 에러들 사이사이에도 섞여 들어갈 수 있음)
        # ----------------------------------------------
        post_low = root_offset + 0.5
        post_high = last_cons_offset + 25
        for _ in range(post_noise_n):
            offset = unique_random_offset(used_offsets, post_low, post_high)
            code, message, severity = random.choice(noise_messages)
            logs.append({
                "incident_id": incident_id,
                "equipment_id": equipment_id,
                "timestamp": incident_start + timedelta(seconds=offset),
                "module": random.choice(modules_all),
                "error_code": code,
                "severity": severity,
                "message": message,
                "event_type": "INFO" if severity == "INFO" else "WARNING",
                "causal_role": "NOISE",
                "root_cause_type": rc_type,
                "is_root_error": False
            })

        # ----------------------------------------------
        # 5-5. 가짜 후보 (Decoy) - 1~3개, incident 구간 아무 데나 랜덤 배치
        #      실제 원인이 아니므로 뒤따르는 연쇄(cascade)를 만들지 않는다.
        # ----------------------------------------------
        n_decoy = random.randint(1, 3)
        for _ in range(n_decoy):
            if random.random() < DECOY_CRITICAL_PROB:
                code, message = random.choice(decoy_critical_pool)
                severity = "CRITICAL"
            else:
                code, message = random.choice(decoy_error_pool)
                severity = "ERROR"

            offset = unique_random_offset(used_offsets, pre_low, post_high)
            decoy_module = random.choice(modules_all)  # Root와 같은 모듈일 수도, 다를 수도 있음 (랜덤)
            logs.append({
                "incident_id": incident_id,
                "equipment_id": equipment_id,
                "timestamp": incident_start + timedelta(seconds=offset),
                "module": decoy_module,
                "error_code": code,
                "severity": severity,
                "message": message,
                "event_type": "ERROR",
                "causal_role": "DECOY",
                "root_cause_type": rc_type,
                "is_root_error": False
            })

            # 가끔은 decoy도 자기만의 "가짜 연쇄"를 달고 나온다 (실제 원인과는 무관한 우연의 일치).
            # 이게 있어야 "진짜 Root가 항상 점수 1위"가 아니게 되고, 순위 매기기가 진짜 의미를 가진다.
            if random.random() < DECOY_FAKE_CASCADE_PROB:
                n_followup = random.randint(1, 3)
                last_followup_offset = offset
                for _ in range(n_followup):
                    gap = round(random.uniform(1.5, 5.0), 3)
                    last_followup_offset = nudge_unique(last_followup_offset + gap, used_offsets)
                    f_code, f_message = random.choice(decoy_followup_pool)
                    logs.append({
                        "incident_id": incident_id,
                        "equipment_id": equipment_id,
                        "timestamp": incident_start + timedelta(seconds=last_followup_offset),
                        "module": decoy_module,  # decoy와 같은 모듈 (같은 "가짜 사건"의 후속처럼 보이게)
                        "error_code": f_code,
                        "severity": "ERROR",
                        "message": f_message,
                        "event_type": "ERROR",
                        "causal_role": "DECOY",
                        "root_cause_type": rc_type,
                        "is_root_error": False
                    })

    # --------------------------------------------------
    # 6. 데이터프레임 변환 및 정렬
    # --------------------------------------------------

    df = pd.DataFrame(logs)
    df = df.sort_values(["incident_id", "timestamp"]).reset_index(drop=True)

    # --------------------------------------------------
    # 7. CSV 파일 저장
    # --------------------------------------------------

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------
    # 8. 생성 결과 확인
    # --------------------------------------------------

    if verbose:
        print("=" * 50)
        print(f"가상 노광장비 로그 데이터 생성 완료 (seed={seed})")
        print("=" * 50)

        print("전체 로그 개수:", len(df))
        print("고장 시나리오 개수:", df["incident_id"].nunique())

        print("\n[로그 유형별 개수]")
        print(df["causal_role"].value_counts())

        print("\n[고장 유형별 시나리오 개수]")
        print(df.groupby("root_cause_type")["incident_id"].nunique())

        print("\n[생성된 로그 데이터 상위 10개]")
        print(df.head(10).to_string(index=False))

        # ---- 개선 효과 자체 점검 ----
        dup_count = int(df.duplicated().sum())

        root_first_flags = []
        for inc_id, g in df.groupby("incident_id"):
            g_sorted = g.sort_values("timestamp")
            root_first_flags.append(g_sorted.iloc[0]["causal_role"] == "ROOT")
        root_first_ratio = sum(root_first_flags) / len(root_first_flags) * 100

        print("\n[개선 전/후 점검]")
        print("완전 중복 로그 수:", dup_count, "(목표: 0)")
        print("Root Error가 항상 1번째인 시나리오 비율: {:.1f}% (목표: 100% 아님)".format(root_first_ratio))

        print("\nCSV 파일 저장 완료:", out_path)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="가상 노광장비 에러 로그 데이터 생성")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본 42)")
    parser.add_argument("--out", type=str, default="virtual_lithography_logs.csv", help="저장할 CSV 경로")
    args = parser.parse_args()

    generate(seed=args.seed, out_path=args.out, verbose=True)
