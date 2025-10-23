
Heston Model Lab — 사용자 매뉴얼 (v1.0)
====================================

1. 개요
-------
Heston Model Lab은 Heston 확률적 변동성 모형을 이용해
(1) 유럽형 옵션 가격 계산, (2) 가격·분산 경로 시뮬레이션, (3) 내재변동성(IV) 스마일 분석을
GUI로 손쉽게 수행할 수 있는 도구입니다.
핵심 파일
 - heston.py : 모델·프라이서·시뮬레이터 (Characteristic Function + QE/Euler)
 - GUI.py    : PyQt 기반 GUI (PyQt5/6 자동 호환)
 - requirements.txt / README.md

2. 실행 환경
------------
- Python 3.10~3.12
- 필수 패키지: numpy, scipy, matplotlib, PyQt5(또는 PyQt6)
- 권장: conda-forge 단일 채널 사용

예시(새 환경):
  conda create -n heston python=3.11 -y
  conda activate heston
  conda install -c conda-forge numpy scipy matplotlib pyqt=5.15.*

실행:
  python GUI.py

3. 화면 구성(상단 공통 파라미터 패널)
------------------------------------
S0         : 현재 기초자산 가격(spot)
v0         : 초기 분산(= 초기 변동성^2). 예: 0.04 → 20% vol
kappa(κ)   : 평균회귀 속도(variance가 theta로 되돌아가려는 힘)
theta(θ)   : 장기평균 분산
xi(ξ)      : vol-of-vol (분산의 변동성)
rho(ρ)     : 가격/분산 브라운운동 상관(-1<ρ<1; 보통 음수)
r          : 무위험이자율
q          : 연속배당수익률

[중요] 안정적 시뮬레이션을 위해 Feller 조건 2 κ θ ≥ ξ^2 를 가급적 만족시키세요.
수정 후 ‘Apply Params’를 눌러야 모든 탭에 반영됩니다.

도구 막대(플롯 상단): Home / Pan / Zoom / Save 이미지 파일

4. Pricing 탭
-------------
입력
- T(년), K(행사가), Type(call/put)
- ‘Compute Price & IV’ 클릭

출력
- Price: Heston 폐형식 적분 가격
- Implied Vol (BS): 해당 가격을 Black–Scholes 역대입으로 계산한 IV

보조 플롯
- Integrand Decay Preview: P1/P2 적분 피적분함수의 감쇠 확인용(실수부). 값이 완만히 0으로 수렴하면 수치적으로 양호합니다.

팁
- 매우 깊은 ITM/OTM, 극단 파라미터에서는 수치 오차가 커질 수 있습니다.
- 필요 시 heston.py 내 적분 상한(기본 200)을 늘리면 정확도 상승(속도 감소) 효과가 있습니다.

5. Simulation 탭
----------------
입력
- T(년), Steps(기간 분할 수), Paths(경로 수)
- Scheme: QE(권장) / Euler
- Antithetic: 난수 반대짝 사용(분산 감소)
- Seed: 재현성 제어

실행
- ‘Run Simulation’ → 내부에서 상관 난수를 생성하여 S(t), v(t) 경로를 만듭니다.
  가격 갱신: dS/S = (r - q - 0.5 v) dt + sqrt(v) dW₁
  분산 갱신: QE(Andersen) 또는 Euler. 음수 분산은 0으로 바닥 처리.

출력
- E[S_T], Std[S_T], E[v_T], Std[v_T]
- 샘플 경로 플롯(최대 25개만 표시하여 UI 속도 보장)

CSV 내보내기
- ‘Export CSV (paths)’
- 컬럼: path_id, time_index, t, S, v
- 대용량 분석은 CSV 저장 후 외부 도구(pandas 등)에서 처리 권장

성능 팁
- 미리보기: Paths를 작게(예: 500~2,000), Steps는 용도에 따라 252~1,000 수준
- 정밀 분석: CSV로 내보낸 뒤 오프라인 처리
- QE가 Euler보다 보통 더 안정/정확

6. IV Smile 탭
---------------
입력
- T(년), Center K(중심행사가), ±Range(배율), #Points(격자 수), Type(call/put)
- ‘Generate Smile’ 클릭

출력
- K에 대한 BS IV 곡선(스마일)
- ‘Export CSV (smile)’: 컬럼 K, IV, T, flag

7. 권장 파라미터 범위(실무 감각치)
---------------------------------
- v0, θ ∈ [0.01, 0.09]  (10%~30% 변동성에 해당)
- κ ∈ [0.5, 4.0]       (평균회귀 속도)
- ξ ∈ [0.2, 1.0]       (vol-of-vol)
- ρ ∈ [-0.95, -0.2]    (주가 하락 시 변동성 상승 반영)
- r, q 는 시장 상황에 맞게(연 0~5% 범위가 흔함)

8. 검증/비교 아이디어
----------------------
- Black–Scholes 근사: ξ≈0, κ를 크게, v가 거의 일정하게 되도록 두면 BS와 근사적으로 일치
- Put-Call Parity 확인: C - P = DF_r(F - K) (GUI 가격이 합리적인지 체크)
- 시뮬레이션 평균 E[S_T] ≈ S0 * exp((r - q)T) 여부 점검

9. 자주 묻는 질문(FAQ) & 트러블슈팅
-----------------------------------
Q1) PyQt/Qt 오류(“cocoa plugin”, “both implemented” 등)가 뜹니다.
 - 한 환경에 PyQt5와 PyQt6가 섞이면 충돌합니다. 둘 중 하나로 정리하세요.
   (conda-forge 기준) PyQt5:  conda install -c conda-forge 'pyqt=5.15.*' 'qt-main=5.15.*'
   PyQt6:                   conda install -c conda-forge 'pyqt6' 'qt6-main'
 - 필요 시 QT_QPA_PLATFORM_PLUGIN_PATH를 Qt 플랫폼 폴더로 지정
   PyQt5: $CONDA_PREFIX/plugins/platforms
   PyQt6: $CONDA_PREFIX/lib/qt6/plugins/platforms

Q2) Implied Vol이 NaN으로 나옵니다.
 - 가격이 비정상/수치적 한계(극단 K, T, 파라미터)일 수 있습니다.
 - T를 키우거나, 파라미터를 완화, 또는 적분 상한을 300~500으로 올려보세요.

Q3) 플롯이 느립니다.
 - Paths를 줄이거나(미리보기 25개만 그림), Steps를 적절히 낮추세요.
 - CSV로 내보내 외부에서 분석하세요.

Q4) 분산이 음수로 내려갑니다.
 - Euler에서는 0으로 바닥 처리됩니다. QE 사용을 권장합니다.
 - Feller 조건 2κθ ≥ ξ^2를 가급적 만족시키세요.

10. 단축/조작 팁
----------------
- 플롯 툴바: Zoom(돋보기), Pan(손바닥), Home(전체 보기), Save(이미지 저장)
- 드래그로 확대 영역 지정, 우클릭/컨텍스트 메뉴는 플랫폼/백엔드에 따라 다를 수 있습니다.

11. 재현성/로깅
----------------
- Simulation 탭의 Seed로 난수 고정
- 동일 파라미터 + 동일 Seed → 동일 결과 재현


부록 A) 수학적 배경(요약)
--------------------------
- 가격 특성함수 ϕ(u; T)를 이용한 Heston 폐형식 적분(“Little Heston Trap”)
- 가격 적분 상한 기본 200(깊은 ITM/OTM 시 상향 조정 가능)
- 시뮬레이션: Andersen(2008) QE 스킴 + 상관 난수. 로그자산은 드리프트 (r - q - 0.5 v) dt

부록 B) 예제 워크플로
----------------------
(1) 상단 파라미터: S0=100, v0=0.04, κ=1.5, θ=0.04, ξ=0.5, ρ=-0.7, r=0.01, q=0
    → Apply Params
(2) Pricing: T=1.0, K=100, call → Compute → 가격과 IV 확인
(3) Simulation: T=1.0, Steps=252, Paths=2000, Scheme=QE, Seed=42 → Run
    → 요약통계 확인, 경로 CSV 저장
(4) IV Smile: T=1.0, CenterK=100, ±Range=0.5, Points=31 → Generate → CSV 저장
