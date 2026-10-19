# Portfolio 재설계 (Frost Blue + KR/EN 토글)

날짜: 2026-10-19 · 대상: `src/pages/portfolio/index.astro`

## 목표
기존 Light Editorial(베이지/amber) 포트폴리오를 사이트 메인 **frost-blue** 톤·시스템 폰트·compact 간격으로 재작성하고, 최신 경력/특허/논문/수상을 반영한다. 본문은 **KR/EN 토글**로 전환.

## 디자인 방향 — "Frost Card" (Direction A)
- 사이트 `global.css` 토큰 재사용: `--accent:#1d4ed8`, `--text:#0f1f3a`, `--bg:#eef4fb`, `--accent2:#0891b2`, `--border:#cfdbe8`, `--card:#fff`.
- 폰트: `-apple-system,"Apple SD Gothic Neo","Pretendard",sans-serif`, 본문 ~13~15px, line-height 1.6~1.8 (사이트와 일치).
- 세리프(Georgia) 사용 안 함. 경력/프로젝트는 **흰 카드**로 구획, 지표는 코발트 뱃지(`.m`)로 강조.
- 클래스는 `p-` prefix로 BaseLayout 전역 스타일과 격리 (기존 규칙 유지).

## 섹션 구성 (순서)
1. **Hero** — label / 이름(이승준) / role(Vision AI Engineer · SKT Enterprise 기술지원팀) / KR·EN 토글 / summary / meta(email·location·github·site)
2. **Education** — KAIST 전기및전자공학 석사(2020.09–2022.08, GPA 4.15/4.3, RITL Lab 김종환) / 한양대 전자공학 학사(2014.03–2020.08, GPA 4.04/4.5, Cum Laude)
3. **Career** — SK Telecom (2023.01 신입 입사 – 현재), 4개 프로젝트 카드:
   - ORBIT — Physical AI (2026.08–2026.12, 메인): Holiday Robotics Spatial VLM 공동개발(Stator Accuracy 32.6→84.2, 실시간성 확보) · BEHAVIOR-1K Robot FM w/SNU(VLA G0.5 100-task Q 0.191→0.450, Submitted) · 시냅스고 플랫폼 기획·개발(총괄 PM+메인). 진행 중: B1K Challenge, Valeo PoC
   - SynapsEgo — Ego-view Agentic AI (2025.06–2025.09, 메인): 경찰청 영상요약·리포트(비디오 의미론적 분할, Qwen3-MoE·Qwen2.5-VL fine-tune, SOTA Gemini-2.5-pro 상회, Recursive Agentic Pipeline) · 연세대 간호대 SOP PoC · 고려대 Ego-Exo Fusion. 성과: 2026 MWC 출품, EgoExoChat IEEE TNNLS Accept(2026.10)
   - SK하이닉스 向 PoC — 산업안전·물리보안·소방안전 (2025.04–2026.07, 메인/서브): 산업안전 지능형 CCTV(오알람 96% 저감) · 물리보안·소방안전 VLM PoC · 인텔리빅스 협업. 성과: 본사업 요건 조기 충족
   - CareVia — 발달장애 AI 돌봄 (2023.05–2026.05, 메인): FR·ReID SOTA·AI Box 운용 · 행동인식 9종 DeepStream·포즈추정(300만장+)·멀티카메라 협동(정확도 71.5→90.9%, 경량화 79%↓). 성과: 과기부 장관상(ICT 2025)·전국 12개소 복지관 상용·법무병원 입찰
4. **Skills** — ① LLM/VLM/VLA Agentic Pipeline ② Computer Vision·ML/DL ③ Languages & Infra (chip 태그)
5. **Publications** — EgoExoChat (IEEE TNNLS 2026, Accepted)
6. **Patents** — 국내 23건(공동발명, 주 특허자 7건), 5분야 그룹 + 대표 특허 + "외 N건":
   - Video Understanding 4건(주 3): 다중시점 비디오 이해 / 비디오 의미론적 분할 / 다중시점·시공간 그리드 VLM 행동인식(등록) · 외 1(요약 보고서)
   - Visual Perception 5건(주 3): 객체 추적 / 폐색 강건 포즈 추정 / 동물 승가행위 탐지 · 외 2(재식별, 행동 관제)
   - Agentic/Multimodal 8건(주 1): 동적 시공간 그래프·시간 역추적 사건 원인 분석(등록) · 외 7
   - Robot/Physical AI 5건(주 0): 고정형·이동형 로봇 카메라 협력(등록) · 외 4
   - Industrial Safety 1건(주 0): 안전점검항목 유형별 작업환경 안전점검
7. **Awards & Challenges** — 이모지 제거, 왼쪽 고정폭 연도 열 + rule 구분, 최신순:
   - 2026 BEHAVIOR Challenge Robot FM (Stanford VL/CoRL, Submitted)
   - 2025 과기부 장관상 ICT Award Korea (CareVia)
   - 2024 CVPR Workshop AVA Challenge 3rd (Keypoint Track)
   - 2024 SKT AI Frontier 4기 최우수상 (Vision Retrieval System)
8. **Certifications** — 동일 연도열 포맷: 2026.09 AWS Certified Generative AI Developer – Professional
9. **Footer**

## 유지 사항
- **잠금화면**: 기존 blue 잠금 UI + 클라이언트 SHA-256 + `sessionStorage['port:auth']` 게이트 그대로.
- **PDF**: `@media print` A4 재조판 + `window.print()` (Hero 버튼 + FAB).
- `:global(.sidebar/.header/...)` print 숨김 규칙 유지.

## KR/EN 토글
- Hero에 토글 UI. JS로 `[data-lang]` 요소를 보이거나, KR/EN 두 텍스트 블록을 토글. 기본 KR. 구조·레이아웃은 동일, 텍스트만 치환.

## 범위 외 (YAGNI)
- 다크모드 별도 대응 안 함(이력서는 라이트 고정, print 친화).
- 특허 23건 전체 나열 안 함(대표+총건수).
