# NVIDIA Physical AI Map — Coursework Design

- Date: 2026-07-06
- Type: coursework (direcf.github.io, coursework-creator)
- Category: `physical-ai` (COMPASS와 같은 섹터), slug `nvidia-physical-ai-map`
- Level: advanced-but-accessible · 개념/아키텍처 중심(코드 가볍게) · NVIDIA 중심(최소 비교) · 13챕터
- Figures: NVIDIA official 문서/블로그 + 논문 다이어그램을 다운로드해 `figures/`에 저장, 섹션에 삽입(출처 caption). arXiv 자동추출은 ANTHROPIC_API_KEY 부재로 스킵.

## Goal
2026년 7월 기준 NVIDIA가 Physical AI 분야를 어떻게 만들어가는지 전체 그림을, 기초 용어·history·관계성부터 최신 연구/도메인까지 insight 중심으로.

## Syllabus (13)
1. Physical AI란 & NVIDIA의 판돈 (three computers)
2. Omniverse & OpenUSD
3. Isaac Sim
4. Isaac Lab & Newton
5. Cosmos (WFM)
6. 네 기둥의 관계성 — 스택 총정리
7. 합성 데이터 문제 (SDG, GR00T-Dreams/Mimic, Data Factory)
8. GR00T Foundation Model 계보
9. 로봇 정책 워크플로우 (R²D²: X-Mobility·COMPASS·HOVER·ReMEmbR)
10. Autonomous Driving 스택 (DRIVE·Alpamayo·Halos)
11. 산업 디지털 트윈 & 로봇 플릿
12. 배치 하드웨어 (Jetson Thor, Reference Humanoid)
13. 2026 최전선 & 전체 그림

## Key facts anchored (2026-07 기준)
- Omniverse: OpenUSD 기반, "Physical AI OS", DSX Blueprint.
- Isaac 계보: Isaac Gym→OmniIsaacGymEnvs→Orbit→Isaac Lab. Isaac Lab 3.0 = Newton backend + kit-less + factory pattern.
- Newton: NVIDIA+Google DeepMind+Disney, 2025.09 Linux Foundation, Warp+OpenUSD, differentiable, Newton 1.0.
- Cosmos: CES 2025.01 출시(9000T tokens), Predict/Transfer/Reason, Cosmos 3(GTC 2026: Nano 16B/Super 64B/Edge 2B, omnimodel 5 modality).
- GR00T: Project GR00T 2024.03 → N1(2025.03, arXiv:2503.14734, dual-system) → N1.5(2025.05, GR00T-Dreams 36h) → N1.6(2025.12) → N1.7(2026.06, EgoScale 20K h human video).
- three computers: DGX(train)/Omniverse+OVX(sim)/Jetson AGX Thor(deploy).
- GTC 2026: Cosmos 3, GR00T N1.7, Alpamayo 1.5, Reference Humanoid(Unitree H2 Plus+Jetson Thor T5000), Physical AI Data Factory Blueprint.

Related prior work in repo: [[project_compass_course]], [[research_compass_cross_embodiment]] — 9장에서 COMPASS 코스로 링크.
