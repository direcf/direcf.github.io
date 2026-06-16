# Syllabus — Temporal Grounding for Video VLMs (2026)

Course slug: `temporal-grounding-vlm`
Category: `multimodal-ai`
Level: advanced
Code language: Python
Audience: ML/CV researchers and engineers who already know transformers and basic video processing; want a current map of 2026 SOTA and concrete research directions.

## Thesis

Temporal grounding이 2024년까지 proposal-based / DETR 계열로 풀던 boundary regression 문제였다면, 2025-2026년에는 **VLM이 timestamp를 직접 emit하는 생성 문제**로 재정의되었다. NeurIPS 2025의 Time-R1·UniTime, ICLR 2026의 VideoMind·MeCo, CVPR 2026의 VideoITG·TimeLens가 모두 같은 방향이다. 그러나 이 전환은 새로운 문제를 가져왔다 — **hallucination**, **search failure at hour scale**, **reward hacking**, **plug-and-play scarcity**. 이 강의는 (a) 무엇이 어디까지 풀렸는지, (b) 새 SOTA paper들이 어떤 trade-off를 안고 있는지, (c) 다음 12개월에 어디에 새 paper를 쓸 수 있는지를 다룬다.

## 10 Chapters

1. 🎯 **What is Temporal Grounding** — task family (TSG / MR / TAL), eval metrics, 왜 VLM 시대에 다시 뜨거워졌나
2. 📏 **Benchmark Landscape & 7 Biases** — Charades-STA부터 ExtremeWhenBench까지 11종, caption-only bias / discrete granularity / negative annotation 부재 등
3. 🏛️ **Pre-VLM Foundations — DETR Era** — 2D-TAN, MS-2D-TAN, M-DETR, QD-DETR, CG-DETR, UniVTG. Boundary regression의 종착점
4. 🧠 **VLM-as-Grounder** — timestamp token generation. UniTime (NeurIPS 2025), MeCo (ICLR 2026), Universal VTG. 왜 verbal generation이 regression head를 대체했나
5. 🎮 **RL Fine-tuning Era** — Time-R1 (NeurIPS 2025), TempSamp-R1, VideoTemp-o3, TimeLens (CVPR 2026). GRPO + verifiable rewards, reward hacking, penalty-aware IoU
6. 🕵️ **Agentic Search for Long-Form** — VideoMind (ICLR 2026, Chain-of-LoRA), AVI, Deep Video Discovery. ExtremeWhenBench의 "85% search failure" finding
7. 📡 **Streaming + Online Grounding** — StreamingHarness (June 2026), CacheFlow, LiveVLM. Sub-second latency, audio modality의 빈 자리
8. 🔌 **Plug-and-Play: VideoITG and the Empty Field** — VideoITG (CVPR 2026 Highlight)이 왜 유일한가. 2026이 plug-and-play에서 멀어진 이유
9. 🛡️ **Trust: Hallucination, Faithfulness, Abstention** — CounterVid, DIQ-H, Step-Level Faithfulness, TempCore. Reliability gap의 정량화
10. 🚀 **Novel Research Directions — 12 Paper Ideas + Data + Feasibility** — capstone. 새 논문을 위한 white space, dataset 가용성, compute 추정, baseline 비교
