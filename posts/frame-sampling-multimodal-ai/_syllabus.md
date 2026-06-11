# Syllabus: Frame Sampling for Multimodal AI (2026)

Course slug: `frame-sampling-multimodal-ai`
Category: `multimodal-ai`
Level: advanced
Language for code: Python
Audience: ML/CV engineers who already understand transformers and basic video processing; want to ship a video-LLM pipeline.

## Course thesis

비디오 LLM의 *진짜* 병목은 모델 크기가 아니라 **어떤 프레임을 LLM의 한정된 context에 통과시키느냐**이다. 연구실에서는 query-aware adaptive sampler들이 SOTA를 깨고 있지만, 상용 서비스(Gemini, Qwen-VL, LLaVA-Video, Twelve Labs)는 여전히 uniform sampling이 표준이다. 이 강의는 (a) 왜 그 갭이 존재하는지, (b) 어디서 SOTA plug-and-play 모듈을 갈아끼울지, (c) 그 모듈을 실제 시스템 안에서 어떻게 swappable한 인터페이스로 묶을지를 다룬다.

## 10 Chapters

1. **🎬 Why Frame Sampling Is the Bottleneck** — context window 한계, token economics, 정보 손실의 정량화. 왜 더 큰 LLM이 아니라 더 똑똑한 sampler가 답인가.

2. **🎯 Classical Baselines and Where They Fail** — uniform, fps-based, PySceneDetect 기반 shot detection. 짧은 비디오에서는 충분, 긴 비디오에서 실패하는 정확한 지점.

3. **🧭 Query-Aware Adaptive Sampling — AKS & BOLT** — CVPR 2025의 두 plug-and-play training-free 모듈. Relevance vs coverage 트레이드오프 정식화. LLaVA-Video-7B+AKS > LLaVA-Video-72B.

4. **🎓 Learned Samplers — Frame-Voyager, M-LLM, GenS** — ICLR 2025 / CVPR 2025 / ACL 2025 의 learned approach. Combinational ranking, spatial+temporal MLLM scoring, generative retriever.

5. **🌐 Relevance × Diversity Frontier — Q-Frame, AdaRD-Key, FOCUS** — ICCV 2025와 ICLR 2026 submissions. log-det diversity, multi-armed bandit. 2026 frontier가 어디로 가는가.

6. **🎞️ Long-Video Sampling — LongVU, Hour-LLaVA, VideoMarathon** — 1시간+ 비디오 처리. MemAug, DINOv2 temporal pruning. HourVideo 벤치마크.

7. **🗜️ Token-Level Compression — VideoChat-Flash, NVILA, FastVID** — 샘플링과 직교하는 압축 축. HiCo 1/50, 10K-frame NIAH 99.1%. 언제 sample 줄이고 언제 token 줄일까.

8. **🏢 Commercial Reality — Gemini, Twelve Labs, OpenAI/Anthropic, Open-source** — 상용 서비스가 실제 무엇을 쓰는가. 왜 uniform이 여전히 이기는가. 연구 SOTA와 production 사이의 영구적 갭.

9. **🔌 Plug-and-Play Architecture Design** — Sampler.select() 계약, swap point, vLLM-Omni disaggregated pipeline, Twelve Labs Embed cache, anti-patterns 6선.

10. **🧪 Reference Architecture & Evaluation** — 직접 swappable pipeline 만들기. Video-MME, MLVU, LongVideoBench, EgoSchema, Multi-Hop NIAH 평가. SOTA 갈아끼우는 운영 워크플로우.
