# Real-time Video LLM 코스 Syllabus

## Mental Model Shift

| Video LLM 종류 | 핵심 문제 | 멘탈 모델 |
|---|---|---|
| **General** (1-3분 비디오) | "어떤 프레임을 선택할 것인가?" | **Sampler 잘 짜기** |
| **Hour-scale** (1시간+ 비디오) | "메모리와 컨텍스트를 어떻게 트레이드오프?" | **Memory System 설계** |
| **Real-time** (실시간 스트림) | "들어오는 데이터를 latency budget 안에서 처리할 것인가?" | **Streaming Pipeline + Adaptive Processing** |

## 시니어 평가 기준 (Real-time)

(a) **Latency vs Quality의 양적 이해**: p50/p99 latency, throughput, 정확도 사이의 명시적 트레이드오프 모델링
(b) **Real-time benchmark의 의미 정확히 짚기**: VStream-Bench, OVO-Bench가 측정하는 능력 (memory persistence? proactive response? sliding context?)
(c) **Micro-design decision 의도 읽기**: VideoLLM-online의 EOS-based stream alignment, Flash-VStream의 STAR memory hierarchy, Streaming Vision Transformer의 cross-frame attention 같은 결정의 배경

## 10챕터 구성

### Foundation (1-2): 문제 정의
1. **The Real-time Video LLM Problem** - 왜 sampler/memory만으로는 부족한가
2. **Streaming Input Pipeline** - 들어오는 데이터를 어떻게 처리할 것인가

### Core Mechanics (3-5): 핵심 메커니즘
3. **Adaptive Frame Sampling under Latency Budget** - 실시간 제약 하의 프레임 선택
4. **Sliding Window & Streaming Context** - 과거 정보의 유지 vs forgetting
5. **Visual Encoder Optimization** - encoder를 어떻게 빠르게 할 것인가

### Advanced Patterns (6-7): 고급 기법
6. **Token Compression for Streaming** - streaming 환경의 토큰 압축 (HiCo, STAR, Streaming Vision Transformer)
7. **KV Cache & Continuous Batching** - LLM inference 측 최적화 (vLLM, TensorRT-LLM, prefix cache)

### System Design (8-9): 시스템 설계
8. **End-to-End Latency Profiling** - bottleneck 분석과 SLA 설정
9. **Producer-Consumer Pipeline Architecture** - 전체 시스템 설계

### Production (10): 프로덕션
10. **Production Deployment & Monitoring** - SLA 관리, A/B 테스트, observability
