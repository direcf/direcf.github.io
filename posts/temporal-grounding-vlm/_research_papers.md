# 2026 Conference Papers on Temporal Grounding for VLMs

(Internal notes — deep-research workflow verified, 2026-06)

## Verified by deep-research (6 papers, all unanimous votes)

### VideoMind (ICLR 2026 Poster)
- **Title**: Chain-of-LoRA agent for temporal-grounded video reasoning
- **Authors**: Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, Mike Zheng Shou
- **arxiv**: https://arxiv.org/abs/2503.13444
- **github**: https://github.com/yeliudev/VideoMind
- **Key idea**: 4-role agentic workflow (planner / grounder / verifier / answerer) coordinated via multiple LoRA adapters on a unified Qwen2-VL base model
- **Trained method** via LoRA fine-tuning; not training-free; not plug-and-play
- **Eval**: 15 benchmarks across Grounded VideoQA + Video Temporal Grounding + General VideoQA

### VideoITG (CVPR 2026 Highlight) ★ The ONLY plug-and-play module identified
- **Title**: Multimodal Video Understanding with Instructed Temporal Grounding
- **Authors**: Shihao Wang, Guo Chen, De-an Huang, Zhiqi Li, Minghan Li, Guilin Liu, Jose M. Alvarez, Lei Zhang, Zhiding Yu (NVIDIA Labs)
- **arxiv**: https://arxiv.org/abs/2507.13353
- **github**: https://github.com/NVlabs/VideoITG
- **CVPR 2026 page**: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VideoITG_Multimodal_Video_Understanding_with_Instructed_Temporal_Grounding_CVPR_2026_paper.html
- **Key idea**: Plug-and-play frame selector that scores 512 uniformly sampled frames and picks Top-K based on user instructions for downstream Video-LLMs
- **Trained on VideoITG-40K** (40K videos, 500K temporal grounding annotations) via VidThinker automated annotation pipeline
- Originally withdrawn from ICLR 2026 → accepted as CVPR 2026 Highlight (2026/04/09)

### MeCo (ICLR 2026)
- **Title**: Measure Twice, Cut Once — A Semantic-Oriented Approach to Video Temporal Localization with Video LLMs
- **Authors**: Zongshang Pang, Yuta Nakashima (Osaka U.), Mayu Otani (CyberAgent)
- **arxiv**: https://arxiv.org/abs/2503.09027
- **github**: https://github.com/pangzss/MeCo
- **Key idea**: Timestamp-free semantic-oriented fine-tuning — structural token generation + query-focused captioning + contrastive structural-token grounding
- **Trained method** (LoRA rank 128, 1 epoch on E.T.Instruct 164K, base: E.T.Chat / Qwen2VL-7B)
- **QVHighlights SOTA**: mAP=45.3, HIT@1=75.1 — surpasses M-DETR, UMT, QD-DETR, CG-DETR, UniVTG
- ⚠️ MeCo zero-shot Charades-STA SOTA claim was REFUTED in verification

### Time-R1 (NeurIPS 2025) ★ Charades-STA + ActivityNet SOTA
- **Title**: RL post-training framework using GRPO with verifiable tIoU + format reward for LVLMs
- **Authors**: Ye Wang et al. (Renmin / Xiaomi, 17 authors)
- **arxiv**: https://arxiv.org/abs/2503.13377
- **github**: https://github.com/xiaomi-research/time-r1
- **project**: https://xuboshen.github.io/Time-R1/
- **NeurIPS proceedings**: https://papers.neurips.cc/paper_files/paper/2025/file/7801b29c93b599b8d0c44138596bdeed-Paper-Conference.pdf
- **Key idea**: RL post-training (GRPO + verifiable rewards) adapts Qwen2.5-VL-7B for temporal video grounding via TimeRFT curated dataset (2.5K)
- **Trained method**, not training-free
- **Zero-shot SOTA**: Charades-STA R1@0.3=78.1 / R1@0.5=60.8 / R1@0.7=35.3; ActivityNet 58.6/39.0/21.4 (surpasses VideoChat-Flash 74.5, VideoMind 73.5, TimeSuite 69.9)
- **Time-R1* (after fine-tune)**: Charades-STA R1@0.5=72.2, R1@0.7=50.1; ActivityNet R1@0.5=55.6, R1@0.7=34.0
- Caveat: zero-shot does NOT surpass TRACE on ActivityNet R1@0.7 (21.4 vs 24.0)

### UniTime (NeurIPS 2025)
- **Title**: Universal Video Temporal Grounding with Generative Multi-modal Large Language Models
- **Authors**: Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, Weidi Xie
- **arxiv**: https://arxiv.org/abs/2506.18883
- **NeurIPS poster**: https://neurips.cc/virtual/2025/poster/119042
- **Key idea**: Robust universal grounding model leveraging vision-language understanding of generative MLLMs

### TimeLens (CVPR 2026)
- **Title**: Trained MLLM family using thinking-free RLVR (Reinforcement Learning with Verifiable Rewards)
- **Authors**: TencentARC
- **Key idea**: Built on Qwen2.5-VL and Qwen3-VL via training on TimeLens-100K data
- (Result truncated in deep-research output — additional verification needed for full details)

## Adjacent papers from open-problems research

### RL-based grounding (2025-2026)
- **TempSamp-R1** (arXiv:2509.18056) — RL fine-tuning, ActivityNet mIoU 49+
- **VideoTemp-o3** (arXiv:2602.07801) — penalty-aware IoU reward, NextGQA mIoU 33.4

### Agentic search
- **AVI (Agentic Video Intelligence)** (arXiv:2511.14446) — Charades-STA [email protected] 88.6
- **Deep Video Discovery** (arXiv:2505.18079) — retrieve-zoom-verify loop

### Long-form / streaming
- **ExtremeWhenBench** (arXiv:2606.12300) — search-problem benchmark (June 2026)
- **Harnessing Streaming Video in the Wild** (arXiv:2606.08615) — Streaming-Train-248K, Streaming-Eval (June 2026)
- **CacheFlow** (arXiv:2511.13644) — KV-cache compression for streaming
- **LiveVLM** (arXiv:2505.15269) — live video VLM

### Hour-scale grounding
- **TAR-TVG** (arXiv:2508.07683) — Charades-STA mIoU 61.1, [email protected] 83.6
- **CONE** (arXiv:2209.10918), **RGNet** (arXiv:2312.06729) — MAD baselines
- **Multi-Scale Contrastive for MAD** (arXiv:2412.07157) — +3.58 [email protected]
- **Hand Trajectory Fusion** (arXiv:2606.02962) — Ego4D NLQ (June 2026)
- **GroundNLQ** (arXiv:2306.15255) — Ego4D NLQ baseline

### Hallucination / abstention / faithfulness
- **CounterVid** (arXiv:2601.04778) — hallucination quantification
- **DIQ-H** (arXiv:2512.03992) — hallucination under temporal degradation
- **Step-Level Faithfulness** (arXiv:2603.06828) — predicts OOD generalization
- **TempCore** (arXiv:2509.01167) — frame-selection sensitivity

### Compositional / causal
- **TimeBlind** (arXiv:2602.00288) — Allen-13-relation compositional (Feb 2026)
- **V-STaR** (arXiv:2503.11495) — spatio-temporal reasoning

### View transfer
- **EgoExo-Con** (arXiv:2510.26113) — synchronized ego-exo pairs
- **Fine-grained Spatiotemporal Grounding on Egocentric Videos** (arXiv:2508.00518)

### Open-vocab / weakly-supervised
- **VTG-MLLM Survey** (arXiv:2508.10922) — limited unseen action generalization
- **Universal VTG** (arXiv:2604.08522) — April 2026
- **Game-perspective WSTSG** (arXiv:2605.26441)
- **Positive Sample Mining WSTSG** (arXiv:2505.06557)
- **VERIFIED** (arXiv:2410.08593) — fine-grained negative augmentation
- **PC-Net** (NeurIPS 2025) — weakly-supervised compositional

### Self-supervised pretraining
- **TEMPURA** (arXiv:2505.01583)
- **NaQ** (arXiv:2301.00746) — narrations as queries

### Other (relevant context)
- **SMORE** vs **SG-DETR** — QVHighlights +4.19% [email protected]
- **SeeRankFilter** (arXiv:2511.22906) — QVHighlights avg mAP 44.05
- **Lighthouse** (arXiv:2408.02901) — moment retrieval library
- **TimeChat**, **VTimeLLM**, **TimeSuite** (arXiv:2410.19702), **TimeScope** (arXiv:2509.26360) — earlier VLM groundres

## Verified SOTA per benchmark (2026-06)

| Benchmark | Best | Score | Paper |
|---|---|---|---|
| Charades-STA R1@0.5 | Time-R1* | 72.2 | arXiv:2503.13377 (NeurIPS 2025) |
| Charades-STA [email protected] | AVI | 88.6 | arXiv:2511.14446 |
| Charades-STA mIoU | TAR-TVG | 61.1 | arXiv:2508.07683 |
| ActivityNet R1@0.5 | Time-R1* | 55.6 | arXiv:2503.13377 |
| ActivityNet mIoU | TempSamp-R1 | 49+ | arXiv:2509.18056 |
| QVHighlights mAP | MeCo (Qwen2VL-7B) | 45.3 | arXiv:2503.09027 (ICLR 2026) |
| QVHighlights HIT@1 | MeCo | 75.1 | arXiv:2503.09027 |
| TACoS [email protected] | UniTime/Universal VTG | 60+ | arXiv:2506.18883 (NeurIPS 2025) |
| MAD [email protected] | Multi-Scale Contrastive | (+3.58 over CONE) | arXiv:2412.07157 |
| Ego4D NLQ | Hand Trajectory Fusion | (+2.54 [email protected]) | arXiv:2606.02962 |
| NextGQA mIoU | VideoTemp-o3-7B-RL | 33.4 | arXiv:2602.07801 |
| ExtremeWhenBench mIoU | Retrieve-then-ground hybrid | 0.354 | arXiv:2606.12300 |

## Key 2026 narrative

1. **VLM as grounder dominates**: 5 of 6 major papers (VideoMind, MeCo, Time-R1, UniTime, TimeLens) are VLM/MLLM-based trained methods
2. **Plug-and-play scarcity**: Only VideoITG (CVPR 2026 Highlight) is true plug-and-play
3. **RL post-training is the new lever**: Time-R1, TempSamp-R1, VideoTemp-o3, TimeLens all use RLVR
4. **Agentic search emerging**: VideoMind (Chain-of-LoRA), AVI, Deep Video Discovery
5. **Hour-scale is a SEARCH problem**: ExtremeWhenBench shows monolithic Video-LLM (mIoU 0.110) loses to CLIP retrieval (0.269) and retrieve-then-ground (0.354)
6. **Training-free still rare**: most modern methods need post-training; few zero-shot competitive options
