# Temporal Grounding Benchmarks Survey (2026-06)

(Internal note bundle. Not part of the final course.)

## Short/Medium-form Benchmarks

### Charades-STA (Gao et al., ICCV 2017) — arXiv:1705.02101
- 12,408 train / 3,720 test sentence-moment pairs, 9,848 indoor videos
- avg video ~30s, avg moment ~8.2s (27% of video)
- Metric: R@N (N=1,5), IoU thresholds {0.3, 0.5, 0.7}, mIoU
- **2026 SOTA**: TAR-TVG (arXiv:2508.07683) mIoU 61.1, [email protected] 83.6. AVI (arXiv:2511.14446) [email protected] 88.6
- **Bias**: Otani et al. (arXiv:2009.00325) — query-only or prior-only achieves strong perf. `start_norm` heavily concentrated near 0

### ActivityNet-Captions (Krishna et al., ICCV 2017) — arXiv:1705.00754
- 20K videos, 100K queries, splits 10K/4.9K/5K (val_1, val_2 as test)
- avg video ~120s, avg moment ~36s (~30% of video)
- 13.48 words/query
- **2026 SOTA**: TempSamp-R1 (arXiv:2509.18056) mIoU 49+, Universal Generative MLLM (arXiv:2506.18883)
- **Bias**: temporal location bias — ActivityNet-CD (arXiv:2101.09028) recommended

### TACoS (Regneri et al., TACL 2013)
- 127 videos, 18,818 video-query pairs (10,146/4,589/4,083 split)
- single domain (MPII-Cooking), avg ~4.79 min, moment ~5.4s (1.8% of video)
- **2026 SOTA**: Universal VTG (arXiv:2506.18883) [email protected] 60+
- **Limit**: domain narrow, small vocabulary, very short moments

### DiDeMo (Hendricks et al., ICCV 2017) — arXiv:1708.01641
- 10,464 videos, 40K moment-query pairs
- max 30s videos, 5s segments × 6 (discrete)
- 72% queries are single-segment
- Metric: Rank@1/5, mIoU (discrete 5s)
- **Limit**: discrete quantization prevents fine-grained IoU

### QVHighlights (Lei et al., NeurIPS 2021) — arXiv:2107.09609
- 10,148 videos, 10,310 queries, 18,367 moments
- avg video 150s, avg moment ~24s, 1.8 disjoint moments/query
- Metric: [email protected]/0.7, mAP@{0.5,0.75,avg 0.5:0.05:0.95}, HIT@1
- **2026 SOTA**: SMORE vs SG-DETR +4.19% [email protected]. SeeRankFilter (arXiv:2511.22906) avg mAP 44.05
- **Limit**: vlog/news domain narrow

## Long-form / Domain-specific

### MAD (Soldan et al., CVPR 2022) — arXiv:2112.00431
- 650+ movies, 1,200+ hours, 384K sentences
- avg video ~110 min, avg moment ~4.1s (0.06% of movie — extreme needle-in-haystack)
- audio description ASR-extracted, free-form sentence
- Metric: R@{1,5,10,50,100}@IoU{0.1,0.3,0.5}
- **2026 SOTA**: Multi-Scale Contrastive Learning (arXiv:2412.07157) [email protected] +3.58 over CONE. RGNet (arXiv:2312.06729)
- **Limit**: raw video undisclosed (copyright) — features only

### Ego4D NLQ (Grauman et al., CVPR 2022) — arXiv:2110.07058
- v2: 19.2K queries, 227 hours (11.3K/3.9K/4.0K split)
- 9 countries 855 wearers egocentric
- avg clip 8.2 min, avg moment 10.5s (~2% of clip)
- 13 query templates
- **2026 SOTA**: Hand Trajectory Fusion (arXiv:2606.02962, June 2026) [email protected] +2.54 on Hand-Object Interaction. NaQ pretraining + GroundNLQ (arXiv:2306.15255)

### Ego4D MQ (Moment Queries)
- 110 action class, ~22.2K instances, 326 hours
- Single label (action class) — essentially temporal action detection

### NaQ (Ramakrishnan et al., CVPR 2023) — arXiv:2301.00746
- 945K training samples, 5,389 clips
- Ego4D narrations auto-converted to NLQ-style
- Pretraining/augmentation only

### GroundVQA / EgoTimeQA (Di & Xie, CVPR 2024) — arXiv:2312.06505
- 303K QA pairs with temporal windows (30× QaEgo4D)
- Joint grounding + QA

### HiREST (Zala et al., CVPR 2023) — arXiv:2303.16406
- 3.4K text-video pairs, 1.1K with moment+step
- HowTo100M-based instructional, hierarchical (video→moment→step→caption)
- **Limit**: small scale

### MomentSeeker (Yuan et al., 2025) — arXiv:2502.12558
- 1.8K queries, 4 meta × 18 sub-tasks
- avg video 1,200s+, max 7,108s (2hr) — 5-300× longer than prior
- Multi-domain (movies, surveillance, ego, sports)
- Multi-modal query (text + image + video-conditioned)

### Video-MME temporal subset (arXiv:2405.21075)
- 900 videos / 254 hr / 2,700 QA, Short/Medium/Long splits
- Temporal Reasoning/Perception task type — multiple-choice QA, not boundary regression
- Follow-up: TimeSuite (arXiv:2410.19702), TimeScope (arXiv:2509.26360)

### 2026 newcomers
- **ExtremeWhenBench** (arXiv:2606.12300, June 2026): hour-long, search-problem reformulation
- **TVGBench** (LVLM-friendly short, 11 balanced query types)
- **TempCore** (arXiv:2509.01167): frame-selection sensitivity for grounded QA validation

## Dataset-level Biases (7 known hacks)

1. **Caption-only / prior-only bias** (Charades-STA, ActivityNet-Captions): query-only or prior-only baseline strong
2. **Word-level shortcut**: first verb predicts temporal interval
3. **Why-not / negative annotation absence**: only positive moments annotated
4. **Temporal localization vs description entanglement**: MAD needs character ID tracking, Ego4D MQ is essentially action detection, QVHighlights forces grounding + saliency joint
5. **Discrete granularity (DiDeMo)**: 5s quantization prevents [email protected] evaluation
6. **Train/test temporal distribution leak**: Charades-CD, ActivityNet-CD recommended for OOD eval
7. **Long-form scarcity**: only MAD / Ego4D / MomentSeeker / ToTG-Bench at hour-scale; each has bias

## 2026 Trends
- (a) **VLM-as-grounder**: timestamp token generation (Universal VTG, TimeChat, VTimeLLM)
- (b) **RL fine-tuning**: Time-R1 (arXiv:2503.13377), TempSamp-R1 (arXiv:2509.18056), VideoTemp-o3 (arXiv:2602.07801)
- (c) **Agentic search**: AVI (arXiv:2511.14446), Deep Video Discovery (arXiv:2505.18079), ToTG
- (d) **Long-context**: MomentSeeker, ToTG-Bench, ExtremeWhenBench
