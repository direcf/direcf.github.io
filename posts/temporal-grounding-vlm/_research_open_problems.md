# Open Problems + Novel Paper Ideas (2026-06)

(Internal note bundle for Chapter 10 — novel ideas capstone.)

## Part 1: Open Problems (12)

### OP-1. Hour-scale grounding is a SEARCH problem and VLMs can't search
- **Evidence**: ExtremeWhenBench (arXiv:2606.12300) — 2,273 queries / 194 videos / mean 75.7 min / max 9 hr
- **Quantified**: Qwen3.5-9B mIoU 0.110, CLIP-only baseline 0.269, Retrieve-then-ground hybrid 0.354 (6.7× over Video-LLM)
- **85% failures are search, 11% localization**

### OP-2. Streaming + audio + sub-second latency
- **Evidence**: "Harnessing Streaming Video in the Wild" (arXiv:2606.08615) — author explicit future work: audio integration + 1 FPS limit
- **Available data**: Streaming-Train-248K (248K samples, per-second alignment), Streaming-Eval (138 video / 15 category)
- **SOTA**: StreamingHarness-8B 61.4% narration win-rate, vision-only

### OP-3. Open-vocabulary verb / novel action
- **Evidence**: VTG-MLLM Survey (arXiv:2508.10922) — "limited generalization to unseen temporal actions"
- **Quantified**: compositional split R1@0.5 drops 15-25%p vs in-domain
- **Gap**: dedicated open-vocab benchmark mostly absent

### OP-4. Multi-event / compositional grounding ("A then B then C")
- **Evidence**: TimeBlind (arXiv:2602.00288) — Allen's 13 temporal relations, first compositional benchmark. PC-Net (NeurIPS '25) weakly-supervised compositional
- **Quantified**: SOTA Video-LLM hardly above random in TimeBlind paired-video setting

### OP-5. Audio-Visual temporal grounding
- **Evidence**: "Audio Does Matter" (arXiv:2508.04273) — "most datasets are visual-centric"
- **Gap**: no benchmark where audio is decisive

### OP-6. Causal temporal grounding ("Why did X happen?")
- **Evidence**: TimeBlind, V-STaR (arXiv:2503.11495)
- **SOTA**: VideoTemp-o3-7B-RL (arXiv:2602.07801) NextGQA mIoU 33.4, Acc 76.4% — gap between grounded and answer accuracy is large

### OP-7. Hallucination — fake moments confidently emitted
- **Evidence**: CounterVid (arXiv:2601.04778), DIQ-H (arXiv:2512.03992)
- **Quantified**: 7B VLM 30-50% false-positive on Charades-STA-Negative split

### OP-8. Ego ↔ Exo viewpoint transfer
- **Evidence**: EgoExo-Con (arXiv:2510.26113) — naive joint fine-tune degrades single-view perf
- **Gap**: view-invariant grounding pretraining

### OP-9. Sparse/point supervision (1/100 cost)
- **Evidence**: Game-perspective WSTSG (arXiv:2605.26441), Positive Sample Mining (arXiv:2505.06557)
- **Quantified**: point-supervised TSG still 10%p+ behind fully-supervised R1@0.5

### OP-10. Token-level rationale (which frame drove the decision?)
- **Evidence**: Step-Level Faithfulness (arXiv:2603.06828), TempCore (arXiv:2509.01167)
- **Gap**: attention rollout unreliable for multimodal multi-layer; gradient attribution quadratic in video length

### OP-11. Universal grounding still loses to domain experts
- **Evidence**: UniversalVTG (arXiv:2604.08522), Universal Video Temporal Grounding (arXiv:2506.18883)
- **Gap**: "one model for all benchmarks" unsolved

### OP-12. Reward hacking in RL-based grounding
- **Evidence**: VideoTemp-o3 (arXiv:2602.07801) — explicitly proposes "penalty-aware IoU" reward to address hacking
- **Gap**: IoU-maximization buys out semantic alignment

## Part 2: Novel Paper Ideas (12 with full method sketches)

### Idea 1. StreamGround: Audio-Synchronized Streaming Temporal Grounding
- **Pitch**: Per-second causal audio tokens into streaming Video-LLM for sound-cued event grounding
- **Hypothesis**: Audio provides implicit 1-2s look-ahead → R1@0.5 +5-7%p
- **Method**: Whisper audio encoder → per-second causal tokens → audio cross-attention layer in StreamingHarness-8B → joint train on Streaming-Train-248K + AudioSet narration
- **Eval**: Streaming-Eval, Charades-STA (audio split), Ego4D NLQ, EpicSounds
- **Baseline**: StreamingHarness-8B (arXiv:2606.08615)
- **Expected gain**: Streaming-Eval SW-F1 +4-6%, Charades-STA-Audio R1@0.5 65→71
- **Why not done**: streaming benchmark released June 2026, audio is author's stated future work
- **Risk**: noisy audio
- **Compute**: 8×H100 2 weeks

### Idea 2. RetGround-Agent: Tool-Using LLM Agent for Hour-Scale Search
- **Pitch**: Iterative retrieve→zoom→verify directly attacks the "85% search failure" finding
- **Hypothesis**: Agentic loop > monolithic Video-LLM by 3-4× mIoU
- **Method**: (1) Coarse CLIP retrieval top-K window, (2) LLM agent decides narrow/expand/move, (3) Frame inspector tool, (4) RL on tool-use policy
- **Eval**: ExtremeWhenBench, MAD, Ego4D-NLQ
- **Baseline**: Retrieve-then-ground hybrid (0.354 mIoU), Deep Video Discovery (arXiv:2505.18079)
- **Expected gain**: ExtremeWhenBench mIoU 0.354 → 0.42-0.48
- **Why not done**: benchmark just released
- **Risk**: agent loop latency blowup
- **Compute**: 4×H100 3 weeks

### Idea 3. GroundDiff: Diffusion Priors over Temporal Segments
- **Pitch**: Multi-event ordering as diffusion-sampled "segment distribution"
- **Hypothesis**: Implicit ordering constraint → TimeBlind composite +6-9%p
- **Method**: (1) k sub-event masks as latents, (2) text-conditioned diffuse, (3) IoU + ordering loss
- **Eval**: TimeBlind, ActivityNet-CG, PC-Net composite
- **Compute**: 8×A100 2 weeks

### Idea 4. AbstainGround: Negative-Aware Temporal Grounding
- **Pitch**: Abstention head + calibrated confidence directly attacks hallucination
- **Hypothesis**: Negative-aware training → Charades-STA-Negative FP rate -30%, in-domain R1@0.5 within 1%p
- **Method**: (1) LLM-synthesized negative queries, (2) abstention head with logit margin loss, (3) temperature calibration
- **Eval**: Charades-STA, ActivityNet, QVHighlights with Negative splits
- **Baseline**: CounterVid, Multi-Modal Hallucination Control (arXiv:2403.14003)
- **Expected**: AUROC for "moment exists" 0.65 → 0.82
- **Why not done**: no negative annotations exist; RL rewards punish abstention
- **Compute**: 4×A100 1 week

### Idea 5. WorldGround: World-Model-Based Causal Temporal Grounding
- **Pitch**: Video world model prediction error as "causal event" signal
- **Method**: (1) V-JEPA prediction loss per frame, (2) surprise curve as query-conditioned attention weight, (3) joint causal QA + grounding
- **Eval**: NextGQA, V-STaR, TimeBlind causal split
- **Baseline**: VideoTemp-o3 (NextGQA mIoU 33.4)
- **Expected**: NextGQA mIoU 33.4 → 38-40
- **Compute**: 8×H100 1 month

### Idea 6. EgoExoGround: View-Invariant Pretraining
- **Pitch**: EgoExo-Con synchronized pairs for view-invariant grounding pretrain
- **Method**: contrastive view-invariant moment embedding + adapter-based per-view fine-tune + view-dropout regularizer
- **Eval**: EgoExo-Con, Ego4D-NLQ, Charades-STA
- **Expected**: cross-view R1@0.5 +8-12%p, in-view -0.5%p
- **Compute**: 4×H100 2 weeks

### Idea 7. ClickGround: Click + LLM-Generated Pseudo-Spans
- **Pitch**: One click + LLM boundary expansion → 95% of fully-supervised
- **Method**: (1) click as center, frozen VLM expands boundary, (2) IoU-aware self-training, (3) consistency regularization
- **Eval**: Charades-STA click-relabeled, ActivityNet click-relabeled
- **Expected**: Click-supervised R1@0.5 50→58 (fully-sup ~62)
- **Compute**: 4×A100 2 weeks

### Idea 8. GroundRAG: Knowledge-Graph-Injected Open-Vocab Grounding
- **Pitch**: ConceptNet/Wikidata verb taxonomy → unseen verb generalization
- **Method**: query → KG node link, subgraph embedding concat, hierarchical contrastive
- **Eval**: Charades-STA compositional, ActivityNet-CG, new OOV-VTG split
- **Expected**: OOV split R1@0.5 +8-12%p
- **Compute**: 4×A100 2 weeks

### Idea 9. FaithGround: Token-Level Faithful Rationale
- **Pitch**: Counterfactual frame ablation → faithfulness reward → OOD generalization
- **Method**: per-frame ablation causal importance + RL reward (α·IoU + β·faithfulness) + faithfulness probe head
- **Eval**: Charades-STA, MAD OOD, TempCore
- **Baseline**: Step-Level Faithfulness (arXiv:2603.06828)
- **Expected**: TempCore frame-sensitivity +0.1, OOD R1@0.5 +4-6%p
- **Compute**: 8×A100 3 weeks

### Idea 10. SyntheticGround: Sora-2/Veo Counterfactual Video for Robustness
- **Pitch**: Generate same-query-different-viewpoint for view-invariant grounding
- **Method**: Sora 2 / Open-Sora re-render with 4 viewpoints, same-query-different-view contrastive, test on camera-motion adversarial
- **Eval**: Charades-STA + synthetic ext, Movie Gen Bench
- **Expected**: adversarial camera-motion R1@0.5 +6-9%p
- **Compute**: Cloud API ~$5K + 4×A100 2 weeks

### Idea 11. MemGround: KV-Cache as Long-Term Memory for Hour-Scale Streaming
- **Pitch**: CacheFlow-style compressed KV as explicit memory for grounding
- **Method**: query-conditioned KV importance score → retain important KV → abstain head for "not in memory"
- **Eval**: ExtremeWhenBench (streaming-converted), Streaming-Eval
- **Baseline**: CacheFlow (arXiv:2511.13644), LiveVLM (arXiv:2505.15269)
- **Expected**: streaming mIoU 0.20 → 0.32-0.35
- **Compute**: 8×H100 3 weeks

### Idea 12. SSPL-TG: Self-Supervised Pretrain via Reversed Future Prediction
- **Pitch**: Random span mask → "find this span" reversed self-sup task
- **Method**: mask 5-30% span, contrastive span vs LLM caption, lightweight grounding head fine-tune
- **Eval**: HowTo100M / InternVid 25M pretrain, downstream Charades-STA/ActivityNet/Ego4D-NLQ
- **Baseline**: TEMPURA (arXiv:2505.01583)
- **Expected**: zero-shot Charades-STA R1@0.5 ~20→28-30, fine-tuned +2-3%p
- **Compute**: 8×H100 1 month (pretrain heavy)

## Part 3: Data Investigation

### Largest current open datasets
- **MAD**: 650 movies × 1,200+ hr × 384K queries (arXiv:2302.13372)
- **Ego4D-NLQ**: ~3,670 hr × 19K queries
- **InternVid**: 234M clips (weak text-clip, no grounding label)
- **ExtremeWhenBench (2026.06)**: 194 videos × mean 75.7 min × 2,273 queries (eval only)
- **Streaming-Train-248K (2026.06)**: 248K per-second-aligned samples

### Synthetic data availability
- **Sora 2 / Open-Sora**: query→video synthesis possible, temporal precision weak
- **Rapidata/sora-video-generation-time-flow** (HF): temporal coherence eval data
- **Open-Sora-Plan** (github.com/hpcaitech/Open-Sora): fully open, viewpoint augmentation
- **VidGen-1M** (HF papers/2408.02629): 1M generated clips for pretrain

### Annotation cost estimates
- Span annotation: ~$0.30-0.60/query → 384K queries ≈ $150K
- Click supervision: 5-10× cheaper → $15-30K for same scale
- LLM-generated synthetic query: ~$0.001/query → 100K queries ≈ $100 (quality verify needed)

### 2026 new datasets
- **ExtremeWhenBench** (2026.06, arXiv:2606.12300)
- **Streaming-Train-248K + Streaming-Eval** (2026.06, arXiv:2606.08615)
- **TimeBlind** (2026.02, arXiv:2602.00288) — Allen-13-relation compositional
- **TimeLens** (2025.12, arXiv:2512.14698)
- **ToG-Bench** (2025.12, arXiv:2512.03666) — egocentric STVG
- **EgoExo-Con** (2025.10, arXiv:2510.26113)
- **DIQ-H** (2025.12, arXiv:2512.03992) — hallucination under temporal degradation
- **VideoTemp-Bench** (2026, with VideoTemp-o3)

### Recommended order (for the user)
1. ExtremeWhenBench leaderboard + Streaming-Eval = freshest white space
2. Idea 1 (Audio-Streaming) and Idea 2 (RetGround-Agent) = highest leverage / lowest novelty risk
3. Idea 4 (AbstainGround), Idea 9 (FaithGround) = reviewer-friendly "trust/responsibility" angle
