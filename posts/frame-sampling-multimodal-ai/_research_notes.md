# Frame Sampling for Multimodal AI — Research Notes

(Internal note bundle used to seed chapter writing. Not part of the final course.)

## 1. Curated Papers (2025–2026)

### Plug-and-play, training-free, query-aware
- **AKS (CVPR 2025)** — Tang et al. — arXiv:2502.21271 — formulates keyframe selection as optimization balancing prompt-relevance vs. coverage; LLaVA-Video-7B+AKS beats LLaVA-Video-72B. https://github.com/ncTimTang/AKS
- **BOLT (CVPR 2025)** — Liu et al. — arXiv:2503.21483 — training-free inverse-transform sampling weighted by query–frame alignment; Video-MME 53.8→56.1, MLVU 58.9→63.4 with 8-frame budget. https://github.com/sming256/BOLT
- **Q-Frame (ICCV 2025)** — Zhang et al. (Xiaomi) — arXiv:2506.22139 — training-free query-aware selection + per-frame multi-resolution under token budget; model-agnostic.
- **AdaRD-Key (ICLR 2026 submission)** — Xian et al. — arXiv:2510.02778 — Relevance–Diversity Max-Volume (log-det diversity + query relevance); gating switches to diversity-only on weak queries.
- **FOCUS (ICLR 2026 submission)** — arXiv:2510.27280 — combinatorial pure-exploration multi-armed bandit (Bernstein bound); +11.9% on 20-min+ LongVideoBench with <2% of frames.

### Learned samplers
- **Frame-Voyager (ICLR 2025)** — Yu et al. — arXiv:2410.03226 — learns to rank frame *combinations* (not single frames) with combinational ranking supervision conditioned on query text.
- **M-LLM Frame Selection (CVPR 2025, Amazon)** — Hu et al. — arXiv:2502.19680 — M-LLM scores spatial single-frame importance + temporal multi-frame coherence, drops into frozen downstream Video-LLM.
- **GenS (ACL 2025 Findings)** — Yao et al. (Salesforce / Li Junnan) — arXiv:2503.09146 — uses VideoLLM itself as generative retriever; +13.4 pts Aria, +13.6 GPT-4o on LongVideoBench.

### End-to-end / long-video / token compression
- **VideoChat-Flash (ICLR 2026)** — Li et al. (OpenGVLab) — arXiv:2501.00574 — Hierarchical Clip→Video token compression (~1/50 ratio); 99.1% on 10K-frame NIAH. https://github.com/OpenGVLab/VideoChat-Flash
- **LongVU (ICLR 2025)** — Shen et al. (Meta) — arXiv:2410.17434 — spatiotemporal adaptive compression with DINOv2 temporal pruning + cross-attention spatial reduction.
- **Hour-LLaVA / VideoMarathon (NeurIPS 2025 Spotlight)** — Lin et al. — 1 FPS hour-long video training with MemAug; releases 9.7K-hour VideoMarathon dataset.
- **NVILA / Cosmos Nemotron (NVIDIA 2025)** — arXiv:2412.04468 — 256-frame "scale-then-compress" approach.
- **FastVID (2025)** — arXiv:2503.11187 — dynamic density pruning for Video-LLMs.

### Agentic loops (2026)
- **FrameThinker (arXiv:2509.24304)**, **FrameMind (arXiv:2509.24008)**, **A.I.R. (arXiv:2510.04428)** — VLM as agent issues `request_frames(t_start, t_end, fps)` tool calls interleaved with chain-of-thought.

### Benchmarks
- **Video-MME** (Long split) — comprehensive video QA, up to 60-min
- **MLVU** — multitask long-video understanding
- **LongVideoBench** — referred-question QA in long videos
- **EgoSchema** — long-form egocentric QA
- **HourVideo** — hour-scale benchmark
- **Multi-Hop NIAH** (VideoChat-Flash) — needle-in-haystack at 10K frames

## 2. Commercial State (2025–2026)

### What ships
| Player | Model | Sampling | Max | Cost |
|---|---|---|---|---|
| Google | Gemini 2.5 Pro | **1 fps default** (configurable), audio 1 Kbps parallel, 258 tokens/sec | 1M ctx ≈ 1hr default / 6hr low-res w/ 2M | ~$0.015/min |
| Twelve Labs | Marengo 3.0 / Pegasus 1.2 | Multi-vector dense embed (joint visual+audio+ASR+motion) per ~6s chunk | multi-hour | ~$0.07/1M input tokens (Bedrock) |
| OpenAI | GPT-4o / GPT-4.1 | No native video; cookbook: client-side ~1 fps frame extract | depends | per image token |
| OpenAI | Sora 2 | Generation only, 10-25s clips | n/a (gen) | n/a |
| Anthropic | Claude | **No video** | n/a | n/a |
| Meta | V-JEPA 2 (1B+) | Tubelet (2 frames × 16×16) + 3D-RoPE; train: 16 frames stride 4 | n/a (pretraining) | n/a |
| Meta | Movie Gen | TAE 8× spatial / 8× temporal compress, DiT | gen | n/a |

### Open-source defaults (all uniform)
| Model | Default sampling | Max frames |
|---|---|---|
| Qwen2.5-VL / Qwen3-VL | 2 fps | 768 (`FPS_MAX_FRAMES`) |
| InternVL2.5 / 3 | Uniform | 16–32 |
| LLaVA-Video-7B/72B | Uniform when >cap | 64 |
| VILA-1.5 / NVILA | Uniform | 64 (16 in ctx) |
| VideoChat-Flash | 1 fps | 768 |

### Retrieval platforms
- **Mixpeek** — scene-detection-based chunking (NOT uniform); `vuse-generic-v1` + transcript + face + OCR
- **Voxel51 FiftyOne** — ffmpeg-based per-frame; integrates Twelve Labs / Pinecone
- **Pinecone + CLIP** — user-defined fps, CLIP/SigLIP per-frame upsert

### Confirmed: research/production gap
Every shipping inference-time video-LLM API defaults to uniform/fixed-fps. Research SOTA (AKS, BOLT, GenS, Frame-Voyager, Q-Frame, AdaRD-Key, FOCUS) shows up to ~93% frame reduction at equal accuracy, but **none default in any commercial API**. Production exceptions are *retrieval* systems (Mixpeek shot/scene, Twelve Labs dense embed) because their cost amortizes at index time, not inference time.

## 3. Architecture Patterns

### Canonical pipeline
```
Input (URL/blob) → Decode (Decord/PyAV/NVDEC) → Sample → Encode (CLIP/SigLIP/ViT) → Token-Reduce → LLM
                                                  ↑ swap point
```

### Standard interfaces
- `decord.VideoReader(path) -> ndarray[T,H,W,3]`
- `sample(video, query?, max_frames) -> List[PIL.Image] | indices`
- `encode(frames) -> Tensor[N, D_v]`
- `LLM.generate(text_tokens, vision_tokens) -> str`

### Hybrid patterns
- **PySceneDetect + adaptive resample** — shot boundaries as anchors, one per shot then uniform top-up
- **CLIP query-aware filter** — SigLIP/CLIP score per frame, top-K (used inside AKS, BOLT)
- **LLM-judged frames** — caption every Nth frame, LLM picks indices
- **Agentic VoT loops** — model emits `request_frames(timestamp_range)` (FrameMind, FrameThinker)

### Production system patterns
- **Disaggregated pipeline (vLLM-Omni)** — encoder, sampler, LLM core as independent stages with per-stage batching; 91.4% JCT reduction reported
- **Frame-feature cache** — per-frame CLIP embeddings stored once (Twelve Labs Embed API), queries become cheap re-ranks
- **GPU pool sharing** — sampler co-locates with encoder; LLM on separate pool (NeMo Curator)
- **Online vs offline** — offline indexes embeddings to vector DB; online does decode→sample→encode on hot path with cache lookups

### Anti-patterns
1. Uniform 8-frame on 1-hour video (BOLT baseline failure)
2. Sampling before shot detection — breaks temporal coherence
3. Selection inside LLM pooling when scene evidence was discarded upstream
4. Fixed token budget independent of video length
5. Re-decoding per query instead of caching features
6. Sampler coupled to specific MLLM tokenizer (breaks plug-and-play)

### Reference modular architecture
```
[Ingest Queue] → [DecodeWorker (Decord+NVDEC)] → FrameStore (S3+Redis)
                                                       │
                  ┌── [SceneSegmenter: PySceneDetect] ─┤
                  ├── [Sampler Plugin]: AKS | BOLT | Frame-Voyager | M-LLM | Agentic
                  ├── [Encoder Pool: SigLIP/Marengo] → EmbedCache
                  ├── [TokenReducer: SlowFast / TokenMerge]
                  └── [LLM Serving: vLLM-Omni graph stage]
```

### Frameworks
- **lmms-eval** — 100+ tasks, `video_decode_backend=decord`, sampler is config field
- **lmms-finetune** — training harness for LLaVA-Next-Video, Qwen2-VL
- **vLLM-Omni** — production serving graph
- **NeMo Video Foundation Maker** — curation + training + inference
- **Twelve Labs Marengo SDK** — managed embed-then-query
