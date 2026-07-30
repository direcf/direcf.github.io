# direcf.github.io

> direct fun, deep study — AI 시대를 살아가며 공부한 것들과 가끔의 사적인 단상.

**Live**: https://direcf.github.io/
**Stack**: Astro 5 (SSG) · GitHub Actions · GitHub Pages
**Author**: Seung Jun Lee

---

## 📚 전체 코스 (Course Index)

AI·시스템·로보틱스 심화 학습 코스 모음. 각 코스는 10~15개 챕터로, 대표 논문·개념 다이어그램·실행 코드와 함께 하나의 주제를 관통합니다. 전체 목록은 [direcf.github.io](https://direcf.github.io/) 에서도 볼 수 있습니다.

#### 💻 Computer Science

- [AI 엔지니어를 위한 백엔드 & Ops](https://direcf.github.io/posts/backend-ops/)
- [AI 엔지니어를 위한 멀티모달 데이터 시스템](https://direcf.github.io/posts/multimodal-data-systems/)
- [Video Codecs — H.264, H.265, AV1, and Beyond](https://direcf.github.io/posts/video-codec/)
- [System Architecture & Network Protocols](https://direcf.github.io/posts/system-architecture/)

#### 🧠 Engineering Philosophy

- [Engineering Philosophy in the AI Agent Era](https://direcf.github.io/posts/engineering-philosophy/)

#### ☁️ Cloud / Infra

- [AWS 기초 완전 정복 — S3·EC2·VPC·Lambda부터 실전 아키텍처까지](https://direcf.github.io/posts/aws-fundamentals/)
- [AWS 심화 완전 정복 — ECS·SageMaker·EventBridge·Kinesis부터 FinOps까지](https://direcf.github.io/posts/aws-advanced/)

#### 👁️ Computer Vision

- [에고-엑소 시점 일관성 — 두 시점을 하나로 잇는 연구의 흐름 (대표 논문 10편)](https://direcf.github.io/posts/ego-exo-view-consistency/)
- [Ego-Exo 연구 — 크로스뷰 학습부터 논문 작성까지](https://direcf.github.io/posts/ego-exo/)

#### 🎨 Multimodal AI

- [Spatial VLM의 진화 — 대표 논문 10편으로 읽는 공간추론 연구의 흐름](https://direcf.github.io/posts/spatial-vlm-evolution/)
- [실시간 VLM 비용 최적화 — 단계별 이벤트 탐지 논문 10편](https://direcf.github.io/posts/realtime-vlm-cost-optimization/)
- [LLM & VLM Post-Training 완전 정복 — SFT부터 최신 RL까지](https://direcf.github.io/posts/llm-vlm-post-training/)
- [Pretrained to Imagine, Fine-Tuned to Act: World-Action Models](https://direcf.github.io/posts/world-action-models/)
- [Temporal Grounding for Video VLMs (2026)](https://direcf.github.io/posts/temporal-grounding-vlm/)
- [Real-time Video LLM](https://direcf.github.io/posts/real-time-video-llm/)
- [World Models & JEPA — LeCun's Path Beyond Generative AI](https://direcf.github.io/posts/jepa-world-models/)
- [Frame Sampling for Multimodal AI](https://direcf.github.io/posts/frame-sampling-multimodal-ai/)

#### ⚙️ AI Engineering

- [Loop Engineering — 프레임워크에서 루프, 그리고 그래프로 (대표 논문 13편)](https://direcf.github.io/posts/loop-engineering/)
- [빅테크 System Design 완전 정복 — 분산 시스템부터 LLM 인프라까지](https://direcf.github.io/posts/system-design-interview/)
- [Claude Code 완전 정복 — CLAUDE.md부터 Multi-agent까지](https://direcf.github.io/posts/claude-code-mastery/)
- [Claude Code 스킬 생태계 & 전문가 워크플로 완전 가이드](https://direcf.github.io/posts/claude-code-skills-guide/)
- [How AI Agents Reshape Knowledge Work](https://direcf.github.io/posts/ai-agents-knowledge-work/)
- [Harness Engineering — Claude Code 하네스 완전 가이드](https://direcf.github.io/posts/harness-engineering/)

#### 🤖 Physical AI

- [COMPASS & Cross-Embodiment Mobility — 하나의 정책으로 모든 로봇을 움직이다](https://direcf.github.io/posts/compass-cross-embodiment/)
- [NVIDIA Physical AI Map — Omniverse·Cosmos·Isaac·GR00T 전체 지도](https://direcf.github.io/posts/nvidia-physical-ai-map/)
- [GR00T & NVIDIA Physical AI 2026 — GR00T는 데이터로 navigation과 manipulation을 다 먹는가](https://direcf.github.io/posts/groot-loco-manipulation/)

---

## 1. 이 사이트가 뭐인가

콘텐츠 중심의 정적 사이트. 두 종류의 글이 있다:

- **Study** — Computer Science · Mathematics for AI · Engineering Philosophy · Machine Learning · Computer Vision · Multimodal AI · AI Engineering. 코스 형태의 깊은 학습 글.
- **Personal** — Diary. 짧은 단상·회고.

전체 8개 카테고리. 좌측 사이드바에서 영역별로 탐색.

---

## 2. 폴더 구조 한눈에

```
direcf.github.io/
├── package.json                ← Astro 의존성
├── astro.config.mjs            ← format: 'directory' (URL이 /foo/ 형태)
├── .gitignore                  ← dist/, node_modules/, .astro/
├── .github/workflows/
│   └── deploy.yml              ← push → build → Pages 자동 배포
│
├── src/                        ← Astro 소스 (편집 대상)
│   ├── layouts/
│   │   └── BaseLayout.astro    ← 모든 페이지의 공통 셸 (사이드바·헤더·OG meta)
│   ├── pages/                  ← 파일 경로 = URL
│   │   ├── index.astro                    → /
│   │   ├── category/[slug].astro          → /category/diary/, /category/computer-science/, …
│   │   └── posts/[course]/
│   │       ├── index.astro                → /posts/engineering-philosophy/  (syllabus)
│   │       └── [chapter].astro            → /posts/engineering-philosophy/chapter-01/
│   ├── data/
│   │   ├── posts.json                     ← 카테고리 정의 + 모든 글의 메타데이터
│   │   └── courses/
│   │       ├── engineering-philosophy.json  ← 코스 전체 내용 (10챕터)
│   │       └── system-architecture.json
│   ├── lib/
│   │   └── courses.ts          ← md(), paragraphs() — 마크다운→HTML 헬퍼
│   └── styles/
│       └── global.css          ← frost blue 디자인 시스템 (574줄)
│
├── public/                     ← 정적 자산 (그대로 root에 복사됨)
│   ├── assets/                 ← 옛 Jekyll 이미지들 (보존용)
│   └── posts/
│       ├── diary/              ← 일기 정적 HTML 3편 (아직 Astro 통합 전)
│       ├── engineering-philosophy/
│       │   ├── cheatsheets/    ← 챕터별 PNG 치트시트 (Nano Banana 생성)
│       │   └── assets/
│       └── system-architecture/
│           └── cheatsheets/
│
├── _legacy_jekyll/             ← 옛 Jekyll 테마 스냅샷 (보존, 빌드엔 영향 X)
└── README.md                   ← 이 파일
```

---

## 3. 왜 Astro인가

### 결론
**콘텐츠 사이트는 SSG(Static Site Generator)가 정답이고, Astro가 그 중 sweet spot.**

### 이전 시도와 학습
| 단계 | 사용 도구 | 문제 |
|---|---|---|
| 1차 (옛날) | Jekyll (minimal-mistakes theme) | 디자인 자유도 낮음, 빌드 느림 |
| 2차 | 수동 SPA (vanilla JS + fetch posts.json) | URL이 `#/category/diary` 형식 → SEO 망함, 카카오·트위터 공유 미리보기 X |
| 3차 (지금) | **Astro 5 (SSG)** | 정적 HTML로 빌드 → 모든 페이지가 진짜 URL, OG 자동 주입, SEO 정상화 |

### SPA를 버린 결정적 이유
- Google 크롤러는 첫 응답만 본다. SPA는 첫 응답이 `<div id="main"></div>` — 빈 껍데기.
- 카카오톡·트위터·링크드인은 OG 태그를 본다. SPA는 모든 페이지가 같은 OG 태그라 공유 미리보기가 다 똑같음.
- Astro는 빌드 타임에 HTML을 완성한다. 위 두 문제가 자동으로 풀림.

### 트레이드오프
Astro 학습 비용은 들지만, 1) 콘텐츠가 많이 쌓이기 전에 마이그레이션한 게 가장 싼 타이밍이었고, 2) 작은 인터랙션(사이드바 토글·캐러셀)은 Astro의 islands로 그대로 가능.

---

## 4. 카테고리 8개의 의미

| Slug | 카테고리 | 의도 |
|---|---|---|
| `computer-science` | 💻 Computer Science | 기초 시스템·네트워크·자료구조 |
| `mathematics-for-ai` | ➗ Mathematics for AI | 선형대수·확률·미분 — AI의 언어 |
| `engineering-philosophy` | 🧠 Engineering Philosophy | AI 시대 엔지니어가 갖춰야 할 사고법 |
| `machine-learning` | 📈 Machine Learning | 딥러닝 기초·학습·평가 |
| `computer-vision` | 👁️ Computer Vision | CNN·detection·segmentation |
| `multimodal-ai` | 🎨 Multimodal AI | CLIP·VLM·text↔image |
| `ai-engineering` | ⚙️ AI Engineering | 에이전트·추론·MLOps |
| `diary` | ✍️ Diary | 기록과 회고, 일상의 단상 |

이 구조는 "기초 → 사고법 → 모델 → 응용 → 시스템" 의 학습 흐름을 담음. 카테고리는 `src/data/posts.json`의 `categories` 배열에서 정의.

---

## 5. 콘텐츠 추가하는 법

### 5-1. 새 코스 추가 (10챕터 학습 자료)

`coursework-creator` 스킬을 사용. Claude Code에서 트리거:

```
"~에 대한 코스 만들어줘"
"~에 대해 공부하고 싶어"
```

작업 흐름:
1. Claude가 `course_data.json` 작성 (10챕터의 overview·sections·analogy·코드·평가·takeaways)
2. `generate_cheatsheet.py` — Nano Banana Pro로 챕터별 PNG 치트시트 생성
3. `build_html.py` — 정적 HTML 생성 (이전 방식, 이제는 참고용)

**Astro 사이트에 통합하려면 추가 작업:**
1. 생성된 `course_data.json` → `src/data/courses/<slug>.json` 복사
2. `src/lib/courses.ts`의 `COURSES` 객체에 import 추가
3. `cheatsheets/` 폴더 → `public/posts/<slug>/cheatsheets/` 로 복사
4. `src/data/posts.json`의 `posts` 배열에 코스 entry 추가:
   ```json
   {
     "title": "코스 제목",
     "category": "machine-learning",
     "date": "2026-XX-XX",
     "type": "course",
     "chapters": 10,
     "excerpt": "한 문장 설명",
     "url": "/posts/<slug>/",
     "cover": "/posts/<slug>/cheatsheets/chapter-01.png"
   }
   ```
5. `course_data.json`에 `"categorySlug": "<slug>"` 추가하면 사이드바에서 해당 카테고리 자동 하이라이트

코스 라우팅은 `src/pages/posts/[course]/[chapter].astro`가 자동으로 모든 코스의 모든 챕터를 생성. `COURSES`에 추가만 하면 끝.

### 5-2. 새 일기 추가 (현재 정적 HTML 방식)

지금은 임시로 정적 HTML로 둠 (`public/posts/diary/YYYY-MM-DD-slug.html`).
- `src/data/posts.json`의 `posts`에 entry 추가 (type: `"diary"`)
- HTML 직접 작성하거나 기존 일기 참고

**다음 단계 (TODO):** 일기를 Astro markdown 콘텐츠로 옮기면 작성이 훨씬 편해짐. 자세한 건 §9 로드맵 참조.

### 5-3. 새 카테고리 추가

`src/data/posts.json`의 `categories` 배열에 항목 추가:
```json
{ "slug": "robotics", "name": "Robotics", "icon": "🤖", "tagline": "..." }
```

`BaseLayout.astro`의 `studySlugs` 배열에 slug 추가 (Personal로 분류하려면 그쪽). 새 카테고리 페이지는 `category/[slug].astro`가 자동 생성.

---

## 6. 로컬 개발

### 필요 환경
- Node 20 이상 (현재 24.14 사용 중, nvm)

### 명령어
```bash
cd ~/Desktop/direcf.github.io

# 의존성 설치 (처음 한 번)
npm install

# 개발 서버 — 변경사항 hot reload
npm run dev
# → http://127.0.0.1:4321/

# 프로덕션 빌드 (dist/ 생성)
npm run build

# 빌드 결과 미리보기
npm run preview
```

### dev 워크플로
1. `src/` 안에서 컴포넌트·페이지 수정
2. 브라우저에서 즉시 확인
3. 만족하면 `git commit && git push`
4. 끝 — Actions가 알아서 build + deploy

---

## 7. 배포 흐름

`.github/workflows/deploy.yml` 가 다음을 처리:

```
git push  →  Actions checkout  →  npm ci  →  npm run build  →  dist/  →  Pages
```

push 후 약 1-2분이면 라이브. 빌드 상태는 https://github.com/direcf/direcf.github.io/actions 에서 확인.

**중요**: dist/ 폴더는 절대 commit하지 않음 (`.gitignore`). Actions가 매번 새로 빌드함. 로컬에서 `npm run build` 한 결과를 push하는 게 아니라 source만 push하면 끝.

### 만약 Pages 설정이 풀려 있다면
GitHub repo → Settings → Pages → **Source: GitHub Actions** 로 설정. 한 번만 하면 됨.

---

## 8. 디자인 시스템 — Frost Blue

`src/styles/global.css`의 `:root`에 정의된 CSS 변수가 전체 톤을 결정.

### 핵심 변수
```css
--bg: #eef4fb;          /* 얼음빛 연파랑 — 배경 */
--bg-card: #ffffff;     /* 카드 흰색 */
--bg-soft: #dde8f3;     /* 소프트 frost 패널 */
--text: #0f1f3a;        /* 딥 네이비 */
--accent: #1d4ed8;      /* 코발트 블루 — 1차 액센트 */
--accent-2: #0891b2;    /* 시안 — 2차 액센트 */
--accent-diary: #7c3aed;/* 보라 — Diary만 다른 톤 (의도적) */
```

### 컴포넌트 단위
모두 같은 셸:
- **App shell**: 좌측 사이드바 + 우측 main (collapsed 토글로 너비 248px ↔ 64px)
- **Sidebar**: 카테고리 트리, localStorage로 접힘 상태 유지 (`direcf:sb-collapsed`)
- **Header**: 브레드크럼 + GitHub 링크
- **Home**: hero(최신 3개) + carousel(그 외)
- **Category**: 글 리스트 또는 "Coming soon" 빈 상태
- **Chapter**: hero → cheatsheet → goals → sections → analogy → code → industry eval → takeaways → chapter nav
- **Syllabus**: 코스 메타 + 10챕터 인덱스

### 마크다운 처리
`src/lib/courses.ts` 의 `md()` 함수가 처리:
- `**bold**` → `<strong>bold</strong>`
- `` `code` `` → `<code>code</code>`

그 외 마크다운(헤더·리스트·링크)은 처리 X. JSON 콘텐츠 작성 시 위 두 가지만 사용.

---

## 9. 알려진 한계 / 로드맵

### 지금 안 되는 것
- [x] ~~**일기 페이지가 정적 HTML 그대로**~~ — Astro 통합 완료. 일기는 `src/data/diary.json` 한 곳에서 작성, `/diary/<slug>/`로 라우팅
- [x] ~~**검색 기능 없음**~~ — Cmd+K 모달 + Fuse.js. 모든 글·챕터·일기가 인덱싱됨
- [x] ~~**다크모드 없음**~~ — 헤더 ☀️/🌙 토글, `prefers-color-scheme` 자동 감지 + localStorage 저장
- [ ] **챕터 읽기 진도 추적 없음** — localStorage로 챕터별 완독 표시 가능
- [ ] **모바일 사이드바 = 숨김** — 햄버거 메뉴 또는 하단 탭바 필요
- [ ] **OG default 이미지 없음** — `/og-default.png` 자리는 있는데 파일 없음
- [ ] **RSS 피드 없음**
- [ ] **댓글 시스템 없음** — Disqus는 기존에 있었지만 제거

### 다음 단계 후보 (우선순위 순)
1. **일기 Astro 통합** — markdown으로 옮기고 BaseLayout 사용 → 디자인 일관성
2. **다크모드 토글** — `prefers-color-scheme` 감지 + localStorage
3. **Cmd+K 검색** — Fuse.js로 posts.json + 챕터 인덱싱
4. **OG default 이미지 생성** — 빈 메타 채우기
5. **모바일 사이드바** — 햄버거 또는 iOS 스타일 탭바
6. **챕터 진도 추적** — 카테고리 카드에 N/10 표시
7. **RSS 피드** — Astro 공식 integration 사용

---

## 10. 작업 히스토리 (메이저 결정)

| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-06-11 | Astro 5로 마이그레이션 (commit `2fbf25e`) | SEO·OG 정상화, 한 리포로 통합 |
| 2026-06-11 | GitHub Actions 자동 배포 셋업 | source push만으로 deploy |
| 2026-06-11 | Diary 카테고리 분리, 카테고리 8개 구조 확정 | "Study + Personal" 두 결 분리 |
| 2026-06-11 | Frost blue 디자인 시스템 채택 | 차분한 톤, Diary만 보라로 따뜻함 |
| 2026-06-11 | 챕터 페이지 Astro 라우팅 (`[course]/[chapter].astro`) | 코스 챕터마다 OG 이미지 자동 |
| 2026-06-05 | System Architecture & Network Protocols 코스 작성 | Computer Science 카테고리 첫 글 |
| 2026-06-11 | Engineering Philosophy in the AI Agent Era 코스 작성 | Engineering Philosophy 카테고리 첫 글 |
| 2021-01-26~28 | 초기 일기 3편 | Diary 카테고리 (옛 콘텐츠 보존) |

---

## 11. 외부 도구

### Claude Code 스킬
- **coursework-creator**: 10챕터 코스 자동 작성. 트리거 표현 "~ 코스 만들어줘".
- **nanobanana**: Gemini Nano Banana Pro로 한글 가독성 좋은 치트시트 이미지 생성.

### Gemini API
- 모델: `gemini-3-pro-image-preview` (한글 텍스트 가독성 우수, 유료 tier 필요)
- 키: `~/.claude/.env`의 `GEMINI_API_KEY`

---

## 12. 기여 / 연락

이 사이트는 개인 학습 기록이라 외부 PR을 받진 않지만, 디자인·콘텐츠 피드백은 GitHub Issues 또는 [github.com/direcf](https://github.com/direcf) 로 환영.
