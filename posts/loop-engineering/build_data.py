import json

data = {
    "topic": "Loop Engineering",
    "topicKr": "루프 엔지니어링",
    "topicSlug": "loop-engineering",
    "level": "advanced",
    "codeLanguage": "Python",
    "categorySlug": "ai-engineering",
    "references": [
        "2210.03629",  # ReAct
        "2303.11366",  # Reflexion
        "2303.17651",  # Self-Refine
        "2305.16291",  # Voyager
        "2304.03442",  # Generative Agents
        "2305.10601",  # Tree of Thoughts
        "2305.18323",  # ReWOO
        "2312.04511",  # LLMCompiler
        "2310.03714",  # DSPy
        "2308.09687",  # Graph of Thoughts
    ],
    "description": "AutoGPT식 monolithic 에이전트 프레임워크에서 '에이전트는 결국 루프다'라는 harness 관점으로, 다시 LangGraph식 명시적 그래프 오케스트레이션으로 넘어가는 흐름을 대표 논문·에세이 13편으로 추적한다. framework → loop → graph 진화사를 통해 2026년 프로덕션 에이전트를 실제로 어떻게 짜는지 배운다.",
    "chapters": [],
}

chapters = []

# ────────────────────────────────────────────────────────────────
# Chapter 1 — 패러다임 전환 (전체 지도, 최대한 쉽게)
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 1,
    "emoji": "🗺️",
    "title": "The Paradigm Shift: Framework → Loop → Graph",
    "titleKr": "패러다임 전환 — 프레임워크에서 루프, 그리고 그래프로",
    "tldr": "AI 에이전트를 '만드는 법' 자체가 3단계로 진화했다. 무거운 프레임워크에 기대던 시대에서, '에이전트는 결국 while 루프 하나다'라는 깨달음으로, 다시 그 루프를 명시적인 그래프로 그리는 시대로. 이 장은 나머지 9장 전체의 지도다.",
    "topics": ["framework/loop/graph 3단계", "harness engineering", "context engineering", "workflow vs agent"],
    "learningGoals": [
        "AI 에이전트 구축 담론이 왜 framework → loop → graph 순으로 이동했는지 한 문장으로 설명할 수 있다",
        "'루프 엔지니어링'과 '그래프 엔지니어링'과 '컨텍스트 엔지니어링'이 각각 무엇을 가리키는지 구분한다",
        "workflow(정해진 경로)와 agent(스스로 경로를 정함)의 차이를 안다",
        "AutoGPT식 프레임워크가 왜 한물갔는지, 무엇이 그 자리를 대체했는지 설명한다",
        "이 코스의 10개 챕터가 이 지도의 어디에 놓이는지 파악한다",
    ],
    "overview": (
        "2023년 초, AI 에이전트를 만들고 싶으면 사람들은 먼저 **프레임워크**를 골랐다. AutoGPT를 깔거나, LangChain의 두꺼운 Agent 클래스를 상속받거나, BabyAGI 코드를 포크했다. 마치 '에이전트'라는 게 대단히 복잡한 소프트웨어라서, 남이 만든 거대한 골격 위에 올라타야만 만들 수 있는 것처럼 느껴졌다.\n\n"
        "그런데 2024년을 지나며 업계는 정반대의 사실을 깨달았다. 에이전트의 본질은 놀랍도록 단순했다. **LLM에게 도구를 쥐여주고, 결과를 다시 보여주고, 이걸 while 루프로 반복**하는 것. 그게 전부였다. 거대한 프레임워크는 이 단순한 진실을 두꺼운 추상화 아래 감추고 있었을 뿐이다. 관심의 초점은 '어떤 프레임워크를 쓸까'에서 '이 루프를 어떻게 잘 돌릴까'로 옮겨갔다. 이것이 **루프 엔지니어링(loop engineering)** 의 등장이다.\n\n"
        "하지만 이야기는 여기서 끝나지 않는다. 하나의 단순한 루프로 모든 걸 처리하려니 한계가 보였다. 복잡한 작업은 병렬로 나눠야 하고, 단계마다 다른 전문가(다른 프롬프트·다른 모델)가 필요하고, 실패하면 특정 지점으로 되돌아가야 했다. 그래서 사람들은 그 루프를 **명시적인 그래프**로 그리기 시작했다 — 노드는 작업 단계, 엣지는 흐름의 방향. 이것이 **그래프 엔지니어링(graph engineering)** 이다. 이 장에서는 이 세 단계의 큰 그림을 먼저 머릿속에 심고, 나머지 챕터가 각각 어디에 해당하는지를 지도 위에 찍어본다."
    ),
    "sections": [
        {
            "title": "세 단계를 한눈에 — 왜 이 순서였나",
            "content": (
                "먼저 세 단어를 확실히 구분하자. 이 코스 전체가 이 세 단어 위에 서 있다.\n\n"
                "**프레임워크 엔지니어링(Framework era)** = 남이 만든 거대한 에이전트 골격(AutoGPT, 초기 LangChain Agent)을 가져다 쓰는 방식. 내가 하는 일은 '설정'에 가깝다. 골격이 알아서 생각하고 도구를 쓴다고 믿는다.\n\n"
                "**루프 엔지니어링(Loop era)** = 프레임워크를 걷어내고, 에이전트의 심장인 `while` 루프를 내 손으로 직접 짜는 방식. '모델 호출 → 도구 실행 → 결과를 다시 넣기'를 언제 멈추고, 무엇을 다시 넣을지를 내가 통제한다.\n\n"
                "**그래프 엔지니어링(Graph era)** = 그 루프가 커지면, 흐름을 노드와 엣지로 이루어진 명시적 그래프로 그리는 방식. 어떤 단계가 병렬로 돌고, 어디서 분기하고, 실패 시 어디로 되돌아가는지를 코드가 아니라 '그래프 구조'로 표현한다.\n\n"
                "이 순서는 우연이 아니다. **추상화가 너무 높아서(프레임워크) → 너무 낮아졌다가(맨손 루프) → 딱 맞는 높이(구조화된 그래프)로 수렴**하는, 소프트웨어 역사에서 반복되는 진자 운동이다. 우리는 프레임워크가 감춘 것을 루프에서 다시 배웠고, 루프가 감당 못 하는 것을 그래프에서 구조로 되찾는다."
            ),
        },
        {
            "title": "핵심 분기점: workflow인가 agent인가",
            "content": (
                "이 지도를 이해하는 데 가장 중요한 개념 하나가 Anthropic의 [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)(2024, 6장에서 정독)에서 나온다. 바로 **workflow와 agent의 구분**이다.\n\n"
                "| 구분 | Workflow | Agent |\n|---|---|---|\n| 경로 결정 | 사람이 코드로 미리 정함 | LLM이 실행 중에 스스로 정함 |\n| 예측 가능성 | 높음 (같은 입력 → 같은 경로) | 낮음 (매번 다를 수 있음) |\n| 비유 | 기차 (정해진 레일) | 택시 (기사가 길을 고름) |\n| 언제 쓰나 | 단계가 뻔한 작업 | 단계를 미리 알 수 없는 작업 |\n\n"
                "> *\"Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage.\"*\n> — Anthropic, Building Effective Agents\n\n"
                "이 구분이 왜 지도의 핵심이냐면, **그래프 엔지니어링은 사실 이 둘을 한 그림 안에서 섞는 기술**이기 때문이다. 큰 뼈대는 예측 가능한 workflow(그래프의 고정된 엣지)로 잡고, 그 안의 특정 노드만 자율적인 agent 루프에게 맡긴다. 순수한 자율 에이전트(모든 게 LLM 마음대로)와 순수한 워크플로우(모든 게 코드로 고정) 사이의 넓은 스펙트럼 — 그 스펙트럼을 다루는 게 이 코스의 후반부다."
            ),
        },
        {
            "title": "루프의 짝꿍: 컨텍스트 엔지니어링",
            "content": (
                "세 단계 위를 관통해서 흐르는 또 하나의 개념이 있다. **컨텍스트 엔지니어링(context engineering)** 이다. 이건 framework/loop/graph와 나란한 '4번째 단계'가 아니라, 루프 시대가 열리면서 자연스럽게 부상한 **운영 규율**이다.\n\n"
                "예전엔 '프롬프트를 어떻게 잘 쓸까'(prompt engineering)가 관심사였다. 프롬프트 한 방으로 끝나던 시절 얘기다. 그런데 에이전트가 루프를 돌면 매 턴마다 컨텍스트 창(context window)에 무엇을 넣을지가 계속 바뀐다 — 이전 대화, 도구 결과, 검색된 문서, 시스템 지침. 이걸 매 턴 **큐레이션**하는 게 진짜 실력이 되었다.\n\n"
                "> *\"Context engineering is the delicate art and science of filling the context window with just the right information for the next step.\"*\n> — Andrej Karpathy (2025)\n\n"
                "Anthropic은 이를 더 정확히 정의한다: *\"the set of strategies for curating and maintaining the optimal set of tokens during LLM inference.\"* 프롬프트 엔지니어링의 자연스러운 후계자이며, **루프가 돌면서 쌓이는 정보를 주기적으로 정제(refine)** 하는 기술이다. 7장에서 이 주제만 따로 깊게 판다. 지금은 '루프 엔지니어링의 쌍둥이 규율' 정도로만 기억하면 된다."
            ),
        },
        {
            "title": "이 코스의 지도 — 어느 챕터가 어디에",
            "content": (
                "이제 10개 챕터를 지도 위에 찍어보자. 각 챕터는 그 단계를 대표하는 논문 또는 에세이 하나(때론 둘)를 앵커로 삼는다.\n\n"
                "- **1장 (지금)**: 전체 지도.\n"
                "- **2장 ReAct**: 루프의 유전자. reason → act → observe. framework에서 loop로 넘어가는 바로 그 기점.\n"
                "- **3장 Reflexion + Self-Refine**: 루프에 '자기반성'을 넣다. 실패를 언어로 기록해 다음 시도를 개선.\n"
                "- **4장 Voyager**: 루프에 '평생 기억'을 넣다. 성공한 코드를 skill library에 쌓아 재사용.\n"
                "- **5장 Generative Agents**: 기억·성찰·계획을 재사용 가능한 primitive로 정립.\n"
                "- **6장 Building Effective Agents**: '에이전트 = 루프'라고 못 박은 결정적 피벗. framework → loop의 선언문.\n"
                "- **7장 Context Engineering**: 루프를 돌리는 진짜 기술 — 매 턴 컨텍스트 큐레이션.\n"
                "- **8장 Tree of Thoughts (+ Graph of Thoughts)**: 선형 루프에서 탐색(트리·그래프)으로. loop → graph의 다리.\n"
                "- **9장 ReWOO + LLMCompiler**: 미리 계획하고 병렬 DAG로 실행. framework → graph의 성능 논거.\n"
                "- **10장 DSPy + LangGraph**: 그래프 엔지니어링의 종착점 — 최적화 가능한 파이프라인과 stateful 그래프 런타임.\n\n"
                "왼쪽(2~5장)은 loop를 **풍부하게** 만드는 이야기(반성·기억·계획), 오른쪽(8~10장)은 loop를 **구조화**하는 이야기(탐색·병렬·그래프)다. 그 한가운데 6·7장이 '왜 루프가 primitive인가'를 못 박는 축으로 서 있다. 이 큰 그림을 잡고 나면, 각 논문이 왜 그 자리에 있는지가 선명하게 보일 것이다."
            ),
        },
    ],
    "analogy": {
        "title": "요리 프랜차이즈 vs 셰프 vs 오픈 키친",
        "content": (
            "**프레임워크 시대**는 냉동 밀키트 프랜차이즈다. 박스를 뜯으면 모든 게 들어 있고, 매뉴얼대로 데우기만 하면 요리가 나온다. 편하지만, 소금을 언제 넣는지·불을 얼마나 올리는지 나는 전혀 모른다. 맛이 이상해도 어디를 고쳐야 할지 알 수가 없다. AutoGPT가 딱 이랬다 — 돌아가긴 하는데, 왜 그렇게 도는지 아무도 몰랐다.\n\n"
            "**루프 시대**는 맨손으로 요리를 배운 셰프다. 재료를 직접 썰고, 간을 보고, 불을 조절한다. '요리란 결국 재료 → 가열 → 간보기의 반복'이라는 본질을 손으로 안다. 프랜차이즈보다 손이 많이 가지만, 무엇이든 만들 수 있고 어디가 잘못됐는지 정확히 안다. 'LLM 호출 → 도구 → 결과 반영'의 while 루프를 내 손으로 짜는 게 이것이다.\n\n"
            "**그래프 시대**는 오픈 키친을 운영하는 셰프다. 이제 혼자가 아니라 여러 스테이션(전채·메인·디저트)을 동시에 돌린다. 어떤 요리는 병렬로, 어떤 건 순서대로, 실패하면 특정 스테이션만 다시. 주방 전체의 '흐름도(그래프)'를 설계하는 일이 요리 실력만큼 중요해진다. LangGraph가 바로 이 주방 흐름도다. 핵심은 — 오픈 키친을 운영하려면 **먼저 맨손 셰프의 감각(루프)** 이 있어야 한다는 것. 그래서 이 코스는 루프부터 시작한다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "말보다 코드가 빠르다. 아래는 '에이전트의 본질'을 15줄로 압축한 것이다. 어떤 프레임워크도 없이, 순수 파이썬 while 루프 하나가 곧 에이전트라는 걸 보여준다. 이 골격을 머릿속에 박아두면 나머지 9장이 전부 '이 루프에 무엇을 더하고 어떻게 구조화하느냐'의 변주로 읽힌다."
        ),
        "code": (
            "def agent_loop(task, tools, llm, max_steps=10):\n"
            "    # 컨텍스트 = 매 턴 LLM에게 보여줄 모든 것 (7장의 주제)\n"
            "    context = [{\"role\": \"user\", \"content\": task}]\n"
            "\n"
            "    for step in range(max_steps):          # ← 이 for/while이 '에이전트'의 전부\n"
            "        reply = llm(context, tools=tools)  # 1) 생각하고 행동을 고름 (2장 ReAct)\n"
            "\n"
            "        if reply.tool_call is None:        # 2) 멈출 때를 스스로 판단\n"
            "            return reply.content           #    도구가 필요 없으면 = 최종 답\n"
            "\n"
            "        # 3) 환경과 상호작용 (act)\n"
            "        observation = tools[reply.tool_call.name](**reply.tool_call.args)\n"
            "\n"
            "        # 4) 결과를 컨텍스트에 되먹임 (observe) → 다음 루프로\n"
            "        context.append({\"role\": \"assistant\", \"content\": reply.raw})\n"
            "        context.append({\"role\": \"tool\", \"content\": observation})\n"
            "\n"
            "    return \"max_steps 도달 — 미완료\"   # 무한루프 방지는 loop engineering의 기본기\n"
        ),
        "walkthrough": (
            "핵심은 딱 세 가지다. **(1) 루프의 몸통**(`for step in range`)이 곧 에이전트다 — 이 골격을 프레임워크가 감추고 있었을 뿐이다. **(2) 멈춤 조건**(`tool_call is None`)을 누가 통제하느냐가 loop engineering의 절반이다. 여기서는 LLM이 '도구가 더 필요 없다'고 판단하면 멈추지만, 실무에선 예산·시간·반복 감지 등 여러 정지 조건을 겹겹이 건다. **(3) 되먹임**(`context.append`)이 나머지 절반이다 — 무엇을, 얼마나, 어떤 형태로 컨텍스트에 다시 넣느냐가 7장 컨텍스트 엔지니어링의 전부다. 이 15줄이 2~7장의 뼈대이고, 8~10장은 이 단일 루프를 트리·DAG·그래프로 펼치는 이야기다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "빅테크 AI 엔지니어링 면접에서 '에이전트를 설계해보라'는 문제가 나오면, 주니어는 곧장 'LangChain의 이 클래스를 쓰겠다'고 답하고, 시니어는 '먼저 이게 workflow로 충분한지 agent가 필요한지 판단하겠다'고 답한다. 이 첫 문장에서 레벨이 갈린다. 면접관은 지원자가 프레임워크의 소비자인지, 에이전트의 본질을 이해한 설계자인지를 본다."
        ),
        "whatEngineersLookFor": [
            "프레임워크를 언급하기 전에 '이 문제가 자율 agent가 필요한지, 고정된 workflow로 충분한지'를 먼저 따지는가",
            "에이전트를 'while 루프 + 도구 + 컨텍스트 관리'로 분해해서 설명할 수 있는가",
            "framework/loop/graph 중 이 문제에 맞는 추상화 수준을 근거를 들어 고를 수 있는가",
            "정지 조건(비용·반복·시간)과 실패 처리를 처음부터 설계에 넣는가",
        ],
        "redFlags": [
            "'에이전트 = AutoGPT/LangChain'처럼 특정 프레임워크와 개념을 동일시함",
            "모든 문제를 최대 자율 에이전트로 풀려 함 (workflow가 더 안전한 경우를 못 봄)",
            "루프의 정지 조건·비용 상한을 언급하지 않음 (프로덕션 경험 부재 신호)",
            "'그래프가 최신이니까 무조건 LangGraph'처럼 유행을 근거로 도구를 고름",
        ],
        "interviewQuestions": [
            "AutoGPT 같은 초기 자율 에이전트 프레임워크가 프로덕션에서 외면받은 이유는 무엇인가?",
            "주어진 작업을 workflow로 짤지 agent로 짤지 어떤 기준으로 판단하나?",
            "'에이전트는 결국 루프다'라는 말의 의미와, 그것이 설계에 주는 실질적 함의는?",
        ],
        "masteryVsFamiliar": (
            "**표면만 아는 사람**은 ReAct·LangGraph 같은 이름을 나열하고 각각이 뭔지 설명한다. **마스터한 사람**은 이 이름들을 framework→loop→graph라는 하나의 진화 서사 위에 배치하고, 주어진 문제에 대해 '여기는 단순 루프로 충분하고 이 노드만 그래프로 분리하겠다'처럼 추상화 수준을 능동적으로 선택한다. 즉, 도구를 아는 것과 '언제 어떤 추상화를 쓸지 판단하는 것'의 차이다."
        ),
    },
    "keyTakeaways": [
        {"title": "3단계 진화", "content": "에이전트 구축법은 framework(남의 골격) → loop(맨손 루프) → graph(구조화된 흐름)로 이동했다."},
        {"title": "에이전트 = 루프", "content": "에이전트의 본질은 'LLM 호출 → 도구 → 결과 되먹임'을 반복하는 while 루프 하나다."},
        {"title": "진자 운동", "content": "추상화가 너무 높았다가(프레임워크) → 너무 낮아졌다가(맨손) → 딱 맞는 높이(그래프)로 수렴하는 반복 패턴이다."},
        {"title": "workflow vs agent", "content": "경로를 코드가 정하면 workflow, LLM이 실행 중 정하면 agent. 그래프는 둘을 한 그림에서 섞는다."},
        {"title": "컨텍스트 엔지니어링", "content": "루프 시대의 쌍둥이 규율 — 매 턴 컨텍스트 창에 무엇을 넣을지 큐레이션하는 기술."},
        {"title": "왼쪽은 풍부화, 오른쪽은 구조화", "content": "2~5장은 루프에 반성·기억·계획을 더하고, 8~10장은 루프를 탐색·병렬·그래프로 구조화한다."},
        {"title": "추상화 선택이 실력", "content": "도구 이름을 아는 것보다 '이 문제에 맞는 추상화 수준을 고르는 판단'이 시니어의 조건이다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 2 — ReAct
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 2,
    "emoji": "🔁",
    "title": "ReAct: The Origin of the Agentic Loop",
    "titleKr": "ReAct — 에이전트 루프의 기원",
    "tldr": "생각(reason)과 행동(act)과 관찰(observe)을 한 루프 안에서 번갈아 돌리는 패턴. 오늘날 거의 모든 에이전트(Claude Code, Cursor, LangGraph)의 유전자가 여기서 나왔다. framework에서 loop로 넘어가는 바로 그 기점.",
    "topics": ["reason-act-observe 인터리빙", "도구 그라운딩", "CoT의 한계 극복", "환각·오류 전파 억제"],
    "learningGoals": [
        "ReAct의 Thought→Action→Observation 루프를 정확히 설명하고 직접 구현할 수 있다",
        "순수 Chain-of-Thought가 왜 환각과 오류 전파에 취약한지, ReAct가 이를 어떻게 막는지 안다",
        "'추론을 외부 도구에 그라운딩한다'는 말의 의미를 이해한다",
        "ReAct가 왜 loop engineering의 출발점으로 불리는지 설명한다",
        "ReAct 루프의 실패 모드(무한 반복, 잘못된 관찰 신뢰)를 진단할 수 있다",
    ],
    "overview": (
        "2022년 10월, Yao 등이 발표한 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)(ICLR 2023)는 오늘날 우리가 '에이전트'라고 부르는 거의 모든 것의 씨앗이다. 제목 그대로 **Reason(추론) + Act(행동)** 의 합성어다.\n\n"
        "그 전까지 LLM의 추론은 Chain-of-Thought(CoT), 즉 '머릿속으로 단계별로 생각하기'가 전부였다. 문제는 이 생각이 순전히 모델 머릿속에서만 일어난다는 것이다. 중간에 사실이 틀리면 그 오류가 눈덩이처럼 다음 단계로 굴러간다(오류 전파). 세상과 대조할 방법이 없으니 그럴싸한 거짓말(환각)을 지어내도 스스로 못 잡는다.\n\n"
        "ReAct의 통찰은 단순하면서 결정적이다. **생각만 하지 말고, 생각 사이사이에 실제 행동을 끼워 넣어 세상과 대조하라.** 위키피디아를 검색하고(Action), 그 결과를 읽고(Observation), 그걸 바탕으로 다시 생각한다(Thought). 이 Thought→Action→Observation의 반복이 바로 1장에서 본 while 루프의 원형이다. 이 장에서 우리는 에이전트 루프의 DNA를 해부한다."
    ),
    "sections": [
        {
            "title": "CoT의 병: 머릿속에만 갇힌 추론",
            "content": (
                "Chain-of-Thought는 강력했지만 치명적 약점이 있었다. 추론의 모든 단계가 **모델 파라미터라는 닫힌 세계 안**에서만 벌어진다는 것이다.\n\n"
                "여기서 두 가지 병이 생긴다. **환각(hallucination)** — 모델이 모르는 사실을 그럴싸하게 지어낸다. **오류 전파(error propagation)** — 중간 한 단계가 틀리면, 이후 모든 추론이 그 틀린 전제 위에 쌓여 통째로 무너진다.\n\n"
                "> *\"ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API.\"*\n> — Yao et al., 2022\n\n"
                "핵심 진단은 이렇다. CoT에는 **현실과 대조하는 순간(reality check)** 이 없다. 시험지에 계산 과정을 적되, 계산기를 한 번도 안 두드리고, 참고서를 한 번도 안 펼치는 학생과 같다. ReAct는 바로 이 '대조하는 순간'을 추론 루프 안에 심는다."
            ),
        },
        {
            "title": "해법: 생각과 행동을 번갈아 짜기(interleaving)",
            "content": (
                "ReAct의 한 스텝은 세 박자로 이루어진다.\n\n"
                "1. **Thought** — 지금 상황에서 무엇을 해야 하는지 언어로 추론한다. (\"파리의 인구를 알아야겠다\")\n"
                "2. **Action** — 그 추론에 따라 외부 도구를 호출한다. (`Search[\"Paris population\"]`)\n"
                "3. **Observation** — 도구가 돌려준 실제 결과를 받는다. (\"약 210만 명\")\n\n"
                "그리고 이 관찰을 컨텍스트에 넣은 채 다시 Thought로 돌아간다. 논문의 표현을 빌리면:\n\n"
                "> *\"reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources.\"*\n\n"
                "즉 추론과 행동이 서로를 돕는 **시너지** 구조다. 추론은 다음에 무슨 행동을 할지 계획을 세우고 예외를 처리하며, 행동은 그 추론을 실제 세계(위키피디아, 계산기, 코드 실행)에 붙들어 맨다. 이 '붙들어 맴'을 **그라운딩(grounding)** 이라 부른다 — 추론이 허공에 떠 있지 않고 관찰된 사실에 닻을 내린 상태다."
            ),
        },
        {
            "title": "왜 이것이 loop engineering의 시작인가",
            "content": (
                "ReAct 논문 자체는 '루프 엔지니어링'이라는 말을 쓰지 않는다. 하지만 후대가 이걸 출발점으로 삼는 이유가 있다.\n\n"
                "ReAct는 에이전트를 **정적인 프롬프트 한 방**이 아니라 **동적인 반복 과정**으로 재정의했다. 답이 나올 때까지 Thought→Action→Observation을 계속 도는 것 — 이게 바로 1장의 `for step in range(max_steps)` 루프다. 오늘날 Claude Code가 파일을 읽고·수정하고·테스트를 돌리는 것도, Cursor가 코드베이스를 탐색하는 것도, 전부 이 루프의 후손이다.\n\n"
                "중요한 건 ReAct가 열어젖힌 **질문들**이다. 루프를 언제 멈출까? 관찰이 너무 길면 어떻게 자를까? 같은 행동을 반복하면 어떻게 감지할까? 도구가 에러를 뱉으면? 이 질문들에 답하는 것이 곧 loop engineering이고, 3장부터의 모든 논문이 이 질문들 중 하나씩을 붙들고 발전시킨 결과물이다. ReAct는 '에이전트는 루프다'라는 명제를 최초로 실증했다."
            ),
        },
    ],
    "analogy": {
        "title": "눈 감고 푸는 학생 vs 참고서 펴는 학생",
        "content": (
            "순수 CoT는 **눈을 감고 암산으로만 시험을 푸는 학생**이다. 머릿속으로 '음, 이 나라 수도는 아마 이거고, 인구는 대략 저 정도일 거야' 하고 쭉 밀고 나간다. 똑똑하면 꽤 맞히지만, 한 번 잘못 기억하면 그 위에 쌓은 모든 답이 연쇄적으로 틀린다. 그리고 자기가 틀렸는지조차 모른다.\n\n"
            "ReAct는 **매 단계 참고서를 펼치고 계산기를 두드리는 학생**이다. '수도가 뭐였지?' 싶으면 바로 찾아보고(Action), 찾은 값을 확인하고(Observation), 그제서야 다음 계산으로 넘어간다. 한 단계 한 단계가 현실에 검증받으니, 틀린 전제 위에 탑을 쌓는 일이 없다.\n\n"
            "결정적 차이는 '생각의 속도'가 아니라 '**대조의 유무**'다. 두 학생 다 똑똑할 수 있다. 하지만 시험이 어려워지고 길어질수록, 매번 사실을 확인하며 나아가는 학생이 압도적으로 안정적이다. 에이전트 작업이 복잡할수록 ReAct 루프가 CoT를 이기는 이유가 정확히 이것이다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "ReAct 루프를 직접 구현해보자. 1장의 골격과 거의 같지만, 이번엔 LLM이 명시적으로 'Thought:'와 'Action:'을 텍스트로 뱉게 하고, 우리가 그걸 파싱해 도구를 실행한 뒤 'Observation:'을 되먹인다. 이 텍스트 프로토콜이 원조 ReAct의 방식이다(요즘은 tool-calling API가 이걸 구조화해준다)."
        ),
        "code": (
            "import re\n"
            "\n"
            "SYSTEM = '''당신은 ReAct 에이전트다. 다음 형식을 반복하라:\n"
            "Thought: <지금 무엇을 왜 해야 하는지>\n"
            "Action: <tool_name>[<query>]\n"
            "(도구 결과는 Observation:으로 주어진다)\n"
            "충분히 알았으면 Action 대신 'Answer: <최종답>'을 출력하라.'''\n"
            "\n"
            "def react(question, tools, llm, max_steps=8):\n"
            "    scratchpad = f\"Question: {question}\\n\"\n"
            "    for _ in range(max_steps):\n"
            "        out = llm(SYSTEM + scratchpad, stop=[\"Observation:\"])  # 관찰 전까지만 생성\n"
            "        scratchpad += out\n"
            "\n"
            "        if \"Answer:\" in out:                       # 정지 조건\n"
            "            return out.split(\"Answer:\")[-1].strip()\n"
            "\n"
            "        m = re.search(r\"Action:\\s*(\\w+)\\[(.*?)\\]\", out)   # Action 파싱\n"
            "        if not m:\n"
            "            scratchpad += \"\\nObservation: (형식 오류 — Action을 다시 출력하라)\\n\"\n"
            "            continue\n"
            "        tool, arg = m.group(1), m.group(2)\n"
            "        obs = tools.get(tool, lambda a: f\"알 수 없는 도구 {tool}\")(arg)  # act\n"
            "        scratchpad += f\"\\nObservation: {obs}\\n\"     # observe → 다음 루프\n"
            "    return \"(max_steps 도달)\"\n"
        ),
        "walkthrough": (
            "주목할 지점 셋. **(1) `stop=[\"Observation:\"]`** — 모델이 스스로 관찰 결과까지 지어내지 못하게 생성을 끊는다. 관찰은 반드시 실제 도구에서 와야 그라운딩이 성립한다. 이걸 빠뜨리면 모델이 도구 결과를 환각하는 고전적 버그가 난다. **(2) scratchpad** — Thought·Action·Observation이 계속 누적되는 이 문자열이 곧 에이전트의 '작업 기억'이다. 길어지면 잘라야 하는데, 그게 7장 컨텍스트 엔지니어링의 문제로 이어진다. **(3) 형식 오류 처리** — 파싱 실패 시 무너지지 않고 재시도를 유도한다. 이런 방어 코드가 loop engineering의 실전 기본기다. 이 30줄이 ReAct의 전부이며, 현대 tool-calling 에이전트는 이 텍스트 프로토콜을 JSON 스키마로 정형화한 것뿐이다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "ReAct는 너무 유명해서 '안다'고 말하기 쉽지만, 면접관은 이름이 아니라 '왜 그 구조여야 하는가'를 묻는다. 특히 CoT 대비 무엇이 근본적으로 달라졌는지, 그리고 프로덕션에서 ReAct 루프가 어떻게 깨지는지를 설명할 수 있느냐가 관건이다."
        ),
        "whatEngineersLookFor": [
            "Thought/Action/Observation 각각의 역할과, 셋이 왜 함께 있어야 시너지가 나는지 설명",
            "그라운딩(추론을 관찰에 닻 내림)이 환각·오류전파를 어떻게 막는지 인과적으로 설명",
            "현대 tool-calling API가 결국 ReAct의 구조화된 버전임을 알아봄",
            "무한 반복·관찰 폭주·형식 오류 같은 실전 실패 모드에 대한 대비책",
        ],
        "redFlags": [
            "ReAct를 그냥 '프롬프트 기법'으로만 이해하고 루프 구조를 못 봄",
            "관찰 결과를 모델이 지어내게 놔두는 stop-sequence 누락 버그를 모름",
            "정지 조건 없이 '답 나올 때까지' 돌리겠다고 함",
            "CoT와의 차이를 '더 똑똑해서'로 설명 (그라운딩의 부재/존재를 못 짚음)",
        ],
        "interviewQuestions": [
            "ReAct와 순수 Chain-of-Thought의 근본적 차이는 무엇이며, 어떤 작업에서 그 차이가 결정적인가?",
            "ReAct 루프가 같은 행동을 무한 반복하는 상황을 어떻게 감지하고 끊겠는가?",
            "현대 함수 호출(function calling) API는 원조 ReAct와 무엇이 같고 무엇이 다른가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'ReAct = 생각하고 행동하기'라고 요약한다. **마스터**는 ReAct의 진짜 기여가 '추론을 외부 관찰에 그라운딩함으로써 오류 전파를 끊은 것'임을 짚고, 그 대가로 생긴 새 문제들(관찰 길이, 반복 감지, 정지 조건)이 이후 모든 에이전트 연구의 의제가 되었다는 계보까지 그린다."
        ),
    },
    "keyTakeaways": [
        {"title": "Reason + Act", "content": "생각(추론)과 행동(도구 호출)을 한 루프에서 번갈아 짜는 것이 ReAct의 핵심."},
        {"title": "그라운딩", "content": "추론을 외부 관찰에 닻 내려, CoT의 환각과 오류 전파를 끊는다."},
        {"title": "루프의 원형", "content": "Thought→Action→Observation 반복이 오늘날 모든 에이전트 while 루프의 조상이다."},
        {"title": "stop sequence의 중요성", "content": "관찰까지 모델이 생성하지 못하게 끊어야 그라운딩이 진짜로 성립한다."},
        {"title": "scratchpad = 작업 기억", "content": "누적되는 Thought/Action/Observation이 에이전트의 단기 기억이며, 길이 관리가 곧 컨텍스트 문제로 이어진다."},
        {"title": "현대 API의 조상", "content": "tool-calling / function calling은 ReAct의 텍스트 프로토콜을 JSON으로 정형화한 버전이다."},
        {"title": "새 질문의 문을 엶", "content": "정지 조건·반복 감지·관찰 관리 등 loop engineering의 의제 전부가 여기서 시작됐다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 3 — Reflexion + Self-Refine
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 3,
    "emoji": "🪞",
    "title": "Reflexion & Self-Refine: The Self-Reflection Loop",
    "titleKr": "Reflexion & Self-Refine — 자기성찰 루프",
    "tldr": "루프에 '반성'을 넣는다. 실패한 시도를 언어로 되짚어 메모리에 적어두고(Reflexion), 혹은 자기 출력을 스스로 비평해 고쳐 쓴다(Self-Refine). 가중치를 건드리지 않고 언어만으로 스스로 나아지는 루프.",
    "topics": ["언어적 피드백(verbal RL)", "episodic memory buffer", "self-critique 반복", "trial-reflect-retry"],
    "learningGoals": [
        "Reflexion의 '언어로 강화학습' 아이디어와 episodic memory buffer를 설명한다",
        "Self-Refine의 generate→critique→refine 단일 LLM 루프를 구현할 수 있다",
        "두 방법이 가중치 업데이트 없이 성능을 올리는 원리를 이해한다",
        "자기성찰 루프의 한계(self-bias, 수렴 실패)를 진단한다",
        "언제 반성을 메모리에 남기고(Reflexion) 언제 즉석에서 고쳐쓸지(Self-Refine) 판단한다",
    ],
    "overview": (
        "ReAct가 '행동하는 루프'를 만들었다면, 다음 질문은 자연스럽다. **실패하면 어떻게 배우지?** 사람은 시험을 망치면 '아, 이 부분을 놓쳤구나' 반성하고 다음엔 다르게 한다. 그런데 LLM을 다시 훈련(가중치 업데이트)하는 건 비싸고 느리다. 더 가벼운 방법은 없을까?\n\n"
        "[Reflexion](https://arxiv.org/abs/2303.11366)(Shinn et al., NeurIPS 2023)의 답은 우아하다. **가중치 대신 언어로 강화하라.** 에이전트가 작업에 실패하면, 무엇이 왜 잘못됐는지를 스스로 문장으로 적어 **episodic memory buffer(일화 기억 버퍼)** 에 저장한다. 다음 시도 때 이 반성문을 컨텍스트에 넣어주면, 같은 실수를 피한다. 훈련 없이, 오직 텍스트로 학습이 일어난다.\n\n"
        "[Self-Refine](https://arxiv.org/abs/2303.17651)(Madaan et al., 2023)은 이걸 더 미니멀하게 만든다. 별도 메모리도, 여러 시도도 필요 없다. **하나의 LLM이 답을 쓰고 → 자기 답을 비평하고 → 그 비평으로 고쳐 쓰기**를 반복한다. 생성자·비평가·수정자가 전부 같은 모델이다. 이 장에서는 '루프에 반성을 심는' 두 가지 방식을 배운다 — 시도 간(Reflexion) 반성과 시도 내(Self-Refine) 반성."
    ),
    "sections": [
        {
            "title": "Reflexion: 언어로 하는 강화학습",
            "content": (
                "전통적 강화학습(RL)은 보상 신호로 가중치를 조금씩 조정한다. 느리고, 많은 시도가 필요하고, 왜 그렇게 바뀌었는지 해석하기 어렵다. Reflexion은 이 루프를 통째로 언어 공간으로 옮긴다.\n\n"
                "> *\"Reflexion, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback.\"*\n> — Shinn et al., 2023\n\n"
                "동작은 이렇다. 에이전트가 작업을 시도한다 → 실패 신호(테스트 불통과, 목표 미달)를 받는다 → **그 실패를 언어로 성찰한다**: \"내가 X를 가정했는데 그게 틀렸다. 다음엔 먼저 Y를 확인해야 한다.\" → 이 반성문을 episodic memory에 저장 → 다음 시도 때 이 메모리를 컨텍스트에 주입.\n\n"
                "핵심은 **반성문이 다음 시도의 지침이 된다**는 것이다. 보상이라는 숫자 한 개(scalar)보다, '무엇을 왜 틀렸는지'라는 문장이 훨씬 정보가 풍부하다. 그래서 단 몇 번의 시도만으로도 극적으로 개선된다. 가중치는 그대로인데, 컨텍스트에 쌓인 반성이 에이전트를 똑똑하게 만든다."
            ),
        },
        {
            "title": "Self-Refine: 혼자서 쓰고, 까고, 고치기",
            "content": (
                "Self-Refine은 더 급진적으로 단순하다. 외부 도구도, 별도 훈련 데이터도, 여러 에피소드도 없다. 단 하나의 LLM이 세 가지 역할을 번갈아 한다.\n\n"
                "> *\"the same LLM provides feedback for its output and uses it to refine itself, iteratively.\"*\n> — Madaan et al., 2023\n\n"
                "루프는 세 박자다.\n\n"
                "1. **Generate** — 초안을 쓴다.\n"
                "2. **Feedback** — 같은 모델이 그 초안을 비평한다. (\"이 함수는 엣지 케이스를 놓쳤다\")\n"
                "3. **Refine** — 그 피드백을 반영해 고쳐 쓴다.\n\n"
                "그리고 만족스러울 때까지 2→3을 반복한다. 훈련이 전혀 없다는 게 논문의 자랑이다: *\"does not require any supervised training data, additional training, or reinforcement learning.\"*\n\n"
                "이게 통하는 이유는 **비평이 생성보다 쉽다**는 비대칭성 때문이다. 처음부터 완벽한 에세이를 쓰긴 어렵지만, 남이 쓴 에세이에서 어색한 문장을 찾긴 쉽다. LLM도 마찬가지여서, '평가자 모드'로 자기 출력을 보면 생성 때 놓친 결함을 종종 잡아낸다."
            ),
        },
        {
            "title": "두 방식의 차이와 공통의 함정",
            "content": (
                "둘 다 '언어 기반 자기개선 루프'지만 결이 다르다.\n\n"
                "| 구분 | Reflexion | Self-Refine |\n|---|---|---|\n| 반성의 위치 | 시도와 시도 **사이** | 한 답 **안에서** |\n| 메모리 | episodic buffer에 축적 | 없음 (즉석) |\n| 외부 신호 | 필요 (성공/실패 판정) | 불필요 (자기 비평만) |\n| 적합한 상황 | 명확한 성공 판정이 있는 반복 과제 | 한 번에 품질을 올리고 싶은 생성 과제 |\n\n"
                "그리고 둘 다 같은 함정에 빠진다. **self-bias(자기편향)** — 모델이 자기 답을 후하게 평가해서 진짜 결함을 못 보는 것이다. 특히 수학처럼 정답이 딱 떨어지는 영역에서, 틀린 답을 '괜찮아 보인다'며 그냥 확정해버리는 실패가 잦다.\n\n"
                "그래서 실무에서는 **비평 신호를 외부에서 주입**하는 식으로 보강한다 — 테스트 실행 결과, 컴파일러 에러, 별도 검증 모델의 판정 등. 자기 자신만 거울로 삼으면 편향에서 못 벗어나지만, 외부 관찰(2장의 그라운딩!)을 반성의 재료로 쓰면 훨씬 튼튼해진다. 이 지점에서 ReAct의 그라운딩과 Reflexion의 반성이 한 루프 안에서 만난다."
            ),
        },
    ],
    "analogy": {
        "title": "오답노트 vs 초고 퇴고",
        "content": (
            "**Reflexion은 오답노트**다. 시험을 한 번 치르고 나서, 틀린 문제마다 '나는 여기서 이런 착각을 했다. 다음엔 이렇게 접근하자'를 노트에 적는다. 다음 시험 전에 이 노트를 다시 읽으면, 같은 함정에 두 번 빠지지 않는다. 시험(시도)과 시험 사이에 반성이 쌓이고, 그 축적이 곧 실력이 된다. 핵심은 노트가 '점수'가 아니라 '문장으로 된 교훈'이라는 점 — 65점이라는 숫자보다 '분수 통분을 빼먹었다'는 문장이 훨씬 쓸모 있다.\n\n"
            "**Self-Refine은 초고 퇴고**다. 에세이 한 편을 쓰는 그 자리에서, 초고를 쓰고 → 스스로 소리 내어 읽어보고 → '이 문단 논리가 약하네' 고치고 → 다시 읽고를 반복한다. 다음 에세이를 위한 노트를 남기는 게 아니라, 지금 이 한 편을 그 안에서 갈고닦는다.\n\n"
            "둘 다 '스스로를 거울에 비춰 고친다'는 점은 같다. 하지만 오답노트는 여러 시험에 걸친 학습이고, 퇴고는 한 작품 안의 완성이다. 그리고 둘 다 위험이 같다 — **자기 눈이 후하면** 오답노트에 틀린 교훈을 적거나, 퇴고에서 결함을 못 본다. 그래서 진짜 고수는 채점표(외부 신호)를 옆에 두고 반성한다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "Self-Refine 루프를 구현해보자. 하나의 LLM이 생성자·비평가·수정자를 번갈아 맡는다. 실전 감각을 위해 '외부 신호로 비평을 보강'하는 훅(테스트 실행)도 함께 넣었다 — 순수 자기비평의 self-bias를 완화하는 실무 패턴이다."
        ),
        "code": (
            "def self_refine(task, llm, run_tests=None, max_iters=4):\n"
            "    draft = llm(f\"다음 작업의 코드를 작성하라:\\n{task}\")  # 1) Generate\n"
            "\n"
            "    for i in range(max_iters):\n"
            "        # 외부 신호로 비평을 그라운딩 (self-bias 완화)\n"
            "        test_signal = run_tests(draft) if run_tests else \"(테스트 없음)\"\n"
            "        if test_signal == \"PASS\":\n"
            "            return draft                       # 정지: 객관적 통과\n"
            "\n"
            "        # 2) Feedback — 같은 모델이 자기 출력을 비평\n"
            "        critique = llm(\n"
            "            f\"작업:\\n{task}\\n\\n현재 코드:\\n{draft}\\n\\n\"\n"
            "            f\"테스트 결과: {test_signal}\\n\"\n"
            "            \"구체적 결함을 항목별로 지적하라. 결함이 없으면 'NONE'.\"\n"
            "        )\n"
            "        if critique.strip() == \"NONE\":\n"
            "            return draft\n"
            "\n"
            "        # 3) Refine — 비평을 반영해 고쳐 씀\n"
            "        draft = llm(\n"
            "            f\"작업:\\n{task}\\n\\n이전 코드:\\n{draft}\\n\\n\"\n"
            "            f\"지적된 결함:\\n{critique}\\n\\n결함을 고친 전체 코드를 출력하라.\"\n"
            "        )\n"
            "    return draft   # max_iters 도달 — 마지막 초안 반환\n"
        ),
        "walkthrough": (
            "이 루프의 설계 포인트. **(1) 외부 신호 우선** — `run_tests`가 PASS를 주면 자기비평을 건너뛰고 즉시 확정한다. 객관적 정답 판정이 있으면 그걸 self-bias보다 신뢰한다. **(2) 비평을 테스트 결과에 그라운딩** — critique 단계에 `test_signal`을 넣어, 모델이 허공에 대고 자화자찬하지 못하게 실패의 물증을 들이민다. 이게 2장 그라운딩과 3장 반성이 만나는 지점이다. **(3) 명시적 종료 신호** — 'NONE'이라는 탈출구를 줘서 불필요한 반복을 막는다. Reflexion으로 확장하려면 이 `critique`를 함수 밖 `memory` 리스트에 append해 다음 '시도'의 프롬프트에 주입하면 된다 — 즉 시도-내 루프가 시도-간 루프로 승격된다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "자기개선 루프는 데모에선 화려하지만 프로덕션에선 함정이 많다. 면접관은 지원자가 self-bias의 위험을 알고, '언제 자기비평이 실제로 도움이 되고 언제 비용만 늘리는지'를 판단할 수 있는지를 본다. 무작정 '반성 루프를 돌리면 좋아진다'는 순진한 낙관은 감점이다."
        ),
        "whatEngineersLookFor": [
            "Reflexion(시도 간)과 Self-Refine(시도 내)의 구조적 차이와 각각의 적합 상황을 구분",
            "self-bias를 인지하고, 외부 신호(테스트·검증 모델)로 비평을 그라운딩하는 설계",
            "반성 루프의 비용(추가 LLM 호출)과 이득을 저울질하는 감각",
            "언제 수렴하지 않는지(무한 퇴고, 진동)를 알고 정지 조건을 설계",
        ],
        "redFlags": [
            "'자기비평을 반복하면 항상 좋아진다'는 무비판적 낙관",
            "정답이 딱 떨어지는 과제에서도 자기비평만 믿고 외부 검증을 안 붙임",
            "반성 루프의 추가 토큰/지연 비용을 고려하지 않음",
            "Reflexion과 Self-Refine을 같은 것으로 뭉뚱그림",
        ],
        "interviewQuestions": [
            "자기비평 루프에서 self-bias는 왜 생기며, 실무에서 어떻게 완화하는가?",
            "Reflexion의 episodic memory와 Self-Refine의 즉석 비평 중, 코드 생성 에이전트에는 어느 쪽이 맞고 왜인가?",
            "자기개선 루프가 오히려 답을 망가뜨리는(진동·퇴행) 경우를 어떻게 감지하고 막겠는가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'LLM이 자기 답을 고치게 하면 좋아진다'고 안다. **마스터**는 그 개선이 '비평이 생성보다 쉽다'는 비대칭성에서 나오되, 자기 자신만 거울로 삼으면 self-bias에 갇힌다는 한계를 알고, 외부 관찰을 반성의 재료로 끌어와(ReAct 그라운딩 + Reflexion 반성) 루프를 튼튼하게 만드는 설계까지 제시한다."
        ),
    },
    "keyTakeaways": [
        {"title": "언어로 하는 RL", "content": "Reflexion은 가중치 대신 언어적 반성으로 에이전트를 강화한다 — 숫자 보상보다 문장 교훈이 정보가 풍부하다."},
        {"title": "episodic memory", "content": "실패의 반성문을 메모리에 쌓아 다음 시도에 주입 — 시도 간 학습이 일어난다."},
        {"title": "generate-critique-refine", "content": "Self-Refine은 하나의 LLM이 쓰고·까고·고치기를 반복하는 시도 내 루프다."},
        {"title": "비평 비대칭성", "content": "생성보다 비평이 쉽기에 자기비평이 통한다 — 하지만 만능은 아니다."},
        {"title": "self-bias 함정", "content": "자기 답을 후하게 봐서 결함을 놓친다. 정답이 명확한 과제에서 특히 위험."},
        {"title": "외부 신호로 그라운딩", "content": "테스트·컴파일러·검증 모델로 비평을 뒷받침하면 self-bias를 완화한다 — 2장 그라운딩과의 결합."},
        {"title": "정지 조건 필수", "content": "무한 퇴고와 진동을 막기 위해 종료 신호(NONE/PASS)와 max_iters를 반드시 건다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 4 — Voyager
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 4,
    "emoji": "🧭",
    "title": "Voyager: The Lifelong Loop & Skill Library",
    "titleKr": "Voyager — 평생학습 루프와 skill library",
    "tldr": "루프에 '영구 기억'을 넣는다. 마인크래프트에서 GPT-4가 스스로 탐험하며, 성공한 행동을 실행 가능한 코드로 skill library에 쌓아 재사용한다. 파인튜닝 없이 경험이 자산으로 축적되는 평생학습 루프.",
    "topics": ["lifelong learning agent", "skill library (코드 축적)", "iterative prompting + self-verification", "임베딩 기반 skill 검색"],
    "learningGoals": [
        "Voyager의 세 축(자동 커리큘럼·skill library·반복 프롬프팅)을 설명한다",
        "'skill을 실행 가능한 코드로 저장'하는 것이 왜 강력한 기억 형태인지 이해한다",
        "파인튜닝 없이 blackbox API만으로 평생학습이 가능한 원리를 안다",
        "skill library를 임베딩으로 검색·조합하는 메커니즘을 구현할 수 있다",
        "3장의 반성 메모리와 4장의 skill 메모리의 차이를 구분한다",
    ],
    "overview": (
        "3장의 Reflexion은 '실패의 교훈'을 텍스트로 기억했다. 하지만 교훈은 추상적이다. 더 강력한 기억은 없을까? [Voyager](https://arxiv.org/abs/2305.16291)(Wang et al., NeurIPS 2023 워크숍, 이후 TMLR)의 답은 대담하다. **성공한 행동을 실행 가능한 코드 그 자체로 저장하라.**\n\n"
        "Voyager는 마인크래프트 속에서 사는 최초의 LLM 기반 '평생학습 에이전트(lifelong learning agent)'다. 사람이 목표를 주지 않아도 스스로 '이번엔 나무를 캐볼까 → 다음엔 돌 곡괭이를 만들까'식으로 커리큘럼을 세우고, 세계를 끝없이 탐험한다. 놀랍게도 GPT-4를 **blackbox API로만** 쓴다 — 파라미터를 단 하나도 파인튜닝하지 않는다.\n\n"
        "비결은 두 가지다. 첫째, 환경 피드백·실행 에러·자기검증을 되먹이는 **반복 프롬프팅 루프**(2·3장의 종합). 둘째, 성공적으로 작동한 행동을 **실행 가능한 코드 함수로 skill library에 저장**하고, 나중에 비슷한 상황에서 임베딩으로 검색해 꺼내 쓰는 것. 경험이 날아가지 않고 재사용 가능한 도구로 굳는다. 이 장에서 우리는 '루프에 영구 기억을 붙이는' 방법을 배운다."
    ),
    "sections": [
        {
            "title": "세 개의 톱니바퀴",
            "content": (
                "Voyager는 세 부품이 맞물려 돈다.\n\n"
                "**자동 커리큘럼(automatic curriculum)** = '지금 내 능력에서 적당히 도전적인 다음 목표'를 GPT-4가 스스로 제안한다. 너무 쉽지도 어렵지도 않은 과제를 연속으로 던져 탐험을 극대화한다. 사람의 목표 지정이 필요 없다.\n\n"
                "**반복 프롬프팅 메커니즘(iterative prompting)** = 목표를 코드로 구현 → 실행 → 결과를 본다. 논문의 표현으로는 환경 피드백, 실행 에러, 그리고 **self-verification(자기검증)** 을 프로그램 개선에 되먹인다. 3장 Self-Refine의 마인크래프트판이다.\n\n"
                "> *\"a new iterative prompting mechanism that incorporates environment feedback, execution errors, and self-verification for program improvement.\"*\n\n"
                "**skill library** = 검증을 통과한 코드를 영구 저장하는 창고. 이 세 번째가 Voyager의 진짜 혁신이라 별도 섹션에서 다룬다. 셋의 흐름은: 커리큘럼이 목표를 던지고 → 반복 루프가 코드를 완성하고 → 완성된 코드가 라이브러리에 적립된다."
            ),
        },
        {
            "title": "skill library: 코드로 된 기억",
            "content": (
                "핵심 통찰은 이것이다. **기억을 '설명'이 아니라 '실행 가능한 코드'로 저장하면, 재사용이 완벽해진다.**\n\n"
                "> *\"an ever-growing skill library of executable code for storing and retrieving complex behaviors.\"*\n\n"
                "예를 들어 Voyager가 '돌 곡괭이 만들기'를 성공하면, 그 절차를 `craftStonePickaxe()`라는 자바스크립트 함수로 정제해 저장한다. 나중에 '철 곡괭이 만들기'라는 더 복잡한 목표가 오면, 라이브러리에서 `craftStonePickaxe`를 **꺼내 조합**해 더 큰 함수를 만든다. 기술이 레고 블록처럼 쌓인다.\n\n"
                "검색은 **임베딩**으로 한다. 각 skill을 그 기능 설명의 임베딩으로 색인해두고, 새 상황이 오면 의미적으로 가까운 skill을 top-k로 꺼낸다. 이건 사실상 **자기가 만든 도구를 자기가 검색해 쓰는 RAG**다.\n\n"
                "이 방식의 위력은 세 가지다. **(1) 복리 효과** — 스킬이 쌓일수록 더 복잡한 걸 더 빨리 해낸다. **(2) 해석 가능성** — 기억이 사람이 읽을 수 있는 코드라 디버깅이 된다. **(3) 이식성** — 라이브러리를 통째로 새 에이전트에 옮겨 부팅할 수 있다. 3장의 텍스트 반성이 '교훈 노트'라면, Voyager의 skill은 '완성된 연장'이다."
            ),
        },
        {
            "title": "파인튜닝 없는 학습의 의미",
            "content": (
                "Voyager가 던진 가장 큰 화두는 이것이다. **학습이 반드시 가중치 안에서 일어나야 하는가?**\n\n"
                "> *\"Voyager interacts with GPT-4 via blackbox queries, which bypasses the need for model parameter fine-tuning.\"*\n\n"
                "전통적 관점에서 '모델이 배운다'는 건 파라미터가 바뀌는 것이다. Voyager는 정반대를 증명했다. 모델(GPT-4)은 얼어붙어 있는데도, **외부에 쌓이는 skill library** 때문에 에이전트 전체는 계속 유능해진다. 학습이 모델의 무게가 아니라 **루프를 감싼 외부 메모리**에서 일어난 것이다.\n\n"
                "이건 loop engineering의 세계관을 정확히 대변한다. 모델은 고정된 '추론 엔진'이고, 지능의 성장은 그 엔진을 감싼 루프·메모리·도구의 설계에서 나온다. 6장에서 볼 Anthropic의 '에이전트 = 루프' 명제가 여기서 이미 실증되고 있다. 값비싼 재훈련 대신, 잘 설계된 루프와 축적되는 메모리가 같은 목적지에 — 때로 더 유연하게 — 도달한다.\n\n"
                "물론 한계도 분명하다. 고정된 모델의 능력 상한을 넘진 못하고, skill library가 커지면 검색 정확도와 컨텍스트 관리가 새 병목이 된다(→ 7장). 하지만 '학습 = 외부 축적'이라는 패러다임의 문을 연 공은 온전히 Voyager의 것이다."
            ),
        },
    ],
    "analogy": {
        "title": "요리사의 개인 레시피 노트북",
        "content": (
            "견습 요리사를 생각해보자. 3장의 Reflexion식 요리사는 '오늘 소스가 짰다. 다음엔 소금을 줄이자' 같은 **교훈을 메모**한다. 유용하지만, 다음에 그 요리를 할 때 여전히 처음부터 감으로 만들어야 한다.\n\n"
            "Voyager식 요리사는 다르다. 어떤 요리를 성공하면, 그 **정확한 레시피를 계량까지 적어 레시피북에 정서**해둔다 — '토마토 소스: 양파 반 개, 마늘 2쪽, 3분 볶고…'. 다음에 그 소스가 필요하면 레시피를 그대로 펼쳐 완벽히 재현한다. 더 나아가 '라자냐'라는 큰 요리를 할 땐, 레시피북에서 '토마토 소스'와 '베샤멜 소스' 페이지를 꺼내 **조합**한다. 이미 완성된 서브레시피들이 블록처럼 결합된다.\n\n"
            "결정적인 건 이 요리사가 **더 똑똑해진 게 아니라 레시피북이 두꺼워졌다**는 점이다. 두뇌(모델)는 그대로인데, 외부 노트(skill library)가 쌓여서 점점 복잡한 요리를 척척 해낸다. 그리고 이 레시피북은 후배에게 통째로 물려줄 수도 있다. Voyager가 증명한 건 — 성장은 머리가 아니라 잘 정리된 노트에서 올 수 있다는 것이다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "Voyager식 skill library의 뼈대를 구현해보자. 핵심은 세 동작이다 — 성공한 코드를 임베딩과 함께 저장하고, 새 목표에 의미적으로 가까운 skill을 검색하고, 그것들을 컨텍스트에 넣어 새 코드를 합성한다. 자기가 만든 도구를 자기가 검색해 쓰는 RAG 루프다."
        ),
        "code": (
            "import numpy as np\n"
            "\n"
            "class SkillLibrary:\n"
            "    def __init__(self, embed):\n"
            "        self.embed = embed          # 텍스트 → 벡터\n"
            "        self.skills = []            # [{name, code, desc, vec}]\n"
            "\n"
            "    def add(self, name, code, desc):\n"
            "        # 검증을 통과한 코드만 적립 (self-verification 이후 호출)\n"
            "        self.skills.append({\"name\": name, \"code\": code,\n"
            "                            \"desc\": desc, \"vec\": self.embed(desc)})\n"
            "\n"
            "    def retrieve(self, goal, k=3):\n"
            "        if not self.skills:\n"
            "            return []\n"
            "        q = self.embed(goal)\n"
            "        sims = [(np.dot(q, s[\"vec\"]), s) for s in self.skills]\n"
            "        return [s for _, s in sorted(sims, reverse=True)[:k]]  # top-k 조합 재료\n"
            "\n"
            "def voyager_step(goal, lib, llm, env):\n"
            "    reusable = lib.retrieve(goal)                       # 1) 관련 skill 검색\n"
            "    primer = \"\\n\\n\".join(s[\"code\"] for s in reusable)  # 조합할 블록들\n"
            "    code = llm(f\"기존 skill:\\n{primer}\\n\\n목표: {goal}\\n\"\n"
            "              \"위 skill들을 활용해 목표를 달성하는 함수를 작성하라.\")\n"
            "\n"
            "    result = env.run(code)                              # 2) 실행 = 자기검증\n"
            "    if result.success:\n"
            "        lib.add(result.fn_name, code, goal)             # 3) 성공하면 적립 (복리!)\n"
            "    return result\n"
        ),
        "walkthrough": (
            "이 20여 줄에 Voyager의 정수가 들어 있다. **(1) 검증 후 적립** — `env.run`이 성공(self-verification)해야만 `lib.add`로 저장한다. 검증되지 않은 코드를 라이브러리에 넣으면 오염된 기억이 미래를 망친다. **(2) 조합(composition)** — 검색된 기존 skill들의 코드를 프롬프트에 primer로 넣어, 새 코드가 그것들을 '호출해 재사용'하게 유도한다. 이게 스킬을 레고처럼 쌓는 복리 메커니즘이다. **(3) 임베딩 검색** — 라이브러리가 커져도 관련된 것만 top-k로 꺼내 컨텍스트를 아낀다(7장 예고). 3장의 메모리는 '실패 교훈 텍스트'였지만, 여기 메모리는 '성공 코드'라 재사용 시 재현율이 100%다 — 텍스트 교훈은 다시 해석해야 하지만, 코드는 그냥 호출하면 된다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "Voyager는 '에이전트 메모리'를 설계할 때 반드시 소환되는 레퍼런스다. 면접관은 지원자가 '메모리 = 벡터DB에 대화 넣기'라는 얕은 이해를 넘어, 기억의 형태(텍스트 vs 코드 vs 파라미터)와 그 트레이드오프를 논할 수 있는지, 그리고 축적형 메모리의 위험(오염·검색 실패·컨텍스트 폭발)을 아는지를 본다."
        ),
        "whatEngineersLookFor": [
            "기억을 실행 가능한 코드로 저장하는 것의 장점(재현율·조합성·이식성)을 설명",
            "'학습 = 파라미터 변경'이 아니라 '외부 메모리 축적'일 수 있음을 이해",
            "검증되지 않은 산출물을 메모리에 넣으면 안 된다는 위생 감각(오염 방지)",
            "skill library가 커질 때의 검색 정확도·컨텍스트 관리 병목을 예상",
        ],
        "redFlags": [
            "메모리를 그냥 '대화 로그를 벡터DB에 다 넣기'로만 생각",
            "검증 없이 에이전트 산출물을 무조건 메모리에 적립 (오염 위험 무시)",
            "고정 모델 + 외부 메모리로도 학습이 일어난다는 점을 이해 못 함",
            "라이브러리 성장에 따른 검색/컨텍스트 병목을 고려하지 않음",
        ],
        "interviewQuestions": [
            "에이전트의 기억을 텍스트 교훈, 실행 가능한 코드, 파인튜닝 중 무엇으로 저장할지 어떤 기준으로 정하나?",
            "파인튜닝 없이 blackbox 모델만으로 에이전트가 '학습'한다는 게 무슨 의미인가?",
            "skill library가 수천 개로 커지면 어떤 문제가 생기고 어떻게 대응하겠는가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'Voyager = 마인크래프트 GPT-4'로 기억한다. **마스터**는 Voyager의 진짜 기여가 '학습을 모델 가중치에서 외부 skill library로 옮긴 것'이며, 이것이 loop engineering의 세계관(고정된 엔진 + 성장하는 루프/메모리)을 최초로 실증했음을 짚고, 코드-형 메모리의 재현율 우위와 오염·검색 병목이라는 대가까지 균형 있게 논한다."
        ),
    },
    "keyTakeaways": [
        {"title": "평생학습 루프", "content": "사람 개입 없이 스스로 커리큘럼을 세우고 세계를 끝없이 탐험하는 최초의 LLM 에이전트."},
        {"title": "코드로 된 기억", "content": "성공한 행동을 실행 가능한 코드로 저장하면 재현율 100% + 조합 가능 + 이식 가능."},
        {"title": "복리 효과", "content": "skill이 쌓일수록 그것들을 조합해 더 복잡한 목표를 더 빨리 해낸다."},
        {"title": "파인튜닝 없는 학습", "content": "모델은 고정, 외부 메모리가 성장 — 학습이 루프를 감싼 메모리에서 일어난다."},
        {"title": "self-verification", "content": "실행 결과로 코드를 검증하고, 통과한 것만 라이브러리에 적립(오염 방지)."},
        {"title": "skill = 자기제작 RAG", "content": "임베딩으로 자기가 만든 도구를 검색해 재사용 — 도구 창고이자 검색 시스템."},
        {"title": "새 병목 예고", "content": "라이브러리가 커지면 검색 정확도와 컨텍스트 관리가 다음 과제가 된다(→7장)."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 5 — Generative Agents
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 5,
    "emoji": "🧠",
    "title": "Generative Agents: Memory + Reflection + Planning as a Primitive",
    "titleKr": "Generative Agents — 기억·성찰·계획을 primitive로",
    "tldr": "스탠퍼드의 '심즈 마을' 실험. 25명의 LLM 에이전트가 기억 스트림·성찰·계획·검색이라는 인지 아키텍처로 믿을 만한 인간적 행동을 창발시킨다. 이 memory-reflection-planning 삼각형이 이후 모든 에이전트의 재사용 primitive가 됐다.",
    "topics": ["memory stream", "reflection(고수준 추론 합성)", "planning & retrieval", "믿을 만한 행동의 창발"],
    "learningGoals": [
        "memory stream · reflection · planning · retrieval 네 요소의 역할과 상호작용을 설명한다",
        "retrieval의 recency·importance·relevance 삼중 점수 메커니즘을 이해한다",
        "낮은 수준 관찰을 고수준 통찰로 합성하는 reflection의 필요성을 안다",
        "이 아키텍처가 왜 '재사용 가능한 primitive'로 불리는지 설명한다",
        "4장 skill 메모리와 5장 경험 메모리의 목적 차이를 구분한다",
    ],
    "overview": (
        "4장 Voyager가 '무엇을 할 줄 아는가(skill)'를 기억했다면, [Generative Agents](https://arxiv.org/abs/2304.03442)(Park et al., UIST 2023)는 '무엇을 겪었고 그래서 어떻게 행동할까'를 기억한다. 스탠퍼드와 구글이 만든 이 실험은 25명의 LLM 에이전트를 작은 마을(Smallville)에 풀어놓았다. 그들은 아침을 차리고, 출근하고, 수다를 떨고, 심지어 한 에이전트가 발렌타인 파티를 기획하자 소문이 퍼지고 사람들이 모여든다 — 아무도 시나리오를 짜주지 않았는데 말이다.\n\n"
        "이 '믿을 만한 인간적 행동(believable behavior)'은 어디서 나올까? 논문의 답은 하나의 **인지 아키텍처**다. 세 개의 기둥으로 이루어진다.\n\n"
        "**(1) Memory stream** — 겪은 모든 것을 자연어로 시간순 기록. **(2) Reflection** — 그 낱낱의 기억을 주기적으로 '고수준 통찰'로 합성. **(3) Planning & retrieval** — 기억을 꺼내 계획을 세우고 행동으로 옮김. 이 memory→reflection→planning 삼각형이 이후 Lilian Weng의 정전급 에이전트 에세이를 비롯해 수많은 프레임워크가 베낀 **재사용 primitive**가 됐다. 이 장에서 우리는 '에이전트의 인지 구조' 그 자체를 설계하는 법을 배운다."
    ),
    "sections": [
        {
            "title": "Memory stream: 모든 것을 자연어로 기록",
            "content": (
                "가장 기초는 **기억 스트림**이다. 에이전트가 관찰하거나 행동한 모든 것을 자연어 문장으로, 타임스탬프와 함께 길게 쌓는다.\n\n"
                "> *\"store a complete record of the agent's experiences using natural language.\"*\n\n"
                "\"오전 8시, 부엌에서 커피를 내렸다\", \"오전 8시 10분, 이웃 John을 마주쳐 인사했다\" 같은 식이다. 문제는 이게 금세 수천 개로 불어난다는 것. 매 순간 이 전부를 컨텍스트에 넣을 순 없다.\n\n"
                "그래서 **검색(retrieval)** 이 핵심이 된다. Generative Agents는 각 기억에 세 점수를 매겨 지금 상황에 꺼낼 것을 고른다.\n\n"
                "| 점수 | 의미 | 예 |\n|---|---|---|\n| Recency | 최근일수록 높음 | 방금 일은 잘 떠오름 |\n| Importance | 중요한 사건일수록 높음 | '이사했다' > '양치했다' |\n| Relevance | 현재 질문과 의미적으로 가까울수록 높음 | 임베딩 유사도 |\n\n"
                "세 점수의 가중합으로 상위 기억을 뽑아 컨텍스트에 넣는다. 이 recency+importance+relevance 공식은 오늘날 거의 모든 장기기억 에이전트가 변형해 쓰는 표준 레시피가 됐다."
            ),
        },
        {
            "title": "Reflection: 점을 이어 통찰로",
            "content": (
                "기억 스트림만으로는 부족하다. '커피를 내렸다', 'John과 인사했다' 같은 낱개 사실은 있지만, '나는 이웃과 잘 지내는 사교적인 사람이다' 같은 **추상적 자기이해**는 없다. 낱개 관찰만 검색해선 일관된 인격이 안 나온다.\n\n"
                "그래서 논문은 **reflection(성찰)** 을 도입한다.\n\n"
                "> *\"synthesize those memories over time into higher-level reflections.\"*\n\n"
                "주기적으로(중요도가 임계치를 넘으면) 에이전트는 최근 기억을 훑고 스스로 묻는다: \"이 관찰들로부터 내릴 수 있는 고수준 결론은 무엇인가?\" 그 답('나는 Klaus라는 인물의 연구를 존경한다' 같은)을 다시 기억 스트림에 **새로운 고수준 기억으로 저장**한다. 이 성찰은 또 다른 성찰의 재료가 되어, 기억이 트리처럼 추상화된다.\n\n"
                "핵심은 이것이 3장 Reflexion과 목적이 다르다는 점이다. Reflexion의 반성은 '실패를 고치기 위한' 것이지만, 여기 reflection은 '흩어진 경험을 일관된 세계관·자아로 통합하기 위한' 것이다. 전자는 성능, 후자는 정체성을 위한 성찰이다. 이 합성 능력이 있어야 에이전트가 그때그때 즉흥적이지 않고 '캐릭터답게' 일관되게 행동한다."
            ),
        },
        {
            "title": "왜 '재사용 primitive'인가",
            "content": (
                "Generative Agents의 진짜 유산은 특정 구현이 아니라 **패턴의 정립**이다. memory + reflection + planning + retrieval이라는 네 부품의 조합이, 이후 에이전트를 설계하는 사람들의 기본 청사진이 됐다.\n\n"
                "Lilian Weng의 널리 인용되는 에이전트 개관 글은 에이전트를 'LLM(두뇌) + Planning + Memory + Tool use'로 정리하는데, 이 Memory와 Planning의 내용물이 바로 Generative Agents에서 왔다. AgentPatterns 같은 카탈로그도 이를 명명된 패턴으로 등재했다.\n\n"
                "왜 이렇게 널리 베껴졌을까? **일반성** 때문이다. 마인크래프트의 Voyager는 도메인(게임)에 묶여 있지만, Generative Agents의 인지 아키텍처는 '경험을 쌓고·통합하고·그로부터 계획하는' 어떤 에이전트에도 적용된다 — 고객 지원 봇이든, 개인 비서든, 코딩 에이전트든. 도메인 독립적인 '에이전트 마음의 골격'을 처음으로 명료하게 제시한 것이다.\n\n"
                "loop engineering의 관점에서 보면, 이 장은 루프에 붙일 수 있는 **기억 서브시스템의 완성형 설계도**를 준다. 2장이 루프의 몸통, 3장이 반성, 4장이 skill 창고였다면, 5장은 그 모든 걸 아우르는 '장기기억 + 자아' 아키텍처다. 8장부터 루프가 그래프로 펼쳐질 때, 이 기억 primitive는 그래프의 상태(state)로 자연스럽게 흡수된다."
            ),
        },
    ],
    "analogy": {
        "title": "일기 + 주말의 회고 + 다이어리 계획",
        "content": (
            "한 사람의 정신적 삶을 생각해보자. 먼저 **일기(memory stream)** 가 있다. 매일 있었던 일을 시시콜콜 적는다 — 누굴 만났고, 뭘 먹었고, 무슨 생각을 했는지. 방대하지만, 매 순간 이 일기 전체를 떠올리며 살진 않는다. 지금 상황에 맞는 몇 페이지만 **펼쳐본다(retrieval)** — 최근 것, 중요한 것, 지금 고민과 관련된 것 위주로.\n\n"
            "그런데 일기만 쓰면 삶이 파편적이다. 그래서 **주말마다 회고(reflection)** 를 한다. 한 주의 일기를 훑고 '아, 나는 요즘 이 프로젝트에 진심이구나', '저 사람과는 거리를 두는 게 낫겠다' 같은 **큰 깨달음**을 뽑아, 그 통찰을 다시 일기에 굵은 글씨로 적어둔다. 이 회고가 흩어진 나날을 '나'라는 일관된 인격으로 꿰맨다.\n\n"
            "마지막으로, 이 일기와 회고를 바탕으로 **내일의 계획(planning)** 을 짠다. 계획은 허공이 아니라 '내가 겪고 깨달은 것' 위에 선다. 이 일기 → 회고 → 계획의 순환이 바로 Generative Agents의 인지 아키텍처다. 그리고 이 순환은 스몰빌 주민에게만 필요한 게 아니다 — 며칠에 걸쳐 한 프로젝트를 돕는 코딩 에이전트에게도 똑같이 필요한, 도메인을 초월한 '마음의 골격'이다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "Generative Agents의 심장인 recency+importance+relevance 검색과 reflection 트리거를 구현해보자. 이 두 메커니즘이 '장기기억을 가진 에이전트'의 표준 레시피다. 실무에서 커스텀 메모리 시스템을 짤 때 거의 이 골격을 변형하게 된다."
        ),
        "code": (
            "import time, numpy as np\n"
            "\n"
            "class MemoryStream:\n"
            "    def __init__(self, embed, llm):\n"
            "        self.embed, self.llm, self.mem = embed, llm, []\n"
            "\n"
            "    def observe(self, text, importance=None):\n"
            "        imp = importance or int(self.llm(  # 중요도를 LLM이 1~10으로 채점\n"
            "            f\"이 사건의 중요도를 1~10 숫자로만: {text}\"))\n"
            "        self.mem.append({\"text\": text, \"t\": time.time(),\n"
            "                         \"imp\": imp, \"vec\": self.embed(text),\n"
            "                         \"last_access\": time.time()})\n"
            "\n"
            "    def retrieve(self, query, k=5, a=1.0, b=1.0, c=1.0):\n"
            "        q, now = self.embed(query), time.time()\n"
            "        scored = []\n"
            "        for m in self.mem:\n"
            "            recency = 0.99 ** ((now - m[\"last_access\"]) / 3600)  # 시간 감쇠\n"
            "            relevance = float(np.dot(q, m[\"vec\"]))\n"
            "            score = a*recency + b*(m[\"imp\"]/10) + c*relevance    # 삼중 가중합\n"
            "            scored.append((score, m))\n"
            "        top = [m for _, m in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]\n"
            "        for m in top: m[\"last_access\"] = now                      # 꺼내면 recency 갱신\n"
            "        return top\n"
            "\n"
            "    def reflect(self, threshold=150):\n"
            "        # 최근 중요도 합이 임계치를 넘으면 고수준 통찰을 합성\n"
            "        recent = self.mem[-20:]\n"
            "        if sum(m[\"imp\"] for m in recent) < threshold:\n"
            "            return\n"
            "        facts = \"\\n\".join(m[\"text\"] for m in recent)\n"
            "        insight = self.llm(f\"다음 관찰들에서 얻을 고수준 통찰 3가지:\\n{facts}\")\n"
            "        self.observe(insight, importance=8)  # 통찰을 다시 기억으로 (재귀적 추상화)\n"
        ),
        "walkthrough": (
            "설계 급소 셋. **(1) 삼중 점수** — `recency`(지수 감쇠), `imp`(LLM 채점 중요도), `relevance`(임베딩 유사도)의 가중합으로 무엇을 떠올릴지 정한다. 가중치 a·b·c를 조절하면 '최근 위주냐, 중요도 위주냐'를 튜닝할 수 있다. **(2) last_access 갱신** — 기억을 꺼낼 때마다 recency를 리셋해, 자주 쓰는 기억이 계속 살아남게 한다(사람의 기억 강화와 같다). **(3) reflection의 재귀성** — 합성한 통찰을 `observe`로 **다시 기억에 넣는다**. 그래서 통찰이 또 다른 통찰의 재료가 되어 추상화 트리가 자란다. 이 골격이 8장 이후 그래프의 `state`로 흡수되면, '장기기억을 가진 그래프 노드'가 된다. 4장의 skill library가 '할 줄 아는 것'의 저장소라면, 이건 '겪고 깨달은 것'의 저장소다 — 목적이 다르니 둘은 보완적이다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "장기기억은 프로덕션 에이전트의 최대 난제 중 하나다. 면접관은 지원자가 '벡터DB에 다 넣고 유사도 검색'이라는 1차원적 답을 넘어, recency·importance·relevance의 균형, 기억의 추상화(reflection), 그리고 기억 폭발 시의 압축·망각 전략까지 설계할 수 있는지를 본다."
        ),
        "whatEngineersLookFor": [
            "검색을 relevance 하나가 아니라 recency·importance와 함께 균형 잡는 설계",
            "낱개 기억을 고수준 통찰로 합성(reflection)해야 일관성이 나온다는 이해",
            "reflection(정체성)과 Reflexion(성능 교정)의 목적 차이를 구분",
            "기억이 폭발할 때의 압축·요약·망각 전략을 제시",
        ],
        "redFlags": [
            "장기기억을 '벡터DB 유사도 검색'으로만 환원 (recency·importance 무시)",
            "reflection 없이 낱개 기억만 검색해 에이전트가 일관성을 잃는 문제를 못 봄",
            "기억 무한 증가에 대한 압축/망각 대책이 없음",
            "5장의 경험 메모리와 4장의 skill 메모리를 같은 것으로 취급",
        ],
        "interviewQuestions": [
            "장기기억 에이전트에서 검색을 relevance만으로 하면 무엇이 잘못되는가?",
            "reflection(고수준 통찰 합성)이 없으면 에이전트 행동에 어떤 문제가 생기는가?",
            "기억이 수십만 건으로 늘어날 때 검색 품질과 비용을 어떻게 관리하겠는가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'Generative Agents = LLM 심즈'로 기억한다. **마스터**는 이 논문의 유산이 특정 데모가 아니라 memory+reflection+planning+retrieval이라는 도메인 독립적 인지 primitive의 정립임을 알고, recency+importance+relevance 검색과 재귀적 reflection을 자기 손으로 설계·튜닝하며, 이 기억 서브시스템이 훗날 그래프 에이전트의 state로 어떻게 흡수되는지까지 연결한다."
        ),
    },
    "keyTakeaways": [
        {"title": "인지 아키텍처", "content": "memory stream + reflection + planning + retrieval의 조합이 믿을 만한 행동을 창발시킨다."},
        {"title": "삼중 검색 점수", "content": "recency + importance + relevance의 가중합으로 지금 꺼낼 기억을 고른다 — 장기기억의 표준 레시피."},
        {"title": "reflection = 추상화", "content": "낱개 관찰을 고수준 통찰로 합성해 다시 기억에 넣어, 일관된 자아·세계관을 만든다."},
        {"title": "정체성 vs 성능", "content": "5장 reflection은 일관성(정체성)을 위한 것, 3장 Reflexion은 실패 교정(성능)을 위한 것 — 목적이 다르다."},
        {"title": "도메인 독립 primitive", "content": "게임에 묶인 Voyager와 달리, 어떤 에이전트에도 적용되는 '마음의 골격'을 제시했다."},
        {"title": "재귀적 추상화", "content": "통찰이 다시 기억이 되어 또 다른 통찰의 재료가 된다 — 기억이 트리처럼 자란다."},
        {"title": "state로 흡수", "content": "이 기억 서브시스템은 8장 이후 그래프 에이전트의 명시적 state로 자연스럽게 흡수된다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 6 — Building Effective Agents (Anthropic)
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 6,
    "emoji": "📐",
    "title": "Building Effective Agents: The Loop-as-Primitive Manifesto",
    "titleKr": "Building Effective Agents — '루프가 곧 primitive'라는 선언문",
    "tldr": "Anthropic이 2024년에 '에이전트란 도구를 쓰며 환경 피드백을 루프로 도는 LLM'이라 못 박고, workflow와 agent를 가르며, 무거운 프레임워크 대신 API를 직접 쓰라고 선언했다. framework에서 loop로 넘어가는 결정적 피벗.",
    "topics": ["agent = LLM + tools in a loop", "workflow vs agent", "anti-framework 원칙", "building blocks (augmented LLM)"],
    "learningGoals": [
        "Anthropic의 agent 정의와 workflow/agent 구분을 정확히 인용·설명한다",
        "'프레임워크보다 API 직접 사용'을 권하는 이유와 그 예외를 안다",
        "다섯 가지 workflow 패턴(prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer)을 구분한다",
        "이 글이 왜 framework→loop 피벗의 선언문으로 불리는지 설명한다",
        "복잡성을 '이득이 증명될 때만' 추가하는 실무 원칙을 적용한다",
    ],
    "overview": (
        "2~5장에서 우리는 루프가 어떻게 풍부해지는지(행동·반성·기억·인지)를 봤다. 하지만 이 모든 걸 관통하는 **하나의 선언**이 2024년 말 Anthropic에서 나왔다. 바로 [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)다. 이 글은 새 알고리즘을 제안하지 않는다. 대신 업계가 어렴풋이 느끼던 것을 **명료한 언어로 못 박았다** — 그래서 loop engineering의 선언문이 됐다.\n\n"
        "핵심 문장은 이것이다.\n\n"
        "> *\"Agents are typically just LLMs using tools based on environmental feedback in a loop.\"*\n\n"
        "'just(그저)'와 'loop(루프)'가 이 문장의 무게중심이다. 에이전트는 신비한 무언가가 아니라 **그저 루프**다. 이 한 문장이 AutoGPT식 거대 프레임워크의 시대를 닫고, '루프를 직접 잘 짜는' 시대를 열었다.\n\n"
        "이 글의 두 번째 기여는 **workflow와 agent를 명확히 가른 것**, 세 번째는 **'프레임워크 대신 API를 직접 쓰라'는 반(反)프레임워크 원칙**이다. 이 장에서 우리는 앞 4개 장의 실험들이 왜 하나의 패러다임으로 수렴하는지, 그리고 그 패러다임이 프로덕션 설계에 주는 구체적 지침을 배운다. 여기가 코스의 정확한 중심축이다."
    ),
    "sections": [
        {
            "title": "'그저 루프': 에이전트의 탈신비화",
            "content": (
                "2023년 AutoGPT 열풍 때, 에이전트는 마치 스스로 사고하는 마법 상자처럼 여겨졌다. 목표만 던지면 알아서 하위 목표를 쪼개고, 도구를 쓰고, 자기를 호출하며 무한히 일하는 — 그러나 실제로는 자주 헛돌고, 비싸고, 디버깅이 불가능한 상자였다.\n\n"
                "Anthropic의 정의는 이 신비를 걷어낸다. 에이전트는 그저 'LLM이 도구를 쓰고, 환경 피드백을 받아, 루프를 도는' 것이다. 2장 ReAct의 while 루프, 그 이상도 이하도 아니다.\n\n"
                "이 탈신비화가 왜 중요한가? **디버깅과 통제가 가능해지기 때문이다.** 루프라면 나는 정확히 안다 — 언제 반복하고, 무엇을 컨텍스트에 넣고, 언제 멈추는지. 마법 상자는 열어볼 수 없지만, 내가 짠 루프는 매 줄을 들여다볼 수 있다.\n\n"
                "그리고 이 정의는 앞 장들을 하나로 묶는다. Reflexion? 루프에 반성 단계를 넣은 것. Voyager? 루프에 skill 메모리를 붙인 것. Generative Agents? 루프에 인지 서브시스템을 단 것. 전부 '루프 + α'다. 루프가 만물의 기본 단위(primitive)라는 것 — 이것이 이 글의 첫 번째 못이다."
            ),
        },
        {
            "title": "Workflow vs Agent: 스펙트럼을 긋다",
            "content": (
                "두 번째 못은 용어의 정리다. 사람들이 '에이전트'라 뭉뚱그려 부르던 것을 Anthropic은 둘로 가른다.\n\n"
                "> *\"Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage.\"*\n\n"
                "| 축 | Workflow | Agent |\n|---|---|---|\n| 흐름 통제 | 개발자의 코드 | LLM의 실시간 판단 |\n| 예측 가능성 | 높음 | 낮음 |\n| 디버깅 | 쉬움 | 어려움 |\n| 적합 상황 | 단계가 알려진 반복 작업 | 단계를 미리 알 수 없는 개방형 작업 |\n\n"
                "핵심 실무 지침은 **'대부분의 경우 workflow로 충분하다'** 는 것이다. 진짜 자율 agent가 필요한 경우는 생각보다 드물다. 개방형이고, 단계 수를 예측할 수 없고, LLM의 판단에 흐름을 맡겨야만 하는 문제 — 그럴 때만 agent를 쓴다. 나머지는 예측 가능하고 디버깅 쉬운 workflow가 낫다.\n\n"
                "이 구분이 8~10장의 그래프 엔지니어링을 예고한다. 그래프는 결국 이 스펙트럼을 **한 시스템 안에 섞는** 도구다 — 큰 뼈대는 workflow(고정 엣지)로, 특정 노드만 agent(자율 루프)로. Anthropic이 그은 이 선이 없었다면, 그래프 오케스트레이션이 무엇을 조율하는지조차 말할 수 없었을 것이다."
            ),
        },
        {
            "title": "반(反)프레임워크: API를 직접 써라",
            "content": (
                "세 번째 못이 가장 논쟁적이다. Anthropic은 **무거운 에이전트 프레임워크를 기본값으로 삼지 말라**고 권한다.\n\n"
                "> *\"We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code ... consider adding complexity only when it demonstrably improves outcomes.\"*\n\n"
                "이유는 명확하다. 프레임워크는 *\"extra layers of abstraction that can obscure the underlying prompts\"* — 밑바닥 프롬프트와 루프를 두꺼운 추상화로 가려, 무엇이 실제로 벌어지는지 알 수 없게 만든다. 1장 밀키트 비유가 그대로다.\n\n"
                "여기서 중요한 뉘앙스: 이건 '프레임워크를 절대 쓰지 마라'가 아니다. **프레임워크를 쓰더라도 그 밑에서 무슨 일이 일어나는지 이해하고, 복잡성은 이득이 증명될 때만 더하라**는 것이다. 기본값을 '프레임워크 먼저'에서 'API 직접 먼저'로 뒤집는 것이 요지다.\n\n"
                "글은 또 유용한 building block과 다섯 workflow 패턴을 제시한다. 토대는 **augmented LLM**(검색·도구·메모리로 증강된 LLM)이고, 그 위에 ① prompt chaining ② routing ③ parallelization ④ orchestrator-workers ⑤ evaluator-optimizer의 다섯 조합 패턴이 있다. 이들은 대부분 몇십 줄 코드로 짤 수 있는 것들이다 — 거대 프레임워크 없이. 이 다섯 패턴이 8~10장 그래프 구조의 원형 어휘가 된다."
            ),
        },
    ],
    "analogy": {
        "title": "오케스트라 지휘자냐, 재즈 즉흥연주냐",
        "content": (
            "음악에는 두 가지 연주 방식이 있다. **오케스트라**는 악보(predefined code path)가 모든 걸 정한다. 언제 바이올린이 들어오고 언제 팀파니가 울리는지 미리 다 적혀 있다. 지휘자는 그 악보를 정확히 실현한다. 예측 가능하고, 매번 거의 같고, 어디가 틀렸는지 악보와 대조하면 안다. 이것이 **workflow**다.\n\n"
            "**재즈 즉흥연주(jam)** 는 다르다. 코드 진행이라는 느슨한 뼈대만 있고, 그 위에서 연주자가 실시간으로 무엇을 칠지 스스로 정한다. 예측 불가능하고, 매번 다르고, 짜릿하지만 통제하기 어렵다. 이것이 **agent**다.\n\n"
            "Anthropic의 메시지는 이렇다 — **대부분의 곡은 오케스트라로 연주하는 게 낫다.** 재즈 즉흥이 필요한 순간은 정말 개방적이고 예측 불가능한 무대뿐이다. 그리고 재즈를 하겠다고 값비싼 '전자동 재즈 머신(프레임워크)'을 살 필요도 없다 — 좋은 연주자(직접 짠 루프)와 기본 악기(API)면 된다. 머신은 무슨 소리를 내는지 감춰버려서, 정작 연주가 이상할 때 고칠 수가 없다. 진짜 실력은 악보와 즉흥 사이에서 '이 곡엔 어느 쪽이 맞는가'를 고르는 판단 — 그게 8장부터 배울 그래프 엔지니어링이다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "Anthropic이 말한 '프레임워크 없이 몇 줄로'를 실증해보자. 다섯 패턴 중 가장 강력한 evaluator-optimizer를 순수 파이썬으로 구현한다. 프레임워크 임포트가 단 하나도 없다는 점, 그리고 이게 사실상 3장 Self-Refine을 workflow로 정형화한 것임에 주목하라 — 개념들이 어떻게 재조합되는지 보인다."
        ),
        "code": (
            "def augmented_llm(llm, prompt, tools=None, memory=None):\n"
            "    # '증강된 LLM' = 모든 패턴의 토대 building block\n"
            "    ctx = (memory or \"\") + prompt\n"
            "    return llm(ctx, tools=tools)\n"
            "\n"
            "def evaluator_optimizer(task, generator, evaluator, max_rounds=3):\n"
            "    \"\"\"Anthropic의 5패턴 중 하나. 생성자와 평가자를 분리한 workflow.\"\"\"\n"
            "    draft = generator(f\"작업을 수행하라:\\n{task}\")\n"
            "\n"
            "    for _ in range(max_rounds):\n"
            "        verdict = evaluator(                       # 평가자 = 별도 역할(별도 프롬프트/모델)\n"
            "            f\"작업:\\n{task}\\n\\n산출물:\\n{draft}\\n\\n\"\n"
            "            \"PASS 또는 구체적 개선 지시를 출력하라.\")\n"
            "\n"
            "        if verdict.strip().startswith(\"PASS\"):     # 예측 가능한 정지 조건\n"
            "            return draft\n"
            "        draft = generator(                         # 피드백을 반영해 재생성\n"
            "            f\"작업:\\n{task}\\n\\n이전:\\n{draft}\\n\\n개선 지시:\\n{verdict}\")\n"
            "    return draft\n"
            "\n"
            "# workflow: 흐름(생성→평가→분기)이 '코드'로 고정 = 예측 가능·디버깅 쉬움\n"
            "# 만약 generator가 스스로 도구·다음 단계를 정하게 하면 → 그 지점이 agent가 됨\n"
        ),
        "walkthrough": (
            "이 예제의 교훈 셋. **(1) 프레임워크 제로** — LangChain도 뭣도 없이, 강력한 에이전트 패턴이 30줄로 끝난다. Anthropic의 'API를 직접 써라'가 과장이 아님을 보여준다. **(2) workflow의 정체** — 흐름(생성→평가→분기→정지)이 전부 파이썬 `for`/`if`로 **코드에 고정**돼 있다. 그래서 예측 가능하고 디버깅이 쉽다. 이것이 agent(LLM이 흐름을 정함)와의 결정적 차이다. **(3) 역할 분리** — `generator`와 `evaluator`를 다른 함수(다른 프롬프트/모델)로 나눈 게 self-bias(3장)를 완화한다. 이 evaluator-optimizer는 사실상 Self-Refine을 'workflow로 승격'한 것 — 앞 장들의 아이디어가 Anthropic의 어휘로 재정리되는 순간이다. 이 다섯 패턴의 흐름을 노드-엣지로 그리면 그대로 8~10장의 그래프가 된다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "이 글은 2025~2026 에이전트 면접의 사실상 표준 교재다. 면접관은 지원자가 'workflow로 충분한가 agent가 필요한가'를 먼저 판단하는 규율을 체화했는지, 프레임워크를 맹신하지 않고 '이득이 증명될 때만 복잡성을 더하는' 절제를 아는지를 본다. 이걸 인용할 수 있으면 신뢰도가 크게 오른다."
        ),
        "whatEngineersLookFor": [
            "설계 첫머리에 workflow/agent를 판별하고, 기본값을 workflow로 두는 규율",
            "'에이전트 = 루프'라는 탈신비화된 관점으로 시스템을 분해",
            "프레임워크를 기본값으로 삼지 않고, 필요성이 증명될 때만 도입하는 절제",
            "다섯 workflow 패턴을 상황에 맞게 고르고 조합하는 능력",
        ],
        "redFlags": [
            "모든 문제를 자율 agent로 풀려 하고 workflow를 고려하지 않음",
            "'무조건 LangChain/프레임워크부터'라는 반사적 선택",
            "복잡성을 이득 증명 없이 선제적으로 쌓음(over-engineering)",
            "workflow와 agent를 구분 못 하고 '에이전트'로 뭉뚱그림",
        ],
        "interviewQuestions": [
            "주어진 문제를 workflow로 짤지 agent로 짤지 어떤 기준으로 결정하고, 그 근거는?",
            "Anthropic이 무거운 프레임워크 대신 API 직접 사용을 권하는 이유와, 그럼에도 프레임워크가 정당한 경우는?",
            "evaluator-optimizer 같은 workflow 패턴은 Self-Refine 같은 논문 기법과 무엇이 같고 다른가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 '에이전트 = 루프'라는 문장을 외운다. **마스터**는 이 글의 세 못(탈신비화·workflow/agent 구분·반프레임워크)을 앞 4개 장의 실험들을 정리하는 렌즈로 쓰고, 실제 설계에서 '여기는 workflow, 이 노드만 agent'를 근거와 함께 가르며, 이 스펙트럼 사고가 어떻게 그래프 엔지니어링으로 이어지는지까지 연결한다."
        ),
    },
    "keyTakeaways": [
        {"title": "에이전트 = 그저 루프", "content": "'LLM이 도구를 쓰며 환경 피드백을 루프로 도는 것' — 에이전트의 탈신비화가 디버깅·통제를 가능케 한다."},
        {"title": "workflow vs agent", "content": "흐름을 코드가 정하면 workflow, LLM이 실시간에 정하면 agent. 대부분은 workflow로 충분하다."},
        {"title": "반프레임워크 원칙", "content": "기본값은 'API 직접 사용'. 복잡성은 이득이 증명될 때만 더한다."},
        {"title": "추상화가 프롬프트를 가린다", "content": "무거운 프레임워크는 밑바닥 프롬프트·루프를 감춰 디버깅을 불가능하게 만든다."},
        {"title": "다섯 패턴", "content": "prompt chaining·routing·parallelization·orchestrator-workers·evaluator-optimizer — 그래프의 원형 어휘."},
        {"title": "앞 장들을 묶는 렌즈", "content": "Reflexion·Voyager·Generative Agents가 전부 '루프 + α'로 통일된다."},
        {"title": "그래프의 전제", "content": "workflow/agent 스펙트럼을 그었기에, 8~10장 그래프가 '무엇을 섞는지'를 말할 수 있게 됐다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 7 — Context Engineering
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 7,
    "emoji": "🎛️",
    "title": "Context Engineering: Curating the Window Each Turn",
    "titleKr": "Context Engineering — 매 턴 컨텍스트를 큐레이션하다",
    "tldr": "루프가 돌면 컨텍스트 창은 매 턴 바뀐다. 프롬프트 한 방을 잘 쓰는 기술(prompt engineering)에서, 매 턴 창에 '딱 필요한 정보만' 채우고 정제하는 기술(context engineering)로. 루프 엔지니어링의 쌍둥이 운영 규율.",
    "topics": ["prompt→context engineering 전환", "컨텍스트 창 큐레이션", "compaction·격리·검색", "context rot / 오염"],
    "learningGoals": [
        "prompt engineering과 context engineering의 차이를 정확히 설명한다",
        "Karpathy·Anthropic의 정의를 인용하고, 왜 루프 시대에 부상했는지 안다",
        "compaction·retrieval·isolation 등 컨텍스트 큐레이션 전략을 구현한다",
        "context rot(맥락 오염)과 lost-in-the-middle 같은 실패 모드를 진단한다",
        "5장 기억 검색과 7장 컨텍스트 조립이 어떻게 맞물리는지 이해한다",
    ],
    "overview": (
        "6장이 '에이전트 = 루프'라고 못 박은 순간, 자연스러운 후속 질문이 떠오른다. **그 루프가 매 턴 LLM에게 정확히 무엇을 보여줄 것인가?** ReAct의 scratchpad는 길어지고, Voyager의 skill library는 불어나고, Generative Agents의 memory stream은 폭발한다. 컨텍스트 창은 유한한데 넣고 싶은 건 무한하다. 이 긴장을 다루는 규율이 [Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)이다.\n\n"
        "2023년의 화두는 'prompt engineering' — 프롬프트 한 문장을 어떻게 잘 쓰느냐였다. 그런데 에이전트가 루프를 돌면 프롬프트는 한 방이 아니라 **매 턴 새로 조립되는 동적 상태**가 된다. Karpathy는 2025년, 이 변화를 포착해 새 용어에 힘을 실었다.\n\n"
        "> *\"+1 for context engineering over prompt engineering ... the delicate art and science of filling the context window with just the right information for the next step.\"*\n> — Andrej Karpathy (2025)\n\n"
        "이 장은 framework/loop/graph 삼단계와 나란한 4번째 단계가 아니다. 루프 시대가 열리면서 필연적으로 부상한 **운영 규율**이다. 아무리 루프를 잘 짜도, 매 턴 창에 쓰레기를 채우면 에이전트는 무너진다. 반대로 창을 정갈하게 큐레이션하면 같은 모델로도 훨씬 유능해진다. 이 장에서 우리는 '루프를 실제로 돌리는 손기술'을 배운다."
    ),
    "sections": [
        {
            "title": "prompt에서 context로: 무엇이 바뀌었나",
            "content": (
                "차이를 명확히 하자.\n\n"
                "**Prompt engineering** = 주로 정적인 한 번의 입력을, 원하는 출력이 나오도록 문구·예시·형식을 다듬는 기술. 대화가 한두 턴일 때 유효하다.\n\n"
                "**Context engineering** = 루프가 도는 내내, 매 턴 컨텍스트 창에 들어갈 **토큰 전체 집합**을 큐레이션·유지하는 기술. Anthropic의 정의는 이렇다.\n\n"
                "> *\"the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference.\"*\n\n"
                "무엇이 컨텍스트에 들어가는가? 단순히 사용자 메시지만이 아니다. Anthropic은 관리 대상을 이렇게 나열한다 — 시스템 지침, 도구 정의, MCP(Model Context Protocol)로 붙는 외부 소스, 검색된 문서, 그리고 누적되는 대화·도구 결과 이력. 이 **전부**를 매 턴 어떻게 구성할지가 컨텍스트 엔지니어링이다.\n\n"
                "그리고 결정적으로, 루프는 이 정보를 계속 **불린다**.\n\n"
                "> *\"An agent running in a loop generates more and more data ... this information must be cyclically refined.\"*\n\n"
                "즉 컨텍스트 엔지니어링은 프롬프트 엔지니어링의 자연스러운 후계자이되, 핵심 동사가 '작성(write)'에서 '정제(refine)'로 바뀐다. 매 턴 쌓이는 것을 주기적으로 솎아내는 것 — 그게 루프 시대의 진짜 기술이다."
            ),
        },
        {
            "title": "왜 창을 '큐레이션'해야 하나: 실패 모드",
            "content": (
                "'컨텍스트 창이 크니 그냥 다 넣으면 되지 않나?'는 순진한 생각이다. 창을 함부로 채우면 세 가지 병이 난다.\n\n"
                "**Lost in the middle** = 긴 컨텍스트에서 모델은 앞과 끝은 잘 보지만 **가운데 정보를 놓친다**. 중요한 사실을 긴 이력 한복판에 묻으면 무시된다.\n\n"
                "**Context rot(맥락 오염)** = 루프가 돌수록 쌓이는 실패한 시도, 낡은 도구 결과, 잘못된 중간 추론이 창을 오염시킨다. 모델은 이 쓰레기까지 '맥락'으로 받아들여 잘못된 방향으로 끌려간다. 3장의 self-bias가 컨텍스트 층위에서 재현되는 셈이다.\n\n"
                "**비용·지연·창 초과** = 토큰이 곧 돈이고 시간이다. 무지성으로 다 넣으면 비싸지고 느려지고, 결국 창 한계를 넘어 터진다.\n\n"
                "그래서 큐레이션이 필수다. 목표는 창을 '가득' 채우는 게 아니라, **다음 스텝에 딱 필요한 최소한의 고품질 토큰**만 남기는 것이다. Karpathy의 표현대로 'just the right information for the next step'. 더도 덜도 아니게."
            ),
        },
        {
            "title": "큐레이션 전략: compaction·retrieval·isolation",
            "content": (
                "실무에서 쓰이는 컨텍스트 큐레이션 전략은 크게 세 갈래다.\n\n"
                "**Compaction(압축)** = 긴 이력을 요약으로 접는다. 예컨대 20턴이 지나면 앞 15턴을 'LLM이 지금까지 한 일 요약' 한 단락으로 대체한다. Claude Code가 긴 세션에서 하는 게 정확히 이것이다. 낱개 사실은 잃지만 창을 되찾는다 — 5장 reflection과 같은 발상(낱개→고수준 합성)의 컨텍스트판이다.\n\n"
                "**Retrieval(검색)** = 모든 걸 창에 상주시키지 않고, 필요할 때만 꺼내온다. skill library(4장)나 memory stream(5장)을 밖에 두고, 지금 질문에 관련된 top-k만 임베딩으로 가져와 넣는다. 창은 '작업대'이고 검색 저장소는 '창고'다.\n\n"
                "**Isolation(격리)** = 관련 없는 맥락을 서브에이전트로 분리한다. 큰 작업의 한 부분을 별도 컨텍스트를 가진 하위 에이전트에게 통째로 맡기고, 그 결과 요약만 메인 창으로 돌려받는다. 메인 창은 하위 작업의 잡음에 오염되지 않는다 — 이것이 8~10장 그래프/멀티에이전트의 강력한 동기다. 각 노드가 자기만의 정갈한 컨텍스트를 갖는 것.\n\n"
                "세 전략의 공통 철학은 하나다 — **창은 유한한 고가의 자원이니, 매 턴 능동적으로 관리하라.** 이 규율 없이는 아무리 정교한 루프도 몇 턴 못 가 오염되어 무너진다."
            ),
        },
    ],
    "analogy": {
        "title": "명탐정의 화이트보드",
        "content": (
            "복잡한 사건을 쫓는 명탐정의 화이트보드를 떠올려보자. 보드는 크지만 무한하지 않다. 그리고 매 순간 이 보드에 **무엇을 붙여둘지**가 수사의 성패를 가른다.\n\n"
            "서툰 형사는 모든 단서를 다 붙인다. 관련 없는 목격담, 이미 배제된 용의자, 낡은 메모까지. 보드는 금세 빽빽해지고, 정작 중요한 단서가 그 잡동사니(context rot) 한복판에 묻혀 안 보인다(lost in the middle). 결국 형사는 엉뚱한 방향으로 수사를 끌고 간다.\n\n"
            "명탐정은 보드를 **끊임없이 큐레이션**한다. 배제된 단서는 떼어내고(정제), 여러 증언을 '피해자는 밤 10시에 살아 있었다'는 한 줄로 요약해 압축하고(compaction), 지금 안 쓰는 자료는 서류함에 넣었다가 필요할 때만 꺼낸다(retrieval). 부하에게 뒷조사를 시킬 땐 세부는 그에게 맡기고 결론만 보드에 옮긴다(isolation). 그래서 보드에는 늘 '다음 한 수에 필요한 것'만 정갈하게 남는다.\n\n"
            "LLM의 컨텍스트 창이 바로 이 화이트보드다. 그리고 루프가 돌수록 단서는 계속 쌓인다. 매 턴 이 보드를 정리하는 손기술 — 그것이 context engineering이고, 좋은 탐정(에이전트)과 나쁜 탐정을 가르는 진짜 실력이다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "컨텍스트 큐레이션의 핵심인 compaction과 예산 기반 조립을 구현해보자. 매 턴 '토큰 예산' 안에서 시스템 지침·검색된 기억·최근 이력을 우선순위대로 채우고, 넘치면 오래된 이력을 요약으로 접는다. 이 assemble 함수가 사실상 loop engineering의 심장 옆에 붙는 심장이다."
        ),
        "code": (
            "def assemble_context(system, memories, history, llm,\n"
            "                     token_budget=8000, count=len):\n"
            "    \"\"\"매 턴 컨텍스트 창을 예산 안에서 큐레이션한다.\"\"\"\n"
            "    # 1) 고정 우선순위: 시스템 지침은 항상 유지\n"
            "    parts, used = [system], count(system)\n"
            "\n"
            "    # 2) retrieval: 관련 기억 top-k (5장 검색 결과를 주입)\n"
            "    for m in memories:                      # 이미 관련도순 정렬됐다고 가정\n"
            "        if used + count(m) > token_budget * 0.4:  # 기억엔 예산의 40%까지\n"
            "            break\n"
            "        parts.append(m); used += count(m)\n"
            "\n"
            "    # 3) compaction: 최근 이력을 넣되, 오래된 건 요약으로 접기\n"
            "    recent, old = history[-6:], history[:-6]\n"
            "    if old:\n"
            "        summary = llm(f\"다음 대화를 3문장으로 요약:\\n{old}\")  # 낡은 맥락 압축\n"
            "        parts.append(f\"[이전 요약] {summary}\")\n"
            "        used += count(summary)\n"
            "\n"
            "    for turn in recent:                     # 최근 턴은 원본 유지\n"
            "        if used + count(turn) > token_budget:\n"
            "            break\n"
            "        parts.append(turn); used += count(turn)\n"
            "\n"
            "    return \"\\n\\n\".join(parts)               # 정갈하게 조립된 이번 턴의 창\n"
        ),
        "walkthrough": (
            "이 함수가 매 루프 턴마다 호출된다고 보면 된다. **(1) 우선순위 예산 배분** — 시스템 지침은 무조건, 검색 기억은 예산의 40%까지, 나머지는 최근 이력. '무엇이 밀려도 되고 무엇은 안 되는지'를 명시적으로 정하는 게 큐레이션의 핵심이다. **(2) compaction** — 오래된 이력(`old`)을 통째로 버리지 않고 3문장 요약으로 접어, 정보는 지키되 토큰은 되찾는다. 5장 reflection의 컨텍스트판이다. **(3) 최근성 보존** — 최근 6턴은 원본 유지해 세밀한 맥락을 잃지 않는다(lost-in-the-middle 완화: 중요한 최신 정보를 끝에 배치). 이 함수는 5장의 `retrieve`(무엇을 기억할지)와 짝을 이룬다 — 5장이 '창고에서 무엇을 꺼낼지'라면, 7장은 '작업대에 어떻게 배치할지'다. 그리고 isolation은 이 창 자체를 서브에이전트별로 분리하는 것, 즉 8~10장 그래프에서 노드마다 별도 `assemble_context`를 도는 것으로 확장된다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "컨텍스트 엔지니어링은 2025~2026 에이전트 실무에서 가장 뜨거운 역량이다. 면접관은 지원자가 '컨텍스트 창 = 그냥 크게 주면 됨'이라는 오해를 넘어, 유한 자원으로서의 창을 능동 관리하는지, context rot·lost-in-the-middle 같은 실패를 알고 compaction·retrieval·isolation으로 대응하는지를 본다."
        ),
        "whatEngineersLookFor": [
            "컨텍스트 창을 유한한 고가 자원으로 보고 매 턴 능동 관리하는 사고",
            "prompt engineering과 context engineering의 차이(정적 작성 vs 동적 정제)를 명확히 구분",
            "context rot·lost-in-the-middle을 인지하고 compaction·retrieval·isolation으로 대응",
            "컨텍스트 격리(서브에이전트)가 왜 그래프/멀티에이전트의 동기가 되는지 이해",
        ],
        "redFlags": [
            "'창이 크니 그냥 다 넣으면 된다'는 무지성 접근",
            "루프가 쌓는 맥락 오염(낡은 시도·잘못된 관찰)을 정제하지 않음",
            "토큰 비용·지연을 컨텍스트 설계에서 고려하지 않음",
            "context engineering을 prompt engineering과 같은 것으로 취급",
        ],
        "interviewQuestions": [
            "긴 에이전트 세션에서 컨텍스트 창이 꽉 찰 때 무엇을, 어떤 기준으로 버리거나 압축하겠는가?",
            "context rot(맥락 오염)이란 무엇이며, 루프 에이전트에서 어떻게 방지하는가?",
            "컨텍스트 격리(서브에이전트로 분리)가 유용한 상황과 그 대가는?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'context engineering = 프롬프트를 잘 쓰기'로 이해한다. **마스터**는 이것이 루프 시대에 '작성'에서 '정제'로 동사가 바뀐 운영 규율임을 알고, compaction·retrieval·isolation을 예산과 실패 모드에 근거해 설계하며, 컨텍스트 격리가 어떻게 그래프/멀티에이전트 아키텍처의 근본 동기가 되는지까지 연결한다."
        ),
    },
    "keyTakeaways": [
        {"title": "작성에서 정제로", "content": "prompt engineering(정적 작성)에서 context engineering(매 턴 동적 정제)으로 — 루프 시대의 필연."},
        {"title": "딱 필요한 것만", "content": "목표는 창을 채우는 게 아니라 '다음 스텝에 필요한 최소한의 고품질 토큰'만 남기는 것."},
        {"title": "context rot", "content": "루프가 쌓는 낡은 시도·잘못된 관찰이 창을 오염시켜 에이전트를 잘못된 방향으로 끈다."},
        {"title": "lost in the middle", "content": "긴 컨텍스트의 가운데 정보는 무시된다 — 중요한 건 끝이나 앞에 배치."},
        {"title": "compaction", "content": "오래된 이력을 요약으로 접어 정보는 지키고 토큰은 되찾는다(5장 reflection의 컨텍스트판)."},
        {"title": "retrieval + isolation", "content": "관련된 것만 검색해 넣고, 무관한 맥락은 서브에이전트로 격리 — 그래프의 동기."},
        {"title": "루프의 쌍둥이", "content": "루프를 아무리 잘 짜도 창 큐레이션이 없으면 몇 턴 못 가 무너진다. 둘은 한 몸이다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 8 — Tree of Thoughts (+ Graph of Thoughts)
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 8,
    "emoji": "🌳",
    "title": "Tree & Graph of Thoughts: From Linear Loop to Search",
    "titleKr": "Tree & Graph of Thoughts — 선형 루프에서 탐색으로",
    "tldr": "지금까지의 루프는 한 줄로 나아갔다. Tree of Thoughts는 '생각'을 트리의 노드로 삼아 여러 갈래를 탐색·평가·백트래킹하고, Graph of Thoughts는 이를 그래프로 일반화한다. loop에서 graph로 넘어가는 결정적 다리.",
    "topics": ["thought as search node", "BFS/DFS 탐색·자기평가·백트래킹", "chain→tree→graph 계보", "생각의 병합·순환"],
    "learningGoals": [
        "ToT가 CoT를 어떻게 일반화하는지(선형→트리) 설명한다",
        "self-evaluation·lookahead·backtracking을 탐색의 요소로 이해한다",
        "chain → tree → graph 추론 구조의 계보를 그린다",
        "GoT가 트리를 그래프로 확장하며 무엇을 더 표현하는지(병합·순환) 안다",
        "탐색 구조가 왜 loop→graph 전환의 다리인지 설명한다",
    ],
    "overview": (
        "2~7장의 루프에는 공통점이 하나 있다. **한 번에 한 줄로** 나아간다는 것이다. Thought 하나, Action 하나, 그리고 다음. 마치 미로에서 한 방향으로만 걷는 것과 같다. 막다른 길을 만나면? 되돌아올 방법이 마땅찮다. 여러 갈래를 동시에 저울질할 방법도 없다.\n\n"
        "[Tree of Thoughts](https://arxiv.org/abs/2305.10601)(Yao et al., NeurIPS 2023)는 이 한계를 정면으로 깬다. 아이디어는 이렇다 — **'생각(thought)'을 트리의 노드로 삼아라.** 한 지점에서 여러 다음 생각을 뻗고(분기), 각 갈래가 얼마나 유망한지 스스로 평가하고(self-evaluation), 나쁜 길은 버리고 좋은 길로 가되, 막히면 되돌아온다(backtracking). 요컨대 추론을 **고전적 탐색 문제(BFS/DFS)** 로 바꾼다.\n\n"
        "> *\"deliberate decision making by considering multiple different reasoning paths and self-evaluating choices ... as well as looking ahead or backtracking when necessary.\"*\n\n"
        "그리고 [Graph of Thoughts](https://arxiv.org/abs/2308.09687)(Besta et al., 2023)가 한 발 더 나간다. 트리는 부모-자식만 있지만, 그래프는 서로 다른 갈래의 생각을 **병합**하거나 **순환**시킬 수 있다. chain → tree → graph로 이어지는 이 계보가 바로 loop에서 graph 엔지니어링으로 넘어가는 개념적 다리다. 이 장에서 우리는 '단일 루프'가 어떻게 '구조화된 탐색'으로 펼쳐지는지를 본다."
    ),
    "sections": [
        {
            "title": "CoT의 일반화: 한 줄에서 트리로",
            "content": (
                "Chain-of-Thought는 생각을 **사슬(chain)** 로 엮는다 — A니까 B, B니까 C, 한 줄로. 문제는 A에서 갈 수 있는 길이 여럿일 때다. CoT는 그중 하나를 골라 끝까지 가버린다. 그 선택이 틀리면 통째로 실패다.\n\n"
                "Tree of Thoughts는 이 사슬을 **트리**로 편다.\n\n"
                "> *\"ToT generalizes over the popular Chain of Thought approach ... enables exploration over coherent units of text (thoughts).\"*\n\n"
                "한 노드(현재까지의 생각)에서 여러 후보 생각을 자식으로 뻗는다. 예컨대 수학 퍼즐이라면 '이 수를 먼저 더한다', '이 수를 먼저 곱한다' 등 여러 첫수를 병렬로 펼친다. 각 갈래를 조금씩 진행해보고, 유망한 쪽으로 탐색을 집중한다.\n\n"
                "핵심 부품이 **자기평가(self-evaluation)** 다. 각 중간 생각에 대해 LLM 스스로 '이 길이 정답에 가까운가?'를 점수 매긴다. 이 점수가 탐색의 나침반이 되어, BFS(넓게 훑기)나 DFS(깊게 파기)로 트리를 뒤진다. 3장 Self-Refine의 자기비평이 '한 답을 고치는' 데 쓰였다면, 여기선 '여러 갈래 중 어디로 갈지 고르는' 데 쓰인다 — 같은 자기평가 능력이 탐색의 엔진이 된다."
            ),
        },
        {
            "title": "탐색의 세 무기: 평가·전망·백트래킹",
            "content": (
                "ToT가 단일 루프보다 강한 건 세 가지 무기 덕분이다.\n\n"
                "**Self-evaluation(자기평가)** = 각 갈래의 유망함을 스스로 점수화. 어디에 자원을 쏟을지 정하는 나침반.\n\n"
                "**Lookahead(전망)** = 한 수 앞을 내다보고, 이 길이 결국 막다른 길인지 미리 가늠. 체스 선수가 몇 수 앞을 읽는 것과 같다.\n\n"
                "**Backtracking(백트래킹)** = 막다른 길에 다다르면 부모 노드로 되돌아가 다른 형제 갈래를 시도. 선형 루프에는 없는 '되돌아가기' 능력이다.\n\n"
                "이 셋이 함께 작동하면, 에이전트는 24게임·창작·계획 같은 '탐색이 필요한' 문제에서 단일 루프를 크게 앞선다. 공식 구현(princeton-nlp/tree-of-thought-llm)은 실제로 BFS/DFS를 코드로 담고 있다.\n\n"
                "대가도 분명하다. **비용**이다. 트리를 넓게 펼치면 LLM 호출이 노드 수만큼 폭증한다. 그래서 ToT는 '탐색의 이득이 비용을 정당화하는' 어려운 문제에만 쓴다. 쉬운 문제엔 단일 루프가 낫다 — 6장의 '복잡성은 이득이 증명될 때만'이 여기서도 관통한다. 탐색 폭(branching factor)과 깊이를 어떻게 제한할지가 실무 튜닝의 핵심이다."
            ),
        },
        {
            "title": "chain → tree → graph: 왜 그래프의 다리인가",
            "content": (
                "이 장이 코스에서 차지하는 자리가 중요하다. 8장은 **loop(선형)에서 graph(구조)로 넘어가는 개념적 경첩**이다.\n\n"
                "추론 구조의 계보를 보자.\n\n"
                "| 구조 | 표현 | 능력 |\n|---|---|---|\n| Chain (CoT) | 한 줄 | 순차 추론 |\n| Tree (ToT) | 분기·백트래킹 | 탐색·비교 |\n| Graph (GoT) | 병합·순환 | 갈래 통합·재사용 |\n\n"
                "Graph of Thoughts는 트리의 한계를 넘는다. 트리에서 두 갈래는 절대 다시 만나지 못한다(부모-자식뿐). 하지만 그래프에서는 **서로 다른 갈래의 생각을 하나로 병합**할 수 있다 — 예컨대 두 부분해를 합쳐 전체해를 만들거나, 한 생각을 여러 곳에서 재사용하거나, 개선 루프를 순환으로 표현한다.\n\n"
                "> Graph of Thoughts는 ToT의 트리를 명시적으로 그래프로 일반화하며, 서베이들은 ToT를 chain→tree→graph 아크 위에 배치한다.\n\n"
                "여기서 결정적 전환이 일어난다. **'생각의 그래프'는 곧 '작업의 그래프'로 자연스럽게 미끄러진다.** 노드가 '중간 생각'에서 '작업 단계'로 바뀌면, 그게 바로 9~10장의 ReWOO·LLMCompiler·LangGraph다. 8장은 추론 층위에서 그래프를 도입해, 오케스트레이션 층위의 그래프로 가는 사고의 다리를 놓는다. 루프를 명시적 구조로 펼치는 첫 걸음이 여기다."
            ),
        },
    ],
    "analogy": {
        "title": "미로 탐험: 한 길 vs 여러 길 vs 지도 다시 그리기",
        "content": (
            "미로를 빠져나가는 세 사람을 보자.\n\n"
            "**Chain-of-Thought 탐험가**는 갈림길마다 직감으로 한 길을 골라 앞만 보고 걷는다. 운이 좋으면 빠르지만, 막다른 길을 만나면 되돌아올 줄 몰라 그냥 벽 앞에 멈춘다. 한 번의 잘못된 선택이 전부를 망친다.\n\n"
            "**Tree of Thoughts 탐험가**는 다르다. 갈림길에서 여러 길을 조금씩 가보고, 각 길이 '출구에 가까워 보이는지' 평가한다(self-evaluation). 유망한 길로 나아가되, 막히면 **되돌아와(backtracking)** 다른 길을 시도한다. 갈림길 몇 개를 미리 내다보기도(lookahead) 한다. 느리지만, 어려운 미로에서 훨씬 확실하게 출구를 찾는다.\n\n"
            "**Graph of Thoughts 탐험가**는 한 발 더 간다. 여러 길을 탐험하다 '아, 이 두 통로가 사실 같은 방으로 이어지네' 하고 **경로를 합친다(병합)**. 서로 다른 탐험에서 얻은 부분 지도를 하나로 꿰매고, 이미 지나온 길을 재활용한다. 미로를 '한 줄의 발자국'이 아니라 '연결된 지도'로 이해한다.\n\n"
            "핵심은 — 미로가 단순하면 첫 번째 탐험가가 제일 빠르다. 하지만 미로가 복잡할수록, 여러 길을 저울질하고 되돌아오고 합칠 줄 아는 탐험가가 이긴다. 그리고 '발자국(선형 루프)'에서 '지도(그래프)'로 사고를 바꾸는 순간, 우리는 그래프 엔지니어링의 문턱에 선다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "Tree of Thoughts의 BFS 탐색을 구현해보자. 핵심은 '한 노드에서 여러 생각을 뻗고(expand) → 각각을 자기평가(evaluate) → 상위 b개만 남겨(prune) 다음 깊이로'를 반복하는 것이다. 선형 루프(2장)와 나란히 놓고 보면, `for` 하나가 '한 줄 전진'에서 '층별 탐색'으로 바뀐 게 보인다."
        ),
        "code": (
            "def tree_of_thoughts(problem, llm, breadth=3, depth=4, beam=2):\n"
            "    \"\"\"BFS 기반 ToT: 각 층에서 상위 beam개 생각만 유지.\"\"\"\n"
            "    frontier = [problem]                       # 현재 살아있는 부분 생각들\n"
            "\n"
            "    for d in range(depth):\n"
            "        candidates = []\n"
            "        for thought in frontier:\n"
            "            # 1) expand: 한 생각에서 여러 다음 생각을 뻗음(분기)\n"
            "            nexts = llm(f\"현재까지 추론:\\n{thought}\\n\\n\"\n"
            "                        f\"가능한 다음 단계 {breadth}가지를 제시하라.\",\n"
            "                        n=breadth)\n"
            "            for nx in nexts:\n"
            "                branch = thought + \"\\n\" + nx\n"
            "                # 2) evaluate: 이 갈래가 정답에 얼마나 가까운지 자기평가\n"
            "                score = float(llm(\n"
            "                    f\"이 추론이 문제 해결에 얼마나 유망한가? 0~10 숫자만:\\n{branch}\"))\n"
            "                candidates.append((score, branch))\n"
            "\n"
            "        if not candidates:\n"
            "            break\n"
            "        # 3) prune: 상위 beam개만 살려 다음 깊이로 (BFS + beam search)\n"
            "        candidates.sort(reverse=True)\n"
            "        frontier = [b for _, b in candidates[:beam]]\n"
            "\n"
            "        best_score = candidates[0][0]\n"
            "        if best_score >= 9.5:                  # 정지: 충분히 좋은 해 발견\n"
            "            return frontier[0]\n"
            "\n"
            "    return frontier[0]                          # 가장 유망한 갈래 반환\n"
        ),
        "walkthrough": (
            "선형 루프와의 차이를 층별로 읽자. **(1) expand = 분기** — 2장의 ReAct는 한 스텝에 행동 하나였지만, 여기선 한 생각에서 `breadth`개의 갈래를 동시에 뻗는다. 이게 트리의 '가지치기'다. **(2) evaluate = 자기평가 나침반** — 각 갈래를 LLM이 0~10으로 채점한다. 3장 자기비평 능력이 '탐색 방향키'로 재활용된다. **(3) prune = beam search** — 모든 갈래를 다 키우면 비용이 지수로 폭발하므로, 매 층에서 상위 `beam`개만 살린다. `breadth`·`depth`·`beam`을 조절하는 게 '탐색의 폭과 비용'을 튜닝하는 손잡이다. 백트래킹은 이 beam이 '더 나은 형제 갈래로 자연히 되돌아가는' 형태로 녹아 있다. 이 코드에서 노드의 내용물을 '중간 생각'에서 '실행할 작업'으로 바꾸면, 그대로 9장의 작업 DAG로 넘어간다 — 추론 그래프에서 작업 그래프로."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "ToT/GoT는 '언제 단일 루프를 넘어 탐색으로 가야 하는가'를 판단하는 감각을 시험한다. 면접관은 지원자가 탐색의 이득(어려운 문제에서의 정확도)과 비용(호출 폭증)을 저울질하는지, chain→tree→graph 계보를 이해하고 이것이 오케스트레이션 그래프로 어떻게 이어지는지를 보는지 확인한다."
        ),
        "whatEngineersLookFor": [
            "탐색이 필요한 문제와 단일 루프로 충분한 문제를 구분하는 판단",
            "self-evaluation·lookahead·backtracking을 탐색의 구성요소로 이해",
            "탐색 폭·깊이·beam으로 비용을 통제하는 실무 감각",
            "chain→tree→graph 계보가 추론에서 오케스트레이션 그래프로 이어짐을 이해",
        ],
        "redFlags": [
            "모든 문제에 ToT를 남발해 비용을 폭증시킴(단일 루프로 충분한 경우 무시)",
            "탐색의 지수적 호출 비용을 고려하지 않음",
            "self-evaluation의 신뢰성 문제(3장 self-bias)를 인지 못 함",
            "ToT를 단발 프롬프트 기법으로만 보고 탐색 구조를 못 봄",
        ],
        "interviewQuestions": [
            "어떤 문제에 Tree of Thoughts가 단일 CoT/ReAct 루프보다 확실히 유리하며, 그 대가는?",
            "ToT의 자기평가가 틀리면 탐색 전체가 어떻게 오도되며, 어떻게 완화하는가?",
            "chain·tree·graph 추론 구조의 표현력 차이는 무엇이고, 이것이 에이전트 오케스트레이션과 어떻게 연결되는가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'ToT = 여러 갈래로 생각하기'로 안다. **마스터**는 ToT가 추론을 고전 탐색(BFS/DFS+평가)으로 환원한 것이며, self-evaluation의 신뢰성과 지수 비용이라는 대가를 폭·깊이·beam으로 통제하고, chain→tree→graph 계보가 어떻게 추론 그래프에서 작업 오케스트레이션 그래프로 미끄러지는지를 코스 전체의 전환점으로 짚는다."
        ),
    },
    "keyTakeaways": [
        {"title": "생각 = 탐색 노드", "content": "ToT는 '생각'을 트리 노드로 삼아 추론을 BFS/DFS 탐색 문제로 바꾼다."},
        {"title": "CoT의 일반화", "content": "한 줄 사슬(chain)을 여러 갈래 트리로 펴서, 한 번의 잘못된 선택이 전부를 망치는 걸 막는다."},
        {"title": "탐색의 세 무기", "content": "self-evaluation(나침반)·lookahead(전망)·backtracking(되돌아가기)이 단일 루프를 넘어서게 한다."},
        {"title": "자기평가의 재활용", "content": "3장 자기비평 능력이 여기선 '어느 갈래로 갈지'를 정하는 탐색 방향키가 된다."},
        {"title": "비용의 대가", "content": "트리를 넓히면 호출이 폭증한다 — 폭·깊이·beam으로 통제하고, 어려운 문제에만 쓴다."},
        {"title": "chain→tree→graph", "content": "GoT는 트리를 그래프로 확장해 갈래의 병합·순환·재사용을 표현한다."},
        {"title": "그래프로의 다리", "content": "'생각의 그래프'는 노드를 작업으로 바꾸면 곧 '작업의 그래프'(9~10장)가 된다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 9 — ReWOO + LLMCompiler
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 9,
    "emoji": "🗂️",
    "title": "ReWOO & LLMCompiler: Plan-Then-Execute as a DAG",
    "titleKr": "ReWOO & LLMCompiler — 계획-후-실행과 병렬 DAG",
    "tldr": "ReAct식 인터리빙 루프는 매 도구 호출마다 멈춰 재프롬프트한다 — 느리고 비싸다. ReWOO는 계획을 통째로 먼저 세워 관찰과 추론을 분리하고, LLMCompiler는 작업을 DAG로 컴파일해 병렬 실행한다. 순차 루프를 명시적 작업 그래프로 대체하는 성능 논거.",
    "topics": ["Planner/Worker/Solver", "reasoning-observation 분리", "작업 DAG 병렬 실행", "지연·비용·토큰 효율"],
    "learningGoals": [
        "ReAct식 인터리빙 루프의 비효율(반복 재프롬프트, 순차성)을 정량적으로 이해한다",
        "ReWOO의 Planner-Worker-Solver 분리와 관찰-추론 디커플링을 설명한다",
        "LLMCompiler의 DAG 컴파일과 병렬 함수 호출 원리를 안다",
        "언제 순차 루프 대신 계획-후-실행 그래프가 유리한지 판단한다",
        "이 두 논문이 왜 framework→graph 전환의 성능 논거인지 설명한다",
    ],
    "overview": (
        "2장 ReAct의 루프에는 숨은 비용이 있다. 매 스텝마다 '생각 → 도구 호출 → **멈춤** → 결과 받기 → 다시 전체 맥락으로 재프롬프트'를 반복한다. 도구를 열 번 부르면 LLM을 열 번 (매번 전체 맥락과 함께) 호출한다. 순차적이라 느리고, 재프롬프트가 반복돼 토큰이 낭비된다.\n\n"
        "[ReWOO](https://arxiv.org/abs/2305.18323)(Xu et al., EMNLP 2023 Findings)의 통찰은 이렇다 — **추론과 관찰을 분리하라(Reasoning WithOut Observation).** 도구 결과를 볼 때마다 매번 추론을 다시 하지 말고, **계획 전체를 처음에 한 번에** 세운다. Planner가 서로 연결된 도구 호출 계획을 짜고, Worker들이 실행하고, Solver가 결과를 종합한다. 도구 결과를 기다리며 LLM을 반복 호출하는 낭비가 사라진다(논문은 최대 5배 토큰 효율을 노린다).\n\n"
        "[LLMCompiler](https://arxiv.org/abs/2312.04511)(Kim et al., ICML 2024)는 이를 컴파일러 원리로 밀어붙인다. 작업을 **DAG(방향성 비순환 그래프)** 로 분해하고, 의존성이 없는 작업들을 **병렬 실행**한다. ReAct 대비 최대 3.7배 빠르고 6.7배 저렴하다고 보고한다. 이 장에서 우리는 '순차 루프를 명시적 작업 그래프로 대체하면 무엇을 얻는가'를 성능의 언어로 배운다 — framework에서 graph로 넘어가는 가장 실용적인 이유다."
    ),
    "sections": [
        {
            "title": "ReAct 루프의 숨은 비용",
            "content": (
                "먼저 문제를 정확히 보자. LLMCompiler 논문은 기존 방식을 이렇게 진단한다.\n\n"
                "> *\"current methods for function calling often require sequential reasoning and acting for each function which can result in high latency, cost, and sometimes inaccurate behavior.\"*\n\n"
                "ReWOO도 같은 곳을 찌른다.\n\n"
                "> *\"an LLM reasons to call an external tool, gets halted to fetch the tool's response, and then decides the next action ... often leads to huge computation complexity from redundant prompts and repeated execution.\"*\n\n"
                "구체적으로 세 가지 병이다. **(1) 순차성** — 서로 독립적인 도구 호출(예: '파리 날씨'와 '도쿄 날씨')도 한 번에 하나씩 순서대로 한다. 병렬로 하면 될 것을. **(2) 반복 재프롬프트** — 매 스텝마다 누적된 전체 맥락을 다시 LLM에 밀어넣어, 같은 정보를 몇 번씩 재처리한다. **(3) 관찰 결합** — 추론이 관찰에 매 스텝 묶여 있어, 도구가 느리면 전체가 느려진다.\n\n"
                "이 병들은 작업이 커질수록 심해진다. 도구 호출이 스무 번인 작업이라면, ReAct는 스무 번의 순차 LLM 호출 + 스무 번의 재프롬프트다. 여기서 '루프를 미리 계획된 구조로 바꾸면 이 낭비를 없앨 수 있지 않을까?'라는 질문이 나온다. 그 답이 ReWOO와 LLMCompiler다."
            ),
        },
        {
            "title": "ReWOO: 먼저 다 계획하고, 관찰을 떼어내다",
            "content": (
                "ReWOO는 에이전트를 세 모듈로 분리한다.\n\n"
                "**Planner** = 문제를 받아, 필요한 모든 추론·도구 호출을 **처음에 한 번에** 계획한다. 이때 아직 도구를 실행하지 않는다. 대신 결과가 들어갈 자리를 변수로 남긴다 — `#E1`, `#E2` 같은 증거 변수(evidence variable). 예: \"Plan: 파리 인구를 검색 → #E1. Plan: 도쿄 인구를 검색 → #E2. Plan: #E1과 #E2를 비교.\"\n\n"
                "**Worker** = Planner가 남긴 도구 호출들을 실제로 실행해 `#E1`, `#E2`에 값을 채운다.\n\n"
                "**Solver** = 채워진 증거들을 종합해 최종 답을 낸다.\n\n"
                "> *\"a modular paradigm ReWOO (Reasoning WithOut Observation) that detaches the reasoning process from external observations, thus significantly reducing token consumption.\"*\n\n"
                "핵심은 **추론(Planner)이 관찰(Worker)로부터 분리**된 것이다. ReAct는 관찰을 볼 때마다 추론을 처음부터 다시 했지만, ReWOO는 추론을 딱 한 번(계획 시)만 한다. 그래서 재프롬프트 낭비가 사라진다. 계획이 이미 '서로 연결된 도구 호출들의 구조'라는 점에 주목하라 — 이건 사실상 작업 그래프의 초안이다. ReWOO는 순차 루프에서 명시적 계획 구조로 가는 첫 걸음이다. 물론 대가도 있다: 계획을 미리 다 세우므로, 중간 결과에 따라 경로가 크게 바뀌어야 하는 문제에는 ReAct의 적응성이 더 낫다."
            ),
        },
        {
            "title": "LLMCompiler: 작업을 DAG로 컴파일해 병렬 실행",
            "content": (
                "LLMCompiler는 ReWOO의 아이디어를 **고전 컴파일러**의 언어로 완성한다. 프로그램을 컴파일할 때 컴파일러가 명령어 간 의존성을 분석해 병렬화하듯, 도구 호출들을 그렇게 다룬다. 세 부품이다.\n\n"
                "**Function Calling Planner** = 작업을 **DAG로 분해**한다. 각 노드는 도구 호출, 엣지는 의존성이다. *\"a DAG of tasks with their inter-dependencies.\"*\n\n"
                "**Task Fetching Unit** = 의존성이 해소된(입력이 준비된) 작업을 골라 실행 큐로 보낸다.\n\n"
                "**Executor** = 서로 독립인 작업들을 **병렬로** 실행한다. *\"executing these tasks in parallel.\"*\n\n"
                "핵심은 **병렬성**이다. '파리 날씨'와 '도쿄 날씨'는 서로 의존하지 않으니 동시에 실행한다. ReAct라면 순서대로 했을 것을. 그 결과가 인상적이다.\n\n"
                "| 지표 | ReAct 대비 LLMCompiler |\n|---|---|\n| 지연(latency) | 최대 3.7배 빠름 |\n| 비용(cost) | 최대 6.7배 저렴 |\n| 정확도 | 약 9% 향상 |\n\n"
                "정확도까지 오르는 이유는, 계획을 구조적으로 세우면 ReAct식 즉흥 루프가 빠뜨리는 단계나 중복 호출이 줄기 때문이다. 여기서 결정적 인식이 온다 — **에이전트의 흐름을 DAG(그래프)로 명시하면, 성능·비용·정확도가 모두 좋아진다.** 이것이 loop에서 graph 엔지니어링으로 넘어가는 가장 강력한 실용적 논거다. 8장이 개념의 다리였다면, 9장은 성능 수치로 그 다리를 건너는 이유를 준다."
            ),
        },
    ],
    "analogy": {
        "title": "장보기: 목록 없이 vs 목록 짜고 vs 여럿이 나눠서",
        "content": (
            "저녁 파티를 위해 열 가지 재료를 사야 한다. 세 가지 방식을 보자.\n\n"
            "**ReAct 방식**은 목록 없이 마트에 가는 것이다. 진열대 앞에서 '음, 파스타를 살까? 그럼 소스도 필요하겠네. 소스를 집었으니 이제 뭐가 필요하지?' 하며 매번 처음부터 다시 생각한다. 한 품목 담을 때마다 멈춰서 전체를 재고한다. 재료 하나 사고 계산대까지 갔다가, 아 치즈를 깜빡했네 하고 되돌아온다. 열 번을 이렇게 순차로 반복하니 하루가 다 간다.\n\n"
            "**ReWOO 방식**은 집에서 **장보기 목록을 통째로 먼저** 짜는 것이다. '파스타, 소스, 치즈, 마늘…' 열 개를 다 적고(계획), 각 재료가 어느 코너에 있는지 표시해둔다(증거 변수). 마트에선 생각 없이 목록대로 담기만 하면 된다. 진열대 앞에서 매번 고민하는 낭비가 사라진다.\n\n"
            "**LLMCompiler 방식**은 목록을 짜되, **가족 세 명이 코너를 나눠** 동시에 장을 보는 것이다. 채소 담당, 유제품 담당, 정육 담당이 병렬로 움직인다. 서로 의존하지 않는 품목(채소와 우유)은 동시에 담기니 시간이 3분의 1로 준다. 물론 '고기 상태를 보고 메뉴를 바꿀지 결정' 같은 의존 관계는 순서를 지킨다(DAG).\n\n"
            "교훈은 명확하다 — 살 게 몇 개 안 되면 목록 없이도 괜찮다. 하지만 품목이 많고 서로 독립적일수록, **미리 계획하고(ReWOO) 병렬로 나누는(LLMCompiler)** 방식이 압도적으로 빠르고 싸다. 에이전트의 도구 호출도 정확히 그렇다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "LLMCompiler의 핵심인 'DAG로 계획하고 의존성 없는 작업을 병렬 실행'을 구현해보자. Planner가 작업 그래프를 만들고, 실행기는 입력이 준비된 작업들을 동시에 돌린다. 2장의 순차 while 루프와 대조하면, 흐름이 '한 줄'에서 '의존성 그래프'로 바뀐 게 핵심이다."
        ),
        "code": (
            "import asyncio\n"
            "\n"
            "# Planner가 만든 작업 DAG (실제로는 LLM이 생성). '$1' = 작업1의 결과 참조\n"
            "PLAN = {\n"
            "    \"1\": {\"tool\": \"search\", \"args\": [\"파리 인구\"],       \"deps\": []},\n"
            "    \"2\": {\"tool\": \"search\", \"args\": [\"도쿄 인구\"],       \"deps\": []},\n"
            "    \"3\": {\"tool\": \"compare\", \"args\": [\"$1\", \"$2\"],      \"deps\": [\"1\", \"2\"]},\n"
            "}\n"
            "\n"
            "async def execute_dag(plan, tools):\n"
            "    results, pending = {}, dict(plan)\n"
            "\n"
            "    while pending:\n"
            "        # 1) 의존성이 모두 해소된 작업들을 한 번에 고름 (Task Fetching Unit)\n"
            "        ready = [tid for tid, t in pending.items()\n"
            "                 if all(d in results for d in t[\"deps\"])]\n"
            "\n"
            "        async def run(tid):\n"
            "            t = pending[tid]\n"
            "            args = [results[a[1:]] if str(a).startswith(\"$\") else a\n"
            "                    for a in t[\"args\"]]          # $1 → 작업1의 실제 결과로 치환\n"
            "            return tid, await tools[t[\"tool\"]](*args)\n"
            "\n"
            "        # 2) 준비된 독립 작업들을 병렬 실행 (Executor)\n"
            "        done = await asyncio.gather(*[run(tid) for tid in ready])\n"
            "        for tid, out in done:\n"
            "            results[tid] = out\n"
            "            del pending[tid]                     # 완료 → 큐에서 제거\n"
            "\n"
            "    return results\n"
            "\n"
            "# 작업 1·2(파리·도쿄 검색)는 서로 독립 → 동시에 실행\n"
            "# 작업 3(비교)은 1·2에 의존 → 둘이 끝난 뒤 실행 (DAG가 순서를 보장)\n"
        ),
        "walkthrough": (
            "이 코드가 순차 루프를 그래프로 바꾼 정수다. **(1) 의존성 기반 스케줄링** — `ready`는 '입력이 다 준비된' 작업만 고른다. 작업 1·2는 `deps`가 비었으니 즉시 실행 가능, 작업 3은 1·2가 끝나야 한다. 흐름이 코드 순서가 아니라 **데이터 의존성**으로 결정된다. **(2) 병렬 실행** — `asyncio.gather`가 독립 작업(파리·도쿄 검색)을 동시에 돌린다. ReAct라면 순차로 두 번 걸렸을 시간이 한 번으로 준다 — 여기서 3.7배 latency 이득이 나온다. **(3) 결과 참조(`$1`)** — Planner가 남긴 변수 참조가 ReWOO의 증거 변수(`#E1`)와 같은 발상이다. 계획이 실행과 분리돼 있다. 이 `execute_dag`를 노드마다 조건 분기·상태·순환까지 갖도록 일반화하면, 그게 바로 10장의 LangGraph다. 순차 루프 → 작업 DAG → 완전한 상태 그래프로 이어지는 마지막 계단이다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "이 장은 '에이전트 성능·비용 최적화'를 다루므로, 프로덕션 경험을 검증하는 단골 주제다. 면접관은 지원자가 ReAct 루프의 순차성·재프롬프트 비용을 정량적으로 이해하는지, 그리고 계획-후-실행/DAG 병렬화가 언제 이득이고 언제 (적응성 손실로) 손해인지 균형 있게 판단하는지를 본다."
        ),
        "whatEngineersLookFor": [
            "ReAct 순차 루프의 지연·토큰 비용을 정량적으로 분석",
            "추론-관찰 분리(ReWOO)와 DAG 병렬화(LLMCompiler)의 원리를 정확히 설명",
            "독립 작업 병렬화가 latency·cost를 어떻게 줄이는지 이해",
            "계획-후-실행의 약점(중간 결과에 따른 경로 변경이 어려움)과 ReAct의 적응성을 저울질",
        ],
        "redFlags": [
            "ReAct 루프의 반복 재프롬프트 비용을 인지하지 못함",
            "모든 작업을 DAG로 미리 계획하려 함(적응이 필요한 개방형 작업에도)",
            "병렬화 가능한 독립 작업과 순차 의존 작업을 구분 못 함",
            "성능 수치(3.7x, 6.7x)를 맥락 없이 절대적으로 신뢰",
        ],
        "interviewQuestions": [
            "ReAct 스타일 인터리빙 루프가 왜 느리고 비싼지, 도구 호출이 20개인 작업으로 설명해보라.",
            "ReWOO의 계획-후-실행이 ReAct보다 불리한 상황은 언제이며 왜인가?",
            "에이전트 작업을 DAG로 병렬화할 때, 무엇이 병렬 가능하고 무엇이 순차인지 어떻게 판별하는가?",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'ReWOO/LLMCompiler = 더 빠른 에이전트'로 안다. **마스터**는 이들이 '추론-관찰 분리'와 'DAG 병렬화'로 ReAct의 순차성·재프롬프트 비용을 공격했음을 정량적으로 설명하고, 그 대가로 잃는 적응성(중간 결과 기반 경로 변경)을 인지해 '계획형 그래프 vs 적응형 루프'를 문제에 따라 고르며, 이것이 loop→graph 전환의 성능 논거임을 짚는다."
        ),
    },
    "keyTakeaways": [
        {"title": "인터리빙 루프의 비용", "content": "ReAct는 매 도구 호출마다 멈춰 전체 맥락을 재프롬프트한다 — 순차적이고 토큰 낭비가 크다."},
        {"title": "추론-관찰 분리", "content": "ReWOO는 계획을 처음에 한 번에 세워 관찰과 추론을 떼어내, 반복 재프롬프트를 없앤다."},
        {"title": "Planner-Worker-Solver", "content": "계획(증거 변수) → 실행 → 종합의 3모듈 분리가 ReWOO의 골격이다."},
        {"title": "작업 DAG", "content": "LLMCompiler는 도구 호출을 의존성 DAG로 컴파일한다 — 노드는 작업, 엣지는 의존성."},
        {"title": "병렬 실행의 이득", "content": "독립 작업을 동시에 돌려 ReAct 대비 최대 3.7x 빠르고 6.7x 저렴, 정확도 ~9% 향상."},
        {"title": "구조가 성능이다", "content": "흐름을 명시적 그래프로 만들면 성능·비용·정확도가 모두 좋아진다 — loop→graph의 실용 논거."},
        {"title": "적응성과의 트레이드오프", "content": "미리 계획하면 빠르지만, 중간 결과로 경로가 크게 바뀌는 문제엔 ReAct의 적응성이 낫다."},
    ],
})

# ────────────────────────────────────────────────────────────────
# Chapter 10 — DSPy + LangGraph
# ────────────────────────────────────────────────────────────────
chapters.append({
    "number": 10,
    "emoji": "🕸️",
    "title": "DSPy & LangGraph: The Destination of Graph Engineering",
    "titleKr": "DSPy & LangGraph — 그래프 엔지니어링의 종착점",
    "tldr": "DSPy는 LLM 파이프라인을 '최적화 가능한 계산 그래프'로 추상화해 손으로 짠 프롬프트를 컴파일러로 대체하고, LangGraph는 State·Node·Edge로 stateful 에이전트를 명시적 그래프로 짠다. framework→loop→graph 여정의 종착점이자, 2026 프로덕션의 현재.",
    "topics": ["LM 파이프라인 = 계산 그래프", "DSPy 컴파일러 최적화", "LangGraph State/Node/Edge", "loop vs graph 하이브리드"],
    "learningGoals": [
        "DSPy가 프롬프트 파이프라인을 최적화 가능한 그래프로 보는 관점을 설명한다",
        "'프롬프트를 손으로 튜닝'에서 '메트릭으로 컴파일'로의 전환을 이해한다",
        "LangGraph의 State·Node·Edge로 stateful 에이전트를 설계할 수 있다",
        "loop와 graph를 하이브리드로 조합하는 2026 프로덕션 패턴을 안다",
        "framework→loop→graph 여정 전체를 하나의 서사로 통합한다",
    ],
    "overview": (
        "여정의 마지막이다. 우리는 framework(남의 골격)에서 출발해, loop(맨손 루프)를 거쳐, graph(구조화된 흐름)에 도착했다. 이 종착점을 대표하는 두 도구가 서로 다른 층위에서 그래프 엔지니어링을 완성한다.\n\n"
        "[DSPy](https://arxiv.org/abs/2310.03714)(Khattab et al., ICLR 2024)는 **파이프라인/최적화 층위**에서 그래프를 다룬다. 핵심 선언은 이렇다 — LLM 파이프라인을 *\"text transformation graphs, i.e. imperative computational graphs where LMs are invoked through declarative modules\"* 로 추상화하라. 그리고 이 그래프를 **컴파일러가 메트릭에 맞춰 최적화**한다. 손으로 프롬프트를 갈아넣는 시대(*\"hard-coded prompt templates ... discovered via trial and error\"*)를 끝내겠다는 것이다.\n\n"
        "[LangGraph](https://www.langchain.com/langgraph)는 **오케스트레이션 층위**에서 그래프를 다룬다. State(상태)·Node(작업)·Edge(흐름)라는 저수준 primitive로 stateful 에이전트를 명시적 그래프로 짠다. 초기 LangChain의 블랙박스 Agent와 정반대로, 흐름의 모든 것을 개발자가 통제한다.\n\n"
        "이 장은 두 가지를 한다. 첫째, 그래프 엔지니어링의 두 얼굴(최적화 그래프 DSPy, 오케스트레이션 그래프 LangGraph)을 배운다. 둘째, 여정 전체를 되짚으며 2026년 현재 프로덕션이 loop와 graph를 어떻게 하이브리드로 쓰는지 — 즉 이 코스가 도착한 실무의 현재를 조망한다."
    ),
    "sections": [
        {
            "title": "DSPy: 프롬프트를 코딩하지 말고 컴파일하라",
            "content": (
                "DSPy의 출발점은 통렬한 진단이다. 우리가 LLM 파이프라인을 만드는 방식이 원시적이라는 것.\n\n"
                "> 현재의 파이프라인은 *\"hard-coded prompt templates, i.e. lengthy strings discovered via trial and error\"* 에 의존한다.\n\n"
                "프롬프트를 손으로 조금씩 바꿔가며 '이게 더 잘 되네' 하는 노가다 — 재현도 안 되고, 모델이 바뀌면 처음부터 다시다. DSPy는 이를 소프트웨어 공학의 언어로 바꾼다.\n\n"
                "핵심 추상화가 셋이다. **Signature** = 모듈의 입출력 명세('질문 → 답'처럼 무엇을 하는지 선언). **Module** = 그 signature를 구현하는 선언적 부품(`ChainOfThought`, `ReAct` 등). **Compiler(Optimizer)** = 파이프라인 전체를 주어진 메트릭에 맞춰 최적화한다 — few-shot 예시를 자동 선택하고, 프롬프트를 자동 생성/개선한다.\n\n"
                "> *\"We design a compiler that will optimize any DSPy pipeline to maximize a given metric.\"*\n\n"
                "관점의 전환이 핵심이다. 파이프라인은 *\"imperative computational graphs\"* — 즉 **계산 그래프**다. 노드는 LM 모듈, 엣지는 데이터 흐름. 이 그래프를 사람이 프롬프트로 손튜닝하는 대신, 컴파일러가 데이터와 메트릭으로 최적화한다. 손으로 짠 few-shot 대비 25~65% 향상을 보고한다. 뉘앙스: 인간의 노력이 사라지는 게 아니라 '프롬프트 문구'에서 '메트릭·모듈·데이터 설계'로 **이동**한다. 이것이 그래프 엔지니어링의 최적화 얼굴이다 — 프롬프트 엔지니어링(7장)의 자동화된 후계자."
            ),
        },
        {
            "title": "LangGraph: State·Node·Edge로 에이전트를 그리다",
            "content": (
                "LangGraph는 다른 얼굴이다. 최적화가 아니라 **오케스트레이션** — 에이전트의 흐름을 명시적 그래프로 짜는 런타임이다. 스스로를 이렇게 규정한다.\n\n"
                "> *\"a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents.\"*\n\n"
                "핵심 primitive 셋. **State** = 그래프 전체가 공유하는 상태(대화 이력, 중간 결과, 5장의 메모리가 여기 산다). **Node** = 하나의 작업 단위(LLM 호출, 도구 실행, 하위 에이전트). **Edge** = 노드 간 흐름, 조건부 분기 포함(상태를 보고 어디로 갈지 결정).\n\n"
                "결정적으로 LangGraph는 **순환(cycle)을 허용**한다. 9장의 DAG는 비순환이었지만, 진짜 에이전트는 '실패하면 되돌아가 재시도'하는 루프가 필요하다. LangGraph의 그래프는 노드로 돌아오는 엣지를 그릴 수 있어, **2장의 while 루프를 그래프의 순환으로 표현**한다. 즉 loop가 graph의 특수한 경우로 흡수된다.\n\n"
                "LangGraph의 자기positioning은 초기 프레임워크에 대한 명시적 반작용이다.\n\n"
                "> *\"Other agentic frameworks ... fall short for complex tasks ... without restricting users to a single black-box cognitive architecture.\"*\n\n"
                "(주의: 이는 벤더 자기포지셔닝이다.) 요지는 6장 Anthropic의 반프레임워크 정신과 통한다 — 블랙박스를 거부하고 저수준 통제권을 개발자에게 돌려준다. 다만 방식이 다르다. Anthropic은 '프레임워크를 걷어내고 루프를 직접'이라면, LangGraph는 '루프를 명시적 그래프 구조로 승격하되 모든 걸 통제 가능하게'다. 둘 다 블랙박스에 대한 거부라는 점에서 한 계보다."
            ),
        },
        {
            "title": "여정의 종합: 2026, loop와 graph의 하이브리드",
            "content": (
                "이제 전체 지도를 완성하자. 우리가 지나온 길은 이렇다.\n\n"
                "| 단계 | 대표 | 핵심 질문 |\n|---|---|---|\n| Framework | AutoGPT | (감춰진 마법 상자) |\n| Loop 기원 | ReAct(2장) | 어떻게 행동하는 루프를 만드나 |\n| Loop 풍부화 | Reflexion·Voyager·Gen.Agents(3~5장) | 루프에 반성·기억·인지를 어떻게 넣나 |\n| Loop 선언 | Anthropic(6장) | 에이전트 = 루프, 프레임워크를 걷어라 |\n| Loop 운영 | Context Eng.(7장) | 매 턴 창을 어떻게 큐레이션하나 |\n| Loop→Graph 다리 | ToT/GoT(8장) | 선형 루프를 탐색 구조로 |\n| Graph 성능 | ReWOO·LLMCompiler(9장) | 순차를 병렬 DAG로 |\n| Graph 종착 | DSPy·LangGraph(10장) | 파이프라인을 최적화/오케스트레이션 그래프로 |\n\n"
                "그런데 2026년 프로덕션의 진실은 '그래프가 루프를 이겼다'가 아니다. **둘은 계층으로 공존한다.** 큰 뼈대는 LangGraph식 명시적 그래프(예측 가능·디버깅 쉬움·병렬)로 짜되, 그래프의 **특정 노드 안에는 6장식 자율 루프**가 돈다. 예컨대 '코드 작성' 노드는 내부적으로 ReAct 루프(2장) + 컨텍스트 큐레이션(7장)을 돌리고, 그 노드가 실패하면 그래프 엣지가 '재계획' 노드로 되돌린다.\n\n"
                "즉 6장의 workflow/agent 스펙트럼이 그래프 안에서 **노드 단위로 실현**된다. 고정된 엣지 = workflow, 자율 루프 노드 = agent. Claude Code·Cursor 같은 실제 프로덕션 하네스가 정확히 이 하이브리드다 — 최상위는 구조화된 흐름, 말단은 유연한 루프.\n\n"
                "그래서 이 코스의 결론은 'graph가 최신이니 무조건 LangGraph'가 아니다. **framework→loop→graph는 대체가 아니라 포섭의 역사**다. 루프의 본질(2~7장)을 손으로 아는 사람만이, 그것을 언제 그래프로 펼치고(8~10장) 언제 단순한 루프로 남길지를 판단할 수 있다. 추상화 수준을 문제에 맞게 고르는 그 판단 — 그것이 loop engineering이자 graph engineering의 진짜 실력이고, 이 여정이 당신에게 남기는 것이다."
            ),
        },
    ],
    "analogy": {
        "title": "건축: 자재 규격화(DSPy)와 건물 설계도(LangGraph)",
        "content": (
            "집을 짓는 두 가지 다른 전문성을 생각해보자.\n\n"
            "**DSPy는 자재를 규격화·최적화하는 엔지니어**다. 예전엔 목수가 현장에서 나무를 눈대중으로 깎아 맞췄다(손튜닝 프롬프트). 재현도 안 되고 목수가 바뀌면 품질이 들쭉날쭉했다. DSPy는 '이 부재는 이런 하중을 견뎌야 한다'는 **명세(signature)** 만 정하면, 최적의 규격을 **자동으로 계산(compile)** 해준다. 목수의 감(感)을 공학으로 대체한다. 관심사는 '각 부품을 어떻게 최적으로 만드나'다.\n\n"
            "**LangGraph는 건물 전체의 설계도를 그리는 건축가**다. 방(node)들을 어떻게 배치하고, 복도(edge)로 어떻게 잇고, 어디서 층을 나눌지(조건 분기), 그리고 필요하면 나선 계단으로 위층에 되돌아가게(cycle) 설계한다. 건물이 어떻게 '작동'하는지 — 사람이 어떤 동선으로 흐르는지를 명시적으로 그린다. 관심사는 '전체 구조를 어떻게 조직하나'다.\n\n"
            "좋은 집은 둘 다 필요하다. 최적화된 자재(DSPy)로 튼튼한 부품을 만들고, 잘 설계된 도면(LangGraph)으로 그것들을 조직한다. 그리고 결정적으로 — **명세서만 보고 지을 수 있는 건축가는, 벽돌을 직접 쌓아본 사람**이다. 2~7장에서 맨손으로 루프를 쌓아본 사람만이, 10장의 그래프 도구를 남용하지 않고 제자리에 쓴다. 도구가 손을 대체하는 게 아니라, 손을 아는 사람이 도구를 지휘한다."
        ),
    },
    "codeExample": {
        "language": "python",
        "intro": (
            "여정의 종합을 코드로 보자. LangGraph 스타일의 명시적 그래프를 짜되, 한 노드 안에는 2·7장의 자율 루프가 돌고, 조건부 엣지로 순환(재시도)을 표현한다. 이것이 2026 프로덕션 하네스의 하이브리드 골격이다 — 최상위는 graph, 말단은 loop."
        ),
        "code": (
            "from typing import TypedDict\n"
            "\n"
            "class State(TypedDict):        # 그래프가 공유하는 상태 (5장 메모리가 여기 산다)\n"
            "    task: str\n"
            "    draft: str\n"
            "    attempts: int\n"
            "    passed: bool\n"
            "\n"
            "def plan_node(s: State) -> State:              # workflow 노드 (고정 흐름)\n"
            "    return {**s, \"task\": decompose(s[\"task\"])}\n"
            "\n"
            "def code_node(s: State) -> State:              # agent 노드 (내부는 자율 루프!)\n"
            "    # 이 노드 안에서 2장 ReAct 루프 + 7장 컨텍스트 큐레이션이 돈다\n"
            "    draft = react_loop(s[\"task\"], tools=CODE_TOOLS, ctx_budget=8000)\n"
            "    return {**s, \"draft\": draft, \"attempts\": s[\"attempts\"] + 1}\n"
            "\n"
            "def test_node(s: State) -> State:              # 그라운딩된 검증 (3장)\n"
            "    return {**s, \"passed\": run_tests(s[\"draft\"]) == \"PASS\"}\n"
            "\n"
            "def route(s: State) -> str:                    # 조건부 엣지 = 순환(cycle) 표현\n"
            "    if s[\"passed\"]:            return \"done\"\n"
            "    if s[\"attempts\"] >= 3:     return \"done\"   # 정지 조건 (loop engineering 기본기)\n"
            "    return \"code\"                              # 실패 → code 노드로 되돌아감(루프)\n"
            "\n"
            "# 그래프 조립: plan → code → test → (조건분기) → code로 순환 or 종료\n"
            "GRAPH = {\n"
            "    \"plan\": (plan_node, lambda s: \"code\"),\n"
            "    \"code\": (code_node, lambda s: \"test\"),\n"
            "    \"test\": (test_node, route),               # route가 순환/종료를 결정\n"
            "}\n"
            "\n"
            "def run(graph, state, start=\"plan\"):           # 미니 그래프 런타임\n"
            "    node = start\n"
            "    while node != \"done\":\n"
            "        fn, edge = graph[node]\n"
            "        state = fn(state)\n"
            "        node = edge(state)\n"
            "    return state\n"
        ),
        "walkthrough": (
            "이 40줄에 코스 전체가 응축돼 있다. **(1) State (5장)** — 그래프가 공유하는 명시적 상태. 메모리·중간 결과가 여기 산다. 초기 LangChain 블랙박스와 달리 모든 상태가 훤히 보인다. **(2) 노드마다 다른 자율성 (6장 스펙트럼)** — `plan_node`는 고정 흐름(workflow), `code_node`는 내부에 `react_loop`가 도는 자율 노드(agent). workflow/agent가 그래프 안에서 노드 단위로 공존한다. **(3) 조건부 엣지 = 순환 (8·9장의 종합)** — `route`가 상태를 보고 '종료냐 재시도냐'를 정한다. 9장 DAG는 비순환이었지만, 여기 `test → code` 되돌이 엣지가 2장의 while 루프를 그래프의 cycle로 승격시킨다. **(4) 정지 조건** — `attempts >= 3`, loop engineering의 처음이자 끝인 기본기가 그래프 층위에서도 그대로다. 최상위는 graph(예측 가능·디버깅), 말단 노드는 loop(유연). 이 하이브리드가 Claude Code·Cursor가 실제로 도는 방식이며, framework→loop→graph 여정이 도착한 2026의 현재다."
        ),
    },
    "industryEvaluation": {
        "overview": (
            "이 장은 코스의 종합이자 시니어 역량의 시험대다. 면접관은 지원자가 DSPy(최적화 그래프)와 LangGraph(오케스트레이션 그래프)를 층위로 구분하는지, 그리고 무엇보다 'loop와 graph를 하이브리드로 조합'하는 2026 프로덕션 감각 — 즉 문제에 맞는 추상화 수준을 고르는 판단이 있는지를 본다. '무조건 최신 그래프 프레임워크'는 오히려 미숙함의 신호다."
        ),
        "whatEngineersLookFor": [
            "DSPy(파이프라인 최적화)와 LangGraph(흐름 오케스트레이션)를 다른 층위로 구분",
            "프롬프트 손튜닝에서 메트릭 기반 컴파일로의 전환 의미를 이해",
            "loop를 graph의 cycle(순환 노드)로 흡수하는 하이브리드 설계",
            "'graph가 loop를 대체'가 아니라 '노드 단위로 공존'하는 프로덕션 현실을 이해",
        ],
        "redFlags": [
            "'최신이니 무조건 LangGraph/그래프 프레임워크'라는 유행 추종",
            "DSPy와 LangGraph를 같은 종류의 도구로 혼동",
            "모든 걸 그래프로 짜려 하고 단순 루프가 나은 경우를 못 봄",
            "framework→loop→graph를 '대체의 역사'로만 이해(포섭·공존을 놓침)",
        ],
        "interviewQuestions": [
            "DSPy와 LangGraph는 각각 무슨 문제를 푸는가? 둘을 한 시스템에서 함께 쓸 수 있는가?",
            "LangGraph에서 순환(cycle)이 필요한 이유는 무엇이며, 이것이 2장 while 루프와 어떻게 연결되나?",
            "2026년 프로덕션 에이전트에서 loop와 graph는 대체 관계인가 공존 관계인가? 근거를 들어 설명하라.",
        ],
        "masteryVsFamiliar": (
            "**표면**은 'LangGraph = 요즘 에이전트 프레임워크'로 안다. **마스터**는 DSPy(최적화 층)와 LangGraph(오케스트레이션 층)를 구분하고, loop가 graph의 cycle로 포섭되는 구조를 설계하며, 2~7장의 루프 감각 위에서만 8~10장의 그래프를 남용 없이 지휘할 수 있음을 안다. 즉 framework→loop→graph를 '대체'가 아니라 '포섭'의 역사로 읽고, 문제에 맞는 추상화 수준을 고르는 판단 자체가 핵심 역량임을 체화한다."
        ),
    },
    "keyTakeaways": [
        {"title": "두 얼굴의 그래프", "content": "DSPy는 최적화 층(파이프라인=계산 그래프), LangGraph는 오케스트레이션 층(흐름=상태 그래프)."},
        {"title": "컴파일 vs 손튜닝", "content": "DSPy는 프롬프트 손튜닝을 메트릭 기반 컴파일러 최적화로 대체한다 — 7장의 자동화된 후계자."},
        {"title": "State·Node·Edge", "content": "LangGraph는 공유 상태·작업 노드·(조건부)엣지로 stateful 에이전트를 명시적으로 그린다."},
        {"title": "loop는 graph의 cycle", "content": "LangGraph의 순환 엣지가 2장 while 루프를 그래프의 특수 경우로 흡수한다."},
        {"title": "블랙박스 거부의 계보", "content": "Anthropic(루프 직접)과 LangGraph(통제 가능한 그래프)는 방식은 달라도 블랙박스 거부라는 한 계보다."},
        {"title": "2026은 하이브리드", "content": "최상위는 graph(예측·디버깅·병렬), 말단 노드는 loop(유연) — Claude Code·Cursor의 실제 구조."},
        {"title": "포섭의 역사", "content": "framework→loop→graph는 대체가 아니라 포섭. 루프를 손으로 아는 사람만이 그래프를 제자리에 쓴다."},
    ],
})

# ────────────────────────────────────────────────────────────────
data["chapters"] = chapters
out = "/Users/1113493/Desktop/direcf.github.io/posts/loop-engineering/course_data.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"wrote {out}: {len(chapters)} chapters")
for c in chapters:
    assert set(["number","title","titleKr","tldr","topics","learningGoals","overview",
                "sections","analogy","codeExample","industryEvaluation","keyTakeaways"]) <= set(c), c["number"]
    assert len(c["sections"]) >= 3, c["number"]
print("schema OK")

