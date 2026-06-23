# Agent 学习路线与项目规划

> 目标：暑期实习。主线 = 看完 Hello-Agents（Datawhale《从零开始构建智能体》）打牢基础，
> 同时做 2~3 个 GitHub 项目作为简历作品。
> 本文分两部分：**一、技能知识**（学什么）；**二、项目**（做什么）。

---

## 一、技能知识

### 1. Hello-Agents 基础知识点（先学透）

来源：https://github.com/datawhalechina/hello-agents

| 模块 | 章节 | 关键知识点 | 自查标准 |
|---|---|---|---|
| 智能体基础 | 第1~3章 | 智能体定义/类型/范式、发展史、LLM 基础（Transformer、提示、局限） | 能讲清「什么是 AI 原生 Agent」与流程驱动 Agent 的区别 |
| 经典范式 | 第4章 | **ReAct、Plan-and-Solve、Reflection** 手写实现 | 不靠框架，能写出「思考→调工具→观察→再思考」的 loop |
| 低代码平台 | 第5章 | Coze / Dify / n8n 的使用 | 知道何时用低代码、何时用代码 |
| 框架实践 | 第6章 | AutoGen、AgentScope、LangGraph | 能用 LangGraph 编一个带分支/循环的工作流 |
| 自研框架 | 第7章 | 从 0 构建一个 Agent 框架（HelloAgents） | 理解 Agent / Tool / Memory / LLM 的抽象分层 |
| 记忆与检索 | 第8章 | 记忆系统、**RAG**、存储 | 能讲 chunking、embedding、检索为什么不准 |
| 上下文工程 | 第9章 | 持续交互的情境理解、上下文管理 | 知道怎么压缩/裁剪上下文控制成本 |
| 通信协议 | 第10章 | **MCP、A2A、ANP** | 能自己写一个 MCP server/client |
| Agentic-RL | 第11章 | 从 **SFT 到 GRPO** 训练 LLM | 理解可验证奖励、为什么用 RL 训 Agent |
| 性能评估 | 第12章 | 核心指标、基准测试、评估框架 | 能给自己的 Agent 设计一套 eval |
| 综合案例 | 第13~15章 | 智能旅行助手、DeepResearch、赛博小镇 | 能端到端跑通并改造一个 |
| 毕业设计 | 第16章 | 完整多智能体应用 | 产出一个自己的项目 |
| 求职 | Extra01 | **Agent 面试题 + 参考答案** | 全部刷完并能复述 |

### 2. 需要拓展 / 我认为重要的知识点（决定深度，面试拉开差距）

#### 原理深挖（读一手论文，能讲透）
- **Agent 范式论文**：ReAct、Reflexion、Toolformer、Tree-of-Thoughts —— 各自适用场景与失败模式。
- **Function Calling / Tool Use**：结构化输出、JSON Schema、工具选择与参数填充的可靠性。
- **Planning**：任务分解、Plan-and-Execute、ReWOO 等。

#### RAG / 记忆（实习高频）
- chunking 策略对比、embedding 选型、**rerank（重排序）**、混合检索（BM25 + 向量）。
- **GraphRAG**、记忆压缩、长期记忆 vs 工作记忆。
- 评估检索质量：召回率、命中率，「为什么这次检索不准 + 怎么优化」。

#### 工程化（区分「会调 API」和「能上线」，实习最看重）
- **可观测性**：Langfuse / LangSmith 做全链路 trace（token、延迟、成本）。
- **Eval 工程化**：自建测试集 + **LLM-as-judge** + 跑公开 benchmark，能报具体指标。
- **可靠性**：重试、超时、降级、限流；输出校验。
- **安全**：**Prompt 注入防护**、护栏（guardrails）、最小权限工具。

#### 多 Agent 与编排
- 协作模式：分工、辩论（debate）、监督者-工人（supervisor）。
- 状态机编排：**LangGraph**（工业界主流，建议吃透）、human-in-the-loop。

#### 前沿方向（聊得出新东西）
- **Agentic RL**：GRPO、verifiers、可验证环境训练。
- **Computer Use / GUI Agent / Web Agent**：视觉 + 操作。
- **Agent 自进化（Self-Evolution）**：从经验中沉淀 **技能库（skill library）** 并在使用中自我改进 —— 代表项目 **Hermes Agent**（Nous Research）的核心就是这套「闭环学习」，建议作为重点案例研究。
- **跨会话记忆 + 用户建模**：长期记忆、会话检索（如 FTS5 全文检索 + LLM 摘要）、对用户的持续建模（如 Honcho dialectic user modeling），是 Hermes 这类「越用越懂你」Agent 的关键。
- **技能标准化**：关注 **agentskills.io** 开放标准 与 **MCP** —— MCP 已成事实标准，重点掌握。
- **Agent 部署形态**：从「绑在笔记本上」到 **常驻云端 / serverless**（Docker、SSH、Modal、Daytona 等后端，IM 入口如 Telegram/Discord），是工程化新趋势。

#### 必备基本功
- Python 工程（类型标注、异步、包管理）、Git、Docker、Linux 基础。
- 至少熟练一个 LLM API（OpenAI / Anthropic / 国内模型）的调用与计费。

---

## 二、项目

> 简历策略：每个 repo 都写好 README（背景 / 架构图 / 怎么跑 / **结果指标** / 你的改进）；
> 至少 1 个有量化数字、1 个有 demo 视频；配 1~2 篇技术博客记录踩坑与优化。
> **方向首选：浏览器 / GUI 自动化（2025 最火、新颖度最高）。**

### 方向一：浏览器 / GUI 自动化（最火 · 新颖度最高 · 首选）
- **browser-use/browser-use** — 让 Agent 像人一样操作浏览器（点击 / 填表 / 抓数据）。
  新颖点：**视觉 + DOM 融合的 web agent**，大厂都在做。可复现、demo 极炫。
  https://github.com/browser-use/browser-use
- **Skyvern-AI/skyvern** — 用 LLM + 视觉自动化网页工作流（RPA 的 AI 版）。
  新颖点：**视觉驱动、抗页面改版**。很贴合企业自动化实习。
  https://github.com/Skyvern-AI/skyvern
- **browserbase/stagehand** — 可控的 AI 浏览器自动化框架，工程质量高。
  https://github.com/browserbase/stagehand

### 方向二：数据分析 / Text-to-SQL（最贴合实习岗 · 易出 demo）
- **vanna-ai/vanna** — 用 RAG 做 Text-to-SQL，自然语言查数据库。
  新颖点：**RAG-on-schema 提升 SQL 准确率**，能讲检索优化。
  https://github.com/vanna-ai/vanna
- **eosphoros-ai/DB-GPT** — 数据库领域的 Agent 平台（含 GBI、数据对话）。功能全，适合做综合应用。
  https://github.com/eosphoros-ai/DB-GPT

### 方向三：自动研究 / DeepResearch（接 Hello-Agents 第14章，延续已学）
- **assafelovic/gpt-researcher** — 自动检索多源资料、产出**带引用**的研究报告。
  新颖点：**planner + executor 多 agent + 引用溯源**。可复现、结果直观。
  https://github.com/assafelovic/gpt-researcher

### 方向四：记忆型个人助手（技术点稀缺 · 面试加分）
- **letta-ai/letta**（原 MemGPT）— 主打**长期记忆 / 自管理上下文**的 Agent。
  新颖点：**把内存管理做成 OS 式分页**，论文级思想，面试很能聊。
  https://github.com/letta-ai/letta

### 方向五：科研 / 数据科学自动化（最高级 · 最唬人 · 门槛偏高）
- **microsoft/RD-Agent** — 微软「研发自动化」Agent，自动做数据科学 / 量化因子挖掘与迭代。
  新颖点：**自动假设 → 实验 → 迭代闭环**，简历上极有分量。
  https://github.com/microsoft/RD-Agent

### 方向六：自进化 Agent · 个人 AI 助手（最新颖 · 故事性最强 · 重点推荐）
- **NousResearch/hermes-agent** — Nous Research 的**自进化 AI Agent**。
  新颖点（简历最能讲的一组）：
  - **闭环学习（self-improving loop）**：复杂任务后**自动生成技能**，技能在使用中持续自我改进；
  - **跨会话记忆 + 用户建模**：FTS5 会话全文检索 + LLM 摘要做跨会话回忆，集成 Honcho 对用户建模，「越用越懂你」；
  - **模型无关**：一条 `hermes model` 切换 OpenRouter / Nous Portal / NVIDIA NIM / 本地端点等，无锁定；
  - **随处运行**：本地 / Docker / SSH / Modal / Daytona，serverless 空闲近乎零成本，能跑在 $5 VPS；
  - **多入口 + 子代理**：Telegram/Discord/Slack 等统一网关，可派生隔离子代理并行、用 Python 经 RPC 调工具。
  可复现：MIT 协议，一行 `curl ... | bash`（Win 有 PowerShell 安装），文档 https://hermes-agent.nousresearch.com/docs/ 。
  https://github.com/NousResearch/hermes-agent

  > 这块正好把上面「Agent 自进化 + 跨会话记忆 + 技能库 + 云端部署」的前沿知识点串成一个能落地、能演示、能讲故事的作品，和「方向四 记忆型助手（letta）」是同一条主线、互为补充。

---

## 三、推荐执行组合

| 项目 | 作用 | 选型 | 产出 |
|---|---|---|---|
| 项目 1（炫 demo） | 展示能落地、效果惊艳 | **browser-use**（首选方向一） | 「自动完成某项网上任务」的 agent + demo 视频 |
| 项目 2（有数字） | 展示有量化结果 | vanna（SQL 准确率）/ 或复现 SWE-agent | README 里报具体指标 + 你的改进 |
| 项目 3（有深度，可选） | 展示原理深度 | **Hermes Agent**（自进化/记忆）/ letta / RD-Agent / gpt-researcher | 技术博客讲清新颖点 |

**最小可行组合：browser-use（惊艳）+ vanna（扎实、可量化）。** 若想再加一个「有深度、最新颖」的，首推 **Hermes Agent**。

### 结合 Hermes Agent 的进阶玩法（把三个方向打通）
用 **Hermes Agent 作为「会自进化的 Agent 运行时」**，把前面的方向挂上去，做成一个有记忆、会成长的个人助手：
- **挂工具**：给 Hermes 接 `browser-use`（网页操作）/ `vanna`（查数据库）作为技能或工具，让它能真正「上网做事、查数据」。
- **用它的闭环**：跑几个真实任务，观察它**自动沉淀出哪些技能**、技能如何自我改进 —— 截图/录屏作为 demo 与博客素材。
- **可量化的实验（简历亮点）**：设计一组任务，对比「开启 vs 关闭自进化/记忆」的**成功率、所需轮数、token 成本**，用数字证明自进化的收益。
- **换模型实验**：用 `hermes model` 切换不同模型（含开源/国内模型），对比效果与成本。
- 简历写法示例：「基于 Hermes Agent 构建可自进化的个人助手，集成 browser-use/vanna 工具；通过 A/B 实验验证自进化使任务成功率从 X% 提升到 Y%、平均轮数下降 Z%。」

执行要点：
1. 先复现跑通 → 再做**自己的扩展**（换模型 / 加工具 / 改策略 / 迁到自己的场景）。
2. 每个项目都补上 **eval + trace**，能报「准确率 / 成功率 / 成本」。
3. README + demo + 博客三件套，简历写法：「复现 X，在 Y 上达到 Z%，并新增 W 功能」。
