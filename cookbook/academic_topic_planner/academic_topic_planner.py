"""
学术研究选题规划工具 (Academic Research Topic Planning Tool)
=============================================================

面向硕士生、博士生和高校教师的智能选题规划系统。
An AI-powered topic planning system for graduate students and university faculty.

Features / 功能:
    - 学术热点分析 (Academic Hotspot Analysis)
    - 研究缺口检测 (Research Gap Detection)
    - 多维选题树生成 (Multi-dimensional Topic Tree)
    - 选题建议报告 + 研究价值评级 (Topic Report + Research Value Rating)

Usage / 使用方法:
    python academic_topic_planner.py --field "计算机视觉" --context "关注医学图像分析"
    python academic_topic_planner.py --field "Natural Language Processing" --lang en
    python academic_topic_planner.py --field "量子计算" --output report.md

Requirements:
    pip install anthropic pydantic rich typer
    export ANTHROPIC_API_KEY=your_api_key
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Data Models  数据模型定义
# ─────────────────────────────────────────────────────────────────────────────

class TrendDirection(str, Enum):
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"


class DifficultyLevel(str, Enum):
    ENTRY = "入门级"
    INTERMEDIATE = "中级"
    ADVANCED = "高级"
    CUTTING_EDGE = "前沿探索"


class ImpactLevel(str, Enum):
    LOW = "低影响"
    MEDIUM = "中等影响"
    HIGH = "高影响"
    TRANSFORMATIVE = "颠覆性影响"


class GapType(str, Enum):
    METHODOLOGICAL = "方法论空白"
    THEORETICAL = "理论空白"
    EMPIRICAL = "实证空白"
    APPLICATION = "应用空白"
    INTERDISCIPLINARY = "跨学科空白"


class Recommendation(str, Enum):
    HIGHLY_RECOMMENDED = "强烈推荐"
    RECOMMENDED = "推荐"
    CONDITIONAL = "条件推荐"
    NOT_RECOMMENDED = "暂不推荐"


class ResearchHotspot(BaseModel):
    """学术热点条目 / Academic hotspot entry"""
    topic: str = Field(description="热点主题名称")
    trend: TrendDirection = Field(description="研究趋势方向")
    importance_score: int = Field(ge=1, le=10, description="重要性评分 1-10")
    yearly_paper_growth: str = Field(description="年论文增长趋势描述")
    key_venues: list[str] = Field(description="主要发表期刊/会议")
    leading_institutions: list[str] = Field(description="主要研究机构")
    core_keywords: list[str] = Field(description="核心关键词")
    description: str = Field(description="热点背景和意义说明")


class ResearchGap(BaseModel):
    """研究缺口条目 / Research gap entry"""
    gap_title: str = Field(description="研究缺口标题")
    description: str = Field(description="缺口的详细描述")
    related_hotspots: list[str] = Field(description="关联的研究热点")
    gap_type: GapType = Field(description="缺口类型")
    difficulty: DifficultyLevel = Field(description="研究难度")
    potential_impact: ImpactLevel = Field(description="潜在学术影响")
    suggested_approaches: list[str] = Field(description="建议的研究路径")
    prerequisites: list[str] = Field(description="所需前置知识/技能")


class TopicNode(BaseModel):
    """选题树节点 / Topic tree node"""
    id: str = Field(description="唯一标识符")
    title: str = Field(description="选题标题")
    level: int = Field(ge=0, le=3, description="层级: 0=根, 1=主干, 2=分支, 3=叶节点")
    parent_id: Optional[str] = Field(default=None, description="父节点ID")
    dimension: str = Field(description="所属研究维度，如：理论/方法/应用/跨学科")
    novelty_score: int = Field(ge=1, le=10, description="创新性评分")
    feasibility_score: int = Field(ge=1, le=10, description="可行性评分")
    impact_score: int = Field(ge=1, le=10, description="预期影响力评分")
    time_to_publish_months: int = Field(description="预计发表周期（月）")
    description: str = Field(description="选题说明")
    research_questions: list[str] = Field(description="核心研究问题")
    suggested_methods: list[str] = Field(description="建议研究方法")
    target_venues: list[str] = Field(description="目标发表期刊/会议")


class ResearchValueRating(BaseModel):
    """研究价值评级 / Research value rating"""
    topic_id: str = Field(description="对应选题节点ID")
    topic_title: str = Field(description="选题标题")
    stars: int = Field(ge=1, le=5, description="综合星级评定 1-5星")
    novelty: int = Field(ge=1, le=5, description="创新性 1-5星")
    feasibility: int = Field(ge=1, le=5, description="可行性 1-5星")
    impact: int = Field(ge=1, le=5, description="学术影响力 1-5星")
    funding_potential: int = Field(ge=1, le=5, description="获批基金可能性 1-5星")
    publication_potential: int = Field(ge=1, le=5, description="高水平发表可能性 1-5星")
    recommendation: Recommendation = Field(description="综合推荐意见")
    strengths: list[str] = Field(description="优势分析")
    risks: list[str] = Field(description="风险与挑战")
    reasoning: str = Field(description="综合评价理由")


class TopicSuggestionReport(BaseModel):
    """完整的选题建议报告 / Complete topic suggestion report"""
    field: str = Field(description="研究领域")
    context: str = Field(description="用户背景与需求")
    generated_at: str = Field(description="报告生成时间")
    executive_summary: str = Field(description="执行摘要")
    hotspots: list[ResearchHotspot] = Field(description="学术热点分析结果")
    research_gaps: list[ResearchGap] = Field(description="研究缺口分析结果")
    topic_tree: list[TopicNode] = Field(description="多维选题树节点列表")
    value_ratings: list[ResearchValueRating] = Field(description="研究价值评级列表")
    top_recommendations: list[str] = Field(description="最优先推荐的选题ID列表")
    conclusion: str = Field(description="总结与下一步行动建议")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates  提示词模板
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位顶尖的学术研究顾问，专门帮助硕士生、博士生和高校教师进行学术选题规划。
你拥有广博的跨学科知识，能深入分析学术热点、识别研究缺口、构建多维选题框架，并提供专业的研究价值评估。

你的分析需要：
1. 基于最新的学术动态和研究趋势
2. 结合用户的实际情况和需求
3. 提供具体、可操作的选题建议
4. 用量化指标支撑评估结论

请始终以严谨、客观、专业的态度进行分析，确保建议的学术质量和实践价值。"""

HOTSPOT_ANALYSIS_PROMPT = """请对研究领域「{field}」进行全面的学术热点分析。

用户背景与需求：{context}

请识别该领域当前最重要的 5-8 个学术热点，每个热点需要：
1. 准确描述当前研究趋势（上升/稳定/下降）
2. 评估其学术重要性（1-10分）
3. 列举主要的期刊/会议发表渠道
4. 说明领先研究机构
5. 提炼核心关键词
6. 深入分析热点的背景和意义

返回 JSON 格式的热点列表，字段结构如下：
{schema}

请确保分析覆盖以下维度：
- 基础理论研究热点
- 方法论创新热点
- 应用落地热点
- 跨学科交叉热点"""

GAP_DETECTION_PROMPT = """基于「{field}」领域的以下学术热点分析结果：

{hotspots_summary}

用户背景：{context}

请深入分析该领域存在的 5-8 个重要研究缺口。每个研究缺口需要：
1. 清晰定位缺口的性质（方法论/理论/实证/应用/跨学科空白）
2. 详细描述缺口存在的原因和现状
3. 评估填补该缺口的研究难度
4. 预估成功填补后的学术影响力
5. 提出可行的研究路径建议
6. 说明所需前置知识和技能

返回 JSON 格式的缺口列表，字段结构如下：
{schema}

优先识别那些：
- 现有研究明显不足的领域
- 方法可迁移但尚未应用的场景
- 不同子领域之间的连接空白
- 理论与实践之间的鸿沟"""

TOPIC_TREE_PROMPT = """基于「{field}」领域的热点分析和缺口检测结果，请构建一个多维选题树。

学术热点摘要：
{hotspots_summary}

研究缺口摘要：
{gaps_summary}

用户背景：{context}

请构建包含 3-4 个主要维度的多层选题树，每个维度下包含 2-4 个具体选题：
- 维度1：理论研究类选题
- 维度2：方法创新类选题
- 维度3：应用研究类选题
- 维度4：综合交叉类选题（可选）

每个叶节点选题需要：
1. 设置唯一ID（格式：dim1_topic1, dim2_topic3 等）
2. 评估创新性、可行性、影响力（1-10分）
3. 估计发表周期（月）
4. 提出 2-4 个核心研究问题
5. 建议研究方法
6. 推荐目标期刊/会议

返回 JSON 格式的节点列表（包含所有层级节点），字段结构如下：
{schema}

树的根节点 ID 为 "root"，层级关系通过 parent_id 字段表达。"""

RATING_PROMPT = """请对「{field}」领域的以下候选选题进行研究价值评级：

{topics_summary}

用户背景：{context}

请为每个叶节点选题（level=3）提供详细的研究价值评级，评级维度包括：
1. 综合星级（1-5星）
2. 创新性（1-5星）：该选题的学术创新程度
3. 可行性（1-5星）：在有限资源/时间内完成的可能性
4. 学术影响力（1-5星）：对领域发展的贡献度
5. 获批基金可能性（1-5星）：申请国家自然科学基金等的成功率
6. 高水平发表可能性（1-5星）：发表于CCF-A/SCI-Q1的概率

同时提供：
- 综合推荐意见（强烈推荐/推荐/条件推荐/暂不推荐）
- 优势分析（3-5点）
- 风险与挑战（2-3点）
- 综合评价理由（3-5句话）

返回 JSON 格式的评级列表，字段结构如下：
{schema}"""

REPORT_SYNTHESIS_PROMPT = """请基于以下分析结果，为「{field}」领域的研究选题规划生成完整报告。

用户背景：{context}

热点分析（{hotspot_count}个热点）、研究缺口（{gap_count}个缺口）、
选题树（{topic_count}个候选选题）和价值评级已完成。

请生成：
1. 执行摘要（200-300字）：总结该领域研究现状和最核心建议
2. 最优先推荐的选题 ID 列表（选3-5个综合评分最高、最适合用户的选题）
3. 总结与行动建议（300-400字）：包括：
   - 推荐的学习路径
   - 建议的时间规划（短期3个月/中期1年/长期3年）
   - 合作资源建议
   - 数据/算力/实验资源准备建议

以 JSON 格式返回，包含字段：executive_summary, top_recommendations（ID列表）, conclusion"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Planner Class  主规划器类
# ─────────────────────────────────────────────────────────────────────────────

class AcademicTopicPlanner:
    """
    学术研究选题规划器

    基于 Claude claude-opus-4-6 大模型，通过多步骤分析生成专业的学术选题建议报告。
    """

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True, mock: bool = False):
        self.mock = mock
        self.model = "claude-opus-4-6"
        self.verbose = verbose
        self._steps_completed = 0
        self._total_steps = 4
        if not mock:
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    def _log(self, msg: str, emoji: str = "📌") -> None:
        if self.verbose:
            print(f"\n{emoji} {msg}", flush=True)

    def _log_progress(self, step_name: str) -> None:
        self._steps_completed += 1
        pct = int(self._steps_completed / self._total_steps * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        if self.verbose:
            print(f"\n[{bar}] {pct}% — {step_name}", flush=True)

    def _stream_and_collect(self, prompt: str, step_label: str) -> str:
        """Stream a Claude response and collect full text, with live output."""
        self._log(f"正在{step_label}...", "🔍")
        if self.mock:
            return self._get_mock_response(step_label)
        collected = []

        with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            in_text = False
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "text":
                        in_text = True
                        if self.verbose:
                            print("  ", end="", flush=True)
                    else:
                        in_text = False
                elif event.type == "content_block_delta":
                    if in_text and event.delta.type == "text_delta":
                        chunk = event.delta.text
                        collected.append(chunk)
                        if self.verbose:
                            print(chunk, end="", flush=True)
                elif event.type == "content_block_stop":
                    if in_text and self.verbose:
                        print()  # newline after each text block

        return "".join(collected)

    def _get_mock_response(self, step_label: str) -> str:
        """Return rich mock JSON for demonstration without API calls."""
        if self.verbose:
            print("  🎭 [Mock 演示数据] ", end="", flush=True)
            for _ in range(5):
                time.sleep(0.2)
                print("·", end="", flush=True)
            print(" ✓", flush=True)

        if "热点" in step_label:
            data = [
                {
                    "topic": "大规模视觉基础模型",
                    "trend": "rising",
                    "importance_score": 10,
                    "yearly_paper_growth": "年均增长 200%+，2023年后呈爆发式增长",
                    "key_venues": ["CVPR", "NeurIPS", "ICLR"],
                    "leading_institutions": ["Google DeepMind", "OpenAI", "清华大学", "北京大学"],
                    "core_keywords": ["ViT", "CLIP", "SAM", "视觉大模型", "预训练"],
                    "description": "以 ViT、CLIP、SAM 为代表的视觉基础模型已成为领域核心范式，通过大规模预训练获得强大的泛化能力，正在重塑计算机视觉的研究格局。现有研究表明，视觉基础模型在零样本迁移、开放词汇识别等任务上展现出惊人的通用性。",
                },
                {
                    "topic": "多模态大语言模型",
                    "trend": "rising",
                    "importance_score": 9,
                    "yearly_paper_growth": "年均增长 300%+，为近三年最热方向",
                    "key_venues": ["ACL", "NeurIPS", "CVPR", "ICLR"],
                    "leading_institutions": ["Meta AI", "Google", "上海AI Lab", "浙江大学"],
                    "core_keywords": ["VLM", "GPT-4V", "LLaVA", "图文对齐", "视觉问答"],
                    "description": "视觉语言模型（VLM）将图像理解与自然语言处理深度融合，实现了图文对话、视觉推理等复杂任务，是当前多模态 AI 研究的核心方向，已在 OCR、图表理解、视觉问答等任务上超越专用模型。",
                },
                {
                    "topic": "医学图像智能分析",
                    "trend": "rising",
                    "importance_score": 8,
                    "yearly_paper_growth": "年均增长 80%，应用场景持续扩大",
                    "key_venues": ["MICCAI", "Medical Image Analysis", "Radiology: AI"],
                    "leading_institutions": ["Mayo Clinic", "斯坦福医学院", "中山大学", "华南理工大学"],
                    "core_keywords": ["病灶分割", "放射组学", "病理切片", "辅助诊断", "联邦学习"],
                    "description": "AI 辅助医学影像分析在癌症早筛、放射科辅助诊断等领域取得重大突破，政策利好（国家医疗AI战略）叠加数据积累加速商业化，是产学研结合最紧密的方向之一。",
                },
                {
                    "topic": "视频理解与时序建模",
                    "trend": "stable",
                    "importance_score": 8,
                    "yearly_paper_growth": "年均增长 40%，稳步发展",
                    "key_venues": ["ICCV", "ECCV", "CVPR", "ACM MM"],
                    "leading_institutions": ["MIT CSAIL", "微软亚洲研究院", "商汤科技", "北京大学"],
                    "core_keywords": ["时序建模", "动作识别", "视频生成", "Mamba", "长视频理解"],
                    "description": "随着短视频和流媒体爆炸式增长，视频理解成为刚需。长时序建模、高效视频 Transformer 等问题仍有大量待解决空间，Mamba 等状态空间模型为长视频处理带来新希望。",
                },
                {
                    "topic": "三维场景理解与重建",
                    "trend": "rising",
                    "importance_score": 7,
                    "yearly_paper_growth": "年均增长 120%，受 NeRF/3DGS 驱动",
                    "key_venues": ["CVPR", "SIGGRAPH", "ICCV", "3DV"],
                    "leading_institutions": ["ETH Zurich", "卡内基梅隆大学", "浙江大学", "腾讯AI Lab"],
                    "core_keywords": ["NeRF", "3D Gaussian Splatting", "点云", "SLAM", "数字孪生"],
                    "description": "NeRF 和 3D Gaussian Splatting 引发三维重建技术革命，在机器人、自动驾驶、元宇宙等领域应用价值巨大，动态场景重建和实时渲染仍是关键挑战。",
                },
                {
                    "topic": "高效模型与轻量化部署",
                    "trend": "stable",
                    "importance_score": 7,
                    "yearly_paper_growth": "年均增长 30%，工程需求驱动",
                    "key_venues": ["MLSys", "CVPR", "ICLR", "DAC"],
                    "leading_institutions": ["ARM Research", "英特尔实验室", "清华大学", "旷视科技"],
                    "core_keywords": ["知识蒸馏", "量化", "剪枝", "NAS", "边缘计算"],
                    "description": "随着 AI 模型规模不断膨胀，如何在资源受限设备（手机、嵌入式端）上高效部署成为产业痛点。轻量化与高效推理研究兼具学术价值和产业需求，国内大量企业有合作意愿。",
                },
            ]
            return json.dumps(data, ensure_ascii=False)

        if "缺口" in step_label:
            data = [
                {
                    "gap_title": "小样本/零样本医学影像标注",
                    "description": "医学图像标注依赖专业医师，成本极高，标注数据严重匮乏。现有大模型在小样本场景下泛化能力不足，如何利用基础模型实现低成本精准医学影像分析仍是核心挑战。现有方法在跨机构、跨设备数据上性能急剧下降。",
                    "related_hotspots": ["医学图像智能分析", "大规模视觉基础模型"],
                    "gap_type": "应用空白",
                    "difficulty": "高级",
                    "potential_impact": "高影响",
                    "suggested_approaches": [
                        "利用 SAM 等通用分割模型进行医学影像迁移微调",
                        "设计医学领域专用视觉提示学习策略",
                        "构建半监督/自监督医学预训练框架",
                    ],
                    "prerequisites": ["深度学习基础", "医学图像处理", "迁移学习"],
                },
                {
                    "gap_title": "视觉模型跨域泛化能力不足",
                    "description": "当前视觉模型在训练分布外数据上性能急剧下降，域偏移问题严重制约实际部署。从自然图像到医学、遥感、工业检测等特殊域的迁移仍需大量标注数据微调，缺乏通用的跨域泛化理论框架。",
                    "related_hotspots": ["大规模视觉基础模型", "高效模型与轻量化部署"],
                    "gap_type": "方法论空白",
                    "difficulty": "高级",
                    "potential_impact": "高影响",
                    "suggested_approaches": [
                        "研究基于不变性表征学习的跨域泛化理论",
                        "设计基于因果推断的域自适应方法",
                        "利用视觉大模型先验知识指导域泛化",
                    ],
                    "prerequisites": ["表示学习", "域自适应", "因果推断基础"],
                },
                {
                    "gap_title": "多模态推理的可解释性缺失",
                    "description": "多模态大模型在视觉推理任务中表现优异，但内部决策机制高度不透明，无法解释为何产生特定回答。这严重阻碍了在医疗诊断、法律取证等高风险领域的落地，监管机构明确要求 AI 决策可解释。",
                    "related_hotspots": ["多模态大语言模型", "医学图像智能分析"],
                    "gap_type": "理论空白",
                    "difficulty": "前沿探索",
                    "potential_impact": "颠覆性影响",
                    "suggested_approaches": [
                        "开发基于注意力的视觉-语言对齐解释方法",
                        "构建多模态推理链标注数据集与评估基准",
                        "研究因果干预下的多模态模型行为分析",
                    ],
                    "prerequisites": ["可解释AI (XAI)", "注意力机制", "多模态模型架构"],
                },
                {
                    "gap_title": "动态场景实时三维重建",
                    "description": "NeRF 和 3DGS 在静态场景重建上取得突破，但对动态场景（人体运动、流体、变形物体）的实时重建面临计算效率与质量的双重瓶颈，当前方法在动态场景下速度慢 10-100 倍，距实用化还有较大差距。",
                    "related_hotspots": ["三维场景理解与重建"],
                    "gap_type": "方法论空白",
                    "difficulty": "高级",
                    "potential_impact": "中等影响",
                    "suggested_approaches": [
                        "设计时空解耦的动态神经辐射场表达",
                        "利用 3DGS 流式更新实现动态场景实时渲染",
                        "探索神经隐式表达与显式点云的混合方法",
                    ],
                    "prerequisites": ["计算机图形学", "NeRF/3DGS", "CUDA 编程"],
                },
                {
                    "gap_title": "长视频的高效时序语义理解",
                    "description": "现有视频模型受限于计算资源，只能处理数秒至数分钟的视频片段，无法有效理解数小时长视频中的因果关系、事件链条和长程依赖，制约了视频内容安全审核、影视分析等应用的上限。",
                    "related_hotspots": ["视频理解与时序建模", "多模态大语言模型"],
                    "gap_type": "理论空白",
                    "difficulty": "高级",
                    "potential_impact": "高影响",
                    "suggested_approaches": [
                        "研究基于状态空间模型（Mamba）的长序列视频编码",
                        "设计分层记忆机制压缩视频时序语义信息",
                        "探索检索增强（RAG）的视频理解新范式",
                    ],
                    "prerequisites": ["序列建模", "Transformer/Mamba", "视频处理"],
                },
                {
                    "gap_title": "隐私保护的视觉联邦学习",
                    "description": "医疗、金融等敏感领域积累了大量视觉数据，但数据孤岛和隐私法规（GDPR、《数据安全法》）阻碍数据共享。现有联邦学习方法在视觉任务中通信效率低、收敛慢、异质性数据处理能力差。",
                    "related_hotspots": ["医学图像智能分析", "高效模型与轻量化部署"],
                    "gap_type": "应用空白",
                    "difficulty": "中级",
                    "potential_impact": "高影响",
                    "suggested_approaches": [
                        "设计通信高效的视觉模型联邦聚合策略",
                        "研究差分隐私与视觉大模型的融合方法",
                        "探索基于提示学习的轻量化联邦微调方案",
                    ],
                    "prerequisites": ["联邦学习", "差分隐私", "分布式训练"],
                },
            ]
            return json.dumps(data, ensure_ascii=False)

        if "选题" in step_label or "树" in step_label:
            data = [
                # ── Root ──
                {"id": "root", "title": "学术研究选题规划", "level": 0, "parent_id": None, "dimension": "根节点", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "多维选题总纲", "research_questions": [], "suggested_methods": [], "target_venues": []},
                # ── Dim 1: 理论研究 ──
                {"id": "dim1", "title": "理论研究维度", "level": 1, "parent_id": "root", "dimension": "理论研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "聚焦基础理论创新与机制解析", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim1_sub1", "title": "视觉表示学习理论", "level": 2, "parent_id": "dim1", "dimension": "理论研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "研究视觉特征表示的理论基础", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim1_sub1_t1", "title": "跨模态语义对齐的几何理论研究", "level": 3, "parent_id": "dim1_sub1", "dimension": "理论研究", "novelty_score": 8, "feasibility_score": 6, "impact_score": 9, "time_to_publish_months": 18, "description": "从信息几何视角建立视觉与语言特征空间对齐的统一理论框架，解释为何预训练能产生跨模态对齐能力，为多模态模型设计提供理论指导。", "research_questions": ["视觉-语言对齐为何在大规模预训练后自然涌现？", "如何用黎曼流形理论刻画跨模态语义空间？", "最优传输理论能否为模态对齐提供更优化算法？"], "suggested_methods": ["信息几何分析", "最优传输理论", "流形学习", "消融实验"], "target_venues": ["ICLR", "NeurIPS", "TPAMI"]},
                {"id": "dim1_sub1_t2", "title": "视觉大模型涌现能力的理论解析", "level": 3, "parent_id": "dim1_sub1", "dimension": "理论研究", "novelty_score": 9, "feasibility_score": 5, "impact_score": 10, "time_to_publish_months": 24, "description": "系统研究视觉大模型在特定规模阈值后出现的涌现能力（零样本泛化、跨任务迁移等），探索其背后的统计规律和可预测性。", "research_questions": ["视觉模型涌现能力的规模临界点在哪？", "如何预测哪些能力会在缩放中涌现？", "涌现能力是否可通过数据配比人为诱导？"], "suggested_methods": ["缩放律分析", "神经切线核理论", "大规模对照实验"], "target_venues": ["NeurIPS", "ICML", "ICLR"]},
                {"id": "dim1_sub2", "title": "可解释视觉AI机制", "level": 2, "parent_id": "dim1", "dimension": "理论研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "研究视觉模型决策机制的可解释性", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim1_sub2_t1", "title": "基于因果推断的多模态模型决策机制研究", "level": 3, "parent_id": "dim1_sub2", "dimension": "理论研究", "novelty_score": 9, "feasibility_score": 7, "impact_score": 9, "time_to_publish_months": 15, "description": "运用结构因果模型（SCM）分析多模态大模型的视觉推理过程，识别模型依赖的虚假相关性，构建可审计的因果推理路径，为高风险决策场景提供理论保障。", "research_questions": ["多模态模型的视觉推理是否存在捷径学习（Shortcut Learning）？", "如何用干预实验识别模型决策的真实因果变量？", "因果图能否指导更鲁棒的多模态架构设计？"], "suggested_methods": ["结构因果模型", "反事实推理", "注意力可视化", "人工标注对比实验"], "target_venues": ["CVPR", "NeurIPS", "IJCAI"]},
                {"id": "dim1_sub2_t2", "title": "视觉注意力机制的认知科学对齐研究", "level": 3, "parent_id": "dim1_sub2", "dimension": "理论研究", "novelty_score": 7, "feasibility_score": 7, "impact_score": 7, "time_to_publish_months": 12, "description": "对比人类视觉认知（眼动追踪数据）与模型注意力分布的异同，研究如何让模型的关注区域更接近人类认知规律，提升决策的可信度与可解释性。", "research_questions": ["Transformer 注意力与人类眼动模式有多大差异？", "强制对齐认知注意力是否能提升模型鲁棒性？", "认知对齐训练是否可作为通用正则化策略？"], "suggested_methods": ["眼动仪数据采集", "注意力可视化", "知识蒸馏", "人类评估实验"], "target_venues": ["CVPR", "Cognitive Science", "CHI"]},
                # ── Dim 2: 方法创新 ──
                {"id": "dim2", "title": "方法创新维度", "level": 1, "parent_id": "root", "dimension": "方法创新", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "聚焦新颖算法设计与技术突破", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim2_sub1", "title": "高效视觉架构设计", "level": 2, "parent_id": "dim2", "dimension": "方法创新", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "设计计算高效的视觉模型架构", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim2_sub1_t1", "title": "面向边缘端的轻量化视觉基础模型", "level": 3, "parent_id": "dim2_sub1", "dimension": "方法创新", "novelty_score": 7, "feasibility_score": 9, "impact_score": 8, "time_to_publish_months": 10, "description": "针对移动端/嵌入式设备，设计参数量 <50M 但保留基础模型泛化能力的轻量化视觉模型，结合结构化剪枝、混合精度量化和知识蒸馏，实现实时推理。", "research_questions": ["如何在大幅压缩参数的同时保留基础模型的零样本能力？", "混合精度量化对视觉基础模型的影响规律是什么？", "能否设计端云协同的自适应推理框架？"], "suggested_methods": ["知识蒸馏", "结构化剪枝", "INT4/INT8量化", "NAS"], "target_venues": ["CVPR", "MLSys", "ICLR"]},
                {"id": "dim2_sub1_t2", "title": "基于 Mamba 的长视频高效时序建模", "level": 3, "parent_id": "dim2_sub1", "dimension": "方法创新", "novelty_score": 8, "feasibility_score": 8, "impact_score": 8, "time_to_publish_months": 10, "description": "将状态空间模型（SSM/Mamba）引入视频理解，设计兼顾长时序建模与计算效率的视频骨干网络，突破 Transformer 二次复杂度对长视频的限制。", "research_questions": ["Mamba 能否完全替代 Transformer 在视频任务上的地位？", "如何设计视觉-时序混合 Mamba 架构？", "SSM 在密集时序预测任务（光流、深度）上的潜力如何？"], "suggested_methods": ["状态空间模型", "因果卷积", "滑动窗口注意力", "基准测试"], "target_venues": ["CVPR", "ICCV", "NeurIPS"]},
                {"id": "dim2_sub2", "title": "数据高效学习方法", "level": 2, "parent_id": "dim2", "dimension": "方法创新", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "研究小数据/无标注场景下的视觉学习方法", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim2_sub2_t1", "title": "基于视觉提示工程的少样本医学分割", "level": 3, "parent_id": "dim2_sub2", "dimension": "方法创新", "novelty_score": 8, "feasibility_score": 9, "impact_score": 9, "time_to_publish_months": 8, "description": "利用 SAM 等通用视觉基础模型的交互式分割能力，设计适用于医学影像的视觉提示策略（点击、边界框、文本描述），实现 1-5 张标注图像即可完成精准器官/病灶分割。", "research_questions": ["哪类视觉提示策略对医学影像分割最有效？", "如何设计自动化提示生成模块减少人工交互？", "少样本医学分割的下界性能是多少？"], "suggested_methods": ["提示学习", "SAM微调", "元学习", "半监督学习"], "target_venues": ["MICCAI", "MedIA", "CVPR"]},
                {"id": "dim2_sub2_t2", "title": "无监督跨数据集目标检测域适应", "level": 3, "parent_id": "dim2_sub2", "dimension": "方法创新", "novelty_score": 7, "feasibility_score": 8, "impact_score": 7, "time_to_publish_months": 9, "description": "无需目标域标注，通过自监督伪标签生成和对抗训练，实现目标检测模型从合成数据/自然图像到医疗/工业/卫星图像的高效无监督迁移。", "research_questions": ["自训练伪标签的质量如何自动评估和筛选？", "视觉基础模型能否作为无监督域适应的通用特征提取器？", "如何应对目标域类别分布不均衡问题？"], "suggested_methods": ["对抗训练", "自训练", "均值教师", "域对齐"], "target_venues": ["CVPR", "ICCV", "ECCV"]},
                # ── Dim 3: 应用研究 ──
                {"id": "dim3", "title": "应用研究维度", "level": 1, "parent_id": "root", "dimension": "应用研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "聚焦高价值场景落地与产业化", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim3_sub1", "title": "智能医疗影像", "level": 2, "parent_id": "dim3", "dimension": "应用研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "AI 驱动的医学影像分析应用研究", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim3_sub1_t1", "title": "联邦学习驱动的多中心病理切片协作诊断", "level": 3, "parent_id": "dim3_sub1", "dimension": "应用研究", "novelty_score": 8, "feasibility_score": 8, "impact_score": 9, "time_to_publish_months": 12, "description": "针对医院间病理数据无法共享的痛点，设计隐私保护的联邦学习框架，使多个医疗中心在不交换原始数据的条件下共同训练高精度癌症病理切片分析模型。", "research_questions": ["联邦训练如何应对医院间染色方式、设备型号差异导致的数据异质性？", "差分隐私机制对病理模型精度的影响有多大？", "如何设计适合超大分辨率病理切片的联邦通信协议？"], "suggested_methods": ["联邦学习（FedAvg/FedProx）", "差分隐私", "多实例学习", "WSI分析"], "target_venues": ["MICCAI", "Nature Machine Intelligence", "MedIA"]},
                {"id": "dim3_sub1_t2", "title": "大模型辅助的放射科报告自动生成", "level": 3, "parent_id": "dim3_sub1", "dimension": "应用研究", "novelty_score": 7, "feasibility_score": 9, "impact_score": 8, "time_to_publish_months": 9, "description": "基于多模态大模型（如 LLaVA-Med），构建能从 CT/MRI/X-Ray 影像自动生成结构化放射科报告的 AI 系统，重点解决幻觉问题与临床术语一致性。", "research_questions": ["如何有效减少医学报告生成中的幻觉（虚构发现）？", "临床术语约束解码能否提升报告的专业准确性？", "报告生成模型如何进行临床有效性评估？"], "suggested_methods": ["指令微调", "RLHF", "医学知识图谱约束", "临床评估协议"], "target_venues": ["Radiology: AI", "MICCAI", "ACL"]},
                {"id": "dim3_sub2", "title": "自动驾驶感知", "level": 2, "parent_id": "dim3", "dimension": "应用研究", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "面向自动驾驶场景的视觉感知研究", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim3_sub2_t1", "title": "基于 3DGS 的自动驾驶闭环仿真数据生成", "level": 3, "parent_id": "dim3_sub2", "dimension": "应用研究", "novelty_score": 9, "feasibility_score": 7, "impact_score": 9, "time_to_publish_months": 12, "description": "利用 3D Gaussian Splatting 从真实驾驶视频中快速重建可编辑的三维场景，通过插入天气、障碍物、行人等元素生成大量逼真的长尾场景仿真数据，解决自动驾驶数据不足问题。", "research_questions": ["如何实现 3DGS 场景中动态物体（行人、车辆）的解耦和可控生成？", "仿真生成数据与真实数据的感知性能迁移差距有多大？", "如何设计闭环评估框架验证数据生成质量？"], "suggested_methods": ["3D Gaussian Splatting", "NeRF", "场景图生成", "域随机化"], "target_venues": ["CVPR", "ICCV", "CoRL"]},
                # ── Dim 4: 跨学科 ──
                {"id": "dim4", "title": "跨学科交叉维度", "level": 1, "parent_id": "root", "dimension": "跨学科交叉", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "聚焦视觉与其他领域的交叉融合", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim4_sub1", "title": "视觉 × 生命科学", "level": 2, "parent_id": "dim4", "dimension": "跨学科交叉", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "计算机视觉与生物医学的深度融合", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim4_sub1_t1", "title": "多模态视觉增强的蛋白质功能预测", "level": 3, "parent_id": "dim4_sub1", "dimension": "跨学科交叉", "novelty_score": 9, "feasibility_score": 6, "impact_score": 9, "time_to_publish_months": 18, "description": "将蛋白质三维结构可视化与序列信息相结合，利用视觉-语言大模型对蛋白质功能进行多模态理解与预测，在 AlphaFold 基础上进一步提升功能注释精度。", "research_questions": ["视觉表示能否捕捉蛋白质结构中的功能相关特征？", "如何设计蛋白质结构的有效视觉编码方式？", "多模态融合相比单一序列模型的增益在哪些任务上最显著？"], "suggested_methods": ["图神经网络", "多模态预训练", "蛋白质语言模型", "分子对接模拟"], "target_venues": ["NeurIPS", "Nature Methods", "ICLR"]},
                {"id": "dim4_sub2", "title": "视觉 × 具身智能", "level": 2, "parent_id": "dim4", "dimension": "跨学科交叉", "novelty_score": 1, "feasibility_score": 1, "impact_score": 1, "time_to_publish_months": 0, "description": "视觉感知与机器人操作的融合", "research_questions": [], "suggested_methods": [], "target_venues": []},
                {"id": "dim4_sub2_t1", "title": "视觉-语言-行动大模型驱动的机器人操作", "level": 3, "parent_id": "dim4_sub2", "dimension": "跨学科交叉", "novelty_score": 10, "feasibility_score": 5, "impact_score": 10, "time_to_publish_months": 24, "description": "构建统一的视觉-语言-行动（VLA）大模型，使机器人能够理解自然语言指令、感知三维环境并规划精细操作动作，实现从厨房烹饪到手术辅助的通用机器人操作。", "research_questions": ["如何设计高效的具身感知-规划-执行统一架构？", "模拟到现实（Sim-to-Real）的视觉感知迁移如何降低噪声干扰？", "如何让 VLA 模型快速适应新工具和新任务？"], "suggested_methods": ["模仿学习", "强化学习", "视觉-语言对齐", "3D 场景理解"], "target_venues": ["CoRL", "RSS", "NeurIPS"]},
            ]
            return json.dumps(data, ensure_ascii=False)

        if "价值" in step_label or "评级" in step_label:
            data = [
                {"topic_id": "dim1_sub1_t1", "topic_title": "跨模态语义对齐的几何理论研究", "stars": 4, "novelty": 4, "feasibility": 3, "impact": 5, "funding_potential": 4, "publication_potential": 5, "recommendation": "推荐", "strengths": ["填补多模态对齐理论空白，学术价值极高", "理论成果可指导工程实践，应用前景广", "该方向顶会接受率相对较高"], "risks": ["理论研究周期长，产出节奏慢", "需要较强的数学背景（微分几何/最优传输）"], "reasoning": "该选题具有重要的理论价值，从几何视角解释多模态对齐是领域的前沿问题。虽然难度较高，但成果一旦产出影响力巨大，适合有扎实数学功底的研究者。建议与信息论/优化方向的研究者合作。"},
                {"topic_id": "dim1_sub1_t2", "topic_title": "视觉大模型涌现能力的理论解析", "stars": 3, "novelty": 5, "feasibility": 3, "impact": 5, "funding_potential": 3, "publication_potential": 4, "recommendation": "条件推荐", "strengths": ["极高创新性，处于研究最前沿", "若成功，影响力可比肩 Scaling Law 等里程碑工作"], "risks": ["研究难度极高，不确定性大", "需要大量计算资源验证缩放律，成本昂贵", "短期内难以发表，不适合有毕业压力的同学"], "reasoning": "该选题创新性最高但风险也最大，属于高风险高回报类型。适合有充足资源和时间、不急于毕业的研究者。建议先在小规模模型上探索规律再扩展。"},
                {"topic_id": "dim1_sub2_t1", "topic_title": "基于因果推断的多模态模型决策机制研究", "stars": 5, "novelty": 5, "feasibility": 4, "impact": 5, "funding_potential": 5, "publication_potential": 5, "recommendation": "强烈推荐", "strengths": ["可解释 AI 是监管重点，国家基金高度支持", "多模态 + 因果推断是两个热点的交叉，发表机会多", "与医疗/法律应用结合紧密，产业合作机会大", "研究链条清晰：理论→实验→应用验证"], "risks": ["因果推断方法迁移到视觉领域有一定技术门槛", "实验设计需要构建高质量对比数据集"], "reasoning": "这是当前最具综合价值的选题之一。可解释 AI 受到国家政策和监管层面的高度重视，多模态因果推断兼具理论深度和应用价值，发表于 CVPR/NeurIPS 的概率极高。强烈推荐作为博士论文核心方向。"},
                {"topic_id": "dim1_sub2_t2", "topic_title": "视觉注意力机制的认知科学对齐研究", "stars": 3, "novelty": 4, "feasibility": 4, "impact": 3, "funding_potential": 3, "publication_potential": 4, "recommendation": "条件推荐", "strengths": ["跨学科视角新颖，认知科学 × AI 交叉热度上升", "眼动数据相对易获取，实验成本低"], "risks": ["期刊接受度分散（CV 会议 vs 认知科学期刊），影响力难聚焦", "与工程应用关联较弱，基金申请有难度"], "reasoning": "该方向有一定新颖性，适合对认知科学有兴趣的研究者，但商业价值和基金申请竞争力偏弱。建议作为辅助方向而非主攻方向。"},
                {"topic_id": "dim2_sub1_t1", "topic_title": "面向边缘端的轻量化视觉基础模型", "stars": 4, "novelty": 4, "feasibility": 5, "impact": 4, "funding_potential": 4, "publication_potential": 4, "recommendation": "推荐", "strengths": ["产业需求极其旺盛，容易获得企业合作和数据支持", "工程实现路径清晰，可行性高", "顶会工程类 paper 接受空间大（CVPR/MLSys）"], "risks": ["竞争激烈，工业界（ARM、高通、华为）资源优势明显", "纯工程贡献可能被认为缺乏理论深度"], "reasoning": "轻量化视觉模型是工业界刚需，研究可行性高且容易发表。建议结合理论分析（如量化误差分析）提升学术深度，同时与国内手机厂商建立合作，形成产学研闭环。"},
                {"topic_id": "dim2_sub1_t2", "topic_title": "基于 Mamba 的长视频高效时序建模", "stars": 4, "novelty": 4, "feasibility": 4, "impact": 4, "funding_potential": 4, "publication_potential": 5, "recommendation": "推荐", "strengths": ["Mamba 是 2024 年后最热架构，顶会对此方向接受度高", "长视频理解需求真实存在，应用场景丰富", "发表周期短（8-12月），适合在读生"], "risks": ["Mamba 是否会被新架构取代存在不确定性", "需要一定 CUDA 工程优化能力"], "reasoning": "Mamba 架构在序列建模领域势头强劲，将其系统引入视频理解是很自然的方向，顶会竞争激烈但发表机会大。适合有一定工程能力的研究者，可快速产出论文。"},
                {"topic_id": "dim2_sub2_t1", "topic_title": "基于视觉提示工程的少样本医学分割", "stars": 5, "novelty": 4, "feasibility": 5, "impact": 5, "funding_potential": 5, "publication_potential": 5, "recommendation": "强烈推荐", "strengths": ["痛点明确，标注成本高是医学AI落地最大障碍", "SAM 提供了强大的基础，站在巨人肩膀上", "MICCAI/MedIA 对此需求迫切，发表概率极高", "可与医院直接合作获取数据，形成研究闭环", "国家自然科学基金医学影像专项支持力度大"], "risks": ["SAM 医学迁移效果可能不稳定，需要大量调试", "数据获取需要医院伦理审批，周期较长"], "reasoning": "这是综合评分最高的选题之一，兼具极高可行性和重要应用价值。建议尽快与附属医院建立合作关系获取数据，先在公开数据集（MSD、BRATS）上验证方法，再扩展到私有数据。强烈推荐作为硕士论文方向。"},
                {"topic_id": "dim2_sub2_t2", "topic_title": "无监督跨数据集目标检测域适应", "stars": 3, "novelty": 3, "feasibility": 4, "impact": 3, "funding_potential": 3, "publication_potential": 3, "recommendation": "条件推荐", "strengths": ["研究思路成熟，实验容易复现", "有多个标准基准数据集可直接使用"], "risks": ["该方向已相对成熟，创新空间收窄", "与新兴基础模型方法相比竞争力偏弱"], "reasoning": "该方向可行性较高但创新空间有限，建议结合视觉基础模型（如用 CLIP 特征替代传统骨干网络）寻找新的技术切入点，否则发表于顶会有一定难度。"},
                {"topic_id": "dim3_sub1_t1", "topic_title": "联邦学习驱动的多中心病理切片协作诊断", "stars": 5, "novelty": 4, "feasibility": 4, "impact": 5, "funding_potential": 5, "publication_potential": 5, "recommendation": "强烈推荐", "strengths": ["响应《数据安全法》政策需求，基金申请优先级极高", "多中心合作数据质量远超单中心，研究结论更可靠", "同时贡献联邦学习和医学图像两个领域，发表机会倍增", "与临床紧密结合，成果具有明确社会效益"], "risks": ["多中心伦理审批周期长（可能 6-12 个月）", "各医院 IT 基础设施差异大，联邦框架部署有挑战"], "reasoning": "这是医学 AI 领域最具战略价值的方向之一。数据隐私是医疗 AI 规模化的核心障碍，联邦学习是目前最可行的解决方案，政策和产业双重驱动。建议先与 2-3 家医院建立合作，小范围验证框架再扩展。"},
                {"topic_id": "dim3_sub1_t2", "topic_title": "大模型辅助的放射科报告自动生成", "stars": 4, "novelty": 3, "feasibility": 5, "impact": 4, "funding_potential": 4, "publication_potential": 4, "recommendation": "推荐", "strengths": ["临床痛点明确，放射科医生工作量超负荷", "基础模型（LLaVA-Med 等）已提供强力基座", "发表周期短，可快速产出成果"], "risks": ["幻觉问题在医疗场景是致命风险，需要严格验证", "已有较多竞争工作（CheXpert、BioVil-T 等）"], "reasoning": "该方向可行性高，有清晰的技术路线。关键差异化点应聚焦在幻觉减少和临床一致性上，建议设计专门的医学报告可信度评估指标，这本身就是一个创新贡献。"},
                {"topic_id": "dim3_sub2_t1", "topic_title": "基于 3DGS 的自动驾驶闭环仿真数据生成", "stars": 4, "novelty": 5, "feasibility": 4, "impact": 5, "funding_potential": 4, "publication_potential": 5, "recommendation": "推荐", "strengths": ["解决自动驾驶长尾场景数据稀缺的核心痛点", "3DGS 速度极快（实时渲染），具备工程价值", "顶会（CVPR/ICCV）和自动驾驶专项会议均有旺盛需求"], "risks": ["需要真实驾驶数据作为重建输入，数据获取有门槛", "动态物体（行人、车辆）的 3DGS 建模还不成熟"], "reasoning": "自动驾驶数据生成是产业界投入最大的方向之一（Tesla、Waymo 均有大量布局），3DGS 技术成熟度恰好在近 1-2 年达到可用水平，时机极佳。建议从静态场景编辑起步，逐步攻克动态物体建模。"},
                {"topic_id": "dim4_sub1_t1", "topic_title": "多模态视觉增强的蛋白质功能预测", "stars": 4, "novelty": 5, "feasibility": 3, "impact": 5, "funding_potential": 4, "publication_potential": 5, "recommendation": "推荐", "strengths": ["生命科学 × AI 是国家重点支持方向", "成果可在 Nature/Science 系列期刊发表", "AlphaFold 之后，蛋白质功能预测是下一个重大挑战"], "risks": ["生物医学领域门槛高，需要跨学科合作伙伴", "数据质量参差不齐，实验验证需要生物实验室支持"], "reasoning": "这是高难度高回报的跨学科选题，适合有生物信息学合作资源的研究者。建议寻找生命科学学院的合作教授，分工明确：CS 方向负责模型设计，生物方向负责数据获取和实验验证。"},
                {"topic_id": "dim4_sub2_t1", "topic_title": "视觉-语言-行动大模型驱动的机器人操作", "stars": 3, "novelty": 5, "feasibility": 3, "impact": 5, "funding_potential": 4, "publication_potential": 4, "recommendation": "条件推荐", "strengths": ["具身智能是未来 10 年最重要的 AI 方向之一", "创新性极高，是真正的前沿探索"], "risks": ["需要机器人实验平台，硬件成本高（50-200万）", "Sim-to-Real 迁移难度大，调试周期极长", "不适合有明确毕业时间限制的研究者"], "reasoning": "VLA 驱动的机器人操作是 AI 领域的终极目标之一，但研究门槛极高，需要机器人硬件平台和充足的时间。建议仅在有成熟机器人实验室支撑的情况下考虑，否则可将重心放在纯仿真环境研究。"},
            ]
            return json.dumps(data, ensure_ascii=False)

        # Synthesis (executive_summary + top_recommendations + conclusion)
        data = {
            "executive_summary": (
                "当前该领域正处于由大规模预训练模型驱动的范式变革期，研究机遇与挑战并存。"
                "热点分析显示，视觉基础模型、多模态大语言模型和医学图像分析是增速最快的三大方向，"
                "年均论文增长均超 80%。研究缺口集中在小样本医学标注、模型可解释性和跨域泛化三个核心问题上，"
                "具有明确的填补价值。选题树从理论、方法、应用、跨学科四个维度构建了 13 个候选选题，"
                "综合评级显示「少样本医学分割」「因果可解释性」和「联邦学习病理诊断」三个方向"
                "兼具高创新性、高可行性和强基金支持，是最优先推荐的方向。"
            ),
            "top_recommendations": ["dim2_sub2_t1", "dim3_sub1_t1", "dim1_sub2_t1"],
            "conclusion": (
                "基于以上分析，给出以下行动建议：\n\n"
                "【短期（0-3个月）】\n"
                "• 精读推荐选题的 10 篇核心文献，掌握领域现状\n"
                "• 在公开数据集（如 MSD、BRATS）上复现 1-2 篇代表性 Baseline\n"
                "• 与导师确认最终选题方向，制定论文计划\n\n"
                "【中期（3-12个月）】\n"
                "• 联系附属医院或合作单位，启动数据共享/合作协议\n"
                "• 提交第一篇会议论文（目标：MICCAI 或 CVPR）\n"
                "• 准备国家自然科学基金青年项目或面上项目申请材料\n\n"
                "【长期（1-3年）】\n"
                "• 构建完整的研究体系，发表 2-3 篇 CCF-A/SCI-Q1 论文\n"
                "• 将核心方法开源，建立学术影响力\n"
                "• 探索与医疗 AI 企业（联影、推想等）的成果转化合作\n\n"
                "资源建议：GPU 算力可申请高校超算中心免费额度；数据获取优先使用公开数据集（PhysioNet、TCIA），"
                "再逐步扩展到合作医院私有数据；代码框架推荐基于 MMDetection/MONAI 二次开发，降低工程成本。"
            ),
        }
        return json.dumps(data, ensure_ascii=False)

    def _extract_json(self, raw: str) -> str:
        """Extract JSON block from markdown-wrapped or raw response."""
        raw = raw.strip()
        # Handle ```json ... ``` wrapping
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.rindex("```")
            return raw[start:end].strip()
        if "```" in raw:
            start = raw.index("```") + 3
            end = raw.rindex("```")
            return raw[start:end].strip()
        # Find first [ or {
        for i, ch in enumerate(raw):
            if ch in ("{", "["):
                return raw[i:]
        return raw

    def _parse_list(self, raw: str, model_class, step_label: str) -> list:
        """Parse a JSON array response into a list of Pydantic model instances."""
        try:
            json_str = self._extract_json(raw)
            data = json.loads(json_str)
            if isinstance(data, dict):
                # Sometimes wrapped in a key like {"hotspots": [...]}
                data = next(iter(data.values()))
            return [model_class.model_validate(item) for item in data]
        except Exception as e:
            self._log(f"解析{step_label}时出错: {e}", "⚠️")
            self._log(f"原始内容片段: {raw[:300]}", "📄")
            return []

    # ── Step 1: Hotspot Analysis ──────────────────────────────────────────────

    def analyze_hotspots(self, field: str, context: str) -> list[ResearchHotspot]:
        schema_example = json.dumps(ResearchHotspot.model_json_schema(), ensure_ascii=False, indent=2)
        prompt = HOTSPOT_ANALYSIS_PROMPT.format(
            field=field, context=context or "无特殊要求", schema=schema_example
        )
        raw = self._stream_and_collect(prompt, "分析学术热点")
        results = self._parse_list(raw, ResearchHotspot, "学术热点")
        self._log_progress(f"学术热点分析完成，识别到 {len(results)} 个热点")
        return results

    # ── Step 2: Gap Detection ─────────────────────────────────────────────────

    def detect_research_gaps(
        self, field: str, context: str, hotspots: list[ResearchHotspot]
    ) -> list[ResearchGap]:
        hotspots_summary = "\n".join(
            f"- 【{h.topic}】趋势:{h.trend.value} 重要性:{h.importance_score}/10 — {h.description[:100]}…"
            for h in hotspots
        )
        schema_example = json.dumps(ResearchGap.model_json_schema(), ensure_ascii=False, indent=2)
        prompt = GAP_DETECTION_PROMPT.format(
            field=field,
            context=context or "无特殊要求",
            hotspots_summary=hotspots_summary,
            schema=schema_example,
        )
        raw = self._stream_and_collect(prompt, "检测研究缺口")
        results = self._parse_list(raw, ResearchGap, "研究缺口")
        self._log_progress(f"研究缺口检测完成，发现 {len(results)} 个缺口")
        return results

    # ── Step 3: Topic Tree ────────────────────────────────────────────────────

    def build_topic_tree(
        self,
        field: str,
        context: str,
        hotspots: list[ResearchHotspot],
        gaps: list[ResearchGap],
    ) -> list[TopicNode]:
        hotspots_summary = "\n".join(
            f"- 【{h.topic}】{h.description[:80]}…"
            for h in hotspots
        )
        gaps_summary = "\n".join(
            f"- 【{g.gap_title}】({g.gap_type.value}) {g.description[:80]}…"
            for g in gaps
        )
        schema_example = json.dumps(TopicNode.model_json_schema(), ensure_ascii=False, indent=2)
        prompt = TOPIC_TREE_PROMPT.format(
            field=field,
            context=context or "无特殊要求",
            hotspots_summary=hotspots_summary,
            gaps_summary=gaps_summary,
            schema=schema_example,
        )
        raw = self._stream_and_collect(prompt, "构建多维选题树")
        results = self._parse_list(raw, TopicNode, "选题树节点")
        self._log_progress(f"选题树构建完成，生成 {len(results)} 个节点")
        return results

    # ── Step 4: Value Rating ──────────────────────────────────────────────────

    def rate_topics(
        self, field: str, context: str, topic_nodes: list[TopicNode]
    ) -> list[ResearchValueRating]:
        leaf_nodes = [n for n in topic_nodes if n.level == 3]
        if not leaf_nodes:
            # Fallback: take all non-root nodes
            leaf_nodes = [n for n in topic_nodes if n.level > 0]

        topics_summary = "\n".join(
            f"- ID:{n.id} 【{n.title}】维度:{n.dimension} "
            f"创新:{n.novelty_score} 可行:{n.feasibility_score} 影响:{n.impact_score}\n"
            f"  研究问题: {'; '.join(n.research_questions[:2])}"
            for n in leaf_nodes
        )
        schema_example = json.dumps(ResearchValueRating.model_json_schema(), ensure_ascii=False, indent=2)
        prompt = RATING_PROMPT.format(
            field=field,
            context=context or "无特殊要求",
            topics_summary=topics_summary,
            schema=schema_example,
        )
        raw = self._stream_and_collect(prompt, "评估研究价值")
        results = self._parse_list(raw, ResearchValueRating, "研究价值评级")
        self._log_progress(f"研究价值评级完成，评估了 {len(results)} 个选题")
        return results

    # ── Step 5: Report Synthesis ──────────────────────────────────────────────

    def _synthesize_report_meta(
        self,
        field: str,
        context: str,
        hotspots: list[ResearchHotspot],
        gaps: list[ResearchGap],
        topic_nodes: list[TopicNode],
        ratings: list[ResearchValueRating],
    ) -> tuple[str, list[str], str]:
        """Generate executive_summary, top_recommendations, conclusion."""
        prompt = REPORT_SYNTHESIS_PROMPT.format(
            field=field,
            context=context or "无特殊要求",
            hotspot_count=len(hotspots),
            gap_count=len(gaps),
            topic_count=len([n for n in topic_nodes if n.level == 3]),
        )
        raw = self._stream_and_collect(prompt, "综合生成报告摘要")
        try:
            json_str = self._extract_json(raw)
            data = json.loads(json_str)
            return (
                data.get("executive_summary", ""),
                data.get("top_recommendations", []),
                data.get("conclusion", ""),
            )
        except Exception as e:
            self._log(f"合成摘要解析失败: {e}", "⚠️")
            return raw[:500], [], ""

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_report(self, field: str, context: str = "") -> TopicSuggestionReport:
        """
        一键生成完整学术选题建议报告。

        Args:
            field: 研究领域，如 "计算机视觉" / "量子计算" / "行为经济学"
            context: 用户背景与特殊需求（可选），如研究阶段、资源限制、偏好方向等

        Returns:
            TopicSuggestionReport: 包含热点、缺口、选题树和价值评级的完整报告
        """
        self._log(f"开始为领域「{field}」生成选题规划报告", "🚀")
        self._log(f"用户背景: {context or '（未指定）'}", "👤")
        start_time = datetime.now()

        # Step 1
        hotspots = self.analyze_hotspots(field, context)

        # Step 2
        gaps = self.detect_research_gaps(field, context, hotspots)

        # Step 3
        topic_nodes = self.build_topic_tree(field, context, hotspots, gaps)

        # Step 4
        ratings = self.rate_topics(field, context, topic_nodes)

        # Step 5: Synthesis (not counted in _total_steps progress bar)
        self._log("正在综合生成报告摘要与最终推荐...", "📝")
        exec_summary, top_recs, conclusion = self._synthesize_report_meta(
            field, context, hotspots, gaps, topic_nodes, ratings
        )

        elapsed = (datetime.now() - start_time).seconds
        self._log(f"报告生成完成！耗时 {elapsed} 秒", "✅")

        return TopicSuggestionReport(
            field=field,
            context=context,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=exec_summary,
            hotspots=hotspots,
            research_gaps=gaps,
            topic_tree=topic_nodes,
            value_ratings=ratings,
            top_recommendations=top_recs,
            conclusion=conclusion,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Report Rendering  报告渲染
# ─────────────────────────────────────────────────────────────────────────────

def _stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def render_markdown(report: TopicSuggestionReport) -> str:
    """Convert a TopicSuggestionReport into a formatted Markdown string."""
    lines: list[str] = []
    sep = "\n---\n"

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# 📚 学术研究选题规划报告",
        f"",
        f"**研究领域：** {report.field}",
        f"**生成时间：** {report.generated_at}",
        f"**用户背景：** {report.context or '未指定'}",
        sep,
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines += [
        "## 📋 执行摘要",
        "",
        report.executive_summary,
        sep,
    ]

    # ── Hotspot Analysis ──────────────────────────────────────────────────────
    lines += ["## 🔥 学术热点分析", ""]
    trend_emoji = {"rising": "📈", "stable": "➡️", "declining": "📉"}
    for i, h in enumerate(report.hotspots, 1):
        lines += [
            f"### {i}. {h.topic} {trend_emoji.get(h.trend.value, '')}",
            f"",
            f"- **重要性：** {'▓' * h.importance_score}{'░' * (10 - h.importance_score)} {h.importance_score}/10",
            f"- **趋势：** {h.trend.value} | **论文增长：** {h.yearly_paper_growth}",
            f"- **主要期刊/会议：** {', '.join(h.key_venues)}",
            f"- **领先机构：** {', '.join(h.leading_institutions)}",
            f"- **核心关键词：** `{'` `'.join(h.core_keywords)}`",
            f"",
            f"{h.description}",
            "",
        ]
    lines.append(sep)

    # ── Research Gaps ─────────────────────────────────────────────────────────
    lines += ["## 🔬 研究缺口检测", ""]
    impact_emoji = {
        "低影响": "🟡", "中等影响": "🟠", "高影响": "🔴", "颠覆性影响": "🟣"
    }
    for i, g in enumerate(report.research_gaps, 1):
        lines += [
            f"### {i}. {g.gap_title}",
            f"",
            f"| 属性 | 值 |",
            f"|------|-----|",
            f"| 缺口类型 | {g.gap_type.value} |",
            f"| 研究难度 | {g.difficulty.value} |",
            f"| 潜在影响 | {impact_emoji.get(g.potential_impact.value, '')} {g.potential_impact.value} |",
            f"| 关联热点 | {', '.join(g.related_hotspots)} |",
            f"",
            f"**描述：** {g.description}",
            f"",
            f"**建议路径：**",
        ]
        for approach in g.suggested_approaches:
            lines.append(f"- {approach}")
        lines += [
            f"",
            f"**前置要求：** {', '.join(g.prerequisites)}",
            "",
        ]
    lines.append(sep)

    # ── Topic Tree ────────────────────────────────────────────────────────────
    lines += ["## 🌳 多维选题树", ""]
    node_map = {n.id: n for n in report.topic_tree}
    root_children = [n for n in report.topic_tree if n.level == 1]
    for branch in root_children:
        lines += [f"### 📐 维度：{branch.dimension} — {branch.title}", ""]
        sub_nodes = [n for n in report.topic_tree if n.parent_id == branch.id]
        for sub in sub_nodes:
            lines += [f"#### {sub.title}", ""]
            leaf_nodes = [n for n in report.topic_tree if n.parent_id == sub.id]
            if not leaf_nodes:
                leaf_nodes = [sub] if sub.level >= 2 else []
            for leaf in leaf_nodes:
                lines += [
                    f"##### 🔖 [{leaf.id}] {leaf.title}",
                    f"",
                    f"- **创新性：** {leaf.novelty_score}/10  |  "
                    f"**可行性：** {leaf.feasibility_score}/10  |  "
                    f"**影响力：** {leaf.impact_score}/10",
                    f"- **预计发表周期：** {leaf.time_to_publish_months} 个月",
                    f"- **目标期刊：** {', '.join(leaf.target_venues)}",
                    f"",
                    f"{leaf.description}",
                    f"",
                    f"**核心研究问题：**",
                ]
                for q in leaf.research_questions:
                    lines.append(f"- {q}")
                lines += [
                    f"",
                    f"**建议方法：** {', '.join(leaf.suggested_methods)}",
                    "",
                ]
    lines.append(sep)

    # ── Value Ratings ─────────────────────────────────────────────────────────
    lines += ["## ⭐ 研究价值评级", ""]
    # Sort by stars descending
    sorted_ratings = sorted(report.value_ratings, key=lambda r: r.stars, reverse=True)
    rec_emoji = {
        "强烈推荐": "🏆", "推荐": "✅", "条件推荐": "⚠️", "暂不推荐": "❌"
    }
    for rating in sorted_ratings:
        is_top = rating.topic_id in report.top_recommendations
        top_badge = "  🌟 **TOP推荐**" if is_top else ""
        lines += [
            f"### {rec_emoji.get(rating.recommendation.value, '')} {rating.topic_title}{top_badge}",
            f"",
            f"**综合评级：** {_stars(rating.stars)} ({rating.stars}/5星)",
            f"",
            f"| 维度 | 评级 |",
            f"|------|------|",
            f"| 创新性 | {_stars(rating.novelty)} |",
            f"| 可行性 | {_stars(rating.feasibility)} |",
            f"| 学术影响力 | {_stars(rating.impact)} |",
            f"| 获批基金可能性 | {_stars(rating.funding_potential)} |",
            f"| 高水平发表可能性 | {_stars(rating.publication_potential)} |",
            f"",
            f"**推荐意见：** {rating.recommendation.value}",
            f"",
            f"**优势：**",
        ]
        for s in rating.strengths:
            lines.append(f"- ✔ {s}")
        lines += ["", "**风险与挑战：**"]
        for r in rating.risks:
            lines.append(f"- ⚡ {r}")
        lines += ["", f"**综合评价：** {rating.reasoning}", ""]
    lines.append(sep)

    # ── Top Recommendations ───────────────────────────────────────────────────
    lines += ["## 🏅 优先推荐选题", ""]
    for rank, tid in enumerate(report.top_recommendations, 1):
        node = node_map.get(tid)
        rating = next((r for r in report.value_ratings if r.topic_id == tid), None)
        title = node.title if node else tid
        stars_str = _stars(rating.stars) if rating else ""
        lines.append(f"{rank}. **[{tid}] {title}** {stars_str}")
    lines += ["", sep]

    # ── Conclusion ────────────────────────────────────────────────────────────
    lines += [
        "## 🎯 总结与行动建议",
        "",
        report.conclusion,
        sep,
        "*本报告由 AI 学术选题规划系统自动生成，仅供参考，请结合导师意见和实际条件综合决策。*",
    ]

    return "\n".join(lines)


def render_console(report: TopicSuggestionReport) -> None:
    """Print a concise console summary (no rich dependency required)."""
    print("\n" + "=" * 70)
    print(f"  📚 学术选题规划报告 — {report.field}")
    print("=" * 70)
    print(f"\n📋 执行摘要\n{report.executive_summary[:400]}…\n")

    print("🔥 识别到的学术热点：")
    for h in report.hotspots:
        print(f"  • {h.topic} ({h.trend.value}, {h.importance_score}/10)")

    print("\n🔬 检测到的研究缺口：")
    for g in report.research_gaps:
        print(f"  • {g.gap_title} [{g.gap_type.value}]")

    print("\n⭐ 研究价值评级（综合排名）：")
    sorted_r = sorted(report.value_ratings, key=lambda r: r.stars, reverse=True)
    for r in sorted_r:
        top = " 🌟" if r.topic_id in report.top_recommendations else ""
        print(f"  {_stars(r.stars)} {r.topic_title}{top}")

    print("\n🏅 最优先推荐选题：")
    node_map = {n.id: n for n in report.topic_tree}
    for i, tid in enumerate(report.top_recommendations, 1):
        node = node_map.get(tid)
        print(f"  {i}. {node.title if node else tid}  (ID: {tid})")

    print(f"\n🎯 总结\n{report.conclusion[:500]}…")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="学术研究选题规划工具 — Academic Topic Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  python academic_topic_planner.py --field "计算机视觉"
  python academic_topic_planner.py --field "量子计算" --context "博士生，关注量子机器学习"
  python academic_topic_planner.py --field "行为经济学" --output report.md
  python academic_topic_planner.py --field "NLP" --context "有GPU资源，熟悉transformer" --output nlp_report.md
        """,
    )
    parser.add_argument(
        "--field", "-f", required=True,
        help="研究领域，如：计算机视觉 / 量子计算 / 行为经济学"
    )
    parser.add_argument(
        "--context", "-c", default="",
        help="用户背景与需求（可选），如：研究阶段、资源条件、偏好方向"
    )
    parser.add_argument(
        "--output", "-o", default="",
        help="输出 Markdown 报告的文件路径（可选，默认仅打印控制台摘要）"
    )
    parser.add_argument(
        "--json", "-j", default="",
        help="输出完整 JSON 数据的文件路径（可选）"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式，不显示流式输出过程"
    )
    parser.add_argument(
        "--mock", "-m", action="store_true",
        help="使用 Mock 演示数据，无需 API Key，快速体验完整流程"
    )
    args = parser.parse_args()

    if args.mock:
        print("🎭 Mock 模式已启用 — 使用演示数据，无需 API Key")

    planner = AcademicTopicPlanner(verbose=not args.quiet, mock=args.mock)

    try:
        report = planner.generate_report(field=args.field, context=args.context)
    except anthropic.AuthenticationError:
        print("❌ API Key 无效，请设置环境变量: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("❌ 网络连接失败，请检查网络设置")
        sys.exit(1)

    # Console summary
    render_console(report)

    # Markdown output
    if args.output:
        md_content = render_markdown(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"✅ Markdown 报告已保存至: {args.output}")

    # JSON output
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2, ensure_ascii=False))
        print(f"✅ JSON 数据已保存至: {args.json}")

    if not args.output:
        print("💡 提示：使用 --output report.md 可保存完整 Markdown 报告")


if __name__ == "__main__":
    main()
