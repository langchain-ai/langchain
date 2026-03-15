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

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-opus-4-6"
        self.verbose = verbose
        self._steps_completed = 0
        self._total_steps = 4

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
    args = parser.parse_args()

    planner = AcademicTopicPlanner(verbose=not args.quiet)

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
