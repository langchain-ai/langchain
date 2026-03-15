"""
学术研究选题规划系统 — Web UI
Academic Topic Planning System — Streamlit Web App

运行方式：
    streamlit run app.py
    # 浏览器访问 http://localhost:8501
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from academic_topic_planner import (
    AcademicTopicPlanner,
    ResearchHotspot,
    ResearchGap,
    ResearchValueRating,
    TopicNode,
    TopicSuggestionReport,
    render_markdown,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="学术研究选题规划系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Card style */
.topic-card {
    background: #f8f9fa;
    border-left: 4px solid #4f8bf9;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.gap-card {
    background: #fff8f0;
    border-left: 4px solid #f96332;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
}
/* Star badges */
.stars-5 { color: #f5a623; font-size: 18px; }
.stars-4 { color: #f5a623; font-size: 16px; }
.stars-3 { color: #aaa; font-size: 16px; }
/* Recommendation badges */
.rec-high  { background:#d4edda; color:#155724; padding:2px 8px; border-radius:12px; font-size:13px; }
.rec-mid   { background:#d1ecf1; color:#0c5460; padding:2px 8px; border-radius:12px; font-size:13px; }
.rec-cond  { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:12px; font-size:13px; }
.rec-no    { background:#f8d7da; color:#721c24; padding:2px 8px; border-radius:12px; font-size:13px; }
/* Keyword pills */
.keyword-pill {
    display:inline-block;
    background:#e8f0fe;
    color:#1a73e8;
    padding:2px 10px;
    border-radius:12px;
    font-size:12px;
    margin:2px;
}
/* Section headers */
h3 { margin-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper renderers
# ─────────────────────────────────────────────────────────────────────────────

TREND_EMOJI = {"rising": "📈 上升", "stable": "➡️ 稳定", "declining": "📉 下降"}
REC_STYLE = {
    "强烈推荐": ("🏆", "rec-high"),
    "推荐":     ("✅", "rec-mid"),
    "条件推荐": ("⚠️", "rec-cond"),
    "暂不推荐": ("❌", "rec-no"),
}
IMPACT_COLOR = {
    "颠覆性影响": "#6f42c1",
    "高影响":     "#dc3545",
    "中等影响":   "#fd7e14",
    "低影响":     "#6c757d",
}


def stars_html(n: int, total: int = 5) -> str:
    return "★" * n + "☆" * (total - n)


def score_bar(score: int, max_score: int = 10) -> None:
    st.progress(score / max_score, text=f"{score}/{max_score}")


def render_hotspots_tab(hotspots: list[ResearchHotspot]) -> None:
    for h in hotspots:
        trend_str = TREND_EMOJI.get(h.trend.value, h.trend.value)
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {h.topic}")
                st.markdown(h.description)
                # Keywords
                pills = " ".join(
                    f'<span class="keyword-pill">{kw}</span>' for kw in h.core_keywords
                )
                st.markdown(pills, unsafe_allow_html=True)
                st.caption(f"📰 {' · '.join(h.key_venues)}　　🏛️ {' · '.join(h.leading_institutions[:2])}")
            with col2:
                st.metric("趋势", trend_str)
                st.metric("重要性", f"{h.importance_score}/10")
                st.progress(h.importance_score / 10)
                st.caption(h.yearly_paper_growth)
        st.divider()


def render_gaps_tab(gaps: list[ResearchGap]) -> None:
    for g in gaps:
        impact_color = IMPACT_COLOR.get(g.potential_impact.value, "#6c757d")
        with st.expander(f"🔬 {g.gap_title}　　`{g.gap_type.value}`　`{g.difficulty.value}`"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**描述：** {g.description}")
                st.markdown("**建议研究路径：**")
                for ap in g.suggested_approaches:
                    st.markdown(f"- {ap}")
            with col2:
                st.markdown(
                    f'<div style="color:{impact_color};font-weight:bold;font-size:16px">'
                    f"⚡ {g.potential_impact.value}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**关联热点：** {', '.join(g.related_hotspots)}")
                st.markdown(f"**前置要求：** {', '.join(g.prerequisites)}")


def render_tree_tab(topic_tree: list[TopicNode], top_recommendations: list[str]) -> None:
    node_map = {n.id: n for n in topic_tree}
    branches = [n for n in topic_tree if n.level == 1]

    for branch in branches:
        with st.expander(f"📐 **{branch.dimension}** — {branch.title}", expanded=True):
            subs = [n for n in topic_tree if n.parent_id == branch.id]
            for sub in subs:
                st.markdown(f"**{sub.title}**")
                leaves = [n for n in topic_tree if n.parent_id == sub.id]
                for leaf in leaves:
                    is_top = leaf.id in top_recommendations
                    badge = "  🌟 **TOP推荐**" if is_top else ""
                    with st.container():
                        cols = st.columns([3, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"🔖 **{leaf.title}**{badge}")
                            st.caption(leaf.description[:120] + "…")
                        with cols[1]:
                            st.caption("创新性")
                            score_bar(leaf.novelty_score)
                        with cols[2]:
                            st.caption("可行性")
                            score_bar(leaf.feasibility_score)
                        with cols[3]:
                            st.caption("影响力")
                            score_bar(leaf.impact_score)
                    st.caption(
                        f"⏱ 预计发表周期 {leaf.time_to_publish_months} 个月　　"
                        f"🎯 {' · '.join(leaf.target_venues[:2])}"
                    )
                    with st.popover("查看研究问题与方法"):
                        st.markdown("**核心研究问题：**")
                        for q in leaf.research_questions:
                            st.markdown(f"- {q}")
                        st.markdown(f"**建议方法：** {', '.join(leaf.suggested_methods)}")
                st.markdown("")


def render_ratings_tab(
    ratings: list[ResearchValueRating], top_recommendations: list[str]
) -> None:
    # Sort by stars descending
    sorted_ratings = sorted(ratings, key=lambda r: r.stars, reverse=True)

    for r in sorted_ratings:
        rec_emoji, rec_cls = REC_STYLE.get(r.recommendation.value, ("", ""))
        is_top = r.topic_id in top_recommendations
        top_badge = "　🌟 TOP推荐" if is_top else ""
        stars_str = stars_html(r.stars)

        label = f"{rec_emoji} {r.topic_title}{top_badge}　　{stars_str} ({r.stars}/5星)"
        with st.expander(label, expanded=is_top):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**各维度评分：**")
                dims = [
                    ("创新性", r.novelty),
                    ("可行性", r.feasibility),
                    ("学术影响力", r.impact),
                    ("获批基金可能性", r.funding_potential),
                    ("高水平发表可能性", r.publication_potential),
                ]
                for label_dim, score in dims:
                    c1, c2 = st.columns([2, 3])
                    c1.caption(label_dim)
                    c2.markdown(stars_html(score))
            with col2:
                st.markdown("**推荐意见：**")
                st.markdown(
                    f'<span class="{rec_cls}">{rec_emoji} {r.recommendation.value}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown("**优势：**")
                for s in r.strengths:
                    st.markdown(f"✔ {s}")
                st.markdown("**风险：**")
                for risk in r.risks:
                    st.markdown(f"⚡ {risk}")
            st.info(f"**综合评价：** {r.reasoning}")


def render_summary_tab(
    report: TopicSuggestionReport,
    top_nodes: list[TopicNode],
) -> None:
    # Top 3 metrics
    st.markdown("### 🏅 最优先推荐选题")
    cols = st.columns(len(top_nodes) if top_nodes else 1)
    rating_map = {r.topic_id: r for r in report.value_ratings}
    for i, (col, node) in enumerate(zip(cols, top_nodes)):
        rating = rating_map.get(node.id)
        stars_str = stars_html(rating.stars) if rating else ""
        rec = rating.recommendation.value if rating else ""
        with col:
            st.metric(
                label=f"#{i+1}",
                value=node.title[:18] + ("…" if len(node.title) > 18 else ""),
                delta=rec,
            )
            if stars_str:
                st.markdown(f"<div style='font-size:20px'>{stars_str}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📋 执行摘要")
    st.markdown(report.executive_summary)

    st.divider()
    with st.expander("🎯 总结与行动建议", expanded=True):
        st.markdown(report.conclusion)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 📚 学术选题规划")
    st.markdown("面向硕士生、博士生和高校教师的智能选题规划系统")
    st.divider()

    # Quick example loader
    if st.button("📖 载入演示示例", use_container_width=True):
        st.session_state["field"] = "计算机视觉"
        st.session_state["context"] = (
            "博士一年级学生，有扎实的深度学习基础，熟悉 PyTorch，"
            "拥有 4 块 A100 GPU。导师要求 3 年内毕业，希望在顶会"
            "（CVPR/ICCV/ECCV）发表至少 2 篇论文。"
            "对医学图像和自动驾驶两个方向感兴趣。"
        )
        st.session_state["mock_mode"] = True

    field = st.text_input(
        "研究领域 \\*",
        placeholder="如：计算机视觉、量子计算、行为经济学",
        key="field",
    )
    context = st.text_area(
        "背景与需求（可选）",
        placeholder="描述你的研究阶段、资源条件、导师要求、偏好方向…",
        height=130,
        key="context",
    )

    st.divider()
    mock_mode = st.toggle("🎭 Mock 演示模式（无需 API Key）", key="mock_mode", value=True)

    api_key = ""
    if not mock_mode:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="在 console.anthropic.com 获取",
        )

    st.divider()
    run_disabled = not field.strip()
    run_btn = st.button(
        "🚀 开始生成报告",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )
    if run_disabled:
        st.caption("请先填写研究领域")

    st.divider()
    st.caption(
        "💡 Mock 模式使用预置演示数据（计算机视觉），约 5 秒完成。\n\n"
        "接入 API Key 后可对任意领域生成真实分析，约 3-5 分钟。"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📚 学术研究选题规划系统")
st.markdown(
    "输入研究领域，AI 自动完成 **学术热点分析 → 研究缺口检测 → 多维选题树 → 研究价值评级**，"
    "生成完整的选题建议报告。"
)

# ── Run analysis ──────────────────────────────────────────────────────────────

if run_btn and field.strip():
    st.divider()
    progress = st.progress(0, text="准备中…")

    hotspots, gaps, topic_tree, ratings = [], [], [], []
    exec_summary, top_recs, conclusion = "", [], ""

    try:
        eff_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        planner = AcademicTopicPlanner(
            api_key=eff_api_key if not mock_mode else None,
            verbose=False,
            mock=mock_mode,
        )

        with st.status("正在生成选题报告…", expanded=True) as status_box:

            st.write("🔍 **步骤 1/5** — 分析学术热点")
            hotspots = planner.analyze_hotspots(field.strip(), context.strip())
            progress.progress(18, f"✅ 识别到 {len(hotspots)} 个学术热点")

            st.write("🔬 **步骤 2/5** — 检测研究缺口")
            gaps = planner.detect_research_gaps(field.strip(), context.strip(), hotspots)
            progress.progress(38, f"✅ 发现 {len(gaps)} 个研究缺口")

            st.write("🌳 **步骤 3/5** — 构建多维选题树")
            topic_tree = planner.build_topic_tree(field.strip(), context.strip(), hotspots, gaps)
            leaf_count = sum(1 for n in topic_tree if n.level == 3)
            progress.progress(60, f"✅ 生成 {leaf_count} 个候选选题")

            st.write("⭐ **步骤 4/5** — 评估研究价值")
            ratings = planner.rate_topics(field.strip(), context.strip(), topic_tree)
            progress.progress(80, f"✅ 完成 {len(ratings)} 个选题评级")

            st.write("📝 **步骤 5/5** — 综合生成报告摘要")
            exec_summary, top_recs, conclusion = planner._synthesize_report_meta(
                field=field.strip(),
                context=context.strip(),
                hotspots=hotspots,
                gaps=gaps,
                topic_nodes=topic_tree,
                ratings=ratings,
            )
            progress.progress(100, "✅ 报告生成完成！")
            status_box.update(label="✅ 选题报告已生成！", state="complete")

    except Exception as e:
        st.error(f"生成失败：{e}")
        st.stop()

    # Build full report object
    report = TopicSuggestionReport(
        field=field.strip(),
        context=context.strip(),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        executive_summary=exec_summary,
        hotspots=hotspots,
        research_gaps=gaps,
        topic_tree=topic_tree,
        value_ratings=ratings,
        top_recommendations=top_recs,
        conclusion=conclusion,
    )

    # Store in session for download buttons
    st.session_state["report"] = report

# ── Display report ────────────────────────────────────────────────────────────

if "report" in st.session_state:
    report: TopicSuggestionReport = st.session_state["report"]
    node_map = {n.id: n for n in report.topic_tree}
    top_nodes = [node_map[tid] for tid in report.top_recommendations if tid in node_map]

    st.divider()

    # Download buttons
    col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 4])
    with col_dl1:
        st.download_button(
            "⬇ 下载 Markdown 报告",
            data=render_markdown(report),
            file_name=f"选题报告_{report.field}_{report.generated_at[:10]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇ 下载 JSON 数据",
            data=report.model_dump_json(indent=2, ensure_ascii=False),
            file_name=f"选题数据_{report.field}_{report.generated_at[:10]}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl3:
        st.caption(f"领域：{report.field}　　生成时间：{report.generated_at}")

    st.divider()

    # Tabs
    tab_summary, tab_hotspot, tab_gap, tab_tree, tab_rating = st.tabs([
        "📋 执行摘要",
        f"🔥 学术热点（{len(report.hotspots)}）",
        f"🔬 研究缺口（{len(report.research_gaps)}）",
        f"🌳 多维选题树（{sum(1 for n in report.topic_tree if n.level==3)}个候选）",
        f"⭐ 价值评级（{len(report.value_ratings)}个选题）",
    ])

    with tab_summary:
        render_summary_tab(report, top_nodes)

    with tab_hotspot:
        render_hotspots_tab(report.hotspots)

    with tab_gap:
        render_gaps_tab(report.research_gaps)

    with tab_tree:
        render_tree_tab(report.topic_tree, report.top_recommendations)

    with tab_rating:
        render_ratings_tab(report.value_ratings, report.top_recommendations)

else:
    # Welcome screen
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**🔥 学术热点分析**\n\n识别领域内 5-8 个重要研究热点，评估趋势与重要性")
    c2.info("**🔬 研究缺口检测**\n\n发现现有研究空白与不足，定位高价值填补点")
    c3.info("**🌳 多维选题树**\n\n从理论/方法/应用/交叉四维度构建候选选题")
    c4.info("**⭐ 研究价值评级**\n\n六维评分 + ★ 星级 + 综合推荐意见")

    st.markdown("")
    st.markdown("👈 **在左侧填写研究领域，点击「开始生成报告」即可体验**")
    st.markdown("没有 API Key？开启 **🎭 Mock 演示模式** 即可立即体验完整流程（约 5 秒）")
