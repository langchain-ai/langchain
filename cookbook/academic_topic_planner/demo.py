"""
学术选题规划工具演示脚本
Academic Topic Planner — Demo Script

运行前请确保设置 API Key:
    export ANTHROPIC_API_KEY=sk-ant-...

运行方式:
    python demo.py
    python demo.py --save           # 保存报告到文件
    python demo.py --field "量子计算" # 自定义领域
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure local module is importable
sys.path.insert(0, os.path.dirname(__file__))

from academic_topic_planner import AcademicTopicPlanner, render_markdown, render_console

# ── Demo scenarios / 演示场景 ─────────────────────────────────────────────────

DEMO_SCENARIOS = {
    "cv": {
        "field": "计算机视觉",
        "context": (
            "博士一年级学生，研究方向待定。"
            "有扎实的深度学习基础，熟悉 PyTorch，拥有 4 块 A100 GPU。"
            "导师要求3年内毕业，希望在顶会（CVPR/ICCV/ECCV）发表至少2篇论文。"
            "对医学图像和自动驾驶两个应用方向感兴趣。"
        ),
    },
    "nlp": {
        "field": "自然语言处理",
        "context": (
            "硕士生，计划申博。有 NLP 基础，主要使用 Hugging Face 生态。"
            "资源有限（仅 2 块 RTX 3090），倾向于高效微调方向。"
            "希望在 ACL/EMNLP/NAACL 发表论文。"
        ),
    },
    "quantum": {
        "field": "量子计算与量子机器学习",
        "context": (
            "理论物理背景的副教授，希望转型跨学科研究。"
            "有量子力学和线性代数基础，但机器学习经验有限。"
            "目标：申请国家自然科学基金面上项目，需要明确的研究方向。"
        ),
    },
    "econ": {
        "field": "行为经济学与数字经济",
        "context": (
            "经济学硕士，研究消费者决策行为。"
            "有计量经济学基础，正在学习因果推断方法。"
            "有某电商平台的数据访问权限。希望在 JEL-A 期刊发表。"
        ),
    },
}


def run_demo(
    field: str,
    context: str,
    save: bool = False,
    output_dir: str = ".",
) -> None:
    """Run the planner and optionally save reports."""
    print(f"\n{'=' * 60}")
    print(f"  🎓 学术选题规划演示")
    print(f"  领域: {field}")
    print(f"{'=' * 60}")

    planner = AcademicTopicPlanner(verbose=True)
    report = planner.generate_report(field=field, context=context)

    # Console summary
    render_console(report)

    if save:
        # Markdown report
        md_path = os.path.join(output_dir, f"report_{field.replace(' ', '_')}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))
        print(f"✅ Markdown 报告: {md_path}")

        # JSON data
        json_path = os.path.join(output_dir, f"report_{field.replace(' ', '_')}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2, ensure_ascii=False))
        print(f"✅ JSON 数据:     {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="学术选题规划演示")
    parser.add_argument(
        "--scenario", "-s",
        choices=list(DEMO_SCENARIOS.keys()),
        default="cv",
        help=f"选择预设场景: {', '.join(DEMO_SCENARIOS.keys())} (默认: cv)",
    )
    parser.add_argument(
        "--field", "-f", default="",
        help="自定义研究领域（覆盖预设场景）"
    )
    parser.add_argument(
        "--context", "-c", default="",
        help="自定义用户背景（覆盖预设场景）"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存报告到 Markdown 和 JSON 文件"
    )
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="报告保存目录（默认：当前目录）"
    )
    args = parser.parse_args()

    scenario = DEMO_SCENARIOS[args.scenario]
    field = args.field or scenario["field"]
    context = args.context or scenario["context"]

    run_demo(field=field, context=context, save=args.save, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
