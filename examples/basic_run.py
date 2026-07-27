"""Example: run ContentWorkflow from the command line.

Usage:
    python examples/basic_run.py <youtube_url>
"""
import sys
import asyncio
import pathlib

# Ensure src on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from content_growth_agent.workflows.content_workflow import ContentWorkflow
from content_growth_agent.models.input_models import VideoInput


async def main(url: str) -> None:
    video = VideoInput(url=url)
    wf = ContentWorkflow()
    result = await wf.execute(str(video.url))
    print("Summary:\n", result.summary)
    print("Titles:\n", result.seo.titles)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/basic_run.py <youtube_url>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
