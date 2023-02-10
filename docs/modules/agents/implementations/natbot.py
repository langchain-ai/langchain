"""Run NatBot."""
import time

from langchain.chains.natbot.base import NatBotChain
from langchain.chains.natbot.crawler import Crawler


def run_cmd(cmd: str, _crawler: Crawler) -> None:
    """Run command."""
    cmd = cmd.split("\n")[0]

    if cmd.startswith("SCROLL UP"):
        _crawler.scroll("up")
    elif cmd.startswith("SCROLL DOWN"):
        _crawler.scroll("down")
    elif cmd.startswith("CLICK"):
        commasplit = cmd.split(",")
        id = commasplit[0].split(" ")[1]
        _crawler.click(id)
    elif cmd.startswith("TYPE"):
        spacesplit = cmd.split(" ")
        id = spacesplit[1]
        text_pieces = spacesplit[2:]
        text = " ".join(text_pieces)
        # Strip leading and trailing double quotes
        text = text[1:-1]

        if cmd.startswith("TYPESUBMIT"):
            text += "\n"
        _crawler.type(id, text)

    time.sleep(2)


if __name__ == "__main__":
    objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
    print("\nWelcome to natbot! What is your objective?")
    i = input()
    if len(i) > 0:
        objective = i
    quiet = False
    nat_bot_chain = NatBotChain.from_default(objective)
    _crawler = Crawler()
    _crawler.go_to_page("google.com")
    try:
        while True:
            browser_content = "\n".join(_crawler.crawl())
            llm_command = nat_bot_chain.execute(_crawler.page.url, browser_content)
            if not quiet:
                print("URL: " + _crawler.page.url)
                print("Objective: " + objective)
                print("----------------\n" + browser_content + "\n----------------\n")
            if len(llm_command) > 0:
                print("Suggested command: " + llm_command)

            command = input()
            if command == "r" or command == "":
                run_cmd(llm_command, _crawler)
            elif command == "g":
                url = input("URL:")
                _crawler.go_to_page(url)
            elif command == "u":
                _crawler.scroll("up")
                time.sleep(1)
            elif command == "d":
                _crawler.scroll("down")
                time.sleep(1)
            elif command == "c":
                id = input("id:")
                _crawler.click(id)
                time.sleep(1)
            elif command == "t":
                id = input("id:")
                text = input("text:")
                _crawler.type(id, text)
                time.sleep(1)
            elif command == "o":
                objective = input("Objective:")
            else:
                print(
                    "(g) to visit url\n(u) scroll up\n(d) scroll down\n(c) to click"
                    "\n(t) to type\n(h) to view commands again"
                    "\n(r/enter) to run suggested command\n(o) change objective"
                )
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, exiting gracefully.")
        exit(0)
