import urllib.request, json, os, ssl, subprocess
ctx = ssl.create_default_context(); ctx.check_hostname = True; ctx.verify_mode = ssl.CERT_REQUIRED
token = os.environ.get("GITHUB_TOKEN")
try:
    subprocess.run(["git", "config", "user.email", "kartavyaniraj.dikshit2021@vitstudent.ac.in"], check=False)
    subprocess.run(["git", "config", "user.name", "KartavyaDikshit"], check=False)
    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "--signoff", "-m", "fix: prevent orphan ToolMessages in SummarizationMiddleware cut point"], check=False)
    subprocess.run(["git", "push", "fork", "fix-summarization-orphan", "--force"], check=False)
except: pass
payload = {"title": "fix: prevent orphan ToolMessages in SummarizationMiddleware cut point", "body": "SummarizationMiddleware._find_safe_cutoff_point could leave orphan ToolMessages in the history after cutting away the preceding AIMessage with tool_calls. This caused provider 400 errors on subsequent API calls. The fix ensures that when determining the safe cut point, ToolMessages without a corresponding AIMessage in the kept window are also removed.\n\nSigned-off-by: KartavyaDikshit <kartavyaniraj.dikshit2021@vitstudent.ac.in>", "head": "KartavyaDikshit:fix-summarization-orphan", "base": "main"}
req = urllib.request.Request("https://api.github.com/repos/langchain-ai/langchain/pulls", data=json.dumps(payload).encode(), headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        pr_data = json.loads(r.read())
        print("[+] PR_CREATED:", pr_data["number"])
except Exception as e: print("[!] PR Failed:", e)
