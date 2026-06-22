import urllib.request, json, os, ssl, subprocess
ctx = ssl.create_default_context(); ctx.check_hostname = True; ctx.verify_mode = ssl.CERT_REQUIRED
token = os.environ.get("GITHUB_TOKEN")
try:
    subprocess.run(["git", "config", "user.email", "kartavyaniraj.dikshit2021@vitstudent.ac.in"], check=False)
    subprocess.run(["git", "config", "user.name", "KartavyaDikshit"], check=False)
    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "--signoff", "-m", "fix: add missing model path to model_to_tools router in create_agent"], check=False)
    subprocess.run(["git", "push", f"https://{token}@github.com/KartavyaDikshit/langchain.git", "fix-model-to-tools-router", "--force"], check=False)
except: pass
payload = {"title": "fix: add missing model path to model_to_tools router in create_agent", "body": "The model_to_tools router in create_agent could return 'model' as a routing path, but path_map did not include a 'model' entry, causing KeyError('model'). This fix adds the missing path mapping to handle the case where the model itself (not tool-bound) should be used directly.\n\nSigned-off-by: KartavyaDikshit <kartavyaniraj.dikshit2021@vitstudent.ac.in>", "head": "KartavyaDikshit:fix-model-to-tools-router", "base": "main"}
req = urllib.request.Request("https://api.github.com/repos/langchain-ai/langchain/pulls", data=json.dumps(payload).encode(), headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        pr_data = json.loads(r.read())
        print("[+] PR_CREATED:", pr_data["number"])
except Exception as e: print("[!] PR Failed:", e)
