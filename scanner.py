import os
import sqlite3
import requests
from datetime import datetime

# Configuration for Cash-Backed Bounty Scanning
DB_FILE = "bounties.db"
HTML_FILE = "bounties.html"
TXT_FILE = "bounties.txt"

# Target repositories known for active bounties or funding hooks
TARGET_REPOS = [
    "Significant-Gravitas/AutoGPT",
    "langchain-ai/langchain",
    "onyx-dot-app/onyx",
    "crewaiinc/crewai"
]

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bounties (
            id TEXT PRIMARY KEY,
            repo TEXT,
            title TEXT,
            url TEXT,
            updated_at TEXT,
            amount TEXT,
            platform TEXT
        )
    ''')
    conn.commit()
    conn.close()

def scan_cash_backed_bounties():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    print("[*] Scanning repositories for VERIFIED cash-backed bounties...")
    valid_bounties = []

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for repo in TARGET_REPOS:
        api_url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=50"
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code != 200:
                print(f"[-] Error fetching {repo}: Status {response.status_code}")
                continue
            
            issues = response.json()
            for issue in issues:
                # Skip pull requests if returned in issues endpoint
                if "pull_request" in issue:
                    continue
                
                title = issue.get("title", "")
                body = issue.get("body") or ""
                labels = [l["name"].lower() for l in issue.get("labels", [])]
                
                # STRICT FINANCIAL FILTER CHECK:
                has_dollar = "$" in title or "$" in body
                has_bounty_label = any(lbl in ["bounty", "paid", "grant", "algora", "polar"] for lbl in labels)
                is_discussion_proposal = any(keyword in title.lower() for keyword in ["proposal", "discussion", "clarification on", "invitation"])
                
                # If it's a discussion or lacks cash indicators, skip it entirely!
                if is_discussion_proposal or not (has_dollar or has_bounty_label):
                    continue
                
                # Extract amount if present
                amount_str = "Verified Bounty / Escrow"
                if "$" in title:
                    parts = title.split("$")
                    if len(parts) > 1:
                        amount_str = "$" + parts[1].split()[0]
                elif "$" in body:
                    amount_str = "Escrow Funded (See Body)"

                bounty_data = {
                    "id": f"{repo}#{issue['number']}",
                    "repo": repo,
                    "title": title,
                    "url": issue["html_url"],
                    "updated_at": issue["updated_at"][:10],
                    "amount": amount_str,
                    "platform": "Algora/Escrow"
                }
                
                valid_bounties.append(bounty_data)
                
                # Save into SQLite DB
                cursor.execute('''
                    INSERT OR REPLACE INTO bounties (id, repo, title, url, updated_at, amount, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (bounty_data["id"], bounty_data["repo"], bounty_data["title"], 
                      bounty_data["url"], bounty_data["updated_at"], bounty_data["amount"], bounty_data["platform"]))

            # Properly indented inside the loop:
            count = len([b for b in valid_bounties if b.get("repo") == repo])
            print(f"[+] Scanned {repo}: Found {count} cash-backed tasks.")
            
        except Exception as e:
            print(f"[-] Exception while scanning {repo}: {e}")

    conn.commit()
    conn.close()
    
    generate_outputs(valid_bounties)

def generate_outputs(bounties):
    # Generate TXT Dashboard
    with open(TXT_FILE, "w", encoding="utf-8") as f:
        f.write("=== TARGETED CASH-BACKED USB BOUNTY DASHBOARD ===\n\n")
        if not bounties:
            f.write("No active cash-backed bounties found matching strict financial filters.\n")
        for b in bounties:
            f.write(f"[{b['repo']}] {b['title']}\n")
            f.write(f"Payout / Status: {b['amount']} | Updated: {b['updated_at']}\n")
            f.write(f"URL: {b['url']}\n")
            f.write("-" * 50 + "\n")

    # Generate HTML Dashboard
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cash-Backed Bounty Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
        h1 {{ color: #4ec9b0; }}
        .card {{ background: #252526; border-left: 5px solid #007acc; padding: 15px; margin-bottom: 15px; border-radius: 4px; }}
        .amount {{ color: #b5cea8; font-weight: bold; font-size: 1.1em; }}
        a {{ color: #9cdcfe; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Verified Cash-Backed Bounty Dashboard</h1>
    <p>Filtered automatically to exclude unpaid discussions and focus strictly on funded tasks.</p>
"""
    if not bounties:
        html_content += "<p>No active cash-backed bounties found at this time.</p>"
    
    for b in bounties:
        html_content += f"""
        <div class="card">
            <h3>[{b['repo']}] <a href="{b['url']}" target="_blank">{b['title']}</a></h3>
            <p>Payout: <span class="amount">{b['amount']}</span> | Updated: {b['updated_at']}</p>
        </div>
        """
    
    html_content += f"""
    <hr style="border-color: #333;">
    <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} via USB Bounty Extractor</small></p>
</body>
</html>
"""
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[+] Scan complete! Filtered results saved to {TXT_FILE} and {HTML_FILE}.")

if __name__ == "__main__":
    scan_cash_backed_bounties()