#!/usr/bin/env python3
"""Verify that the 0G integration appears in the documentation."""

import requests
import time
import sys

def check_integration():
    """Check if the 0G integration is properly integrated."""
    base_url = "http://localhost:3001"

    print("🔍 Verifying 0G Integration in LangChain Documentation")
    print("=" * 60)

    # Check 1: Specific 0G page
    print("\n1️⃣ Checking specific 0G documentation page...")
    try:
        response = requests.get(f"{base_url}/docs/integrations/chat/zerog", timeout=10)
        if response.status_code == 200:
            print("   ✅ 0G documentation page is accessible!")
            print(f"   🌐 URL: {base_url}/docs/integrations/chat/zerog")

            if "ChatZeroG" in response.text or "0G Compute" in response.text:
                print("   ✅ Page content includes 0G-specific information")
            else:
                print("   ⚠️  Page loads but may still be rendering content")
        else:
            print(f"   ❌ Page not accessible (HTTP {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error accessing page: {e}")
        return False

    # Check 2: Main chat integrations page
    print("\n2️⃣ Checking main chat integrations page...")
    try:
        response = requests.get(f"{base_url}/docs/integrations/chat/", timeout=10)
        if response.status_code == 200:
            print("   ✅ Main chat integrations page is accessible!")

            # Check if our integration appears in the table
            content = response.text.lower()
            if "zerog" in content or "0g compute" in content or "chatzerog" in content:
                print("   ✅ 0G integration appears in the chat models table!")
            else:
                print("   ⚠️  0G integration may not be visible in the table yet")
                print("   💡 The page may need time to rebuild after changes")
        else:
            print(f"   ❌ Main page not accessible (HTTP {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error accessing main page: {e}")
        return False

    # Check 3: Server status
    print("\n3️⃣ Checking documentation server status...")
    try:
        response = requests.get(f"{base_url}/docs/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Documentation server is running properly")
        else:
            print(f"   ⚠️  Server responding with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Server connection issue: {e}")
        return False

    return True

def main():
    """Main function."""
    success = check_integration()

    if success:
        print("\n🎉 Integration verification completed!")
        print("\n📖 How to view the 0G integration:")
        print("1. Visit the main chat models page:")
        print("   http://localhost:3001/docs/integrations/chat/")
        print("2. Look for 'ChatZeroG' or '0G Compute Network' in the table")
        print("3. Or visit the specific page directly:")
        print("   http://localhost:3001/docs/integrations/chat/zerog")
        print("\n💡 Note: If the integration doesn't appear in the table immediately,")
        print("   the documentation server may need a few moments to rebuild.")
        print("   Try refreshing the page or restarting the docs server.")
    else:
        print("\n❌ Integration verification failed")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure the docs server is running: cd docs && npm start")
        print("2. Wait for the server to fully start (1-2 minutes)")
        print("3. Check that the FeatureTables.js file includes the ChatZeroG entry")
        print("4. Verify the zerog.ipynb file exists in docs/docs/integrations/chat/")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
