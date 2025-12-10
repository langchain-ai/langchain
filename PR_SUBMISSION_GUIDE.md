# PR Submission Guide for LangChain Perplexity Cost Tracking

**Issue:** https://github.com/langchain-ai/langchain/issues/31647
**Branch:** `feat/perplexity-cost-tracking`
**Status:** Code complete, ready for PR submission

---

## Step 1: Fork the Repository (CURRENT STEP)

1. Open your browser
2. Go to: https://github.com/langchain-ai/langchain
3. Click the **"Fork"** button (top right corner)
4. Keep all defaults
5. Click **"Create fork"**
6. Wait for it to complete
7. You now have: `https://github.com/YOUR_USERNAME/langchain`

**When done, tell me your GitHub username so we can proceed to Step 2.**

---

## Step 2: Rename Origin and Add Your Fork

```bash
cd /home/joshua/langchain-contrib
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/langchain.git
git remote -v
```

---

## Step 3: Push Your Branch to Your Fork

```bash
git push -u origin feat/perplexity-cost-tracking
```

---

## Step 4: Create the Pull Request

Go to: `https://github.com/YOUR_USERNAME/langchain`
Click "Compare & pull request" button

**Title:** `partners: Add comprehensive cost tracking to ChatPerplexity`

**Description:**
```
## Summary
- Add real-time cost tracking with `PerplexityCostTracker` callback handler
- Add budget management with configurable warnings and hard limits
- Add detailed cost breakdowns for all token types (input, output, reasoning, citation)
- Add pre-call cost estimation with `estimate_cost()`
- Add centralized pricing data for all 5 Perplexity models

Closes #31647

## Test plan
- [x] Unit tests for all cost tracking classes and functions
- [x] Edge case tests (zero tokens, unknown models, concurrent access)
- [x] Budget warning and error tests
- [x] Thread-safety tests
- [x] Pricing data integrity tests

AI agents were involved in developing this contribution.
```

---

## Step 5: Wait for CI and Review

- CI checks will run automatically
- Maintainers will review (usually 1-2 weeks)
- Respond to any feedback

---

## Step 6: Respond to Feedback (if any)

Make changes locally, commit, and push again. The PR will update automatically.
