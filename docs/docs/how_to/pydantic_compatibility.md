# 📦 Using LangChain with Different Pydantic Versions

As of the `v0.3` release, **LangChain fully adopts Pydantic v2** for its internal data modeling.

## ✅ Recommended Setup

To ensure full compatibility and avoid runtime issues:

- **Install Pydantic v2** in your environment.
- **Avoid using the `pydantic.v1` namespace**, even though it's available in Pydantic v2 for backward compatibility.

LangChain APIs are optimized for the Pydantic v2 interface and may not behave correctly with older-style models from `pydantic.v1`.

---

## 🕰 Working with Older Versions

If you are using **LangChain v0.2.x or earlier**, those versions still use **Pydantic v1** under the hood.

➡️ Refer to this detailed guide on version compatibility:  
🔗 [Pydantic Compatibility Guide](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility)

---

## ⚠️ Summary of Key Guidelines

| LangChain Version | Required Pydantic Version | Notes                            |
|-------------------|---------------------------|----------------------------------|
| `>= 0.3.0`        | **Pydantic v2**           | Do **not** use `pydantic.v1`    |
| `< 0.3.0`         | **Pydantic v1**           | Follow legacy model patterns     |

---

## 💡 Tips

- If you're migrating an existing codebase from LangChain v0.2 to v0.3+, test your Pydantic models for v2 compatibility.
- Use the [Pydantic v2 migration guide](https://docs.pydantic.dev/latest/migration/) to refactor legacy models.
- Avoid mixing both `BaseModel` versions in a shared LangChain workflow.

---

## 🧪 Check Your Pydantic Version

```bash
pip show pydantic
