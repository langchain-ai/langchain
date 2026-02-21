# 🦜🔗 LangChain

> 构建可靠智能体的平台

---

## 快速开始

```bash
pip install langchain
```

---

## 什么是 LangChain？

LangChain 是一个用于构建智能体和大语言模型(LLM)驱动应用程序的框架。它帮助你链式组合可互操作的组件和第三方集成，简化 AI 应用开发，在技术演进时确保决策面向未来。

如需更高级的定制或智能体编排，请查看 LangGraph 文档。

---

## 为什么选择 LangChain？

LangChain 帮助开发者通过标准接口构建 LLM 驱动的应用程序，支持模型、嵌入、向量存储等。

### 使用场景

**实时数据增强**：将 LLM 连接到多样化数据源和内外部系统。

**模型互操作性**：在工程团队实验时轻松切换模型，找到最适合应用需求的选择。

**快速原型开发**：使用经过实战检验的可组合工具和组件库加速开发。

---

## 主要功能

- 模型接口：支持 OpenAI、Anthropic、Google、Azure 等主流 LLM
- 链式组合：将多个组件链接成复杂工作流
- 检索增强生成(RAG)：向量存储集成、文档加载器和分割器
- 工具集成：200+ 预构建工具和集成

---

## 快速示例

```python
from langchain import OpenAI, LLMChain, PromptTemplate

llm = OpenAI(temperature=0.7)

template = """将以下文本翻译成中文：{text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("Hello, World!")
print(result)  # 输出：你好，世界！
```

---

## 安装

```bash
# 基础安装
pip install langchain

# 包含常用集成
pip install langchain[community]

# 开发版本
pip install langchain --pre
```

---

## 文档

- 官方文档：docs.langchain.com
- API 参考：api.python.langchain.com
- Chat LangChain：chat.langchain.com

---

## 社区

- 论坛：forum.langchain.com
- Discord：discord.gg/langchain
- Twitter：x.com/langchain

---

## 贡献

欢迎各种形式的贡献！请查看 CONTRIBUTING.md 了解如何参与。

---

## 许可证

本项目基于 MIT 许可证开源。

用 ❤️ 构建，为开发者而生
