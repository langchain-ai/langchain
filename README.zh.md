1. # 🦜🔗 LangChain

2. 

3. [![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain)](https://github.com/langchain-ai/langchain/stargazers)

4. [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

5. [![PyPI](https://img.shields.io/pypi/v/langchain)](https://pypi.org/project/langchain/)

6. 

7. [English](README.md) | **中文** | [日本語](README.ja.md)

8. 

9. > 构建可靠智能体的平台

10. 

11. [📖 文档](https://docs.langchain.com) | [💬 论坛](https://forum.langchain.com) | [🐦 Twitter](https://x.com/langchain)

12. 

13. ---

14. 

15. ## 快速开始

16. 

17. ```bash

18. pip install langchain

什么是 LangChain？
LangChain 是一个用于构建智能体和大语言模型（LLM）驱动应用程序的框架。它帮助你链式组合可互操作的组件和第三方集成，简化 AI 应用开发——同时在底层技术演进时保持决策的前瞻性。
如果你需要更高级的定制或智能体编排，请查看 https://docs.langchain.com/oss/python/langgraph/overview，这是我们用于构建可控智能体工作流的框架。
为什么选择 LangChain？
LangChain 帮助开发者通过标准接口构建由 LLM 驱动的应用程序，支持模型、嵌入、向量存储等。
使用场景
实时数据增强：轻松将 LLM 连接到多样化的数据源和外部/内部系统，利用 LangChain 与模型提供商、工具、向量存储、检索器等的大量集成库。
模型互操作性：在你的工程团队实验时轻松切换模型，为应用程序需求找到最佳选择。随着行业前沿的发展，快速适应——LangChain 的抽象让你保持前进动力而不失去势头。
快速原型开发：使用可组合的工具和组件库加速开发，这些工具和组件经过实战检验，可实现快速实验。
主要功能
🚀 模型接口
支持 OpenAI、Anthropic、Google、Azure 等主流 LLM
统一的模型调用接口
流式输出支持
🔗 链式组合
将多个组件链接成复杂工作流
条件分支和错误处理
可重用的组件库
📚 检索增强生成（RAG）
向量存储集成（Pinecone、Weaviate、Chroma 等）
文档加载器和分割器
上下文感知的回答生成
🛠️ 工具集成
200+ 预构建工具和集成
自定义工具支持
智能体工具选择和执行
快速示例
1. from langchain import OpenAI, LLMChain, PromptTemplate

2. 

3. # 初始化模型

4. llm = OpenAI(temperature=0.7)

5. 

6. # 创建提示模板

7. template = """

8. 将以下文本翻译成中文：

9. {text}

10. """

11. 

12. prompt = PromptTemplate(template=template, input_variables=["text"])

13. 

14. # 创建链

15. chain = LLMChain(llm=llm, prompt=prompt)

16. 

17. # 运行

18. result = chain.run("Hello, World!")

19. print(result)  # 输出：你好，世界！

安装
1. # 基础安装

2. pip install langchain

3. 

4. # 包含常用集成

5. pip install langchain[community]

6. 

7. # 开发版本

8. pip install langchain --pre

文档
官方文档 - 完整文档，包括概念概述和指南
API 参考 - LangChain 包的 API 参考文档
Chat LangChain - 与 LangChain 文档对话，获取问题答案
社区
论坛 - 与社区联系，分享技术问题、想法和反馈
Discord - 实时聊天和社区支持
Twitter - 关注最新动态
贡献
我们欢迎各种形式的贡献！请查看我们的CONTRIBUTING.md了解如何参与。
许可证
本项目基于 MIT 许可证 开源。
<p align="center">
用 ❤️ 构建，为开发者而生
</p>
1. 
2. ---
3. 
4. **使用方法：**
5. 1. 长按上面内容 → 全选 → 复制
6. 2. 打开GitHub，粘贴到 README.zh.md 文件
7. 3. 提交即可！




