#!/usr/bin/env python3
"""
简单的测试 - 不需要 pytest
"""

import sys
import os
import traceback

# 添加当前目录到 Python 路径
sys.path.insert(0, os.getcwd())

print("=" * 50)
print("开始测试 UnDatasIO 集成")
print("=" * 50)

# 测试1: 检查基本导入
try:
    from libs.community.langchain_community.document_loaders.undatasio import UnDatasIODocumentLoader
    print("✓ 1. DocumentLoader 导入成功")
except ImportError as e:
    print(f"✗ 1. DocumentLoader 导入失败: {e}")
    sys.exit(1)

# 测试2: 检查工具导入
try:
    from libs.community.langchain_community.tools.undatasio import UnDatasIOWorkspaceListTool
    print("✓ 2. Tool 导入成功")
except ImportError as e:
    print(f"✗ 2. Tool 导入失败: {e}")
    sys.exit(1)

# 测试3: 创建实例
try:
    loader = UnDatasIODocumentLoader(
        token="test_token",
        task_id="test_task",
        file_id="test_file",
        auto_workflow=False
    )
    print("✓ 3. DocumentLoader 实例创建成功")
except Exception as e:
    print(f"✗ 3. DocumentLoader 实例创建失败: {e}")
    sys.exit(1)

# 测试4: 创建工具实例 - 使用正确的初始化方式
try:
    tool = UnDatasIOWorkspaceListTool(token="sk-mA_ksxX8A-nAeVAN3CEzr6JnZh9u5xdH5LAQAaacFrE")
    print("✓ 4. Tool 实例创建成功")
except Exception as e:
    print(f"✗ 4. Tool 实例创建失败: {e}")
    print(traceback.format_exc())
    sys.exit(1)

print("=" * 50)
print("🎉 所有基本测试通过！")
print("=" * 50)
