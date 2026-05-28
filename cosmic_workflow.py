"""
COSMIC 功能点分解器 - LangGraph 版本
======================================

这个程序替代了 n8n 工作流 "My workflow copy"，把原本在 n8n 里的
AI Agent → Code → AI Agent → Code → Excel 流程，用 LangGraph 重写。

为什么用 LangGraph 替代 n8n：
- 没有节点数量限制（n8n 免费版有节点上限）
- 没有 10 分钟超时限制
- 不需要公开 webhook
- 完全本地运行，数据不外传
- Excel 直接输出到本地文件夹
- 代码可控，可以用 Git 做版本管理

流程:
  用户输入 → AI Agent1(提取功能点表格) → Code(解析表格) → 
  AI Agent2(COSMIC分解) → Code(解析结果) → 输出Excel

运行方式:
  python cosmic_workflow.py                           # 交互模式（一问一答）
  python cosmic_workflow.py "用户需求"                 # 单次运行
  python cosmic_workflow.py --mock "用户需求"          # Mock模式（不调用LLM，测试用）

配置:
  修改 .env 文件来切换 LLM 后端（本地 LM Studio / OpenRouter / 其他）
"""

# =============================================================================
# 导入依赖
# =============================================================================
import re       # 正则表达式，用来解析 LLM 输出的 Markdown 表格
import os       # 读取环境变量
import sys      # 读取命令行参数
import json
from typing import TypedDict, List, Optional  # 类型提示，让代码更清晰
from pathlib import Path  # 跨平台路径处理

import httpx  # HTTP 请求库，用来调用 LLM API（比 requests 轻量）
from dotenv import load_dotenv  # 读取 .env 文件中的配置
import pandas as pd  # 数据处理，最后导出 Excel 就用它

from langgraph.graph import StateGraph, END  # LangGraph 核心


# =============================================================================
# 配置加载
# =============================================================================
# 从 .env 文件读取配置（API Key、模型名称、后端地址）
load_dotenv()

# --- LLM 连接配置 ---
# 默认连本地 LM Studio (http://127.0.0.1:1234/v1)
# 如果想用 OpenRouter，改成:
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1
#   OPENAI_API_KEY=sk-or-v1-你的key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-27b-uncensored-hauhaucs-aggressive")

# --- Mock模式开关 ---
# 传 --mock 参数时启用，用预置数据代替 LLM 调用，方便快速测试
MOCK_MODE = "--mock" in sys.argv or os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")

# 如果传了 --mock，把它从参数列表删掉，免得影响后面的参数处理
if "--mock" in sys.argv:
    sys.argv.remove("--mock")


# =============================================================================
# LangGraph 状态定义
# =============================================================================
# TypedDict 是 Python 的类型提示机制，定义了"状态"里有哪些字段和各自的类型
# LangGraph 的核心概念：每个节点接收一个状态，返回更新后的状态

class CosmicState(TypedDict):
    """整个工作流在运行时携带的数据"""
    user_input: str                    # 用户输入的原始需求文字
    raw_llm_output: str                # AI Agent1 返回的原始 Markdown 表格
    parsed_functions: List[dict]       # 解析后的功能点列表，如 [{"序号": "1", "功能点名称": "xxx", ...}]
    cosmic_output: str                 # AI Agent2 返回的 COSMIC 分解 Markdown 表格
    parsed_cosmic: List[dict]          # 解析后的 COSMIC 子过程列表
    func_summary_output: str           # AI Agent3 返回的功能需求总结 Markdown 表格
    parsed_func_summary: List[dict]    # 解析后的"功能需求→功能点名称"映射列表
    error: Optional[str]               # 如果出错，这里存错误信息


# =============================================================================
# Prompt 模板（AI 的指令）
# =============================================================================
# 这些是从 n8n 工作流里直接复制过来的，完全相同的 prompt 逻辑

# ---- 第一轮 Prompt：让 AI 从用户需求中提取功能点列表 ----
# AI 需要以 Markdown 表格格式输出，表格包含序号、功能点名称和功能点描述三列
# 禁止词列表来自 n8n 工作流的原始配置，这些词语在 COSMIC 方法论中不被视为有效功能点
PROMPT_EXTRACT_FUNCTIONS = """你是一个功能点提取专家。

用户会告诉你需要拆分成多少个功能点，请严格按照用户指定的数量拆，不能多也不能少。
请将以下需求以 markdown 表格形式输出，表头包括 |序号|功能点名称|功能点描述|。
禁止使用以下词语：临时表、缓存表、redis、界面、改造、新增表字段、日志、操作记录、标题、公告、确认、下一页、上一页、配置、接口调用、数据计算、逻辑计算、排序、大小写转换、格式转换、读取格式、解析、封装、美化布局、调整字体大小、记录不存在信息、调用失败操作日志、路径信息的渲染。
注意表格开始和结束都需要有 |。
格式示例：
|1|新建室分名称支持七级地址查询选择|针对"是否新建"为"是"的室分资源点，用户在填写名称时可通过查询框选择七级地址|

用户需求：
{user_input}
"""

# ---- 第二轮 Prompt：让 AI 对每个功能点做 COSMIC 分解 ----
# COSMIC 方法把软件功能分解为 4 种数据移动类型：
#   E (Entry)       → 数据进入系统（动词：输入）
#   R (Read)        → 系统读取已存数据（动词：读取）
#   W (Write)       → 系统保存数据（动词：保存）
#   X (eXit/Output) → 数据离开系统（动词：输出）
# 每个功能点必须：以 E 开头，以 W 或 X 结尾，4~6 个子过程
PROMPT_COSMIC_DECOMPOSE = """{序号}{功能点名称} :{功能点描述} .

将该功能点按 COSMIC 方法拆分成表格，按 E(输入) R(读取) W(写入) X(输出) 四个步骤编写子功能过程。
要求：
1. 第一子过程必须是 E，最后一个子过程必须是 W 或 X
2. 子功能过程个数 4 到 6 个
3. 禁止使用以下词语：临时表、缓存表、redis、界面、改造、新增表字段、日志、操作记录、标题、公告、确认、下一页、上一页、配置、接口调用、数据计算、逻辑计算、排序、大小写转换、格式转换、读取格式、解析、封装、美化布局、调整字体大小、记录不存在信息、调用失败操作日志、路径信息的渲染
4. 每行子过程描述使用 1 到 2 个高级名词组成完整的动宾结构
5. E -> 动词使用"输入"；R -> 动词使用"读取"；W -> 动词使用"保存"；X -> 动词使用"输出"
6. 每个子过程数据组名字各不相同，数据属性 4 到 6 个
7. 子过程描述避免重复，描述长度不等

表头：|序号|功能点名称|触发事件|子过程描述|数据移动类型|数据组|数据属性|
注意每个字段输出都不要为空，触发事件每个功能点之间最好不同，所有字段以中文输出。

示例：
|1|室分规划阶段名称绑定七级地址|室分规划任务提交时触发绑定校验|输入室分规划基本信息|E|室分规划输入数据|规划编号、室分名称、所属站点ID、规划层级、建设类型、创建时间|
"""
# ---- 第三轮 Prompt：对功能点名称总结成 6~10 个功能需求 ----
# 功能点名称可能有 50+ 个，需要归类总结成更上层的功能需求
# AI 需要输出功能需求名称，以及该需求下包含的所有功能点名称列表
PROMPT_SUMMARIZE_REQUIREMENTS = """你是一个需求分析专家。

下面是一个软件项目的所有功能点名称列表（共 {count} 个功能点），请将它们总结归类为 6 到 10 个上层功能需求。

要求：
1. 将功能点按业务逻辑归类，每个功能需求下包含多个功能点
2. 输出格式为 Markdown 表格，表头为：|功能需求|包含的功能点名称|
3. 功能需求名称要准确概括该组功能点的共同业务目标
4. 每个功能需求名称尽量不要重复
5. 【重要】每个功能点的完整名称只能出现在一个功能需求中，绝对不能遗漏任何一个功能点
6. 功能需求数量必须严格在 6 到 10 个之间
7. 输出完毕后，请复核一遍：检查列表中的每个功能点是否都被分配到了某个功能需求中，确保 {count} 个功能点无一遗漏

格式示例：
|功能需求|包含的功能点名称|
|数据接入|家宽历史数据接入, 资费历史数据接入, 无线投诉记录接入|
|数据分析与标签|高频投诉标签生成, 5G低驻留标签生成, 高价值用户标签生成|

功能点名称列表（共 {count} 个）：
{function_names}

请以 Markdown 表格输出。务必覆盖所有 {count} 个功能点，不能少任何一个。"""


# =============================================================================
# 表格解析函数
# =============================================================================
# 这些函数的作用：LLM 输出的是 Markdown 格式的字符串，我们需要把里面的表格
# 解析成 Python 的列表（List[dict]），才能做后续的处理和导出。

def remove_think_tags(text: str) -> str:
    """
    移除 LLM 回复中的 标签及其内容。
    
    某些模型（如 deepseek-r1、部分 qwen 版本）会在正式回复之前输出
    一段"思考过程"，包在 标签里。这些内容不是最终输出，需要去掉。
    
    参数:
        text: LLM 返回的原始文本
    返回:
        移除 标签后的纯文本
    """
    return re.sub(r' thinking.*? response', '', text, flags=re.DOTALL)


def extract_tables(text: str) -> List[str]:
    """
    从文本中提取所有的 Markdown 表格。
    
    Markdown 表格的特征：
    - 每行以 | 开头，以 | 结尾
    - 至少有两行（表头 + 至少一行数据）
    
    这个函数分两步：
    1. 先精确匹配（行首行尾直接是 |）
    2. 如果没找到，用更宽松的方式（先 strip 再匹配），因为有些 AI 会加缩进
    
    参数:
        text: 可能包含表格的文本
    返回:
        每个完整表格的字符串列表
    """
    # 第一步：去掉 thinking 标签，免得被干扰
    text = remove_think_tags(text)
    lines = text.split('\n')
    
    current: List[str] = []
    tables: List[str] = []
    
    # 第一轮：严格匹配（行以 | 开头和结尾）
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            current.append(line)
        else:
            if len(current) >= 2:  # 至少表头+分隔行或表头+数据
                tables.append('\n'.join(current))
            current = []
    if len(current) >= 2:
        tables.append('\n'.join(current))
    
    # 第二轮：如果没找到，用更宽松的匹配（先去掉首尾空格的版本）
    # 有些 LLM 输出的表格带有缩进，strip 后才以 | 开头
    if not tables:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                current.append(stripped)
            else:
                if len(current) >= 2:
                    tables.append('\n'.join(current))
                current = []
        if len(current) >= 2:
            tables.append('\n'.join(current))
    
    return tables


def parse_markdown_table_to_dicts(table_text: str, known_headers: Optional[List[str]] = None) -> List[dict]:
    """
    把一段 Markdown 表格文本解析成字典列表。
    
    支持多种情况：
    1. 标准 Markdown 表格（有分隔行、有表头）
    2. 无分隔行的纯数据行（AI 可能省略 |---| 行）
    3. 第一行就能判定是表头（包含中文关键词）还是数据行
    
    参数:
        table_text: 一个完整的 Markdown 表格字符串
        known_headers: 可选，已知的表头名称列表（如果第一行不是表头就使用这个）
    返回:
        字典列表，每个字典对应表格的一行
    """
    # 按行分割并去掉空行
    lines = [l.strip() for l in table_text.split('\n') if l.strip()]
    
    # 至少需要 2 行
    if len(lines) < 2:
        return []
    
    # 判断第一行是不是真正的表头行
    first_line = lines[0]
    headers = [c.strip().replace('**', '') for c in first_line.split('|') if c.strip()]
    
    # 检查第一行是否看起来像表头（包含中文关键词、全是非数字文本等）
    def looks_like_header(cols: List[str]) -> bool:
        """判断这列是不是表头（不包含纯数字行且包含中文关键词）"""
        # 表头关键词集合
        header_keywords = {'序号', '功能点', '名称', '描述', '触发', '事件', '子过程', 
                          '数据移动', '数据组', '数据属性', '类型', '步骤', '说明'}
        non_numeric_count = 0
        for c in cols:
            c_stripped = c.strip()
            if re.match(r'^\d+$', c_stripped):  # 纯数字 = 数据行
                continue
            non_numeric_count += 1
            # 包含表头关键词 = 确定是表头
            for kw in header_keywords:
                if kw in c_stripped:
                    return True
        # 如果所有列都不是纯数字，也视为表头
        return non_numeric_count >= len(cols)
    
    is_header = looks_like_header(headers)
    
    # 如果第一行不是表头，但有 known_headers，就用已知表头
    if not is_header and known_headers:
        headers = known_headers
        # 第一行其实是数据行，需要重新处理所有行（包括第一行）
        start_line = 0
    elif is_header:
        start_line = 1  # 跳过表头行
    else:
        # 没有已知表头，用 Mock 数据中的标准表头
        headers = ['序号', '功能点名称', '功能点描述']
        start_line = 0
    
    # 处理列解析：注意 split('|') 的结果首尾会有空字符串
    def parse_line(line: str) -> List[str]:
        cols = [c.strip().replace('**', '') for c in line.split('|')]
        cols = [c for c in cols if c]  # 去掉空值
        return cols
    
    result = []
    for line in lines[start_line:]:
        # 跳过分隔行，如 |---|---|
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        
        cols = parse_line(line)
        
        if not cols:
            continue
        
        row = {}
        for j, h in enumerate(headers):
            row[h] = cols[j] if j < len(cols) else ''
        result.append(row)
    
    # 如果第一行没有被当作表头，可能解析出非常少的结果
    # 检查：如果结果太少（<3）但行数很多，可能是表头识别错了
    if not is_header and len(result) < 3 and len(lines) > 3:
        # 尝试用每个列的第一个非空值"自动"当表头
        pass  # 先不管这种情况，极少出现
    
    return result


# =============================================================================
# Mock 数据（测试用，不需要 LLM）
# =============================================================================
# 这些是预制的示例数据，当用 --mock 模式运行时使用。
# 3 个功能点，每个功能点 5 个 COSMIC 子过程，展示了完整的输出格式。

MOCK_FUNCTIONS_OUTPUT = """|序号|功能点名称|功能点描述|
|1|新建室分名称支持七级地址查询选择|针对"是否新建"为"是"的室分资源点，用户在填写名称时可通过查询框选择七级地址|
|2|修改室分地址信息|当用户修改室分资源点的地址信息时，系统自动更新关联的地理编码数据|
|3|删除室分配置校验|删除室分资源点时，系统校验是否存在关联设备或业务数据|
"""

MOCK_COSMIC_OUTPUT = """|序号|功能点名称|触发事件|子过程描述|数据移动类型|数据组|数据属性|
||1|新建室分名称支持七级地址查询选择|新建室分任务提交时触发地址查询选择|输入新建室分申请信息|E|室分新建申请数据|室分名称、所属区域、建设类型、规划层级、创建人员、创建时间|
||1|新建室分名称支持七级地址查询选择|新建室分任务提交时触发地址查询选择|读取七级地址数据库|R|七级地址编码数据|省级编码、市级编码、区级编码、街道编码、社区编码、地址详情|
||1|新建室分名称支持七级地址查询选择|新建室分任务提交时触发地址查询选择|读取室分资源池现有列表|R|室分资源列表|室分配置、所属站点ID、设备数量、安装日期、运行状态|
||1|新建室分名称支持七分地址查询选择|新建室分任务提交时触发地址查询选择|保存新建室分资源记录|W|新建室分资源信息|室分ID、名称、完整地址、站点归属、规划层级、创建时间戳|
||1|新建室分名称支持七级地址查询选择|新建室分任务提交时触发地址查询选择|输出新建室分结果确认|X|室分新建结果反馈|室分编号、关联站点、地址信息、创建状态、创建时间|
"""

MOCK_SUMMARY_OUTPUT = """|功能需求|包含的功能点名称|
|室分地址管理|新建室分名称支持七级地址查询选择, 修改室分地址信息|
|室分配置校验|删除室分配置校验|
"""


# =============================================================================
# LLM 调用封装
# =============================================================================
# 全局 HTTP 客户端连接池，避免频繁创建/关闭连接导致 Windows socket 资源耗尽
# timeout=3600 是因为 52 个功能点的 LLM 推理可能耗时 30-60 分钟
_HTTP_CLIENT = httpx.Client(timeout=3600)

def call_llm(prompt: str) -> str:
    """
    调用 LLM（大语言模型）并返回回复文本。
    
    兼容两种回复格式：
    1. 标准 OpenAI 格式：content 字段包含回复
    2. reasoning 格式（qwen/deepseek 等等）：模型先在 reasoning_content 里
       做思考，然后把最终输出放在 content。但有些模型（或配置）把内容全放在
       reasoning 里，content 为空。这个函数探测这种情况并自动取 reasoning。
    
    参数:
        prompt: 发给 LLM 的完整指令文本
    返回:
        LLM 回复的文本内容
    异常:
        如果 HTTP 请求失败（API 不可用、超时等），会抛出异常
    """
    # 构造 OpenAI API 兼容的请求
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,    # 低温度 = 输出更稳定，不易发散
        "max_tokens": 32768,   # 最多输出 32768 个 token（52 个功能点 + COSMIC 分解可能很大）
    }

    # 使用全局客户端连接池复用连接，避免 Windows socket 资源泄漏
    # timeout=3600 兼容超长推理（52个功能点的 LLM 调用可能耗时 30-60 分钟）
    resp = _HTTP_CLIENT.post(url, json=payload, headers=headers, timeout=3600)
    resp.raise_for_status()  # HTTP 错误时抛出异常
    data = resp.json()

    # ---- 处理多种回复格式 ----
    msg = data["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # 场景1: content 为空但有 reasoning → 取 reasoning
    # 场景2: 两者都有内容，且 content 不含表格但 reasoning 含表格 → 取 reasoning
    # 核心判断依据：谁有 Markdown 表格就取谁
    if not content.strip() and reasoning.strip():
        content = reasoning
    elif content.strip() and reasoning.strip():
        has_table_c = "|" in content and "---" in content[:500]
        has_table_r = "|" in reasoning and "---" in reasoning[:500]
        if not has_table_c and has_table_r:
            content = reasoning
        elif has_table_c and not has_table_r:
            pass  # content 已经有表格，直接用
        elif has_table_c and has_table_r:
            # 两者都有表格，取更长的那个
            pass  # 默认 content 优先

    return content


# =============================================================================
# LangGraph 节点函数
# =============================================================================
# 每个节点函数接收当前的 CosmicState，做自己的工作，然后返回更新的字段。
# LangGraph 会自动合并这些返回值到全局状态中。

def node_extract_functions(state: CosmicState) -> dict:
    """
    节点1: 第一轮 AI 调用 — 提取功能点
    
    输入: state.user_input（用户需求）
    输出: state.raw_llm_output（AI 返回的功能点 Markdown 表格）
    
    处理:
    1. 如果是 Mock 模式，直接返回预制数据
    2. 否则，构造 prompt 并调用 LLM
    """
    print("  -> AI Agent1: 提取功能点...")

    if MOCK_MODE:
        print("  [Mock] 使用预置数据")
        return {"raw_llm_output": MOCK_FUNCTIONS_OUTPUT, "error": None}

    try:
        prompt = PROMPT_EXTRACT_FUNCTIONS.format(user_input=state["user_input"])
        print(f"  Prompt 长度: {len(prompt)} 字符")
        output = call_llm(prompt)
        print(f"  AI 返回长度: {len(output)} 字符")
        # 调试：打印前 300 字符和后 100 字符
        print(f"  AI 返回开头: {output[:300]}")
        print(f"  AI 返回结尾: {output[-200:]}")
        return {"raw_llm_output": output, "error": None}
    except Exception as e:
        return {"error": f"AI Agent1 调用失败: {str(e)}"}


def node_parse_functions(state: CosmicState) -> dict:
    """
    节点2: 解析 AI 返回的 Markdown 表格 → 结构化数据
    
    输入: state.raw_llm_output（Markdown 表格字符串）
    输出: state.parsed_functions（解析后的功能点列表）
    
    这个节点对应 n8n 工作流中的 "Code" 节点。
    """
    print("  -> Code: 解析功能点表格...")
    
    # 如果上一节点报错了，跳过处理
    if state.get("error"):
        return {}

    raw = state.get("raw_llm_output", "")
    tables = extract_tables(raw)
    
    all_functions = []
    for table in tables:
        all_functions.extend(parse_markdown_table_to_dicts(table))

    print(f"  -> 解析到 {len(all_functions)} 个功能点")
    if all_functions:
        for i, f in enumerate(all_functions[:3]):
            print(f"     {i+1}. {f.get('功能点名称', '?')}")
        if len(all_functions) > 3:
            print(f"     ... 还有 {len(all_functions) - 3} 个")
    
    return {"parsed_functions": all_functions}


def node_cosmic_decompose(state: CosmicState) -> dict:
    """
    节点3: 第二轮 AI 调用 — 对每个功能点做 COSMIC 分解
    
    输入: state.parsed_functions（功能点列表）
    输出: state.cosmic_output（所有功能点的 COSMIC 分解 Markdown 表格）
    
    处理:
    1. 遍历每个功能点
    2. 对每个功能点构造 COSMIC 分解 prompt
    3. 调用 LLM 得到分解结果
    4. 合并所有功能的分解结果
    """
    print("  -> AI Agent2: COSMIC 分解...")

    if state.get("error"):
        return {}

    functions = state.get("parsed_functions", [])
    if not functions:
        return {"cosmic_output": "无功能点需要分解", "error": "无输入数据"}

    if MOCK_MODE:
        print("  [Mock] 使用预置数据")
        return {"cosmic_output": MOCK_COSMIC_OUTPUT}

    all_cosmic_results = []
    for func in functions:
        seq = func.get("序号", "")
        name = func.get("功能点名称", "")
        desc = func.get("功能点描述", "")
        
        # 构造 prompt，填入该功能点的序号、名称和描述
        prompt = PROMPT_COSMIC_DECOMPOSE.format(序号=seq, 功能点名称=name, 功能点描述=desc)
        print(f"    分解: [{seq}] {name}")
        
        try:
            output = call_llm(prompt)
            all_cosmic_results.append({"功能点": name, "原始输出": output})
        except Exception as e:
            all_cosmic_results.append({"功能点": name, "原始输出": f"调用失败: {str(e)}"})

    # 把所有功能点的分解结果拼成一段文本，用 --- 分隔
    combined = "\n\n---\n\n".join(
        f"## {r['功能点']}\n{r['原始输出']}" for r in all_cosmic_results
    )
    return {"cosmic_output": combined}


def node_parse_cosmic(state: CosmicState) -> dict:
    """
    节点4: 解析 COSMIC 分解表格 → 结构化数据
    
    输入: state.cosmic_output（Markdown 表格字符串）
    输出: state.parsed_cosmic（解析后的 COSMIC 子过程列表）
    
    这个节点对应 n8n 工作流中第二个 "Code" 节点。
    逻辑和 node_parse_functions 完全一样，只是处理的源数据不同。
    """
    print("  -> Code: 解析 COSMIC 结果...")
    
    if state.get("error"):
        return {}

    raw = state.get("cosmic_output", "")
    tables = extract_tables(raw)
    
    all_cosmic = []
    for table in tables:
        all_cosmic.extend(parse_markdown_table_to_dicts(table))

    print(f"  -> 解析到 {len(all_cosmic)} 个 COSMIC 子过程")
    return {"parsed_cosmic": all_cosmic}


def node_summarize_function_names(state: CosmicState) -> dict:
    """
    节点5: 第三轮 AI 调用 — 将功能点名称总结成 6~10 个功能需求

    输入: state.parsed_functions（功能点列表）
    输出: state.func_summary_output, state.parsed_func_summary

    流程:
    1. 从 parsed_functions 提取所有功能点名称
    2. 调用 LLM 总结归类
    3. 解析 AI 返回的 Markdown 表格
    """
    print("  -> AI Agent3: 总结功能需求...")

    if state.get("error"):
        return {}

    functions = state.get("parsed_functions", [])
    if not functions:
        return {"func_summary_output": "", "parsed_func_summary": []}

    # 提取所有功能点名称
    func_names = [f.get("功能点名称", f.get("名称", "")) for f in functions]
    func_names = [n for n in func_names if n]  # 去空
    func_name_lines = "\n".join(func_names)

    if MOCK_MODE:
        print(f"  [Mock] 使用预置数据")
        # 解析 Mock 总结表格
        tables = extract_tables(MOCK_SUMMARY_OUTPUT)
        parsed = []
        for table in tables:
            parsed.extend(parse_markdown_table_to_dicts(table))
        return {"func_summary_output": MOCK_SUMMARY_OUTPUT, "parsed_func_summary": parsed}

    try:
        prompt = PROMPT_SUMMARIZE_REQUIREMENTS.format(
            count=len(func_names),
            function_names=func_name_lines
        )
        print(f"  Prompt 长度: {len(prompt)} 字符")
        output = call_llm(prompt)
        print(f"  AI 返回长度: {len(output)} 字符")

        # 解析 AI 返回的表格
        tables = extract_tables(output)
        parsed = []
        for table in tables:
            parsed.extend(parse_markdown_table_to_dicts(table))
        print(f"  -> 解析到 {len(parsed)} 个功能需求")

        return {"func_summary_output": output, "parsed_func_summary": parsed}
    except Exception as e:
        print(f"  AI Agent3 调用失败: {e}")
        return {"func_summary_output": f"调用失败: {str(e)}", "parsed_func_summary": []}


def node_output(state: CosmicState) -> dict:
    """
    节点6: 最终输出 — 打印到终端 + 导出 Excel（含"所属功能需求"列）

    输入: state.parsed_functions, state.parsed_cosmic, state.parsed_func_summary
    输出: 无（只做输出，不改状态）

    做四件事:
    1. 打印功能点列表
    2. 打印功能需求总结表格
    3. 打印完整的 COSMIC 子过程表格（含所属功能需求列）
    4. 导出到 output/cosmic_result.xlsx
    """
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

    if state.get("error"):
        print(f"错误: {state['error']}")
        return state

    functions = state.get("parsed_functions", [])
    cosmic = state.get("parsed_cosmic", [])
    summary = state.get("parsed_func_summary", [])

    print(f"\n原始功能点: {len(functions)} 个")
    print(f"COSMIC 子过程: {len(cosmic)} 个")
    print(f"功能需求分类: {len(summary)} 个")

    # ---- 打印功能点列表 ----
    if functions:
        print("\n--- 功能点列表 ---")
        for f in functions:
            print(f"  [{f.get('序号','?')}] {f.get('功能点名称','')}")

    # ---- 打印功能需求总结表格 ----
    if summary:
        print("\n--- 功能需求总结 ---")
        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False))

    # ---- 构建"功能点名称 → 功能需求"映射字典 ----
    # parsed_func_summary 格式: [{"功能需求": "数据接入", "包含的功能点名称": "名称1, 名称2, ..."}, ...]
    func_to_requirement = {}  # {功能点名称: 功能需求名称}
    for s in summary:
        req_name = s.get("功能需求", "")
        # 获取"包含的功能点名称"列（可能列名有差异）
        included = s.get("包含的功能点名称", s.get("包含的功能点", ""))
        if not included:
            # 尝试其他可能的列名
            for key in s:
                if "功能点" in key and key != "功能需求":
                    included = s[key]
                    break
        if req_name and included:
            # 分割功能点名称（逗号分隔）
            parts = [p.strip() for p in included.replace("，", ",").split(",")]
            parts = [p for p in parts if p]
            for p in parts:
                func_to_requirement[p] = req_name

    # ---- 打印 COSMIC 表格（含所属功能需求列） ----
    if cosmic:
        print("\n--- COSMIC 子过程（含所属功能需求） ---")
        df = pd.DataFrame(cosmic)

        # ---- 先将 AI 总结中的"包含的功能点名称"条目去首数字，构建精简索引 ----
        # 这样 "8分客群满意度占比统计模块" 也能匹配 "8分客群满意度分布统计模块"
        def strip_leading_digit(name: str) -> str:
            return re.sub(r'^\d+', '', name)

        # 构建模糊索引：key -> (原key, 功能需求)
        # 索引所有可能的别名形式
        fuzzy_index = []  # [(striped_key, original_key, req_name)]
        for fn_key, req_val in func_to_requirement.items():
            fuzzy_index.append((fn_key, fn_key, req_val))
            stripped_k = strip_leading_digit(fn_key)
            if stripped_k != fn_key:
                fuzzy_index.append((stripped_k, fn_key, req_val))

        def fuzzy_match(func_name: str) -> str:
            """多级模糊匹配，返回功能需求名称或空字符串"""
            # 0级：精确匹配
            if func_name in func_to_requirement:
                return func_to_requirement[func_name]

            # 1级：去首数字后匹配
            stripped = strip_leading_digit(func_name)
            if stripped in func_to_requirement:
                return func_to_requirement[stripped]

            # 2级：遍历所有索引 key，用最长公共子串判断
            # 先算出 COSMIC 表中的名称去首数字后的核心词
            func_core = stripped  # 如 "分客群满意度分布统计模块"

            best_match = ""
            best_len = 0
            for idx_key, orig_key, req_val in fuzzy_index:
                idx_core = strip_leading_digit(idx_key) if idx_key != orig_key else idx_key
                # 双向包含：一方含另一方（处理"占比" vs "分布"这类差异）
                if idx_core in func_core or func_core in idx_core:
                    match_len = max(len(idx_core), len(func_core))
                    if match_len > best_len:
                        best_len = match_len
                        best_match = req_val
                else:
                    # 最长公共子串匹配（处理部分字不同的情况）
                    # 如 "分客群满意度占比统计模块" vs "分客群满意度分布统计模块"
                    # 只有1个字不同，公共子串维度很高
                    shorter = min(len(idx_core), len(func_core))
                    if shorter >= 6:  # 至少6个字长才算
                        # 计算两字符串的公共字符比例（忽略顺序）
                        common = len(set(idx_core) & set(func_core))
                        ratio = common / len(set(func_core))
                        if ratio >= 0.75:  # 公共字符比例 ≥ 75%
                            if shorter > best_len:
                                best_len = shorter
                                best_match = req_val

            if best_match:
                return best_match

            # 3级：子串匹配（更宽松）
            # 取总结 key 中的核心词（去掉序号后），看是否在 func_name 中
            func_lower = func_name.lower()
            for idx_key, orig_key, req_val in fuzzy_index:
                key_lower = idx_key.lower()
                # 确保 key 至少有 4 个字以上避免误配
                if len(key_lower) >= 4 and key_lower in func_lower:
                    return req_val

            return ""

        # ---- 动态兜底：对于总结表中未出现的功能点，按关键词归类 ----
        # 从已匹配的 func_to_requirement 提取关键词 → 功能需求的对应关系
        # 这样 "集团系统报告上传" 即使总结表漏了，也能通过关键词匹配
        keyword_to_req = {}  # {关键词: 功能需求名称}
        for fn_key, req_val in func_to_requirement.items():
            for keyword in fn_key.replace(",", "，").split("，"):
                keyword = keyword.strip()
                if len(keyword) >= 3:
                    keyword_to_req[keyword] = req_val

        # 对缺失的功能点按关键词归类
        def fallback_match(func_name: str) -> str:
            """兜底：从已匹配的功能点名称中找关键词匹配"""
            for keyword, req_val in keyword_to_req.items():
                if keyword in func_name or func_name in keyword:
                    return req_val
            # 再试：对功能点名称按常用业务关键词归类
            business_keywords = {
                "报告": "满意度报告自动化生成与管理",
                "邮件": "满意度报告自动化生成与管理",
                "地市级": "满意度报告自动化生成与管理",
                "省级": "满意度报告自动化生成与管理",
                "质检": "语音质检与工单闭环处理",
                "录音": "语音质检与工单闭环处理",
                "回访": "语音质检与工单闭环处理",
                "投诉": "语音质检与工单闭环处理",
                "工单": "语音质检与工单闭环处理",
                "接入": "数据接入与外部系统对接",
                "上传": "数据接入与外部系统对接",
                "同步": "数据接入与外部系统对接",
                "权限": "系统基础配置与权限控制",
                "配置": "系统基础配置与权限控制",
                "模板": "系统基础配置与权限控制",
                "看板": "可视化监控与态势感知",
                "态势": "可视化监控与态势感知",
                "监控": "可视化监控与态势感知",
                "展示": "可视化监控与态势感知",
                "首页": "可视化监控与态势感知",
                "刷新": "可视化监控与态势感知",
                "诊断": "多维数据分析与诊断洞察",
                "分析": "多维数据分析与诊断洞察",
                "画像": "多维数据分析与诊断洞察",
                "标签": "多维数据分析与诊断洞察",
                "识别": "多维数据分析与诊断洞察",
                "关联": "多维数据分析与诊断洞察",
                "根因": "多维数据分析与诊断洞察",
                "引擎": "多维数据分析与诊断洞察",
            }
            for kw, req_val in business_keywords.items():
                if kw in func_name:
                    return req_val
            return ""

        req_column = []
        missing_names = set()
        for _, row in df.iterrows():
            func_name = row.get("功能点名称", "")
            req = fuzzy_match(func_name)
            if not req:
                req = fallback_match(func_name)
            if not req:
                missing_names.add(func_name)
            req_column.append(req)

        if missing_names:
            print(f"\n  [警告] {len(missing_names)} 个功能点未能匹配到功能需求:")
            for n in sorted(missing_names):
                print(f"    - {n}")

        df["所属功能需求"] = req_column
        print(df.to_string(index=False))

        # ---- 导出 Excel ---
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        excel_path = output_dir / "cosmic_result.xlsx"
        df.to_excel(excel_path, index=False, sheet_name='COSMIC子过程')
        print(f"\nExcel 已导出: {excel_path.resolve()}")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  新增列: 所属功能需求")

    return state


# =============================================================================
# 构建 LangGraph
# =============================================================================
# LangGraph 用图（Graph）来定义工作流。
# 每个节点是一个处理步骤，有向边定义了节点的执行顺序。

def build_cosmic_graph() -> StateGraph:
    """
    构建 COSMIC 分解工作流图。

    图的结构:
    extract_functions → parse_functions → cosmic_decompose → parse_cosmic
    → summarize_requirements → output_results → END

    这是一个单纯的"流水线"（pipeline），没有条件分支（if/else）或循环。
    LangGraph 支持更复杂的图（循环、条件分支、并行），但这个工作流用
    线性 pipeline 就够了。

    返回:
        一个可编译的 StateGraph 对象
    """
    builder = StateGraph(CosmicState)

    # 注册节点
    builder.add_node("extract_functions", node_extract_functions)        # 第1步: AI提取功能点
    builder.add_node("parse_functions", node_parse_functions)            # 第2步: 解析表格
    builder.add_node("cosmic_decompose", node_cosmic_decompose)          # 第3步: AI作COSMIC分解
    builder.add_node("parse_cosmic", node_parse_cosmic)                  # 第4步: 解析表格
    builder.add_node("summarize_requirements", node_summarize_function_names)  # 第5步: AI总结功能需求
    builder.add_node("output_results", node_output)                      # 第6步: 输出结果

    # 设置入口点和边
    builder.set_entry_point("extract_functions")
    builder.add_edge("extract_functions", "parse_functions")             # 第1步→第2步
    builder.add_edge("parse_functions", "cosmic_decompose")              # 第2步→第3步
    builder.add_edge("cosmic_decompose", "parse_cosmic")                 # 第3步→第4步
    builder.add_edge("parse_cosmic", "summarize_requirements")           # 第4步→第5步
    builder.add_edge("summarize_requirements", "output_results")         # 第5步→第6步
    builder.add_edge("output_results", END)                              # 第6步→结束

    return builder


# =============================================================================
# 运行入口
# =============================================================================

def run_once(user_input: str) -> dict:
    """
    单次运行：输入用户需求，执行整个工作流，返回结果。
    
    参数:
        user_input: 用户输入的需求描述文字
    返回:
        最终的 CosmicState（包含解析结果）
    """
    builder = build_cosmic_graph()
    graph = builder.compile()  # "编译"图，准备执行
    
    # 构造初始状态
    initial_state: CosmicState = {
        "user_input": user_input,
        "raw_llm_output": "",
        "parsed_functions": [],
        "cosmic_output": "",
        "parsed_cosmic": [],
        "func_summary_output": "",
        "parsed_func_summary": [],
        "error": None,
    }

    # 执行整个图（自动跑 6 个节点）
    return graph.invoke(initial_state)


def run_interactive():
    """
    交互模式：类似 n8n 的 Chat Trigger。

    输入需求后先展示内容，提供确认/修改/重输的机会。
    支持多行粘贴（输入空行或文件结束符结束输入）。
    按 Ctrl+C 退出。
    """
    print("=" * 60)
    print("COSMIC 功能点分解器 (LangGraph)")
    if MOCK_MODE:
        print("  Mode: [Mock] 无需API Key")
    else:
        print(f"  Model: {LLM_MODEL}")
        print(f"  API:   {OPENAI_BASE_URL}")
    print("=" * 60)
    print()
    print("输入功能点需求，可粘贴大段文字。")
    print("多行输入时，输入空行（直接按 Enter）结束输入。")
    print("按 Ctrl+C 退出")
    print()

    while True:
        try:
            # ---- 收集输入（支持多行粘贴，允许中间有空行段落） ----
            # 结束标志：连续按两次 Enter（产生一个空行）
            # 如果粘贴内容本身有段落空行，连续按三次 Enter 结束
            print("-" * 40)
            lines = []
            empty_line_count = 0
            first_line_skipped = False
            while True:
                try:
                    line = input()
                except EOFError:
                    # Windows git-bash: Ctrl+Z + Enter 触发 EOF
                    break
                if not line:
                    if not first_line_skipped:
                        # 第一行就是空行，忽略（跳过开头的空行）
                        continue
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        # 连续两个空行 → 结束输入
                        break
                    # 第一个空行保留（段落之间的分隔符）
                    lines.append("")
                else:
                    first_line_skipped = True
                    empty_line_count = 0  # 非空行重置计数
                    lines.append(line)

            # 去掉结尾多余的空行
            while lines and lines[-1] == "":
                lines.pop()

            user_input = "\n".join(lines).strip()

            # ---- 对单行输入的兼容处理 ----
            # 如果用户只输入了一行（然后按了两次 Enter 结束），也接受
            if not user_input and lines:
                user_input = lines[0].strip()

            if not user_input:
                continue

            # ---- 展示确认 ----
            print()
            print("=" * 60)
            print("已收到需求（共 %d 字）：" % len(user_input))
            print("=" * 60)
            # 展示前 10 行（太长就截断）
            preview_lines = user_input.split("\n")
            for i, l in enumerate(preview_lines[:10]):
                print("  | " + l)
            if len(preview_lines) > 10:
                print("  | ... (还有 %d 行)" % (len(preview_lines) - 10))
            print()

            # ---- 确认/修改/重输 ----
            while True:
                action = input("操作：[Enter] 直接执行  [e] 编辑修改  [r] 重新输入  [q] 取消: ").strip().lower()

                if action == "":
                    # 直接执行
                    print()
                    result = run_once(user_input)
                    if result.get("error"):
                        print(f"\n处理出错: {result['error']}")
                    print("\n" + "=" * 60)
                    break

                elif action == "e":
                    # 编辑修改 - 显示原文让用户输入替换
                    print()
                    print("当前需求内容（可复制后修改）：")
                    print(user_input)
                    print()
                    print("请输入完整的新需求（或输入空行取消修改）：")
                    edit_lines = []
                    while True:
                        e_line = input()
                        if not e_line and edit_lines:
                            break
                        if not e_line and not edit_lines:
                            edit_lines = [""]
                            break
                        edit_lines.append(e_line)
                    new_input = "\n".join(edit_lines).strip()
                    if new_input:
                        user_input = new_input
                        print(f"\n已更新为（{len(user_input)} 字）：{user_input[:100]}...")
                    else:
                        print("取消修改，保持原内容")
                    break

                elif action == "r":
                    # 重新输入
                    print()
                    print("请重新输入需求（多行输入时空行结束）：")
                    break  # 跳出确认循环，回到主循环顶部

                elif action == "q":
                    # 取消
                    print("\n已取消")
                    break

            if action == "r":
                # 重新输入，回到主循环顶部重新读取输入
                continue
            elif action == "q":
                # 取消，也回到主循环顶部
                continue

        except KeyboardInterrupt:
            print("\n\n退出。")
            break


if __name__ == "__main__":
    """
    程序入口。
    
    使用方式:
    python cosmic_workflow.py                    → 交互模式
    python cosmic_workflow.py "需求描述"          → 单次运行
    python cosmic_workflow.py --mock "需求描述"    → Mock模式测试
    """
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        print(f"输入: {user_input}")
        if MOCK_MODE:
            print("模式: Mock (无需 API Key)")
        run_once(user_input)
    else:
        run_interactive()
