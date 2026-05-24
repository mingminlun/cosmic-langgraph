"""
COSMIC 功能点分解 - LangGraph 版本
简化测试版
"""
import re, json, os, sys

# ============================================================
# 表格解析函数
# ============================================================

def remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_tables(text: str):
    text_without_think = remove_think_tags(text)
    lines = text_without_think.split('\n')
    current_table, tables = [], []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            current_table.append(line)
        else:
            if len(current_table) >= 2:
                tables.append('\n'.join(current_table))
            current_table = []
    if len(current_table) >= 2:
        tables.append('\n'.join(current_table))
    return tables

def parse_markdown_table(table_text: str):
    lines = [l.strip() for l in table_text.split('\n') if l.strip()]
    if len(lines) < 2:
        return []
    headers = [c.strip().replace('**', '') for c in lines[0].split('|') if c.strip()]
    result = []
    for line in lines[1:]:
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        cols = [c.strip().replace('**', '') for c in line.split('|')]
        cols = [c for c in cols if c]
        if cols:
            row = {}
            for j, h in enumerate(headers):
                row[h] = cols[j] if j < len(cols) else ''
            result.append(row)
    return result

# ============================================================
# 测试 LLM 调用
# ============================================================

import httpx

def call_llm(prompt: str, model: str = "qwen3.6-27b-uncensored-hauhaucs-aggressive",
             base_url: str = "http://127.0.0.1:1234/v1"):
    """调用本地 LM Studio"""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    resp = httpx.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    # 测试输入
    user_input = "新建室分时名称支持七级地址查询选择，修改室分时更新地址信息"

    # 第一轮：提取功能点
    print("=" * 50)
    print("第一轮: 提取功能点...")
    prompt1 = f"""以markdown表格形式输出,表头包括 |序号|功能点名称|功能点描述|,
不要使用"临时表"、"缓存表"、"redis"、"界面"、"改造"、"新增表字段"、"日志"、
"操作记录"、"标题"、"公告"这些词语，
注意表格开始结束都需要有|，
例如：|1|新建室分名称支持七级地址查询选择|针对"是否新建"为"是"的室分资源点，
用户在填写名称时可通过查询框选择七级地址

用户需求：
{user_input}"""

    output1 = call_llm(prompt1)
    print(f"AI 输出:\n{output1}\n")

    # 解析表格
    tables = extract_tables(output1)
    functions = []
    for t in tables:
        functions.extend(parse_markdown_table(t))
    
    print(f"解析到 {len(functions)} 个功能点:")
    for f in functions:
        print(f"  [{f.get('序号','?')}] {f.get('功能点名称','')}")

    if not functions:
        print("未解析到功能点，退出")
        return

    # 第二轮：COSMIC 分解
    print("\n" + "=" * 50)
    print("第二轮: COSMIC 分解...")
    
    all_cosmic = []
    for func in functions[:1]:  # 只处理第一个用于演示
        seq = func.get("序号", "")
        name = func.get("功能点名称", "")
        desc = func.get("功能点描述", "")
        
        prompt2 = f"""{seq}{name}:{desc}.
将该功能点按COSMIC拆分成表格,按E R W X四个步骤,
第一子过程必须是E，最后一个子过程必须是W或者X,
子过程个数4到6个,
表头为 |序号|功能点名称|触发事件|子过程描述|数据移动类型|数据组|数据属性|,
所有字段以中文输出
        
例如：
|1|室分规划阶段名称绑定七级地址|室分规划任务提交时触发|输入室分规划基本信息|E|室分规划输入数据|规划编号、室分名称|"""

        print(f"处理: {name}")
        output2 = call_llm(prompt2)
        print(f"AI 输出:\n{output2}\n")

        tables2 = extract_tables(output2)
        for t in tables2:
            all_cosmic.extend(parse_markdown_table(t))

    print(f"\n总共 {len(all_cosmic)} 个 COSMIC 子过程")
    if all_cosmic:
        # 打印结果
        headers = list(all_cosmic[0].keys())
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "|" + "|".join("---" for _ in headers) + "|"
        print(f"\n{header_line}")
        print(sep_line)
        for row in all_cosmic:
            vals = [row.get(h, "") for h in headers]
            print("| " + " | ".join(vals) + " |")

if __name__ == "__main__":
    main()
