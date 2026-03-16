import os
import json
import time
import sqlite3
import requests
import textwrap
import pandas as pd
import yfinance as yf
import streamlit as st
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any

# 加载当前目录下的 .env 文件
load_dotenv()

# 从环境变量获取 Key (本地读取 .env，云端读取 Secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# 如果缺少 Key，在页面上提示并停止运行
if not OPENAI_API_KEY or not ALPHAVANTAGE_API_KEY:
    st.error("⚠️ Missing API Keys! Please ensure OPENAI_API_KEY and ALPHAVANTAGE_API_KEY are set in your .env file or Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
DB_PATH = "stocks.db"

# 模型常量
MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL  # 初始默认模型


@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)   # tool names in order called
    raw_data     : dict  = field(default_factory=dict)   # tool name → raw tool output
    confidence   : float = 0.0                           # set by evaluator / critic
    issues_found : list  = field(default_factory=list)   # set by evaluator / critic
    reasoning    : str   = ""

    def summary(self):
        # 这个方法在 app 开发中可选，但在调试时很有用
        print(f"\n{'─'*54}")
        print(f"Agent      : {self.agent_name}")
        print(f"Tools used : {', '.join(self.tools_called) or 'none'}")
        print(f"Confidence : {self.confidence:.0%}")
        if self.issues_found:
            print(f"Issues     : {'; '.join(self.issues_found)}")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "stocks.db"

# 2. Tools (7 functions)
# ── Tool 1 ── Provided ────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """
    % price change for a list of tickers over a period.
    Valid periods: '1mo', '3mo', '6mo', 'ytd', '1y'
    Returns: { TICKER: {start_price, end_price, pct_change, period} }
    """
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results

# ── Tool 2 ── Provided ────────────────────────────────────────
def get_market_status() -> dict:
    """Open / closed status for global stock exchanges."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 3 ── Provided ────────────────────────────────────────
def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 4 ── Provided ────────────────────────────────────────
def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """
    Latest headlines + Bullish / Bearish / Neutral sentiment for a ticker.
    Returns: { ticker, articles: [{title, source, sentiment, score}] }
    """
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }

# ── Tool 5 ── Provided ────────────────────────────────────────
def query_local_db(sql: str) -> dict:
    """
    Run any SQL SELECT on stocks.db.
    Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange
    market_cap values: 'Large' | 'Mid' | 'Small'
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


# ── Tool 6 — YOUR IMPLEMENTATION ─────────────────────────────
def get_company_overview(ticker: str) -> dict:
    ### YOUR CODE HERE

    try:
        data = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW"
            f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}",
            timeout=10
        ).json()

        print(data)
        if "Name" not in data:
            return {"error": f"No overview data for {ticker}"}

        return {
            "ticker"    : ticker,
            "name"      : data.get("Name"),
            "sector"    : data.get("Sector"),
            "pe_ratio"  : data.get("PERatio"),
            "eps"       : data.get("EPS"),
            "market_cap": data.get("MarketCapitalization"),
            "52w_high"  : data.get("52WeekHigh"),
            "52w_low"   : data.get("52WeekLow"),
        }
    except Exception:
        return {"error": f"No overview data for {ticker}"}


# ── Tool 7 — YOUR IMPLEMENTATION ─────────────────────────────
def get_tickers_by_sector(sector: str) -> dict:
    ### YOUR CODE HERE
    try:
        conn = sqlite3.connect(DB_PATH)

        # 1. exact match on sector
        sql_sector = """
            SELECT ticker, company, industry
            FROM stocks
            WHERE LOWER(sector) = LOWER(?)
        """
        df = pd.read_sql_query(sql_sector, conn, params=(sector,))

        # 2. fallback to industry LIKE
        if df.empty:
            sql_industry = """
                SELECT ticker, company, industry
                FROM stocks
                WHERE LOWER(industry) LIKE LOWER(?)
            """
            df = pd.read_sql_query(sql_industry, conn, params=(f"%{sector}%",))

        conn.close()

        return {
            "sector": sector,
            "stocks": df.to_dict(orient="records")
        }

    except Exception as e:
        return {
            "sector": sector,
            "stocks": [],
            "error": str(e)
        }

def _s(name, desc, props, req):
    return {"type":"function","function":{
        "name":name,"description":desc,
        "parameters":{"type":"object","properties":props,"required":req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector":{"type":"string","description":"Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers":{"type":"array","items":{"type":"string"}},
     "period":{"type":"string","default":"1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker":{"type":"string","description":"Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker":{"type":"string"},"limit":{"type":"integer","default":5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql":{"type":"string","description":"A valid SQL SELECT statement"}}, ["sql"])


# All 7 schemas in one list — used by single agent
ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
               SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]
    
# Dispatch map — maps the tool name string the LLM returns → the Python function to call
ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance"  : get_price_performance,
    "get_company_overview"   : get_company_overview,
    "get_market_status"      : get_market_status,
    "get_top_gainers_losers" : get_top_gainers_losers,
    "get_news_sentiment"     : get_news_sentiment,
    "query_local_db"         : query_local_db,
}

MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_SQL]



# 3. Agent Functions (核心搬运对象)

# - run_specialist_agent
def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    max_iters    : int  = 8,
    verbose      : bool = True,
) -> AgentResult:
    """
    Core agentic loop used by every agent in this project.

    How it works:
      1. Sends system_prompt + task to the LLM
      2. If the LLM requests a tool call → looks up the function in ALL_TOOL_FUNCTIONS,
         executes it, appends the result to the message history, loops back to step 1
      3. When the LLM produces a response with no tool calls → returns an AgentResult

    Parameters
    ----------
    agent_name    : display name for logging
    system_prompt : the agent's persona, rules, and focus area
    task          : the specific question or sub-task for this agent
    tool_schemas  : list of schema dicts this agent is allowed to use
                    (pass [] for no tools — used by baseline)
    max_iters     : hard cap on iterations to prevent infinite loops
    verbose       : print each tool call as it happens
    """
    ### YOUR CODE HERE ###
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools_called = []
    raw_data = {}
    final_answer = ""
    reasoning = ""

    for step in range(max_iters):
        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None,
        )

        msg = response.choices[0].message


        if getattr(msg, "tool_calls", None):
            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")

                if verbose:
                    print(f"[{agent_name}] tool call → {tool_name}({tool_args})")

                if tool_name not in ALL_TOOL_FUNCTIONS:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = ALL_TOOL_FUNCTIONS[tool_name](**tool_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                tools_called.append(tool_name)
                raw_data[f"{tool_name}_{len(tools_called)}"] = tool_result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })

            continue


        final_answer = msg.content if msg.content else ""
        reasoning = final_answer
        break

    else:
        final_answer = f"{agent_name} stopped after reaching max_iters={max_iters}."
        reasoning = final_answer

    return AgentResult(
        agent_name=agent_name,
        answer=final_answer,
        tools_called=tools_called,
        raw_data=raw_data,
        reasoning=reasoning,
    )
print("✅ run_specialist_agent ready")

# - run_single_agent (带历史记录支持)
# ── YOUR SINGLE AGENT IMPLEMENTATION ─────────────────────────
# Write your system prompt and run_single_agent() function here.
# Comments above explain what to think about — the implementation is yours.

### YOUR CODE HERE
SINGLE_AGENT_PROMPT = """
You are a financial analysis agent.

Your job is to answer the user's question accurately using the available tools when needed.

Rules:
- Use tools whenever the question requires live market data, company fundamentals, news sentiment,
  sector/industry lookup, or database filtering.
- Do not make up numbers if tool data is needed.
- If the question involves multiple companies, compare them clearly.
- If the question requires screening/filtering stocks, use the database tools first before answering.
- Base your final answer on the actual tool outputs.
- Keep the final answer concise but informative.
"""

def run_single_agent(question: str, chat_history: list = None, verbose: bool = True) -> AgentResult:
    context_str = ""
    if chat_history:
        for msg in chat_history:
            context_str += f"{msg['role']}: {msg['content']}\n"

    if context_str:
        augmented_task = f"Conversation History:\n{context_str}\nCurrent Question:\n{question}\n\nResolve any references to previous messages."
    else:
        augmented_task = question

    return run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=augmented_task,
        tool_schemas=ALL_SCHEMAS,  # Use all 7 tools provided by the system directly
        max_iters=10,
        verbose=verbose,
    )

print("✅ single-agent ready with chat history support!")

# - run_multi_agent (带编排器逻辑)

# 2. Define roles
ORCHESTRATOR_PROMPT = """
You are the Orchestrator for a financial AI system.
Your job is to read the user's question and break it into sub-tasks for specialized agents.

Available Agents:
- Market: Handles prices, market status, top movers, and sector lookups.
- Fundamentals: Handles company overview (P/E, EPS, market cap) and SQL filtering.
- Sentiment: Handles news sentiment and headlines.

Rules:
- Resolve references like "it", "they", or "those" using the conversation history.
- Return ONLY a valid JSON object with a "tasks" list.
Example: {"tasks": [{"agent": "Market", "task": "Find 1y return for AAPL"}, {"agent": "Fundamentals", "task": "Get P/E for AAPL"}]}
"""

CRITIC_PROMPT = """
You are the Critic. Your job is to verify if the Agent's answer matches the raw tool data.
Check for hallucinations, incorrect numbers, or missing data.

Return ONLY a valid JSON:
{"confidence": float (0.0 to 1.0), "issues_found": ["issue 1", "issue 2"]}
If no issues, return an empty list.
"""

SYNTHESIZER_PROMPT = """
You are the Synthesizer. Combine the specialist answers into one clear, concise final response.
Acknowledge any issues or uncertainties flagged by the Critic.
"""

# 3. 辅助调用函数
def call_orchestrator(question: str, chat_history: list = None) -> list:
    history_str = ""
    if chat_history:
        for m in chat_history:
            history_str += f"{m['role']}: {m['content']}\n"
    
    prompt = f"History:\n{history_str}\n\nQuestion: {question}"
    
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[{"role": "system", "content": ORCHESTRATOR_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content).get("tasks", [])

def call_critic(task: str, answer: str, raw_data: dict) -> dict:
    prompt = f"Task: {task}\nAnswer: {answer}\nRaw Tool Data: {json.dumps(raw_data)}"
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[{"role": "system", "content": CRITIC_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def call_synthesizer(question: str, agent_results: list) -> str:
    summary = ""
    for r in agent_results:
        summary += f"\nAgent {r.agent_name} (Conf: {r.confidence:.0%}): {r.answer}\nIssues: {r.issues_found}"
    
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[{"role": "system", "content": SYNTHESIZER_PROMPT}, 
                  {"role": "user", "content": f"Q: {question}\nData: {summary}"}]
    )
    return response.choices[0].message.content

# 4. 主入口函数 run_multi_agent
def run_multi_agent(question: str, chat_history: list = None, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    if verbose: print(f"策划中: {question}")
    
    # 1. 编排任务
    tasks = call_orchestrator(question, chat_history)
    
    agent_results = []
    # 2. 专家执行与评论家审核
    for t in tasks:
        agent_name = t['agent']
        sub_task = t['task']
        
        # 分配工具组
        tools = MARKET_TOOLS if agent_name == "Market" else (FUNDAMENTAL_TOOLS if agent_name == "Fundamentals" else SENTIMENT_TOOLS)
        sys_prompt = f"You are the {agent_name} Agent. " + (MARKET_AGENT_PROMPT if agent_name == "Market" else "") # 简化版
        
        # 执行专家任务
        res = run_specialist_agent(agent_name, sys_prompt, sub_task, tools, verbose=verbose)
        
        # 评论家审核
        eval_data = call_critic(sub_task, res.answer, res.raw_data)
        res.confidence = eval_data.get("confidence", 0.0)
        res.issues_found = eval_data.get("issues_found", [])
        
        agent_results.append(res)
        
    # 3. 合成答案
    final_answer = call_synthesizer(question, agent_results)
    
    # 遵循返回合同
    return {
        "final_answer"  : final_answer,
        "agent_results" : agent_results,
        "elapsed_sec"   : round(time.time() - t0, 2),
        "architecture"  : "orchestrator-critic"
    }


# 4. Streamlit UI Logic
# - st.sidebar (选择模型和架构)
# - st.chat_message (渲染历史)
# - st.chat_input (获取新问题并调用智能体)
import streamlit as st

# --- 1. 页面配置与标题 ---
st.set_page_config(page_title="FinTech Agentic AI Explorer", layout="wide")
st.title("🏦 Agentic AI in FinTech")
st.markdown("Comparing Baseline, Single-Agent, and Multi-Agent Architectures with Real-time Financial Data.")

# --- 2. 侧边栏配置 (Sidebar) ---
st.sidebar.header("System Settings")

# 架构选择器
agent_choice = st.sidebar.selectbox(
    "Choose Agent Architecture", 
    ["Single Agent", "Multi-Agent"],
    help="Single Agent uses one context; Multi-Agent uses Orchestrator-Critic logic."
)

# 模型选择器
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["gpt-4o-mini", "gpt-4o"],
    index=0
)

# 【关键点】动态更新全局变量 ACTIVE_MODEL，确保 Agent 逻辑调用正确的模型
import __main__
__main__.ACTIVE_MODEL = model_choice

# 清除对话按钮
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.info("This app maintains context up to 3+ turns for follow-up resolution.")

# --- 3. 初始化对话记忆 (Conversational Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # 存储格式: {"role": "user/assistant", "content": "...", "metadata": {...}}

# --- 4. 渲染对话历史 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果是助手回复，显示其使用的架构和模型
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            st.caption(f"⚙️ **Architecture:** {meta['arch']} | 🧠 **Model:** {meta['model']}")

# --- 5. 处理用户输入与 Agent 响应 ---
if prompt := st.chat_input("What is NVIDIA's P/E ratio?"):
    # 1. 展示并存储用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用智能体生成回答
    with st.chat_message("assistant"):
        with st.spinner(f"Running {agent_choice} logic..."):
            # 记录开始时间
            try:
                # 【核心要求】将历史消息传入以实现指代消解
                if agent_choice == "Single Agent":
                    result = run_single_agent(prompt, chat_history=st.session_state.messages)
                    answer = result.answer
                else:
                    # Multi-Agent 返回的是字典
                    result_data = run_multi_agent(prompt, chat_history=st.session_state.messages)
                    answer = result_data["final_answer"]
                
                # 展示回答
                st.markdown(answer)
                
                # 3. 存储助手回复及元数据
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": {"arch": agent_choice, "model": model_choice}
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Ensure your .env keys and stocks.db are correctly configured.")

# 自动滚动到底部
st.empty()