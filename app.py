import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI, Timeout
from supabase import create_client, Client
from datetime import datetime
import base64
from PIL import Image
import io
import akshare as ak
import os
import difflib
from collections import defaultdict

# ====================== 使用 Streamlit Secrets 管理密钥 ======================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
BAIDU_OCR_API_KEY = st.secrets["BAIDU_OCR_API_KEY"]
BAIDU_OCR_SECRET_KEY = st.secrets["BAIDU_OCR_SECRET_KEY"]
# =============================================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
st.set_page_config(page_title="基金智能管理系统", layout="wide")
st.title("📈 基金智能管理系统")

# ---------------------- 百度OCR（高精度优先，失败降级通用）----------------------
def get_baidu_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_OCR_API_KEY,
        "client_secret": BAIDU_OCR_SECRET_KEY
    }
    try:
        response = requests.post(url, params=params, timeout=10)
        res_json = response.json()
        if "access_token" in res_json:
            st.sidebar.success("✅ OCR Token获取成功")
            return res_json["access_token"]
        else:
            st.sidebar.error(f"❌ OCR Token失败: {res_json}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ OCR Token请求异常: {e}")
        return None

def ocr_image(image_file):
    access_token = get_baidu_access_token()
    if not access_token:
        st.error("OCR初始化失败，请检查百度API密钥")
        return ""
    try:
        img = Image.open(image_file)
        img.thumbnail((2048, 2048))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"图片处理失败: {e}")
        return ""

    # 优先高精度，失败降级通用
    for ocr_type, url in [("高精度", "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"),
                          ("通用", "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic")]:
        try:
            params = {"access_token": access_token}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {"image": img_base64, "probability": "false"}
            response = requests.post(url, params=params, headers=headers, data=data, timeout=15)
            result = response.json()
            if "words_result" in result:
                full_text = "\n".join([item["words"] for item in result["words_result"]])
                st.sidebar.subheader(f"📝 OCR原始识别结果（{ocr_type}）")
                st.sidebar.text_area("OCR全文", full_text, height=200)
                return full_text
            else:
                if ocr_type == "高精度":
                    st.sidebar.warning(f"⚠️ 高精度OCR失败，降级尝试通用OCR... ({result.get('error_msg', '')})")
                    continue
                else:
                    st.error(f"OCR识别失败: {result}")
                    return ""
        except Exception as e:
            if ocr_type == "高精度":
                st.sidebar.warning(f"⚠️ 高精度OCR异常，降级尝试通用OCR... ({e})")
                continue
            else:
                st.error(f"OCR请求异常: {e}")
                return ""
    return ""

# ---------------------- DeepSeek AI 零脑补提取 ----------------------
def get_deepseek_client():
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=Timeout(30.0, connect=10.0, read=30.0),
        max_retries=2
    )

def parse_portfolio_by_ai(ocr_text: str) -> list:
    client = get_deepseek_client()
    prompt = f"""
你是一个严格的基金持仓信息提取工具，必须100%遵循以下规则，禁止任何自由发挥：

1. 从以下支付宝基金持仓页面的OCR文字中，提取每一只基金的【完整原始名称】和【持仓市值】。
2. 基金名称规则：
   - 必须完整保留OCR原文中的所有文字，包括括号、QDII、C、ETF、芯片、纳斯达克100、优选、先锋等所有关键词，绝对禁止修改、缩写、替换、脑补名称。
   - 基金名称可能分为上下两行，必须把相邻的基金名称行合并成完整名称，禁止拆分。
3. 持仓市值规则：
   - 提取基金名称右侧的第一个带逗号的数字，单位为元，只保留纯数字，去掉逗号和¥符号。
   - 只提取持仓总市值，忽略昨日收益、持有收益、收益率等其他数字。
4. 过滤规则：
   - 忽略“市场解读”、“定投”、“基金市场”、“持有”、“自选”、“机会”等无关内容。
   - 只提取持仓市值大于1000元的基金，低于1000元的直接过滤。

返回格式：纯JSON数组，每个元素严格按照以下结构，禁止输出任何其他文字、解释、注释：
[
    {{"name": "基金完整原始名称", "market_value": 持仓市值数字}}
]
如果没有识别到有效基金，返回空数组[]。

OCR原始文字内容：
{ocr_text}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        st.sidebar.subheader("🤖 AI提取结果")
        st.sidebar.code(content)
        
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            funds = json.loads(json_match.group())
            valid_funds = []
            for f in funds:
                if isinstance(f, dict) and "name" in f and "market_value" in f:
                    if isinstance(f["market_value"], (int, float)) and f["market_value"] > 1000:
                        valid_funds.append(f)
            return valid_funds
        return []
    except Exception as e:
        st.error(f"AI解析失败: {e}")
        st.sidebar.error(f"AI异常详情: {e}")
        return []

# ---------------------- 全量基金列表加载+标准化（含币种字段）----------------------
@st.cache_data(ttl=3600)
def load_full_fund_list():
    raw_df = pd.DataFrame()
    if os.path.exists("fund_full_list.csv"):
        try:
            raw_df = pd.read_csv("fund_full_list.csv", dtype={"基金代码": str})
            raw_df.columns = raw_df.columns.str.strip()
            if not raw_df.empty and "基金代码" in raw_df.columns and "基金简称" in raw_df.columns:
                st.sidebar.success(f"✅ 本地CSV加载成功 (共{len(raw_df)}只)")
            else:
                st.sidebar.error("❌ CSV列名错误，必须包含【基金代码, 基金简称】")
                raw_df = pd.DataFrame()
        except Exception as e:
            st.sidebar.error(f"❌ CSV读取失败: {e}")
            raw_df = pd.DataFrame()
    
    if raw_df.empty:
        st.sidebar.info("🔄 正在使用AkShare获取最新基金列表...")
        try:
            raw_df = ak.fund_name_em()
            raw_df = raw_df[["基金代码", "基金简称"]].copy()
        except Exception as e:
            st.sidebar.error(f"❌ AkShare获取失败: {e}")
            return pd.DataFrame()

    def standardize_fund_name(full_name):
        name = str(full_name)
        name = name.translate(str.maketrans('（）【】', '()[]'))
        company_list = ["建信", "华夏", "广发", "易方达", "嘉实", "汇添富", "南方", "博时", "富国", "工银瑞信"]
        company = ""
        for comp in company_list:
            if comp in name:
                company = comp
                break
        share_type = "C" if "C" in name or "C类" in name else "A"
        fund_type = ""
        if "ETF联接" in name:
            fund_type = "ETF联接"
        elif "指数增强" in name:
            fund_type = "指数增强"
        elif "混合" in name:
            fund_type = "混合"
        elif "股票" in name:
            fund_type = "股票"
        currency = "美元" if "美元" in name else "人民币"
        core_target = name
        core_target = re.sub(r'^(' + '|'.join(company_list) + ')', '', core_target)
        core_target = re.sub(r'(混合|ETF联接|指数增强|股票|QDII|A类|C类|A|C|发起式|美元|人民币|\(|\)|\s)', '', core_target)
        clean_name = re.sub(r'[\(\)\[\]\s\-_\.，,。·]', '', name).lower()
        return {
            "company": company,
            "share_type": share_type,
            "fund_type": fund_type,
            "currency": currency,
            "core_target": core_target,
            "clean_name": clean_name,
            "original_name": name,
            "name_length": len(name)
        }

    standardize_result = raw_df["基金简称"].apply(standardize_fund_name).apply(pd.Series)
    full_df = pd.concat([raw_df, standardize_result], axis=1)
    st.sidebar.success(f"✅ 基金列表标准化完成 (共{len(full_df)}只)")
    return full_df

# ---------------------- 智能匹配逻辑（币种强制+去冗余+兜底优化）----------------------
def query_fund_code_smart(keyword: str) -> dict:
    if not keyword:
        return {}
    
    full_df = load_full_fund_list()
    if full_df.empty:
        st.sidebar.error("❌ 基金列表为空，无法匹配")
        return {}
    
    debug_info = []
    debug_info.append(f"🔹 原始输入名称: {keyword}")

    if str(keyword).isdigit() and len(str(keyword)) == 6:
        matched = full_df[full_df["基金代码"] == str(keyword)]
        if not matched.empty:
            row = matched.iloc[0]
            st.sidebar.success(f"🎯 6位代码精确匹配: {row['基金简称']} ({row['基金代码']})")
            return {"code": row["基金代码"], "name": row["基金简称"]}

    def standardize_keyword(kw):
        kw = str(kw)
        kw = kw.translate(str.maketrans('（）【】', '()[]'))
        company_list = ["建信", "华夏", "广发", "易方达", "嘉实", "汇添富", "南方", "博时", "富国", "工银瑞信"]
        company = ""
        for comp in company_list:
            if comp in kw:
                company = comp
                break
        share_type = "C" if "C" in kw or "C类" in kw else "A"
        fund_type = ""
        if "ETF联接" in kw:
            fund_type = "ETF联接"
        elif "指数增强" in kw:
            fund_type = "指数增强"
        elif "混合" in kw:
            fund_type = "混合"
        elif "股票" in kw:
            fund_type = "股票"
        currency = "美元" if "美元" in kw else "人民币"
        core_target = kw
        core_target = re.sub(r'^(' + '|'.join(company_list) + ')', '', core_target)
        core_target = re.sub(r'(混合|ETF联接|指数增强|股票|QDII|A类|C类|A|C|发起式|美元|人民币|\(|\)|\s)', '', core_target)
        clean_kw = re.sub(r'[\(\)\[\]\s\-_\.，,。·]', '', kw).lower()
        return {
            "company": company,
            "share_type": share_type,
            "fund_type": fund_type,
            "currency": currency,
            "core_target": core_target,
            "clean_kw": clean_kw
        }

    kw_std = standardize_keyword(keyword)
    debug_info.append(f"🔹 提取公司: {kw_std['company']}")
    debug_info.append(f"🔹 提取份额类型: {kw_std['share_type']}")
    debug_info.append(f"🔹 提取基金类型: {kw_std['fund_type']}")
    debug_info.append(f"🔹 提取币种: {kw_std['currency']}")
    debug_info.append(f"🔹 提取核心标的: {kw_std['core_target']}")

    candidates = full_df.copy()
    if kw_std["company"]:
        candidates = candidates[candidates["company"] == kw_std["company"]]
        debug_info.append(f"✅ 公司筛选后剩余: {len(candidates)} 只")
    
    candidates = candidates[candidates["share_type"] == kw_std["share_type"]]
    debug_info.append(f"✅ 份额类型筛选后剩余: {len(candidates)} 只")
    
    if kw_std["currency"] == "人民币":
        candidates = candidates[candidates["currency"] == "人民币"]
        debug_info.append(f"✅ 币种筛选（强制人民币）后剩余: {len(candidates)} 只")
    
    if kw_std["fund_type"]:
        candidates = candidates[candidates["fund_type"] == kw_std["fund_type"]]
        debug_info.append(f"✅ 基金类型筛选后剩余: {len(candidates)} 只")
    
    if kw_std["core_target"]:
        candidates = candidates[candidates["core_target"].str.contains(kw_std["core_target"], na=False)]
        debug_info.append(f"✅ 核心标的筛选后剩余: {len(candidates)} 只")

    result = {}
    if not candidates.empty:
        candidates["similarity"] = candidates["clean_name"].apply(
            lambda x: difflib.SequenceMatcher(None, kw_std["clean_kw"], x).ratio()
        )
        candidates = candidates.sort_values(["similarity", "name_length"], ascending=[False, True])
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 最终匹配候选列表")
        show_cols = ["基金代码", "基金简称", "similarity", "currency", "core_target", "name_length"]
        st.sidebar.dataframe(candidates[show_cols].rename(columns={"similarity": "相似度", "currency": "币种", "name_length": "名称长度"}), use_container_width=True)
        
        best = candidates.iloc[0]
        result = {"code": best["基金代码"], "name": best["基金简称"]}
        debug_info.append(f"✅ 最佳匹配: {best['基金简称']} ({best['基金代码']})，相似度: {best['similarity']:.2f}")
    else:
        debug_info.append("❌ 强约束无匹配，启动兜底模糊匹配")
        # 先筛选币种（如果明确非美元），再计算相似度排序
        fallback_df = full_df.copy()
        if kw_std["currency"] == "人民币":
            fallback_df = fallback_df[fallback_df["currency"] == "人民币"]
        fallback_df["similarity"] = fallback_df["clean_name"].apply(
            lambda x: difflib.SequenceMatcher(None, kw_std["clean_kw"], x).ratio()
        )
        fallback_df = fallback_df.sort_values(["similarity", "name_length"], ascending=[False, True])
        top_3 = fallback_df.head(3)
        st.sidebar.subheader("⚠️ 兜底匹配Top3")
        st.sidebar.dataframe(top_3[["基金代码", "基金简称", "similarity", "currency"]], use_container_width=True)
        
        best = top_3.iloc[0]
        if best["similarity"] > 0.6:
            result = {"code": best["基金代码"], "name": best["基金简称"]}
            debug_info.append(f"✅ 兜底匹配成功: {best['基金简称']}")
        else:
            debug_info.append("❌ 无有效匹配，请手动输入基金代码")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 匹配全流程日志")
    for line in debug_info:
        st.sidebar.text(line)

    return result

# ---------------------- 基金数据获取 ----------------------
def get_fund_info(fund_code):
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
    try:
        response = requests.get(url, timeout=10)
        text = response.text
        json_str = re.search(r'jsonpgz\((.*)\);', text).group(1)
        data = json.loads(json_str)
        return {
            "code": fund_code,
            "name": data.get("name", ""),
            "net_value": float(data.get("dwjz", 0)),
            "estimate_value": float(data.get("gsz", 0)),
            "estimate_change": float(data.get("gszzl", 0)),
            "update_time": data.get("jzrq", "")
        }
    except:
        return None

def get_historical_nav(fund_code, days=365):
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        if df.empty:
            return None
        df = df[["净值日期", "单位净值"]].rename(columns={"净值日期": "date", "单位净值": "nav"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=days)]
        return df.sort_values("date")
    except:
        return None

# ---------------------- 指标计算 ----------------------
def calculate_metrics(nav_df):
    if nav_df is None or len(nav_df) < 2:
        return {}
    nav_df = nav_df.set_index("date")
    daily_returns = nav_df["nav"].pct_change().dropna()
    if len(daily_returns) < 10:
        return {}
    total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1)
    years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_volatility = daily_returns.std() * np.sqrt(252)
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    risk_free_rate = 0.03
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
    windows = {"1周": 5, "2周": 10, "1月": 21, "2月": 42, "3月": 63, "6月": 126}
    window_returns = {}
    for name, w in windows.items():
        if len(nav_df) >= w:
            ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[-w] - 1
        else:
            ret = None
        window_returns[name] = ret
    return {
        "annual_return": annual_return, "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown, "sharpe": sharpe, "sortino": sortino,
        "window_returns": window_returns, "nav_df": nav_df
    }

def calculate_score(metrics):
    if not metrics:
        return 0, 0, 0
    window_returns = metrics["window_returns"]
    weights = {"1周": 0.25, "2周": 0.2, "1月": 0.2, "2月": 0.15, "3月": 0.1, "6月": 0.1}
    recent_score = 0
    total_weight = 0
    for w_name, weight in weights.items():
        ret = window_returns.get(w_name)
        if ret is not None:
            ret_score = min(max(ret * 100, 0), 50)
            recent_score += ret_score * weight
            total_weight += weight
    if total_weight > 0:
        recent_score = recent_score / total_weight
    sharpe_score = min(max(metrics["sharpe"] * 25, 0), 25)
    sortino_score = min(max(metrics["sortino"] * 25, 0), 25)
    volatility_score = max(25 - metrics["annual_volatility"] * 100, 0)
    drawdown_score = max(25 - abs(metrics["max_drawdown"]) * 100, 0)
    long_term_score = sharpe_score + sortino_score + volatility_score + drawdown_score
    comprehensive = recent_score * 0.6 + long_term_score * 0.4
    return comprehensive, recent_score, long_term_score

def load_strategy_config():
    try:
        res = supabase.table("strategy_config").select("*").execute()
        return {row["rule_name"]: row["rule_value"] for row in res.data} if res.data else {"T_SELL_THRESHOLD": 2.0}
    except:
        return {"T_SELL_THRESHOLD": 2.0}

def ai_chat(messages, funds_context):
    client = get_deepseek_client()
    system_prompt = f"你是专属我的基金分析师，基于以下持仓数据回答问题：\n{funds_context}"
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=full_messages, stream=False)
        return response.choices[0].message.content
    except:
        return "AI调用失败"

# ---------------------- 侧边栏导航 ----------------------
st.sidebar.title("功能导航")
page = st.sidebar.radio("选择页面", [
    "📊 持仓总览", "📋 每日操作建议", "📁 持仓管理", "⚙️ 策略参数配置", "🤖 AI基金分析师"
])

# ---------------------- 持仓总览 ----------------------
if page == "📊 持仓总览":
    st.header("📊 我的持仓总览")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if not df_portfolio.empty:
            total_cost = (df_portfolio["shares"] * df_portfolio["cost_price"]).sum()
            total_market_value = 0
            display_data = []
            for _, row in df_portfolio.iterrows():
                info = get_fund_info(row["fund_code"])
                if info:
                    market_value = row["shares"] * info["net_value"]
                    profit = market_value - row["shares"] * row["cost_price"]
                    profit_pct = (info["net_value"] / row["cost_price"] - 1) * 100 if row["cost_price"] > 0 else 0
                    total_market_value += market_value
                    display_data.append({
                        "代码": row["fund_code"], "名称": row["fund_name"], "类型": row["category"],
                        "份额": f"{row['shares']:,.2f}", "成本价": f"{row['cost_price']:.4f}",
                        "最新净值": f"{info['net_value']:.4f}", "今日涨幅": f"{info['estimate_change']:.2f}%",
                        "市值": f"¥{market_value:,.2f}", "盈亏": f"¥{profit:,.2f}", "盈亏%": f"{profit_pct:.2f}%"
                    })
                else:
                    display_data.append({"代码": row["fund_code"], "名称": row["fund_name"], "最新净值": "获取失败"})
            col1, col2, col3 = st.columns(3)
            col1.metric("总成本", f"¥{total_cost:,.2f}")
            col2.metric("总市值", f"¥{total_market_value:,.2f}")
            col3.metric("总盈亏", f"¥{total_market_value - total_cost:,.2f}",
                       delta=f"{(total_market_value/total_cost-1)*100:.2f}%" if total_cost>0 else "0%")
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        else:
            st.info("暂无持仓数据，请前往「📁 持仓管理」添加")
    except Exception as e:
        st.error(f"数据库错误：{e}")

# ---------------------- 每日操作建议（已修复乱码）----------------------
elif page == "📋 每日操作建议":
    st.header("📋 每日操作建议")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df_portfolio = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df_portfolio.empty:
            st.warning("请先添加持仓")
        elif st.button("🚀 生成今日操作建议", type="primary"):
            config = load_strategy_config()
            signals = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df_portfolio.iterrows()):
                info = get_fund_info(row["fund_code"])
                if info:
                    change = info["estimate_change"]
                    if change >= config.get("T_SELL_THRESHOLD", 2.0):
                        signals.append(f"⚠️ {row['fund_name']}：涨幅{change:.2f}%，建议做T卖出1/3")
                    elif info["net_value"] >= row["cost_price"] and (info["net_value"]/row["cost_price"]-1)*100 < 1:
                        signals.append(f"ℹ️ {row['fund_name']}：已回本，可考虑减仓1/2")
                progress.progress((i+1)/len(df_portfolio))
            progress.empty()
            if signals:
                for s in signals:
                    if "⚠️" in s:
                        st.warning(s)
                    else:
                        st.info(s)
            else:
                st.success("今日无触发操作，持有不动")
    except Exception as e:
        st.error(f"生成失败：{e}")

# ---------------------- 持仓管理（智能匹配 + 自动估算份额成本）----------------------
elif page == "📁 持仓管理":
    st.header("📁 持仓管理")

    with st.expander("📸 上传支付宝持仓截图，智能更新持仓", expanded=True):
        uploaded_file = st.file_uploader("选择截图", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            with st.spinner("OCR识别中..."):
                ocr_text = ocr_image(uploaded_file)
                if ocr_text:
                    funds_parsed = parse_portfolio_by_ai(ocr_text)
                    if funds_parsed:
                        st.success(f"识别到 {len(funds_parsed)} 只基金")
                        
                        for fund in funds_parsed:
                            if "code" not in fund or not fund["code"]:
                                matched = query_fund_code_smart(fund["name"])
                                if matched:
                                    fund["code"] = matched["code"]
                                    fund["name"] = matched["name"]
                        
                        res_existing = supabase.table("portfolio").select("*").execute()
                        df_existing = pd.DataFrame(res_existing.data) if res_existing.data else pd.DataFrame()
                        
                        preview_data = []
                        for fund in funds_parsed:
                            code = fund.get("code", "")
                            name = fund["name"]
                            market_value = fund["market_value"]
                            existing = df_existing[df_existing["fund_code"] == code] if code and not df_existing.empty else pd.DataFrame()
                            action = "更新" if not existing.empty else ("新增" if code else "需补充代码")
                            preview_data.append({
                                "状态": action,
                                "基金代码": code if code else "⚠️ 未识别",
                                "基金名称": name,
                                "识别市值": f"¥{market_value:,.2f}",
                                "当前份额": existing.iloc[0]["shares"] if action=="更新" else "-",
                                "当前成本": existing.iloc[0]["cost_price"] if action=="更新" else "-"
                            })
                        
                        df_preview = pd.DataFrame(preview_data)
                        edited_df = st.data_editor(
                            df_preview,
                            column_config={
                                "状态": st.column_config.TextColumn(disabled=True),
                                "基金代码": st.column_config.TextColumn(),
                                "基金名称": st.column_config.TextColumn(),
                                "识别市值": st.column_config.TextColumn(disabled=True),
                                "当前份额": st.column_config.TextColumn(disabled=True),
                                "当前成本": st.column_config.TextColumn(disabled=True),
                            },
                            use_container_width=True,
                            key="fund_editor"
                        )
                        
                        if st.button("✅ 确认更新到我的持仓", type="primary"):
                            for _, row in edited_df.iterrows():
                                code = row["基金代码"].strip()
                                name = row["基金名称"].strip()
                                if not code or code == "⚠️ 未识别" or not name:
                                    continue
                                market_str = row["识别市值"].replace("¥", "").replace(",", "")
                                market_value = float(market_str) if market_str else 0.0
                                
                                existing = df_existing[df_existing["fund_code"] == code] if not df_existing.empty else pd.DataFrame()
                                if not existing.empty:
                                    update_dict = {
                                        "fund_code": code, "fund_name": name,
                                        "shares": existing.iloc[0]["shares"],
                                        "cost_price": existing.iloc[0]["cost_price"],
                                        "category": existing.iloc[0]["category"]
                                    }
                                else:
                                    info = get_fund_info(code)
                                    if info and info["net_value"] > 0:
                                        shares = market_value / info["net_value"]
                                        cost_price = info["net_value"]
                                    else:
                                        shares = 0.0
                                        cost_price = 0.0
                                    update_dict = {
                                        "fund_code": code, "fund_name": name,
                                        "category": "盈利底仓",
                                        "shares": round(shares, 2),
                                        "cost_price": round(cost_price, 4),
                                        "buy_date": str(datetime.now().date())
                                    }
                                supabase.table("portfolio").upsert(update_dict, on_conflict="fund_code").execute()
                            st.success("持仓已更新！")
                            st.rerun()
                    else:
                        st.warning("未能从截图中解析出基金信息")

    with st.expander("➕ 手动添加持仓"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_code = st.text_input("基金代码")
        with col2:
            new_name = st.text_input("基金名称")
        with col3:
            new_category = st.selectbox("类型", ["盈利底仓", "亏损做T仓", "观察仓"])
        col4, col5, col6 = st.columns(3)
        with col4:
            new_shares = st.number_input("份额", min_value=0.0, step=100.0)
        with col5:
            new_cost = st.number_input("成本价", min_value=0.0, step=0.0001, format="%.4f")
        with col6:
            new_date = st.date_input("买入日期")
        if st.button("添加持仓"):
            if new_code and new_name:
                supabase.table("portfolio").upsert({
                    "fund_code": new_code, "fund_name": new_name, "category": new_category,
                    "shares": new_shares, "cost_price": new_cost, "buy_date": str(new_date)
                }, on_conflict="fund_code").execute()
                st.success("已添加")
                st.rerun()

    st.subheader("当前持仓")
    try:
        res = supabase.table("portfolio").select("*").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if not df.empty:
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            if st.button("💾 保存修改"):
                for _, row in edited_df.iterrows():
                    supabase.table("portfolio").update({
                        "fund_name": row["fund_name"], "category": row["category"],
                        "shares": row["shares"], "cost_price": row["cost_price"]
                    }).eq("fund_code", row["fund_code"]).execute()
                st.success("✅ 保存成功")
                st.rerun()
        else:
            st.info("暂无持仓")
    except Exception as e:
        st.error(f"读取持仓失败：{e}")

# ---------------------- 策略配置 ----------------------
elif page == "⚙️ 策略参数配置":
    st.header("⚙️ 策略参数")
    try:
        res = supabase.table("strategy_config").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            edited = st.data_editor(df[["rule_name", "rule_value", "description"]], disabled=["rule_name", "description"])
            if st.button("保存"):
                for _, row in edited.iterrows():
                    supabase.table("strategy_config").update({"rule_value": row["rule_value"]}).eq("rule_name", row["rule_name"]).execute()
                st.success("已更新")
                st.rerun()
    except:
        st.info("配置表暂不可用")

# ---------------------- AI对话 ----------------------
elif page == "🤖 AI基金分析师":
    st.header("🤖 AI基金分析师")
    try:
        res = supabase.table("portfolio").select("*").execute()
        context = "我的持仓：\n"
        if res.data:
            for row in res.data:
                context += f"{row['fund_name']}({row['fund_code']})，份额{row['shares']}，成本{row['cost_price']}\n"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("问我任何问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    reply = ai_chat(st.session_state.messages, context)
                    st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"AI出错：{e}")