import streamlit as st
import pandas as pd
import requests
import json
import re
from datetime import datetime, timedelta, timezone, time as dt_time
import base64
from PIL import Image
import io
import difflib
import akshare as ak
from openai import OpenAI, Timeout
from supabase import create_client, Client
import os

# ====================== 常量定义 ======================
TZ = timezone(timedelta(hours=8))
OCR_THUMBNAIL_SIZE = (2048, 2048)
CACHE_TTL_FUND_LIST = 3600
CACHE_TTL_FUND_INFO = 300
CACHE_TTL_INDUSTRY_DAYS = 7
TIMEOUT_OCR_TOKEN = 10
TIMEOUT_OCR_REQUEST = 15
TIMEOUT_FUND_API = 10
DEEPSEEK_TIMEOUT = Timeout(60.0, connect=10.0, read=60.0)
DEFAULT_T_SELL_THRESHOLD = 2.0
SELL_SHARE_THRESHOLD = 0.01
MAX_SELL_SHARE_RATIO = 1.0
BATCH_UPSERT_SIZE = 10

CN_HOLIDAYS = {
    "2026-01-01", "2026-01-02", "2026-01-03",
    "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16",
    "2026-04-05",
    "2026-05-01", "2026-05-02", "2026-05-03",
    "2026-06-19",
    "2026-09-27",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
}

# ====================== Render 部署必备 ======================
port = os.environ.get("PORT", "8501")

# ====================== Streamlit 配置 ======================
st.set_page_config(
    page_title="基金智能管理系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== 移动端样式 ======================
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    @media (max-width: 768px) {
        .metric-card { padding: 12px; }
        .stApp { overflow-x: hidden; }
    }
    .positive { color: #e31b23; }
    .negative { color: #2e8b57; }
</style>
""", unsafe_allow_html=True)

# ====================== 密钥（Render 环境变量自动读取）======================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
BAIDU_OCR_API_KEY = st.secrets["BAIDU_OCR_API_KEY"]
BAIDU_OCR_SECRET_KEY = st.secrets["BAIDU_OCR_SECRET_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ====================== 工具函数 ======================
def now_cn() -> datetime:
    return datetime.now(TZ)

def is_trading_day(date: datetime = None) -> bool:
    if date is None:
        date = now_cn()
    if date.weekday() >= 5:
        return False
    date_str = date.strftime("%Y-%m-%d")
    if date_str in CN_HOLIDAYS:
        return False
    return True

def is_trading_time() -> bool:
    if not is_trading_day():
        return False
    now = now_cn()
    current_time = now.time()
    start = dt_time(9, 30)
    end = dt_time(15, 0)
    return start <= current_time <= end

def safe_json_parse(text: str, pattern: str) -> dict | list | None:
    try:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None

# ====================== DeepSeek ======================
def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=DEEPSEEK_TIMEOUT,
        max_retries=2
    )

# ====================== 百度 OCR ======================
def get_baidu_access_token() -> str | None:
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_OCR_API_KEY,
        "client_secret": BAIDU_OCR_SECRET_KEY
    }
    try:
        resp = requests.post(url, params=params, timeout=TIMEOUT_OCR_TOKEN)
        return resp.json().get("access_token")
    except Exception:
        return None

def ocr_image(image_file) -> str:
    token = get_baidu_access_token()
    if not token:
        return ""
    try:
        img = Image.open(image_file)
        img.thumbnail(OCR_THUMBNAIL_SIZE)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode()
    except:
        return ""

    for ocr_type, url in [
        ("高精度", "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"),
        ("通用", "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic")
    ]:
        try:
            resp = requests.post(url,
                params={"access_token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"image": img_base64},
                timeout=TIMEOUT_OCR_REQUEST)
            result = resp.json()
            if "words_result" in result:
                return "\n".join([item["words"] for item in result["words_result"]])
        except:
            continue
    return ""

# ====================== AI 解析持仓（你之前缺失的！）======================
def parse_portfolio_by_ai(ocr_text: str) -> list:
    client = get_deepseek_client()
    prompt = f"""
你是严格的基金持仓提取工具，只提取基金名称和市值，返回JSON数组：
[{{"name":"基金名","market_value":数字}}]
只提取市值>1000元的。
OCR：{ocr_text}
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0)
        content = resp.choices[0].message.content.strip()
        data = safe_json_parse(content, r'\[.*\]')
        return data if isinstance(data, list) else []
    except:
        return []

def parse_operation_by_ai(ocr_text: str) -> dict:
    client = get_deepseek_client()
    prompt = f"""提取交易信息：{{"fund_name":"","operation":"买入/卖出","amount":0,"date":"YYYY-MM-DD"}}
OCR：{ocr_text}"""
    try:
        resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0)
        data = safe_json_parse(resp.choices[0].message.content, r'\{.*\}')
        return data if isinstance(data, dict) else {}
    except:
        return {}

# ====================== 基金列表（Render 只读模式，不写本地文件）======================
def process_fund_list(raw_df: pd.DataFrame) -> pd.DataFrame:
    def standardize(row):
        name = str(row["基金简称"])
        companies = ["建信","华夏","广发","易方达","嘉实","汇添富","南方","博时","富国","工银瑞信"]
        company = next((c for c in companies if c in name), "")
        share_type = "C" if "C" in name else "A"
        fund_type = ""
        if "ETF联接" in name: fund_type = "ETF联接"
        elif "指数增强" in name: fund_type = "指数增强"
        elif "混合" in name: fund_type = "混合"
        elif "股票" in name: fund_type = "股票"
        currency = "美元" if "美元" in name else "人民币"
        core = re.sub(r'(混合|ETF联接|指数增强|股票|QDII|A|C|发起式|美元|人民币|\(|\))','',name)
        clean = re.sub(r'[\(\)\[\]\s\-_\.，,。·]','',name).lower()
        return pd.Series([company, share_type, fund_type, currency, core, clean, len(name)])
    std_df = raw_df.apply(standardize, axis=1)
    std_df.columns = ["company","share_type","fund_type","currency","core_target","clean_name","name_length"]
    return pd.concat([raw_df, std_df], axis=1)

@st.cache_data(ttl=CACHE_TTL_FUND_LIST)
def load_full_fund_list() -> pd.DataFrame:
    try:
        raw_df = ak.fund_name_em()
        raw_df = raw_df[["基金代码","基金简称"]].copy()
        return process_fund_list(raw_df)
    except:
        return pd.DataFrame()

def query_fund_code_smart(keyword: str) -> dict:
    if not keyword:
        return {}
    df = load_full_fund_list()
    if df.empty:
        return {}
    if keyword.isdigit() and len(keyword)==6:
        match = df[df["基金代码"]==keyword]
        if not match.empty:
            return {"code":match.iloc[0]["基金代码"],"name":match.iloc[0]["基金简称"]}
    return {}

# ====================== 基金信息 ======================
@st.cache_data(ttl=CACHE_TTL_FUND_INFO)
def get_fund_info(fund_code: str) -> dict | None:
    url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
    try:
        resp = requests.get(url, timeout=TIMEOUT_FUND_API)
        json_str = re.search(r'jsonpgz\((.*)\);', resp.text).group(1)
        data = json.loads(json_str)
        return {
            "code":fund_code,
            "name":data.get("name",""),
            "net_value":float(data.get("dwjz",0)),
            "estimate_value":float(data.get("gsz",0)),
            "estimate_change":float(data.get("gszzl",0)),
            "update_time":data.get("jzrq","")
        }
    except:
        return None

def get_historical_nav(fund_code: str, days=365) -> pd.DataFrame | None:
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        df = df[["净值日期","单位净值"]].rename(columns={"净值日期":"date","单位净值":"nav"})
        df["date"] = pd.to_datetime(df["date"])
        return df[df["date"] >= df["date"].max()-timedelta(days=days)]
    except:
        return None

def get_fund_holdings(fund_code: str) -> pd.DataFrame | None:
    try:
        return ak.fund_portfolio_holdings_em(symbol=fund_code)
    except:
        return None

# ====================== 行业缓存 ======================
def get_stock_industry(stock_code: str) -> str:
    try:
        res = supabase.table("stock_industry").select("*").eq("stock_code",stock_code).execute()
        if res.data:
            row = res.data[0]
            updated = datetime.fromisoformat(row["updated_at"].replace("Z","+00:00"))
            if now_cn() - updated < timedelta(days=CACHE_TTL_INDUSTRY_DAYS):
                return row["industry"]
    except:
        pass
    industry = "其他"
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        industry = info[info["item"]=="行业"]["value"].iloc[0] if not info.empty else "其他"
    except:
        pass
    try:
        supabase.table("stock_industry").upsert({
            "stock_code":stock_code,
            "industry":industry,
            "updated_at":now_cn().isoformat()
        }, on_conflict="stock_code").execute()
    except:
        pass
    return industry

# ====================== 市场动态 ======================
def get_market_dynamics() -> dict:
    if not is_trading_day():
        return {}
    dyn = {}
    try:
        df = ak.stock_zh_index_spot_em()
        for idx in ["上证指数","深证成指","创业板指","沪深300"]:
            match = df[df["名称"]==idx]
            if not match.empty:
                dyn[idx] = {"current":match.iloc[0]["最新价"],"change_pct":match.iloc[0]["涨跌幅"]}
    except:
        pass
    return dyn

# ====================== 策略 ======================
def load_strategy_config():
    try:
        res = supabase.table("strategy_config").select("*").execute()
        return {r["rule_name"]:float(r["rule_value"]) for r in res.data}
    except:
        return {"T_SELL_THRESHOLD":DEFAULT_T_SELL_THRESHOLD}

def save_strategy_config(config):
    for k,v in config.items():
        supabase.table("strategy_config").upsert({
            "rule_name":k,"rule_value":v,"updated_at":now_cn().isoformat()
        }, on_conflict="rule_name").execute()

# ====================== AI 策略顾问 ======================
def strategy_advisor(messages: list, context: str) -> str:
    client = get_deepseek_client()
    sys = f"你是策略顾问，只基于持仓给建议，不荐股。\n{context}"
    try:
        resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"system","content":sys}]+messages)
        return resp.choices[0].message.content
    except:
        return "AI调用失败"

# ====================== 买卖逻辑 ======================
def update_portfolio_on_buy(fund_code, fund_name, amount, price, op_date):
    if price <=0: return
    shares = amount/price
    if shares <= SELL_SHARE_THRESHOLD: return
    res = supabase.table("portfolio").select("*").eq("fund_code",fund_code).execute()
    if res.data:
        old = res.data[0]
        new_shares = old["shares"]+shares
        new_cost = (old["shares"]*old["cost_price"] + amount)/new_shares
        supabase.table("portfolio").update({
            "shares":new_shares,"cost_price":new_cost
        }).eq("fund_code",fund_code).execute()
    else:
        supabase.table("portfolio").insert({
            "fund_code":fund_code,
            "fund_name":fund_name,
            "category":"盈利底仓",
            "shares":shares,
            "cost_price":price,
            "buy_date":op_date,"realized_profit":0
        }).execute()
    supabase.table("buy_batches").insert({
        "fund_code":fund_code,"buy_date":op_date,
        "shares":shares,"cost_price":price,"remaining_shares":shares
    }).execute()

def update_portfolio_on_sell(fund_code, amount, price, op_date):
    if price <=0: return
    res = supabase.table("portfolio").select("shares").eq("fund_code",fund_code).execute()
    if not res.data: return
    max_sell = float(res.data[0]["shares"])
    sell_shares = min(amount/price, max_sell)
    if sell_shares <= SELL_SHARE_THRESHOLD: return

    batches = supabase.table("buy_batches").select("*").eq("fund_code",fund_code).gt("remaining_shares",0).order("buy_date").execute()
    if not batches.data: return
    remaining = sell_shares
    profit = 0
    for b in batches.data:
        if remaining <=0: break
        s = min(b["remaining_shares"], remaining)
        profit += s*(price - b["cost_price"])
        supabase.table("buy_batches").update({"remaining_shares":b["remaining_shares"]-s}).eq("id",b["id"]).execute()
        remaining -= s

    p = supabase.table("portfolio").select("*").eq("fund_code",fund_code).execute().data[0]
    supabase.table("portfolio").update({
        "shares": p["shares"]-sell_shares,
        "realized_profit": p.get("realized_profit",0)+profit
    }).eq("fund_code",fund_code).execute()

# ====================== 页面导航 ======================
st.sidebar.title("导航")
page = st.sidebar.radio("选择页面",[
    "📊 持仓总览","📋 每日操作建议","📁 持仓管理","⚙️ 策略配置","💬 AI分析师"
])

# ====================== 页面1：持仓总览 ======================
if page == "📊 持仓总览":
    st.header("📊 持仓总览")
    label1 = "实时预估总资产" if is_trading_time() else "上一交易日总资产"
    try:
        res = supabase.table("portfolio").select("*").execute()
        if not res.data:
            st.info("暂无持仓")
        else:
            total_val = 0.0
            total_profit = 0.0
            total_real = 0.0
            rows = []
            for r in res.data:
                code = r["fund_code"]
                info = get_fund_info(code)
                if info:
                    nav = info["net_value"]
                    est = info["estimate_value"]
                    change = info["estimate_change"]
                    val = r["shares"]*nav
                    real = val - r["shares"]*r["cost_price"]
                    est_p = r["shares"]*(est-nav)
                    total_val += val
                    total_profit += est_p
                    total_real += real
                    rows.append({
                        "基金名称":r["fund_name"],
                        "基金代码":code,
                        "持仓分类":r["category"],
                        "持有份额":f"{r['shares']:.2f}",
                        "成本净值":f"{r['cost_price']:.4f}",
                        "单位净值":f"{nav:.4f}",
                        "持仓盈亏":f"{real:+,.2f}",
                        "持仓收益率":f"{real/(r['shares']*r['cost_price'])*100:+.2f}%" if r['shares']*r['cost_price']>0 else "0%",
                        "日涨跌幅":f"{change:+.2f}%"
                    })
            c1,c2,c3,c4 = st.columns(4)
            c1.metric(label1, f"{total_val:,.2f}")
            c2.metric("当日预估收益", f"{total_profit:+,.2f}")
            c3.metric("持仓盈亏(确定)", f"{total_real:+,.2f}")
            c4.metric("收益率", f"{total_real/(total_val-total_real)*100:+.2f}%" if (total_val-total_real)>0 else "0%")

            def colorit(v):
                if isinstance(v,str):
                    if "+" in v: return "color:#e31b23"
                    if "-" in v: return "color:#2e8b57"
                return ""
            st.dataframe(pd.DataFrame(rows).style.applymap(colorit, subset=["持仓盈亏","持仓收益率","日涨跌幅"]), use_container_width=True)
    except Exception as e:
        st.error(str(e))

# ====================== 页面2：每日建议 ======================
elif page == "📋 每日操作建议":
    st.header("📋 每日操作建议")
    dyn = get_market_dynamics()
    if dyn:
        with st.expander("📈 市场",expanded=True):
            c = st.columns(4)
            for i,idx in enumerate(["上证指数","深证成指","创业板指","沪深300"]):
                if idx in dyn:
                    c[i].metric(idx, dyn[idx]["current"], f"{dyn[idx]['change_pct']:+.2f}%")
    res = supabase.table("portfolio").select("*").execute()
    if res.data and st.button("🚀 生成建议"):
        cfg = load_strategy_config()
        t = cfg["T_SELL_THRESHOLD"]
        for r in res.data:
            info = get_fund_info(r["fund_code"])
            if info:
                est = info["estimate_change"]
                cat = r["category"]
                cost = r["cost_price"]
                nav = info["net_value"]
                op = "无操作"
                if cat == "亏损做T仓" and est >= t:
                    op = "做T卖出"
                elif nav >= cost and (nav/cost-1)*100 <1:
                    op = "回本减仓"
                st.write(f"• {r['fund_name']} | {op}")

# ====================== 页面3：持仓管理 ======================
elif page == "📁 持仓管理":
    st.header("📁 持仓管理")
    with st.expander("📸 上传持仓截图"):
        f = st.file_uploader("上传",type=["png","jpg"])
        if f:
            txt = ocr_image(f)
            funds = parse_portfolio_by_ai(txt)
            if funds:
                st.dataframe(pd.DataFrame(funds))
                if st.button("✅ 确认入库"):
                    for fund in funds:
                        match = query_fund_code_smart(fund["name"])
                        if match:
                            code = match["code"]
                            name = match["name"]
                            info = get_fund_info(code)
                            price = info["net_value"] if info else 0
                            shares = fund["market_value"]/price if price>0 else 0
                            supabase.table("portfolio").upsert({
                                "fund_code":code,
                                "fund_name":name,
                                "category":"盈利底仓",
                                "shares":shares,
                                "cost_price":price,
                                "buy_date":now_cn().strftime("%Y-%m-%d")
                            }, on_conflict="fund_code").execute()
                    st.success("完成")

    with st.expander("📎 上传交易截图"):
        f2 = st.file_uploader("交易截图",type=["png","jpg"])
        if f2:
            op = parse_operation_by_ai(ocr_image(f2))
            if op:
                st.write(op)
                if st.button("✅ 确认交易"):
                    match = query_fund_code_smart(op["fund_name"])
                    if match:
                        code = match["code"]
                        name = match["name"]
                        price = get_fund_info(code)["net_value"] if get_fund_info(code) else 0
                        if op["operation"] == "买入":
                            update_portfolio_on_buy(code,name,op["amount"],price,op["date"])
                        else:
                            update_portfolio_on_sell(code,op["amount"],price,op["date"])
                        st.success("完成")

# ====================== 页面4：策略配置 ======================
elif page == "⚙️ 策略配置":
    st.header("⚙️ 策略配置")
    cfg = load_strategy_config()
    v = st.number_input("做T阈值 %", value=cfg.get("T_SELL_THRESHOLD",2.0))
    if st.button("保存"):
        save_strategy_config({"T_SELL_THRESHOLD":v})
        st.success("已保存")

# ====================== 页面5：AI分析师 ======================
elif page == "💬 AI分析师":
    st.header("💬 AI分析师")
    res = supabase.table("portfolio").select("*").execute()
    ctx = "我的持仓：\n"+"\n".join([f"{r['fund_name']} {r['fund_code']}" for r in res.data]) if res.data else "无持仓"
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    i = st.chat_input("提问")
    if i:
        st.session_state.msgs.append({"role":"user","content":i})
        with st.chat_message("user"):
            st.write(i)
        reply = strategy_advisor(st.session_state.msgs, ctx)
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.msgs.append({"role":"assistant","content":reply})