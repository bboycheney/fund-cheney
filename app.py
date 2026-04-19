import streamlit as st
import pandas as pd
import requests
import json
import re
import plotly.express as px
from datetime import datetime, timedelta, timezone
import base64
from PIL import Image
import io
from openai import OpenAI
from supabase import create_client, Client

# ====================== 基础配置 ======================
st.set_page_config(page_title="基金智能管理系统", page_icon="📈", layout="wide")
TZ = timezone(timedelta(hours=8))

# ====================== 密钥配置 ======================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
    BAIDU_OCR_API_KEY = st.secrets.get("BAIDU_OCR_API_KEY", "")
    BAIDU_OCR_SECRET_KEY = st.secrets.get("BAIDU_OCR_SECRET_KEY", "")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"❌ 密钥配置错误: {e}")
    st.stop()

# ====================== 工具函数 ======================
def now_cn():
    return datetime.now(TZ)

@st.cache_data(ttl=300)
def get_fund_info(code):
    """直接通过接口获取基金数据，去掉akshare依赖"""
    try:
        url = f"http://fundgz.1234567.com.cn/js/{code}.js"
        r = requests.get(url, timeout=10)
        data = json.loads(re.search(r'jsonpgz\((.*)\);', r.text).group(1))
        return {
            "nav": float(data.get("dwjz", 0)),
            "chg": float(data.get("gszzl", 0))
        }
    except Exception as e:
        # st.warning(f"获取数据失败: {e}")
        return {"nav": 0, "chg": 0}

# ====================== OCR 识别 ======================
def ocr_image(file):
    if not (BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY):
        return ""
    try:
        img = Image.open(file).convert("RGB")
        img.thumbnail((1000, 1000))
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        token_r = requests.post("https://aip.baidubce.com/oauth/2.0/token",
            params={"grant_type":"client_credentials",
                    "client_id":BAIDU_OCR_API_KEY,
                    "client_secret":BAIDU_OCR_SECRET_KEY}, timeout=10)
        token = token_r.json().get("access_token")
        
        ocr_r = requests.post("https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
            params={"access_token":token}, data={"image":b64}, timeout=15)
        return "\n".join([i["words"] for i in ocr_r.json().get("words_result",[])])
    except:
        return ""

# ====================== 持仓操作 ======================
def buy_fund(code, name, amount, price):
    try:
        shares = amount / price
        r = supabase.table("portfolio").select("*").eq("fund_code", code).execute()
        if r.data:
            old = r.data[0]
            new_shares = old["shares"] + shares
            new_cost = (old["shares"] * old["cost_price"] + amount) / new_shares
            supabase.table("portfolio").update({
                "shares": new_shares, "cost_price": new_cost
            }).eq("fund_code", code).execute()
        else:
            supabase.table("portfolio").insert({
                "fund_code": code, "fund_name": name or "未知基金",
                "shares": shares, "cost_price": price,
                "buy_date": now_cn().strftime("%Y-%m-%d")
            }).execute()
        return True
    except:
        return False

def sell_fund(code, sell_shares):
    try:
        r = supabase.table("portfolio").select("*").eq("fund_code", code).execute()
        if not r.data: return False
        current = r.data[0]["shares"]
        if sell_shares > current: return False
        if current - sell_shares > 0:
            supabase.table("portfolio").update({"shares": current - sell_shares}).eq("fund_code", code).execute()
        else:
            supabase.table("portfolio").delete().eq("fund_code", code).execute()
        return True
    except:
        return False

# ====================== 页面导航 ======================
st.sidebar.title("基金管理")
page = st.sidebar.radio("", ["📊 持仓总览", "📁 持仓管理", "⚙️ 策略配置", "💬 AI分析师"])

# ====================== 持仓总览 ======================
if page == "📊 持仓总览":
    st.title("📊 持仓总览")
    try:
        r = supabase.table("portfolio").select("*").execute()
        if not r.data:
            st.info("暂无持仓")
        else:
            total, profit, day_profit = 0, 0, 0
            rows = []
            for item in r.data:
                info = get_fund_info(item["fund_code"])
                nav = info["nav"]
                val = item["shares"] * nav
                cost_val = item["shares"] * item["cost_price"]
                p = val - cost_val
                day_p = val * info["chg"] / 100
                
                total += val
                profit += p
                day_profit += day_p
                rows.append([item["fund_name"], item["fund_code"], f"{item['shares']:.2f}", 
                            f"{item['cost_price']:.4f}", f"{nav:.4f}", f"{p:+,.2f}"])
            
            df = pd.DataFrame(rows, columns=["基金名称", "代码", "份额", "成本", "净值", "盈亏"])
            col1, col2, col3 = st.columns(3)
            col1.metric("总资产", f"{total:,.2f}")
            col2.metric("总盈亏", f"{profit:+,.2f}")
            col3.metric("当日收益", f"{day_profit:+,.2f}")
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"加载失败: {e}")

# ====================== 持仓管理 ======================
elif page == "📁 持仓管理":
    st.title("📁 持仓管理")
    
    # OCR导入
    with st.expander("📸 截图导入持仓"):
        f = st.file_uploader("上传截图", type=["png","jpg"])
        if f:
            text = ocr_image(f)
            if text:
                st.text_area("识别结果", text, height=150)
    
    # 手动买入
    st.subheader("➕ 买入基金")
    code = st.text_input("基金代码")
    name = st.text_input("基金名称（可选，填了更方便）")
    amount = st.number_input("买入金额", min_value=0.01)
    if st.button("确认买入") and code:
        info = get_fund_info(code)
        if buy_fund(code, name, amount, info["nav"]):
            st.success("买入成功")
            st.rerun()
    
    # 卖出基金
    st.subheader("➖ 卖出基金")
    r = supabase.table("portfolio").select("*").execute()
    if r.data:
        options = [f"{i['fund_name']} ({i['fund_code']})" for i in r.data]
        sel = st.selectbox("选择基金", options)
        code = sel.split("(")[1].replace(")","")
        shares = st.number_input("卖出份额", min_value=0.01)
        if st.button("确认卖出"):
            if sell_fund(code, shares):
                st.success("卖出成功")
                st.rerun()

# ====================== 策略配置 ======================
elif page == "⚙️ 策略配置":
    st.title("⚙️ 策略配置")
    try:
        r = supabase.table("strategy_config").select("*").execute()
        val = 1.0
        if r.data: val = r.data[0]["rule_value"]
        
        new_val = st.number_input("做T卖出阈值(%)", value=val, min_value=0.1)
        if st.button("保存"):
            supabase.table("strategy_config").upsert({
                "rule_name": "T_SELL_THRESHOLD", "rule_value": new_val
            }).execute()
            st.success("保存成功")
    except:
        st.info("首次使用自动创建配置")

# ====================== AI分析师 ======================
elif page == "💬 AI分析师":
    st.title("💬 AI分析师")
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    
    q = st.chat_input("请输入问题")
    if q and DEEPSEEK_API_KEY:
        st.session_state.msgs.append({"role":"user","content":q})
        with st.chat_message("user"):
            st.write(q)
        
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            res = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":q}])
            reply = res.choices[0].message.content
        except:
            reply = "AI服务暂不可用"
        
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.msgs.append({"role":"assistant","content":reply})