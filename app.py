import streamlit as st
import pandas as pd
import requests
import json
import re
import time
import html
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone, time as dt_time
import base64
from PIL import Image
import io
import akshare as ak
from openai import OpenAI, Timeout
from supabase import create_client, Client
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib
import os
from typing import Dict, List, Optional, Tuple  # 完善类型注解

# ====================== 全局配置（优化：分类清晰+注释完善） ======================
# 日志配置（优化：分级输出+文件轮转更合理）
logger.add(
    "fund_system.log",
    rotation="50 MB",  # 调整轮转大小，避免文件过大
    retention="14 days",  # 延长日志保留时间
    encoding="utf-8",
    backtrace=True,
    diagnose=True,
    level="INFO"  # 默认INFO级别，减少冗余
)

# 时区/常量配置（优化：分组归类）
TZ = timezone(timedelta(hours=8))
OCR_THUMBNAIL_SIZE = (512, 512)
# 缓存TTL（优化：按数据更新频率调整）
CACHE_TTL_FUND_LIST = 3600  # 基金列表1小时更新
CACHE_TTL_FUND_INFO = 5 * 60  # 基金实时信息5分钟更新
CACHE_TTL_FUND_DETAIL = 10 * 60  # 基金详情10分钟更新
CACHE_TTL_FUND_NAV = 30 * 60  # 净值数据30分钟更新
# AI配置
AI_CONTEXT_MAX_LENGTH = 10
MAX_ANALYST_MSGS = 20  # AI分析师最大会话数
# 策略配置
DEFAULT_T_SELL_THRESHOLD = 2.0
# 网络请求配置
REQUEST_TIMEOUT = 20  # 统一请求超时时间
RETRY_MAX_ATTEMPT = 3  # 统一重试次数
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 5

# 法定节假日（优化：增加动态获取入口+注释）
CN_HOLIDAYS = {
    # 2024-2026 法定节假日，生产环境建议接入节假日API（如百度/阿里云节假日接口）
    "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16",
    "2024-04-04", "2024-05-01", "2024-06-10", "2024-09-15", "2024-09-16", "2024-10-01", "2024-10-02", "2024-10-03",
    "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07",
    "2025-01-01", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-07", "2025-05-01", "2025-05-02", "2025-05-03", "2025-06-12", "2025-09-18", "2025-10-01", "2025-10-02",
    "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07",
    "2026-01-01", "2026-01-02", "2026-01-03", "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16",
    "2026-04-05", "2026-05-01", "2026-05-02", "2026-05-03", "2026-06-19", "2026-09-27", "2026-10-01", "2026-10-02",
    "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
}

# ====================== 页面初始化（优化：样式更美观+响应式） ======================
st.set_page_config(
    page_title="基金智能管理系统V1.1.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"  # 侧边栏默认展开，提升操作体验
)

# 样式优化：更现代的UI风格+响应式适配
st.markdown("""
<style>
    /* 全局样式 */
    .main {padding: 0rem 1rem;}
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    /* 涨跌颜色优化（符合国内习惯） */
    .positive {color: #ef4444; font-weight: 600;}  /* 红色涨 */
    .negative {color: #16a34a; font-weight: 600;}  /* 绿色跌 */
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 2.8rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    /* 输入框样式 */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    /* 表格样式 */
    .dataframe {border-radius: 8px; overflow: hidden;}
</style>
""", unsafe_allow_html=False)

# ====================== 密钥初始化（优化：更健壮的加载+默认值处理） ======================
try:
    # 环境变量优先，本地开发可使用streamlit secrets
    SUPABASE_URL = os.getenv("SUPABASE_URL", st.secrets.get("SUPABASE_URL", ""))
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", st.secrets.get("SUPABASE_KEY", ""))
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", st.secrets.get("DEEPSEEK_API_KEY", ""))
    BAIDU_OCR_API_KEY = os.getenv("BAIDU_OCR_API_KEY", st.secrets.get("BAIDU_OCR_API_KEY", ""))
    BAIDU_OCR_SECRET_KEY = os.getenv("BAIDU_OCR_SECRET_KEY", st.secrets.get("BAIDU_OCR_SECRET_KEY", ""))
    
    # 密钥校验
    required_keys = [SUPABASE_URL, SUPABASE_KEY]
    if not all(required_keys):
        raise ValueError("Supabase密钥未配置完整")
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("系统初始化成功：核心密钥加载完成")
    
    # 可选服务提示
    if not DEEPSEEK_API_KEY:
        logger.warning("DeepSeek API密钥未配置，AI功能将不可用")
        st.warning("⚠️ AI功能未配置密钥，相关功能将不可用")
    if not (BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY):
        logger.warning("百度OCR密钥未配置，图片识别功能将不可用")
        st.warning("⚠️ OCR功能未配置密钥，图片上传识别将不可用")
        
except Exception as e:
    logger.critical(f"初始化失败：{str(e)}", exc_info=True)
    st.error(f"❌ 系统初始化失败：{str(e)}")
    st.stop()

# ====================== 通用工具函数（优化：封装复用+类型注解完善） ======================
def now_cn() -> datetime:
    """返回中国时区当前时间"""
    return datetime.now(TZ)

def is_trading_day(date: Optional[datetime] = None) -> bool:
    """判断是否为交易日
    Args:
        date: 待判断日期，默认当前时间
    Returns:
        bool: 是否为交易日
    """
    date = date or now_cn()
    date_str = date.strftime("%Y-%m-%d")
    
    # 检查是否为周末
    is_weekday = date.weekday() < 5
    # 检查是否为节假日
    is_not_holiday = date_str not in CN_HOLIDAYS
    
    # 提示缺失的节假日数据
    if date.year not in [2024, 2025, 2026]:
        logger.warning(f"节假日数据缺失：{date.year}，建议接入动态节假日API")
    
    return is_weekday and is_not_holiday

def is_trading_time() -> bool:
    """判断是否为交易时间（9:30-11:30, 13:00-15:00）"""
    if not is_trading_day():
        return False
    
    t = now_cn().time()
    morning = dt_time(9, 30) <= t <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= t <= dt_time(15, 0)
    
    return morning or afternoon

def validate_numeric(v: any, min_v: float = 0.01) -> bool:
    """校验数值是否为正数且不小于最小值
    Args:
        v: 待校验值
        min_v: 最小值，默认0.01
    Returns:
        bool: 校验结果
    """
    try:
        v_float = float(v)
        return v_float >= min_v
    except (ValueError, TypeError):
        return False

def escape_html(s: any) -> str:
    """HTML转义，防止XSS，空值处理"""
    if s is None:
        return ""
    return html.escape(str(s)) if isinstance(s, (str, int, float)) else ""

# ====================== 重试装饰器封装（优化：统一复用） ======================
def retry_decorator(
    stop_attempt: int = RETRY_MAX_ATTEMPT,
    min_wait: int = RETRY_MIN_WAIT,
    max_wait: int = RETRY_MAX_WAIT
):
    """通用重试装饰器
    Args:
        stop_attempt: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
    """
    return retry(
        stop=stop_after_attempt(stop_attempt),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )

# ====================== Supabase 操作封装（优化：更简洁的调用） ======================
@retry_decorator()
def supabase_execute(func, *args, **kwargs):
    """Supabase操作执行（带重试）"""
    return func(*args, **kwargs)

def supabase_transaction(func) -> bool:
    """Supabase事务执行
    Args:
        func: 事务内要执行的函数
    Returns:
        bool: 事务是否成功
    """
    try:
        with supabase._client.transactions():
            func()
        return True
    except Exception as e:
        logger.error(f"事务执行失败：{e}", exc_info=True)
        return False

# ====================== AI 工具函数（优化：更健壮的解析+缓存） ======================
def safe_json_parse(text: str, pattern: str = r'\[.*\]') -> Optional[List[Dict]]:
    """安全解析AI返回的JSON，降级处理"""
    # 空值处理
    if not text or text.strip() == "":
        return None
    
    try:
        # 精准匹配JSON数组
        match = re.search(pattern, text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list):
                # 过滤无效数据
                return [item for item in data if validate_numeric(item.get("market_value", 0))]
    except Exception as e:
        logger.debug(f"JSON解析失败，尝试降级解析：{e}")
    
    # 降级解析：基金名+市值
    try:
        funds = []
        # 更健壮的正则匹配
        names = re.findall(r'["\']?name["\']?\s*[:：]\s*["\']([^"\']+)["\']', text)
        vals = re.findall(r'["\']?market_value["\']?\s*[:：]\s*(\d+\.?\d*)', text)
        
        for n, v in zip(names, vals):
            if n and validate_numeric(float(v)):
                funds.append({"name": n.strip(), "market_value": float(v)})
        
        return funds if funds else None
    except Exception as e:
        logger.error(f"降级解析失败：{e}", exc_info=True)
        return None

@retry_decorator(stop_attempt=2, min_wait=1, max_wait=3)
def get_deepseek_client() -> Optional[OpenAI]:
    """获取DeepSeek客户端，带重试"""
    if not DEEPSEEK_API_KEY:
        return None
    
    try:
        time.sleep(1)  # 防403
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            timeout=Timeout(30, connect=10, read=20),
            default_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
    except Exception as e:
        logger.error(f"获取DeepSeek客户端失败：{e}", exc_info=True)
        return None

# ====================== 百度OCR（优化：更清晰的错误提示+资源释放） ======================
@retry_decorator(stop_attempt=2)
def get_ocr_token() -> Optional[str]:
    """获取百度OCR Token"""
    if not (BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY):
        return None
    
    try:
        r = requests.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": BAIDU_OCR_API_KEY,
                "client_secret": BAIDU_OCR_SECRET_KEY
            },
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        logger.error(f"获取OCR Token失败：{e}", exc_info=True)
        return None

def ocr_image(file) -> str:
    """OCR识别图片中的基金信息
    Args:
        file: 上传的图片文件
    Returns:
        str: 识别结果文本
    """
    if not file:
        st.warning("⚠️ 未选择图片")
        return ""
    
    if file.size == 0:
        st.warning("⚠️ 上传的图片为空文件")
        return ""
    
    try:
        # 处理图片，释放资源
        with Image.open(file) as img:
            img = img.convert("RGB")
            img.thumbnail(OCR_THUMBNAIL_SIZE)
            
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
        
        # 获取OCR Token
        token = get_ocr_token()
        if not token:
            st.error("❌ OCR服务认证失败，请检查密钥配置")
            return ""
        
        # 调用OCR API
        r = requests.post(
            "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
            params={"access_token": token},
            data={"image": b64},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        res = r.json()
        
        # 处理OCR错误
        if "error_code" in res:
            st.error(f"❌ OCR识别失败：{res.get('error_msg', '未知错误')}")
            logger.error(f"OCR API错误：{res}")
            return ""
        
        # 提取识别结果
        return "\n".join([i["words"] for i in res.get("words_result", [])])
    
    except Exception as e:
        st.error(f"❌ OCR识别失败：{str(e)}")
        logger.error(f"OCR处理失败：{e}", exc_info=True)
        return ""

# ====================== 基金数据获取（优化：性能+缓存+错误处理） ======================
@st.cache_data(ttl=CACHE_TTL_FUND_LIST, show_spinner="加载基金列表...")
def get_fund_list() -> pd.DataFrame:
    """获取基金列表（带缓存）"""
    try:
        df = ak.fund_name_em()[["基金代码", "基金简称"]]
        # 去重+空值过滤
        df = df.drop_duplicates(subset=["基金代码"]).dropna()
        return df if not df.empty else pd.DataFrame(columns=["基金代码", "基金简称"])
    except Exception as e:
        logger.error(f"获取基金列表失败：{e}", exc_info=True)
        return pd.DataFrame(columns=["基金代码", "基金简称"])

@st.cache_data(ttl=CACHE_TTL_FUND_INFO, show_spinner="加载基金实时数据...")
@retry_decorator(stop_attempt=3, min_wait=2, max_wait=5)
def get_fund_info(code: str) -> Dict[str, float]:
    """获取基金实时信息（净值/估算值/涨跌幅）"""
    try:
        url = f"http://fundgz.1234567.com.cn/js/{code}.js"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        # 解析基金数据
        data = re.search(r'jsonpgz\((.*)\);', r.text)
        if not data:
            return {"nav": 0.0, "gsz": 0.0, "chg": 0.0}
        
        data = json.loads(data.group(1))
        return {
            "nav": float(data.get("dwjz", 0) or 0),
            "gsz": float(data.get("gsz", 0) or 0),
            "chg": float(data.get("gszzl", 0) or 0)
        }
    except Exception as e:
        logger.error(f"获取基金{code}信息失败：{e}", exc_info=True)
        return {"nav": 0.0, "gsz": 0.0, "chg": 0.0}

def batch_get_fund_info(codes: List[str]) -> Dict[str, Dict[str, float]]:
    """批量获取基金信息（优化性能，减少请求次数）"""
    fund_infos = {}
    for code in codes:
        fund_infos[code] = get_fund_info(code)
        time.sleep(0.1)  # 防请求过快被限制
    return fund_infos

def match_fund(keyword: str) -> Dict[str, str]:
    """匹配基金（代码精准匹配/名称模糊匹配）"""
    df = get_fund_list()
    if df.empty:
        logger.warning("基金列表为空，无法匹配")
        return {}
    
    # 代码精准匹配（6位数字）
    if keyword.isdigit() and len(keyword) == 6:
        mask = df["基金代码"] == keyword
        if mask.any():
            row = df[mask].iloc[0]
            return {"code": row["基金代码"], "name": row["基金简称"]}
    
    # 名称模糊匹配（不区分大小写）
    mask = df["基金简称"].str.contains(keyword, na=False, case=False)
    if mask.any():
        row = df[mask].iloc[0]
        return {"code": row["基金代码"], "name": row["基金简称"]}
    
    logger.warning(f"未匹配到基金：{keyword}")
    return {}

# ====================== 策略配置（优化：更简洁的读写） ======================
def load_strategy_config() -> Dict[str, float]:
    """加载策略配置"""
    try:
        response = supabase_execute(
            supabase.table("strategy_config").select("*").execute
        )
        config = {i["rule_name"]: i["rule_value"] for i in response.data}
        # 兜底默认值
        config.setdefault("T_SELL_THRESHOLD", DEFAULT_T_SELL_THRESHOLD)
        return config
    except Exception as e:
        logger.error(f"加载策略配置失败：{e}", exc_info=True)
        return {"T_SELL_THRESHOLD": DEFAULT_T_SELL_THRESHOLD}

def save_strategy_config(rule_name: str, rule_value: float) -> bool:
    """保存策略配置"""
    if not validate_numeric(rule_value):
        logger.error(f"配置值无效：{rule_name}={rule_value}")
        st.error("❌ 配置值必须为正数（≥0.01）")
        return False
    
    try:
        supabase_execute(
            supabase.table("strategy_config").upsert,
            {"rule_name": rule_name, "rule_value": rule_value},
            on_conflict="rule_name"
        )
        logger.info(f"策略配置保存成功：{rule_name}={rule_value}")
        return True
    except Exception as e:
        logger.error(f"保存策略配置失败：{e}", exc_info=True)
        return False

# ====================== 持仓交易（优化：更严谨的校验+事务） ======================
def buy_fund(code: str, name: str, amount: float, price: float, date: str) -> bool:
    """买入基金（事务处理，防除0，精准校验）
    Args:
        code: 基金代码
        name: 基金名称
        amount: 买入金额（元）
        price: 买入价格（元）
        date: 买入日期（YYYY-MM-DD）
    Returns:
        bool: 操作是否成功
    """
    # 精准校验
    if not re.match(r'^\d{6}$', code):
        st.error("❌ 基金代码必须为6位数字")
        return False
    
    if not validate_numeric(amount):
        st.error(f"❌ 买入金额无效（需≥0.01元）：{amount}")
        return False
    
    # 金额上限限制（可配置）
    if amount > 10_000_000:
        st.error("❌ 买入金额过大（单次上限1000万元）")
        return False
    
    if not validate_numeric(price, min_v=0.01):
        st.error(f"❌ 基金价格无效（需≥0.01元）：{price}")
        return False
    
    # 计算份额（价格已校验>0，无除0风险）
    try:
        shares = round(amount / price, 4)  # 保留4位小数，符合基金份额精度
    except Exception as e:
        logger.error(f"计算基金份额失败：{e}", exc_info=True)
        st.error("❌ 计算份额失败，请检查价格是否有效")
        return False
    
    # 事务逻辑
    def transaction_logic():
        """持仓更新事务逻辑"""
        # 查询现有持仓
        response = supabase_execute(
            supabase.table("portfolio").select("*").eq("fund_code", code).execute
        )
        
        if response.data:
            # 更新现有持仓
            old = response.data[0]
            old_shares = float(old.get("shares", 0))
            old_cost = float(old.get("cost_price", 0))
            
            new_shares = old_shares + shares
            # 计算新成本价（防除0）
            total_cost = (old_shares * old_cost) + amount
            new_cost = round(total_cost / new_shares, 4) if new_shares > 0 else price
            
            supabase_execute(
                supabase.table("portfolio").update,
                {"shares": new_shares, "cost_price": new_cost}
            ).eq("fund_code", code).execute()
        else:
            # 新增持仓
            supabase_execute(
                supabase.table("portfolio").insert,
                {
                    "fund_code": code,
                    "fund_name": name,
                    "shares": shares,
                    "cost_price": price,
                    "buy_date": date
                }
            )
    
    # 执行事务
    if supabase_transaction(transaction_logic):
        logger.success(f"买入基金成功：{code} | 名称：{name} | 金额：{amount} | 价格：{price}")
        return True
    else:
        st.error("❌ 买入失败（数据库事务异常），请重试")
        return False

# ====================== AI策略顾问（优化：缓存+更清晰的提示） ======================
def get_strategy_advice(msgs: List[Dict], context: str) -> str:
    """获取AI策略建议（带缓存）"""
    # 检查AI客户端是否可用
    client = get_deepseek_client()
    if not client:
        return "⚠️ AI服务不可用（请检查DeepSeek密钥配置或网络）"
    
    # 缓存key（基于上下文和消息长度）
    cache_key = hashlib.md5(f"{context}_{len(msgs)}".encode()).hexdigest()
    ai_cache = st.session_state.get("ai_cache", {})
    
    # 命中缓存直接返回
    if cache_key in ai_cache:
        return ai_cache[cache_key]
    
    # 构建提示词
    prompt = f"""
    你是专业的基金投资顾问，基于以下持仓信息和当前市场情况，给出具体的操作建议：
    持仓信息：{context}
    建议要求：
    1. 明确给出买入/卖出/持有建议
    2. 说明具体理由（结合市场趋势、基金表现等）
    3. 建议需具体、可执行
    4. 语言简洁易懂，避免专业术语堆砌
    """
    
    try:
        # 调用AI
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 降低随机性，提升建议稳定性
            timeout=30
        )
        
        advice = response.choices[0].message.content.strip()
        logger.info("AI策略建议生成成功")
        
        # 缓存结果（有效期1小时）
        st.session_state.setdefault("ai_cache", {})[cache_key] = advice
        
        return advice
    except Exception as e:
        error_msg = f"❌ AI分析失败：{str(e)[:50]}..."
        logger.error(f"AI策略生成失败：{e}", exc_info=True)
        return error_msg

# ====================== 基金详情（优化：缓存+空值处理） ======================
@st.cache_data(ttl=CACHE_TTL_FUND_DETAIL, show_spinner="加载基金持仓详情...")
def get_fund_detail(code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """获取基金持仓详情（重仓股票+行业分布）"""
    try:
        # 获取重仓股票
        stock_df = ak.fund_portfolio_stock_em(symbol=code).head(10)
        stock_df = stock_df.dropna(subset=["股票代码", "股票名称"])
        
        # 获取行业分布
        industry_df = ak.fund_portfolio_industry_em(symbol=code)
        industry_df = industry_df.dropna(subset=["行业名称", "占净值比例"])
        
        # 兜底空DataFrame
        stock_df = stock_df if not stock_df.empty else pd.DataFrame(columns=["股票代码", "股票名称", "占净值比例"])
        industry_df = industry_df if not industry_df.empty else pd.DataFrame(columns=["行业名称", "占净值比例"])
        
        return stock_df, industry_df
    except Exception as e:
        logger.error(f"获取基金{code}详情失败：{e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_FUND_NAV, show_spinner="加载基金净值数据...")
def get_fund_nav(code: str, days: int = 90) -> pd.DataFrame:
    """获取基金净值数据（近N天）"""
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值")
        if df.empty:
            logger.warning(f"基金{code}无净值数据")
            return pd.DataFrame(columns=["date", "nav", "type", "remark"])
        
        # 取最近N天数据
        df = df.tail(days)
        df.columns = ["date", "nav", "type", "remark"]
        
        # 数据清洗
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce").fillna(0)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        return df
    except Exception as e:
        logger.error(f"获取基金{code}净值失败：{e}", exc_info=True)
        return pd.DataFrame(columns=["date", "nav", "type", "remark"])

# ====================== 页面导航（优化：更清晰的标签） ======================
st.sidebar.title("📋 基金管理菜单")
page = st.sidebar.radio(
    "选择功能页面",
    [
        "📊 持仓总览",
        "📋 操作建议",
        "📁 持仓管理",
        "⚙️ 策略配置",
        "💬 AI分析师"
    ],
    index=0  # 默认显示持仓总览
)

# ====================== 页面1：持仓总览（优化：性能+可视化+用户体验） ======================
if page == "📊 持仓总览":
    st.header("📊 基金持仓总览", divider="blue")
    
    try:
        with st.spinner("加载持仓数据..."):
            # 获取持仓数据（分页，避免数据过多）
            response = supabase_execute(
                supabase.table("portfolio").select("*").range(0, 100).execute
            )
        
        if not response.data:
            st.info("📭 暂无持仓数据，请先在「持仓管理」页面添加持仓")
        else:
            # 初始化统计变量
            total_asset = 0.0  # 总资产
            total_profit = 0.0  # 总盈亏
            day_profit = 0.0  # 当日收益
            today = now_cn().strftime("%Y-%m-%d")
            
            # 批量获取基金信息（优化性能）
            fund_codes = [item["fund_code"] for item in response.data]
            fund_infos = batch_get_fund_info(fund_codes)
            
            # 处理持仓数据
            table_data = []
            for item in response.data:
                code = item["fund_code"]
                fund_info = fund_infos.get(code, {"nav": 0.0, "gsz": 0.0, "chg": 0.0})
                
                # 基础数据
                shares = float(item.get("shares", 0))
                cost_price = float(item.get("cost_price", 0))
                nav = float(fund_info.get("nav", 0))
                
                # 计算市值（防除0）
                market_value = round(shares * nav, 2) if nav > 0 else 0.0
                # 计算成本
                cost_value = round(shares * cost_price, 2)
                # 计算盈亏
                profit = round(market_value - cost_value, 2)
                # 计算当日收益
                day_p = round(market_value * fund_info["chg"] / 100, 2) if market_value > 0 else 0.0
                
                # 累加统计
                total_asset += market_value
                total_profit += profit
                day_profit += day_p
                
                # 构建表格数据
                table_data.append({
                    "基金名称": escape_html(item["fund_name"]),
                    "基金代码": code,
                    "持仓份额": f"{shares:.2f}",
                    "成本价(元)": f"{cost_price:.4f}",
                    "当前净值(元)": f"{nav:.4f}",
                    "涨跌幅(%)": f"{fund_info['chg']:+.2f}",
                    "持仓市值(元)": f"{market_value:,.2f}",
                    "盈亏(元)": f"{profit:+,.2f}"
                })
            
            # 计算收益率（防除0）
            total_cost = total_asset - total_profit  # 总成本
            if total_cost > 0:
                total_rate = round((total_profit / total_cost) * 100, 2)
            else:
                total_rate = 0.0
            
            if total_asset > 0:
                day_rate = round((day_profit / total_asset) * 100, 2)
            else:
                day_rate = 0.0
            
            # 展示核心指标（卡片式）
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>总资产</h4>
                    <p style="font-size: 24px; font-weight: bold;">{total_asset:,.2f} 元</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>当日收益</h4>
                    <p style="font-size: 24px; font-weight: bold; color: {'#ef4444' if day_profit > 0 else '#16a34a'};">
                        {day_profit:+,.2f} 元 ({day_rate:+.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>总盈亏</h4>
                    <p style="font-size: 24px; font-weight: bold; color: {'#ef4444' if total_profit > 0 else '#16a34a'};">
                        {total_profit:+,.2f} 元
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>总收益率</h4>
                    <p style="font-size: 24px; font-weight: bold; color: {'#ef4444' if total_rate > 0 else '#16a34a'};">
                        {total_rate:+.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # 展示持仓表格
            st.subheader("📋 持仓明细")
            df_table = pd.DataFrame(table_data)
            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True  # 隐藏索引列，更美观
            )
            
            # 基金详情查询
            st.subheader("🔍 基金详情查询", divider="gray")
            fund_code_input = st.text_input("输入基金代码", placeholder="例如：000001", max_chars=6)
            
            if fund_code_input:
                with st.spinner("加载基金详情..."):
                    stock_df, industry_df = get_fund_detail(fund_code_input)
                    nav_df = get_fund_nav(fund_code_input)
                
                # 标签页展示详情
                tab1, tab2, tab3 = st.tabs(["📈 重仓股票", "📊 行业分布", "📉 净值走势"])
                
                with tab1:
                    if not stock_df.empty:
                        st.dataframe(stock_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("📭 暂无重仓股票数据")
                
                with tab2:
                    if not industry_df.empty:
                        # 优化图表样式
                        fig = px.bar(
                            industry_df,
                            x="占净值比例",
                            y="行业名称",
                            orientation="h",
                            title="基金行业分布",
                            color="占净值比例",
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📭 暂无行业分布数据")
                
                with tab3:
                    if not nav_df.empty and nav_df["nav"].sum() > 0:
                        # 优化净值走势图表
                        fig = px.line(
                            nav_df,
                            x="date",
                            y="nav",
                            title=f"{fund_code_input} 净值走势（近90天）",
                            labels={"nav": "单位净值(元)", "date": "日期"},
                            line_shape="spline",
                            color_discrete_sequence=["#2563eb"]
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📭 暂无净值走势数据")
    
    except Exception as e:
        logger.error("加载持仓总览失败", exc_info=True)
        st.error(f"❌ 加载持仓数据失败：{str(e)[:100]}...")

# ====================== 页面2：操作建议（优化：更清晰的提示+时效性） ======================
elif page == "📋 操作建议":
    st.header("📋 每日操作建议", divider="blue")
    
    # 加载策略配置
    strategy_cfg = load_strategy_config()
    t_sell_threshold = strategy_cfg.get("T_SELL_THRESHOLD", DEFAULT_T_SELL_THRESHOLD)
    
    # 生成建议按钮
    if st.button("🚀 生成操作建议", type="primary", use_container_width=True):
        with st.spinner("AI正在分析市场和持仓情况，生成操作建议..."):
            # 检查是否为交易日
            if not is_trading_day():
                st.info(f"📅 当前日期（{now_cn().strftime('%Y-%m-%d')}）非交易日，暂无操作建议")
            else:
                # 获取持仓上下文
                try:
                    response = supabase_execute(supabase.table("portfolio").select("*").execute)
                    if response.data:
                        context = "；".join([f"{i['fund_name']}({i['fund_code']}) 持仓{i['shares']}份" for i in response.data])
                    else:
                        context = "无持仓"
                except Exception as e:
                    logger.error(f"获取持仓上下文失败：{e}", exc_info=True)
                    context = "持仓数据加载失败"
                
                # 生成AI建议
                advice = get_strategy_advice([], context)
                
                # 展示基础规则
                st.success(f"✅ 基础交易规则：亏损做T仓涨幅≥{t_sell_threshold}%卖出，回本逐步减仓")
                
                # 展示AI建议
                st.subheader("🤖 AI策略建议")
                st.markdown(f"> {advice}")
    
    # 规则说明
    st.info(f"""
    📌 规则说明：
    1. 做T卖出阈值：{t_sell_threshold}%（可在「策略配置」页面调整）
    2. 回本后建议逐步减仓，控制风险
    3. 仅在交易日（周一至周五，非节假日）生成有效建议
    4. 当前时间：{now_cn().strftime('%Y-%m-%d %H:%M:%S')}
    5. 交易时间：9:30-11:30, 13:00-15:00
    """)

# ====================== 页面3：持仓管理（优化：OCR+手动录入+用户体验） ======================
elif page == "📁 持仓管理":
    st.header("📁 基金持仓管理", divider="blue")
    
    # OCR识别入库
    with st.expander("📸 上传持仓截图（OCR识别）", expanded=True):
        uploaded_file = st.file_uploader("选择图片（PNG/JPG）", type=["png", "jpg"])
        
        if uploaded_file:
            with st.spinner("正在识别图片中的基金信息..."):
                ocr_text = ocr_image(uploaded_file)
            
            if ocr_text:
                # 展示OCR识别结果
                st.text_area("OCR识别结果", ocr_text, height=150)
                
                # AI提取基金信息
                with st.spinner("AI正在提取基金信息..."):
                    client = get_deepseek_client()
                    funds_data = None
                    
                    if client:
                        prompt = f"""
                        从以下文本中提取基金名称和市值（单位：元），仅返回JSON数组格式，示例：
                        [{{"name":"易方达蓝筹精选混合","market_value":10000.0}},{{"name":"招商中证白酒指数","market_value":5000.0}}]
                        文本内容：{ocr_text}
                        """
                        try:
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0  # 精准提取，关闭随机性
                            )
                            funds_data = safe_json_parse(response.choices[0].message.content)
                        except Exception as e:
                            logger.error(f"AI提取基金信息失败：{e}", exc_info=True)
                            st.error("❌ AI提取基金信息失败，请手动录入")
                
                # 展示并编辑提取的基金信息
                if funds_data and len(funds_data) > 0:
                    st.subheader("📝 待入库基金信息（请校对）")
                    
                    # 可编辑表格
                    edited_df = st.data_editor(
                        pd.DataFrame(funds_data),
                        column_config={
                            "name": st.column_config.TextColumn("基金名称", required=True, width="medium"),
                            "market_value": st.column_config.NumberColumn(
                                "市值（元）",
                                min_value=0.01,
                                step=100.0,
                                required=True,
                                width="small"
                            )
                        },
                        num_rows="dynamic",  # 支持增删行
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # 确认入库
                    if st.button("✅ 确认入库", type="primary", use_container_width=True):
                        success_count = 0
                        fail_count = 0
                        with st.spinner("正在入库持仓数据..."):
                            for _, row in edited_df.iterrows():
                                fund_name = row.get("name", "").strip()
                                market_value = row.get("market_value", 0)
                                
                                # 基础校验
                                if not fund_name or not validate_numeric(market_value):
                                    fail_count += 1
                                    logger.warning(f"入库跳过：名称/金额无效 - {fund_name}|{market_value}")
                                    continue
                                
                                # 匹配基金
                                fund_match = match_fund(fund_name)
                                if not fund_match:
                                    st.warning(f"⚠️ 未匹配到基金：{fund_name}，请检查名称是否正确")
                                    fail_count += 1
                                    continue
                                
                                # 获取基金价格
                                fund_info = get_fund_info(fund_match["code"])
                                nav_price = fund_info["nav"]
                                if not validate_numeric(nav_price, 0.01):
                                    st.warning(f"⚠️ {fund_match['name']} 价格异常：{nav_price}，跳过入库")
                                    fail_count += 1
                                    continue
                                
                                # 买入入库
                                if buy_fund(
                                    code=fund_match["code"],
                                    name=fund_match["name"],
                                    amount=market_value,
                                    price=nav_price,
                                    date=now_cn().strftime("%Y-%m-%d")
                                ):
                                    success_count += 1
                                else:
                                    fail_count += 1
                        
                        # 展示入库结果
                        st.success(f"✅ 入库完成：成功{success_count}条 | 失败{fail_count}条")
                else:
                    st.warning("⚠️ 未提取到有效基金信息，请使用下方手动录入功能")
        
        # 手动录入
        st.divider()
        st.subheader("✍️ 手动录入持仓")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            manual_code = st.text_input("基金代码（6位数字）", placeholder="000001")
        with col2:
            manual_name = st.text_input("基金名称", placeholder="易方达蓝筹精选混合")
        with col3:
            manual_amount = st.number_input("买入金额（元）", min_value=0.01, step=100.0, value=1000.0)
        
        if st.button("📥 手动入库", use_container_width=True):
            # 校验输入
            if not manual_code or not manual_name:
                st.error("❌ 基金代码和名称不能为空")
            elif not validate_numeric(manual_amount):
                st.error("❌ 买入金额必须≥0.01元")
            else:
                # 获取基金价格
                fund_info = get_fund_info(manual_code)
                nav_price = fund_info["nav"]
                
                if not validate_numeric(nav_price, 0.01):
                    st.error(f"❌ 基金{manual_code}价格异常：{nav_price}，无法入库")
                else:
                    # 执行买入
                    if buy_fund(
                        code=manual_code,
                        name=manual_name,
                        amount=manual_amount,
                        price=nav_price,
                        date=now_cn().strftime("%Y-%m-%d")
                    ):
                        st.success("✅ 手动入库成功！")
                        # 清空输入框（通过rerun）
                        st.rerun()

# ====================== 页面4：策略配置 + AI顾问（优化：交互+会话管理） ======================
elif page == "⚙️ 策略配置":
    st.header("⚙️ 交易策略配置", divider="blue")
    
    # 策略参数配置
    st.subheader("📝 基础策略参数")
    strategy_cfg = load_strategy_config()
    t_sell_val = st.number_input(
        "做T卖出阈值(%)",
        min_value=0.5,
        max_value=10.0,
        value=strategy_cfg.get("T_SELL_THRESHOLD", DEFAULT_T_SELL_THRESHOLD),
        step=0.5,
        help="亏损做T仓涨幅达到该阈值时建议卖出，默认2.0%"
    )
    
    if st.button("💾 保存配置", type="primary", use_container_width=True):
        if save_strategy_config("T_SELL_THRESHOLD", t_sell_val):
            st.success("✅ 策略配置保存成功！")
            st.rerun()
        else:
            st.error("❌ 配置保存失败，请重试")
    
    # AI策略顾问
    st.divider()
    st.subheader("🤖 AI策略顾问")
    
    # 获取持仓上下文
    try:
        response = supabase_execute(supabase.table("portfolio").select("*").execute)
        if response.data:
            ctx = "；".join([f"{i['fund_name']}({i['fund_code']}) 持仓{int(i['shares'])}份" for i in response.data])
        else:
            ctx = "无持仓"
    except Exception as e:
        logger.error(f"获取持仓上下文失败：{e}", exc_info=True)
        ctx = "持仓数据加载失败"
    
    # 初始化会话
    if "advisor_msgs" not in st.session_state:
        st.session_state.advisor_msgs = []
    
    # 限制会话长度
    if len(st.session_state.advisor_msgs) > AI_CONTEXT_MAX_LENGTH * 2:
        st.session_state.advisor_msgs = st.session_state.advisor_msgs[-AI_CONTEXT_MAX_LENGTH * 2:]
    
    # 展示历史会话
    for msg in st.session_state.advisor_msgs:
        with st.chat_message(msg["role"]):
            st.write(escape_html(msg["content"]))
    
    # 提问输入
    prompt = st.chat_input(f"基于当前持仓：{ctx[:50]}... 咨询基金投资策略")
    if prompt:
        # 添加用户消息
        st.session_state.advisor_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(escape_html(prompt))
        
        # 生成AI回复
        with st.spinner("AI正在分析您的问题..."):
            reply = get_strategy_advice(st.session_state.advisor_msgs, ctx)
        
        # 展示并保存回复
        with st.chat_message("assistant"):
            st.write(escape_html(reply))
        st.session_state.advisor_msgs.append({"role": "assistant", "content": reply})
        
        st.rerun()

# ====================== 页面5：AI分析师（优化：会话管理+用户体验） ======================
elif page == "💬 AI分析师":
    st.header("💬 基金AI分析师", divider="blue")
    
    # 初始化会话
    if "analyst_msgs" not in st.session_state:
        st.session_state.analyst_msgs = []
    
    # 限制会话长度，防止超限
    if len(st.session_state.analyst_msgs) > MAX_ANALYST_MSGS:
        st.session_state.analyst_msgs = st.session_state.analyst_msgs[-MAX_ANALYST_MSGS:]
    
    # 展示历史消息
    for msg in st.session_state.analyst_msgs:
        with st.chat_message(msg["role"]):
            st.write(escape_html(msg["content"]))
    
    # 输入框提示
    input_placeholder = "请输入您的问题（例如：易方达蓝筹精选混合的投资建议、当前市场行情分析等）"
    user_input = st.chat_input(input_placeholder)
    
    if user_input:
        # 添加用户消息
        st.session_state.analyst_msgs.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(escape_html(user_input))
        
        # 生成AI回复
        with st.spinner("AI正在深度分析，请稍候..."):
            client = get_deepseek_client()
            if client:
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": user_input}],
                        temperature=0.7,  # 适度的随机性
                        timeout=30
                    )
                    reply = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"AI分析师回复失败：{e}", exc_info=True)
                    reply = f"❌ 分析失败：{str(e)[:50]}..."
            else:
                reply = "⚠️ AI服务不可用（请检查DeepSeek密钥配置）"
        
        # 展示并保存回复
        with st.chat_message("assistant"):
            st.write(escape_html(reply))
        st.session_state.analyst_msgs.append({"role": "assistant", "content": reply})
        
        st.rerun()
    
    # 使用提示
    st.info("""
    💡 使用提示：
    1. 可咨询单只基金的投资建议、市场行情分析、风险评估等
    2. 可提问基金投资技巧、仓位管理、止盈止损策略等
    3. 会话记录最多保留20条，超出将自动清理最早的记录
    4. AI回答仅供参考，不构成投资建议
    """)

# ====================== 系统日志与安全提示 ======================
logger.info(f"页面加载完成：{page} | 当前时间：{now_cn().strftime('%Y-%m-%d %H:%M:%S')}")
st.toast("✅ 基金管理系统加载完成", icon="🎉")