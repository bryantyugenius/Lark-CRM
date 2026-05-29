import json
import os
import re
from datetime import date, timedelta
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class ExtractedContact(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    type: Optional[str] = "customer"
    notes: Optional[str] = None


class ExtractedInteraction(BaseModel):
    contact: Optional[ExtractedContact] = None
    content: Optional[str] = None
    intent: Optional[str] = None
    next_action: Optional[str] = None
    next_action_at: Optional[str] = None


EXTRACT_PROMPT = """你是一个 CRM 助理。用户会用中文语音输入一段关于客户/合作伙伴的跟进记录。
请把语音转写内容提取为结构化数据。

规则：
- type 只能是 "customer"（客户）或 "partner"（合作伙伴），如果用户没有明确说明，默认 "customer"
- next_action_at 格式为 YYYY-MM-DD，如果无法判断具体日期则填 null
- 如果语音内容没有明显客户信息（比如闲聊），contact 字段可以填 null
- 只输出 JSON，不要输出其他内容，不要加 markdown 代码块标记

输出 JSON 格式如下：
{
  "contact": {
    "name": "姓名或称呼",
    "company": "公司名",
    "role": "职位",
    "type": "customer",
    "phone": "",
    "email": "",
    "notes": ""
  },
  "content": "跟进内容摘要",
  "intent": "跟进/报价/签约/介绍/其他",
  "next_action": "下一步行动描述",
  "next_action_at": "YYYY-MM-DD 或 null"
}"""


async def extract_structured(text: str) -> ExtractedInteraction:
    """用 AI 提取结构化信息，余额不足时降级到规则引擎"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"语音转写内容：\n{text}"},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if data.get("contact") is None:
            data["contact"] = None
        return ExtractedInteraction(**data)
    except Exception as e:
        err = str(e)
        if "402" in err or "Insufficient" in err or "Balance" in err:
            print(f"  ⚠️ AI API 余额不足，使用规则引擎降级处理")
            return rule_based_extract(text)
        raise


# ── 规则引擎兜底 ──

# 相对日期解析
_RELATIVE_DAYS = {
    "今天": 0, "明天": 1, "后天": 2,
    "大后天": 3,
}
_WEEKDAYS = {
    "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
    "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
}

# 公司关键词
_COMPANY_SUFFIX = r"(生物|科技|医药|医疗|技术|制药|化工|器械|基因|细胞|诊断|检测|试剂|仪器|公司|集团|实验室|研究所|中心|平台)"


def _parse_date(text: str) -> Optional[str]:
    """解析文本中的日期表达式，返回 YYYY-MM-DD"""
    today = date.today()

    # 今天/明天/后天
    for kw, delta in _RELATIVE_DAYS.items():
        if kw in text:
            d = today + timedelta(days=delta)
            return d.isoformat()

    # 下周一/下周二... / 下星期X
    m = re.search(r"(下)(?:周|星期)([一二三四五六日天])", text)
    if m:
        day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "天": 6, "日": 6}
        target = day_map[m.group(2)]
        current = today.weekday()  # 0=Monday
        days_ahead = target - current + 7
        if days_ahead <= 0:
            days_ahead += 7
        d = today + timedelta(days=days_ahead)
        return d.isoformat()

    # 本周一/本周二...
    m = re.search(r"(?:本)?周([一二三四五六日天])", text)
    if m:
        day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "天": 6, "日": 6}
        target = day_map[m.group(1)]
        current = today.weekday()
        days_ahead = target - current
        if days_ahead < 0:
            days_ahead += 7
        d = today + timedelta(days=days_ahead)
        return d.isoformat()

    # 下个月X号
    m = re.search(r"下(?:个)?月(\d{1,2})(?:号|日)?", text)
    if m:
        day_num = int(m.group(1))
        if today.month == 12:
            d = date(today.year + 1, 1, day_num)
        else:
            d = date(today.year, today.month + 1, day_num)
        return d.isoformat()

    # X月X号
    m = re.search(r"(\d{1,2})月(\d{1,2})(?:号|日)?", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        d = date(today.year, month, day)
        return d.isoformat()

    return None


def _extract_contact(text: str) -> Optional[ExtractedContact]:
    """规则提取联系人信息"""
    name = None
    company = None
    role = None
    contact_type = "customer"

    # 提取人名：跟/和/与/给 + 名字
    m = re.search(r"[跟和与给]([^\s，。,.]{2,4})(?:约|打|通|聊|见|发|说|谈|联系)", text)
    if m:
        name = m.group(1)

    # 提取公司名
    m = re.search(rf"([^\s，。,]{{2,8}}{_COMPANY_SUFFIX})", text)
    if m:
        company = m.group(1)

    # 提取职位（跟在人名或公司后面）
    m = re.search(r"(经理|主管|总监|老师|博士|教授|院长|主任|总|老板)", text)
    if m:
        role = m.group(1)

    # 合作伙伴判断
    if re.search(r"(合作|伙伴|供应商|代理)", text):
        contact_type = "partner"

    if name or company:
        return ExtractedContact(
            name=name, company=company, role=role, type=contact_type
        )
    return None


def _guess_intent(text: str) -> Optional[str]:
    if re.search(r"(报价|询价|价格|多少钱)", text): return "报价"
    if re.search(r"(签约|合同|签了|协议|订单)", text): return "签约"
    if re.search(r"(拜访|见面|约了|电话|聊聊|沟通|碰头)", text): return "跟进"
    if re.search(r"(介绍|推荐|引荐|认识一下)", text): return "介绍"
    if re.search(r"(付款|打款|汇款|收款)", text): return "付款"
    if re.search(r"(发货|物流|快递|寄送)", text): return "发货"
    return "跟进"


def rule_based_extract(text: str) -> ExtractedInteraction:
    """规则引擎：不依赖 AI，纯正则提取"""
    contact = _extract_contact(text)
    next_action_at = _parse_date(text)
    intent = _guess_intent(text)

    # 提取下一步行动
    next_action = None
    if contact and contact.name:
        next_action = f"联系{contact.name}"
        if intent == "跟进":
            next_action = f"跟进{contact.name}"
        elif intent == "报价":
            next_action = f"给{contact.name}报价"
        elif intent == "签约":
            next_action = f"与{contact.name}推进签约"

    print(f"  📋 Rule-based: contact={contact}, intent={intent}, next_action_at={next_action_at}")

    return ExtractedInteraction(
        contact=contact,
        content=text,
        intent=intent,
        next_action=next_action,
        next_action_at=next_action_at,
    )
