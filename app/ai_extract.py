import json
import os
import re
from datetime import date, timedelta
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# gpt-4o-mini: 便宜（$0.15/1M input），JSON 格式遵循度极高，比 DeepSeek 稳
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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


# ── 高质量 Few-shot Prompt ──
# 关键：给 AI 看"正确长什么样"，比任何规则都有用

SYSTEM_PROMPT = """你是一个 CRM 数据提取专家。用户会用中文描述一段客户/合作伙伴的跟进情况。
你的任务是把口语化描述精确转为 JSON。

今天是 {today}，{weekday}。

# 提取规则

1. **联系人识别**
   - 从"跟/和/与 XX 约了"、"拜访了 XX"、"XX 说"中提取姓名
   - 如果提到公司名（如"广州永津生物"），填入 company
   - 如果提到职位（经理、博士、教授等），填入 role
   - type 默认 "customer"，明确提到"合作/伙伴/供应商/代理"则用 "partner"
   - 没有明确联系人信息时 contact 填 null

2. **日期计算（这是最容易犯错的）**
   - 基于今天 {today} ({weekday}) 计算
   - "下周一" = 下周的周一（不是本周）
   - "明天" = {today} + 1 天
   - "下个月5号" = 下个月的5号
   - "月底" = 当月最后一天
   - 只输出确定的日期，不确定就填 null
   - 格式必须是 YYYY-MM-DD

3. **意图分类**（严格六选一）
   - "跟进"：约了沟通、拜访、电话、见面
   - "报价"：询价、报价、价格、多少钱
   - "签约"：签约、合同、协议、订单、成交
   - "介绍"：介绍、推荐、引荐、认识
   - "付款"：付款、打款、汇款、开票
   - "发货"：发货、物流、快递、寄送

4. **内容重组**
   - content 是一句话摘要，清晰完整（如"与XX公司XX经理约了下周一电话沟通XX项目"）
   - next_action 是明确的下一步行动（如"打电话给黄康"、"发报价给王博士"）
   - 不要让 next_action 等于 content

# 示例

示例 1:
输入：跟广州永津生物黄康约了下周一电话
输出：
{{"contact":{{"name":"黄康","company":"广州永津生物","role":null,"type":"customer","phone":null,"email":null,"notes":null}},"content":"与广州永津生物黄康约了下周一电话沟通","intent":"跟进","next_action":"电话联系黄康","next_action_at":"2026-06-01"}}

示例 2:
输入：今天拜访了北京生物王博士，他对NK细胞培养基很感兴趣，让我尽快给个报价
输出：
{{"contact":{{"name":"王博士","company":"北京生物","role":"博士","type":"customer","phone":null,"email":null,"notes":"对NK细胞培养基感兴趣"}},"content":"拜访北京生物王博士，其对NK细胞培养基有需求，需尽快报价","intent":"报价","next_action":"给王博士发NK细胞培养基报价","next_action_at":null}}

示例 3:
输入：和上海赛维尔生物的刘经理签了年度合作协议，他们是我们新的经销商
输出：
{{"contact":{{"name":"刘经理","company":"上海赛维尔生物","role":"经理","type":"partner","phone":null,"email":null,"notes":"新经销商，年度合作协议"}},"content":"与上海赛维尔生物刘经理签署年度合作协议，成为新的经销商合作伙伴","intent":"签约","next_action":"跟进刘经理首批订单","next_action_at":null}}

示例 4:
输入：今天天气不错
输出：
{{"contact":null,"content":"闲聊，无业务信息","intent":null,"next_action":null,"next_action_at":null}}

示例 5:
输入：黄康说他们实验室还需要一批抗体，让我下周三之前报价
输出：
{{"contact":{{"name":"黄康","company":null,"role":null,"type":"customer","phone":null,"email":null,"notes":"需要一批抗体"}},"content":"黄康所在实验室需要一批抗体，要求下周三前提供报价","intent":"报价","next_action":"准备抗体报价方案，下周三前发给黄康","next_action_at":"2026-06-03"}}

# 关键要求
- 只输出 JSON，不要加任何解释、不要 markdown 代码块标记
- next_action_at 必须是真实计算出的日期或 null
- 不确定的信息宁可留 null 也不要瞎编"""


_WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# ── 输出校验 ──

VALID_INTENTS = {"跟进", "报价", "签约", "介绍", "付款", "发货"}


def _validate_and_fix(result: ExtractedInteraction, text: str) -> ExtractedInteraction:
    """校验 AI 输出，自动修正常见错误"""
    today = date.today()

    # 1. 意图修正：不在枚举内就用规则引擎兜底
    if result.intent and result.intent not in VALID_INTENTS:
        print(f"  ⚠️ Invalid intent '{result.intent}', fixing via rule")
        result.intent = _guess_intent(text)

    # 2. 日期修正：离谱的日期用规则引擎覆盖
    if result.next_action_at:
        try:
            parsed = date.fromisoformat(result.next_action_at)
            delta = abs((parsed - today).days)
            if delta > 730:
                print(f"  ⚠️ Date hallucination: {result.next_action_at}, fixing")
                result.next_action_at = _parse_date(text)
        except ValueError:
            print(f"  ⚠️ Invalid date format: {result.next_action_at}")
            result.next_action_at = _parse_date(text)

    # 3. 处理 AI 输出 "无" / "None" 字符串
    if result.next_action and result.next_action in ("无", "None", "null", "暂无"):
        result.next_action = None

    # 4. content 为空时用原文
    if not result.content or result.content.strip() in ("", "无", "闲聊"):
        result.content = text[:200]

    return result


async def extract_structured(text: str) -> ExtractedInteraction:
    """用 AI 提取结构化信息，带校验层"""
    today = date.today()
    weekday = _WEEKDAY_NAMES[today.weekday()]
    prompt = SYSTEM_PROMPT.format(today=today.isoformat(), weekday=weekday)

    try:
        print(f"🧠 Calling {MODEL} @ {OPENAI_BASE_URL}...")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # 低温度，减少幻觉
        )
        content = resp.choices[0].message.content
        print(f"  🤖 AI raw output: {content[:300]}")

        data = json.loads(content)
        if data.get("contact") is None or data.get("contact") == {}:
            data["contact"] = None

        result = ExtractedInteraction(**data)
        result = _validate_and_fix(result, text)

        print(f"  ✅ Extracted: name={result.contact.name if result.contact else 'N/A'}, "
              f"intent={result.intent}, date={result.next_action_at}")
        return result

    except Exception as e:
        err = str(e)
        if "402" in err or "Insufficient" in err or "Balance" in err:
            print(f"  ⚠️ AI API 余额不足，使用规则引擎降级处理")
            return rule_based_extract(text)
        raise


# ════════════════════════════════════════════════
#  规则引擎兜底（AI 不可用时启用）
# ════════════════════════════════════════════════

def _parse_date(text: str) -> Optional[str]:
    """解析文本中的日期表达式，返回 YYYY-MM-DD"""
    today = date.today()

    rel = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    for kw, delta in rel.items():
        if kw in text:
            return (today + timedelta(days=delta)).isoformat()

    # 下周一～下周日
    m = re.search(r"下(?:周|星期)([一二三四五六日天])", text)
    if m:
        day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "天": 6, "日": 6}
        target = day_map[m.group(1)]
        days_ahead = target - today.weekday() + 7
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).isoformat()

    # 本周一～本周日
    m = re.search(r"(?:本)?周([一二三四五六日天])", text)
    if m:
        day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "天": 6, "日": 6}
        target = day_map[m.group(1)]
        days_ahead = target - today.weekday()
        if days_ahead < 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).isoformat()

    # 下个月X号
    m = re.search(r"下(?:个)?月(\d{1,2})(?:号|日)?", text)
    if m:
        day_num = int(m.group(1))
        y, mth = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return date(y, mth, day_num).isoformat()

    # X月X号
    m = re.search(r"(\d{1,2})月(\d{1,2})(?:号|日)?", text)
    if m:
        return date(today.year, int(m.group(1)), int(m.group(2))).isoformat()

    # 月底
    if "月底" in text:
        if today.month == 12:
            d = date(today.year, 12, 31)
        else:
            d = date(today.year, today.month + 1, 1) - timedelta(days=1)
        return d.isoformat()

    return None


_COMPANY_SUFFIX = r"(生物|科技|医药|医疗|技术|制药|化工|器械|基因|细胞|诊断|检测|试剂|仪器|公司|集团|实验室|研究所|中心|平台)"


def _extract_contact(text: str) -> Optional[ExtractedContact]:
    name = company = role = None
    contact_type = "customer"

    m = re.search(r"[跟和与给]([^\s，。,.]{2,4})(?:约|打|通|聊|见|发|说|谈|联系)", text)
    if m:
        name = m.group(1)

    m = re.search(rf"([^\s，。,]{{2,8}}{_COMPANY_SUFFIX})", text)
    if m:
        company = m.group(1)

    m = re.search(r"(经理|主管|总监|老师|博士|教授|院长|主任|总|老板)", text)
    if m:
        role = m.group(1)

    if re.search(r"(合作|伙伴|供应商|代理|经销商)", text):
        contact_type = "partner"

    if name or company:
        return ExtractedContact(name=name, company=company, role=role, type=contact_type)
    return None


def _guess_intent(text: str) -> Optional[str]:
    if re.search(r"(报价|询价|价格|多少钱)", text): return "报价"
    if re.search(r"(签约|合同|签了|协议|订单|成交)", text): return "签约"
    if re.search(r"(拜访|见面|约了|电话|聊聊|沟通|碰头)", text): return "跟进"
    if re.search(r"(介绍|推荐|引荐|认识一下)", text): return "介绍"
    if re.search(r"(付款|打款|汇款|收款|开票)", text): return "付款"
    if re.search(r"(发货|物流|快递|寄送)", text): return "发货"
    return "跟进"


def rule_based_extract(text: str) -> ExtractedInteraction:
    contact = _extract_contact(text)
    next_action_at = _parse_date(text)
    intent = _guess_intent(text)

    next_action = None
    if contact and contact.name:
        next_action = f"跟进{contact.name}"
        if intent == "报价":
            next_action = f"给{contact.name}报价"
        elif intent == "签约":
            next_action = f"与{contact.name}推进签约"

    print(f"  📋 Rule-based: contact={contact}, intent={intent}, date={next_action_at}")
    return ExtractedInteraction(
        contact=contact, content=text, intent=intent,
        next_action=next_action, next_action_at=next_action_at,
    )
