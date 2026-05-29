import json
import os
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
    """用 GPT-4o 提取结构化信息"""
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
    # 处理 contact 为 null 的情况
    if data.get("contact") is None:
        data["contact"] = None
    return ExtractedInteraction(**data)
