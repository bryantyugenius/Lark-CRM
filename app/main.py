from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import json
import os
import asyncio

from app.database import init_db, get_db, Contact, Interaction, Relationship
from app.lark_client import LarkClient
from app.ai_extract import extract_structured, ExtractedInteraction


app = FastAPI(title="Lark CRM Bot")
lark = LarkClient()


@app.on_event("startup")
async def startup():
    init_db()
    print("✅ Database initialized")
    print("✅ Lark CRM Bot started")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "Lark CRM Bot running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def decrypt_lark_message(encrypt_key: str, body: dict) -> dict:
    return body


@app.post("/webhook/lark")
async def lark_webhook(request: Request):
    """
    Lark 事件订阅 Webhook 入口。
    """
    body = await request.json()
    print(f"📩 Raw event: {json.dumps(body, ensure_ascii=False)[:300]}")

    # URL 验证挑战
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge"), "token": body.get("token")}

    # 解密（如配置了 ENCRYPT_KEY）
    encrypt_key = os.getenv("LARK_ENCRYPT_KEY", "")
    if encrypt_key and body.get("encrypt"):
        body = decrypt_lark_message(encrypt_key, body)

    header = body.get("header", {})
    event_type = header.get("event_type")
    event = body.get("event", {})

    print(f"📩 Received event: {event_type}")

    if event_type == "im.message.receive_v1":
        await handle_message(event)

    return JSONResponse(content={"success": True})


async def handle_message(event: dict):
    message = event.get("message", {})
    msg_type = message.get("message_type")
    chat_id = message.get("chat_id")
    message_id = message.get("message_id")
    sender_id = message.get("sender", {}).get("sender_id", {}).get("user_id", "")

    print(f"  msg_type={msg_type}, chat_id={chat_id}, sender={sender_id}")

    # 文字消息：给个提示
    if msg_type == "text":
        text_content = json.loads(message.get("content", "{}")).get("text", "")
        print(f"  text: {text_content[:100]}")
        await lark.reply_text(
            chat_id, message_id,
            "👋 我收到了你的消息！\n\n目前我主要处理**语音消息**——请发一段语音，我会自动识别并记录跟进信息。"
        )
        return

    # 只处理语音消息
    if msg_type != "audio":
        return

    try:
        # Step 1: 获取消息详情（拿 file_key）
        msg_detail = await lark.get_message_content(message_id)
        print(f"  msg_detail: {json.dumps(msg_detail, ensure_ascii=False)[:200]}")

        # Lark 消息结构：msg_type=audio 时，audio 字段里有 file_key
        file_key = None
        audio_info = msg_detail.get("data", {}).get("audio", {})
        if audio_info:
            file_key = audio_info.get("file_key")
        # 也尝试从 message 本体直接拿
        if not file_key:
            file_key = message.get("audio", {}).get("file_key")

        if not file_key:
            await lark.reply_text(chat_id, message_id, "❌ 无法获取语音文件，请重新发送")
            return

        # Step 2: 用 Lark 自带语音识别转文字（免费）
        await lark.reply_text(chat_id, message_id, "🔄 正在识别语音...")
        text = await lark.speech_to_text(file_key)
        print(f"  ASR result: {text}")

        if not text or len(text.strip()) < 2:
            await lark.reply_text(chat_id, message_id, "❌ 语音识别结果为空，请重新发送")
            return

        # Step 4: AI 提取结构化信息
        await lark.reply_text(
            chat_id, message_id, f"🔄 正在理解内容...\n识别文字：{text[:100]}"
        )
        extracted = await extract_structured(text)
        print(f"  Extracted: contact={extracted.contact}, content={extracted.content}")

        # Step 5: 写入数据库
        db = next(get_db())
        try:
            contact = None
            if extracted.contact and extracted.contact.name:
                contact = (
                    db.query(Contact)
                    .filter(Contact.name == extracted.contact.name)
                    .first()
                )
                if not contact:
                    contact = Contact(
                        name=extracted.contact.name,
                        company=extracted.contact.company,
                        role=extracted.contact.role,
                        type=extracted.contact.type or "customer",
                        phone=extracted.contact.phone,
                        email=extracted.contact.email,
                        notes=extracted.contact.notes,
                    )
                    db.add(contact)
                    db.flush()

            interaction = Interaction(
                contact_id=contact.id if contact else 0,
                content=extracted.content or text,
                raw_text=text,
                intent=extracted.intent,
                next_action=extracted.next_action,
                next_action_at=extracted.next_action_at,
                created_by=sender_id,
            )
            db.add(interaction)
            db.commit()
            contact_name = contact.name if contact else "N/A"
            print(
                f"  ✅ Saved to DB: contact={contact_name}, "
                f"interaction_id={interaction.id}"
            )

        finally:
            db.close()

        # Step 6: 回复确认卡片
        card = build_confirmation_card(extracted, contact, text)
        await lark.send_card(chat_id, message_id, card)

    except Exception as e:
        print(f"  ❌ Error handling message: {e}")
        import traceback
        traceback.print_exc()
        await lark.reply_text(chat_id, message_id, f"❌ 处理失败：{str(e)[:200]}")


def build_confirmation_card(
    extracted: ExtractedInteraction, contact, raw_text: str
) -> dict:
    """构建 Lark interactive card（dict，不是 JSON 字符串）"""
    elements = []

    c = extracted.contact
    if c and getattr(c, "name", None):
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**姓名**：{c.name}"}}
        )
    if c and getattr(c, "company", None):
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**公司**：{c.company}"},
            }
        )
    if c and getattr(c, "role", None):
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": f"**职位**：{c.role}"}}
        )
    if c and getattr(c, "type", None):
        type_label = "👤 客户" if c.type == "customer" else "🤝 合作伙伴"
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**类型**：{type_label}"},
            }
        )

    elements.append({"tag": "hr"})

    if extracted.content:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**跟进内容**：{extracted.content}"},
            }
        )
    if extracted.intent:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**意图**：{extracted.intent}"},
            }
        )
    if extracted.next_action:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**下一步**：{extracted.next_action}",
                },
            }
        )
    if extracted.next_action_at:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**时间**：{extracted.next_action_at}",
                },
            }
        )

    elements.append({"tag": "hr"})
    elements.append(
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"📝 原始识别：{raw_text[:100]}"},
        }
    )

    # 操作按钮
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 确认"},
                    "value": json.dumps({"action": "confirm"}),
                    "type": "primary",
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✏️ 修改"},
                    "value": json.dumps({"action": "edit"}),
                },
            ],
        }
    )

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "✅ 已记录跟进信息"},
            "template": "green",
        },
        "elements": elements,
    }
    return card
