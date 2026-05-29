from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import json
import os
import re
from typing import Optional
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


@app.api_route("/webhook/lark", methods=["GET", "POST", "HEAD"])
async def lark_webhook(request: Request):
    """
    Lark 事件订阅 Webhook 入口。
    """
    # 记录所有请求
    print(f"📥 [{request.method}] {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    # GET/HEAD 请求：用于健康检查
    if request.method in ("GET", "HEAD"):
        return JSONResponse(content={"status": "ok", "message": "Lark webhook endpoint"}, status_code=200)

    # POST 请求处理
    try:
        body = await request.json()
    except Exception:
        body_text = await request.body()
        print(f"📩 Non-JSON body ({len(body_text)} bytes): {body_text[:200]}")
        return JSONResponse(content={"error": "invalid json"}, status_code=400)

    print(f"📩 Event body keys: {list(body.keys())}")
    print(f"📩 Raw event (first 500 chars): {json.dumps(body, ensure_ascii=False)[:500]}")

    # URL 验证挑战（优先处理，不管是否加密）
    challenge = body.get("challenge")
    if body.get("type") == "url_verification" or challenge:
        print(f"✅ URL verification: challenge={challenge[:20] if challenge else 'N/A'}...")
        # Lark 只期望 {"challenge": "xxx"}
        resp = {"challenge": challenge}
        print(f"✅ Returning: {resp}")
        return JSONResponse(content=resp, status_code=200)

    # 解密（如配置了 ENCRYPT_KEY）
    encrypt_key = os.getenv("LARK_ENCRYPT_KEY", "")
    if encrypt_key and body.get("encrypt"):
        print("🔐 Encrypted body detected, decrypting...")
        body = decrypt_lark_message(encrypt_key, body)

    header = body.get("header", {})
    event_type = header.get("event_type")
    event = body.get("event", {})

    print(f"📩 Received event: {event_type}")

    if event_type == "im.message.receive_v1":
        await handle_message(event)

    return JSONResponse(content={"success": True})


# ── 指令检测 ──

_META_COMMANDS = [
    (r"帮(我|忙).*(做|弄|搞|写|开发|创建|建|搭建|生成)", "dev_request"),
    (r"(你能|你会|你有).*(做什么|啥|什么功能|哪些功能)", "capability"),
    (r"(怎么用|如何用|使用说明|帮助|help)", "help"),
    (r"(查询|搜索|找一下|列出|显示|看看).*", "query"),
]


def _detect_meta_command(text: str) -> Optional[str]:
    """检测是否为对机器人的指令，而非 CRM 数据录入"""
    for pattern, cmd_type in _META_COMMANDS:
        if re.search(pattern, text):
            return cmd_type
    return None


async def handle_message(event: dict):
    message = event.get("message", {})
    msg_type = message.get("message_type")
    chat_id = message.get("chat_id")
    message_id = message.get("message_id")
    sender_id = message.get("sender", {}).get("sender_id", {}).get("user_id", "")

    print(f"  msg_type={msg_type}, chat_id={chat_id}, sender={sender_id}")

    # 文字消息
    if msg_type == "text":
        text = json.loads(message.get("content", "{}")).get("text", "")
        print(f"  text: {text[:100]}")

        # 先检测是否为对机器人的指令
        cmd_type = _detect_meta_command(text)
        if cmd_type:
            print(f"  🎯 Meta command detected: {cmd_type}")
            await handle_meta_command(chat_id, message_id, cmd_type, text)
            return

        await process_text_input(chat_id, message_id, sender_id, text)
        return

    # 语音消息
    if msg_type == "audio":
        try:
            content_data = json.loads(message.get("content", "{}"))
            file_key = content_data.get("file_key")
            print(f"  content parsed: {json.dumps(content_data, ensure_ascii=False)[:200]}")
            print(f"  file_key: {file_key}")

            if not file_key:
                await lark.reply_text(chat_id, message_id, "❌ 无法获取语音文件，请重新发送")
                return

            # 语音识别
            await lark.reply_text(chat_id, message_id, "🔄 正在识别语音...")
            text = await lark.speech_to_text(file_key)
            print(f"  ASR result: {text}")

            if not text or len(text.strip()) < 2:
                await lark.reply_text(chat_id, message_id, "❌ 语音识别结果为空，请重新发送")
                return

            await process_text_input(chat_id, message_id, sender_id, text)

        except Exception as e:
            print(f"  ❌ Error handling audio: {e}")
            import traceback
            traceback.print_exc()
            await lark.reply_text(chat_id, message_id, f"❌ 处理失败：{str(e)[:200]}")


async def handle_meta_command(chat_id: str, message_id: str, cmd_type: str, text: str):
    """处理对机器人的指令类消息"""
    if cmd_type == "dev_request":
        await lark.reply_text(
            chat_id, message_id,
            "🤖 我目前是一个**语音CRM录入助手**，专用于：\n"
            "• 接收你的语音/文字跟进记录\n"
            "• 自动提取客户、意图、下一步行动\n"
            "• 写入结构化数据库\n\n"
            "如果你需要**搭建新功能**（如多维表格同步、可视化面板等），"
            "请直接在 WorkBuddy 里告诉余柏阳，他会帮你开发部署。\n\n"
            "现在你可以试试直接发一段跟进记录给我，比如：\n"
            "「跟广州永津生物黄康约了下周一电话」"
        )
    elif cmd_type == "capability":
        await lark.reply_text(
            chat_id, message_id,
            "🤖 我是你的 **CRM 语音助手**，可以：\n"
            "• 📝 接收语音/文字跟进记录\n"
            "• 👤 自动识别客户姓名、公司、职位\n"
            "• 📅 解析日期（明天、下周一、下个月5号）\n"
            "• 🏷️ 分类意图（报价/签约/跟进/介绍）\n"
            "• 💾 存入数据库，随时可查\n\n"
            "直接发文字或语音就行，我会自动处理。"
        )
    elif cmd_type == "help":
        await lark.reply_text(
            chat_id, message_id,
            "📋 使用方式：\n\n"
            "**发文字**：直接描述跟进情况\n"
            "示例：「今天拜访了北京生物王博士，他对NK细胞培养基很感兴趣，报价已发」\n\n"
            "**发语音**：长按说话，我会自动转文字并提取\n\n"
            "我会自动识别：客户名 → 公司 → 意图 → 下一步 → 日期\n\n"
            "有问题？在 WorkBuddy 里找余柏阳 👨‍💻"
        )
    elif cmd_type == "query":
        await lark.reply_text(
            chat_id, message_id,
            "🔍 查询功能还在开发中，暂时请到 WorkBuddy 让余柏阳帮你查。"
        )
    else:
        await lark.reply_text(chat_id, message_id, "🤔 没太理解你的意思，试试直接发一段跟进记录？")


async def process_text_input(chat_id: str, message_id: str, sender_id: str, text: str):
    """统一处理文字输入（直接文字 或 语音转文字后的结果）"""
    try:
        # AI 提取结构化信息
        await lark.reply_text(chat_id, message_id, f"🔄 正在理解内容...\n{text[:100]}")
        extracted = await extract_structured(text)
        print(f"  Extracted: contact={extracted.contact}, content={extracted.content}")

        # 写入数据库
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

        # 回复确认卡片
        card = build_confirmation_card(extracted, contact, text)
        await lark.send_card(chat_id, message_id, card)

    except Exception as e:
        print(f"  ❌ Error processing input: {e}")
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
