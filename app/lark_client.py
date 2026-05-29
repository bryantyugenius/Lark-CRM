import httpx
import time
import os
import json
from typing import Optional


LARK_API_BASE = "https://open.larksuite.com/open-apis"


class LarkClient:
    def __init__(self):
        self.app_id = os.getenv("LARK_APP_ID", "")
        self.app_secret = os.getenv("LARK_APP_SECRET", "")
        self._access_token = None
        self._token_expires_at = 0

    async def get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at:
            return self._access_token

        url = f"{LARK_API_BASE}/auth/v3/tenant_access_token/internal"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            })
            data = resp.json()
            if data.get("code") == 0:
                self._access_token = data["tenant_access_token"]
                self._token_expires_at = now + data.get("expire", 7200) - 60
                return self._access_token
            else:
                raise Exception(f"Lark token error: {data}")

    async def get_message_content(self, message_id: str) -> dict:
        token = await self.get_access_token()
        url = f"{LARK_API_BASE}/im/v1/messages/{message_id}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={"partition_type": "1"},
            )
            return resp.json()

    async def download_voice(self, file_key: str) -> bytes:
        token = await self.get_access_token()
        url = f"{LARK_API_BASE}/im/v1/audio/{file_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.content

    async def reply_text(self, chat_id: str, msg_id: str, content: str):
        token = await self.get_access_token()
        url = f"{LARK_API_BASE}/im/v1/messages"
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={"receive_id_type": "chat_id"},
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": content}),
                    "reply": {"message_id": msg_id},
                },
            )

    async def send_card(self, chat_id: str, msg_id: str, card: dict):
        token = await self.get_access_token()
        url = f"{LARK_API_BASE}/im/v1/messages"
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={"receive_id_type": "chat_id"},
                json={
                    "receive_id": chat_id,
                    "msg_type": "interactive",
                    "content": json.dumps(card, ensure_ascii=False),
                    "reply": {"message_id": msg_id},
                },
            )
