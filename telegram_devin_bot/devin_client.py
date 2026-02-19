import aiohttp
from config import DEVIN_API_KEY, DEVIN_API_BASE


class DevinClient:
    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {DEVIN_API_KEY}",
            "Content-Type": "application/json",
        }

    async def create_session(self, prompt: str) -> dict:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(
                f"{DEVIN_API_BASE}/sessions",
                json={"prompt": prompt},
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def send_message(self, session_id: str, message: str) -> dict:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(
                f"{DEVIN_API_BASE}/sessions/{session_id}/message",
                json={"message": message},
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def get_session(self, session_id: str) -> dict:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(
                f"{DEVIN_API_BASE}/sessions/{session_id}",
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def list_sessions(self, limit: int = 10) -> dict:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(
                f"{DEVIN_API_BASE}/sessions",
                params={"limit": limit},
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
