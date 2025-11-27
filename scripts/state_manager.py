import asyncio
import os
import json
import logging
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """Manages resume state with efficient saving on shutdown only."""

    def __init__(self, resume_file: str, collection: str):
        self.resume_file = resume_file
        self.collection = collection
        self.state = {}
        self.lock = asyncio.Lock()

    async def load_state(self) -> dict[str, Any]:
        if not self.resume_file or not os.path.exists(self.resume_file):
            return {}
        try:
            with open(self.resume_file, "r") as f:
                all_state = json.load(f)
                return all_state.get(self.collection, {})
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load resume state, starting fresh")
            return {}

    async def update_state(self, range_key: str, state_data: dict[str, Any]) -> None:
        async with self.lock:
            self.state[range_key] = state_data

    async def save_state(self) -> None:
        if not self.resume_file:
            return
        async with self.lock:
            all_state = {}
            if os.path.exists(self.resume_file):
                try:
                    with open(self.resume_file, "r") as f:
                        all_state = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            all_state[self.collection] = self.state.copy()
            temp_file = f"{self.resume_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(all_state, f, indent=2)
            os.replace(temp_file, self.resume_file)

    async def start(self) -> None:
        self.state = await self.load_state()

    async def stop(self) -> None:
        await self.save_state()

    def get_state(self, range_key: str) -> dict[str, Any]:
        return self.state.get(range_key, {})

def build_state_update(
        total_posted: int,
        next_page_url: str | None,
        current_item_offset: int,
        total_items_on_page: int,
        current_page_url: str | None = None,
        completed: bool = False
    ) -> dict[str, Any]:

    if completed:
        return {
            "completed": True,
            "next_page_url": None,
            "item_offset": 0,
            "items_posted": total_posted,
            "completed_at": datetime.now().isoformat()
        }
    elif current_item_offset >= total_items_on_page:
        return {
            "completed": False,
            "next_page_url": next_page_url,
            "item_offset": 0,
            "items_posted": total_posted,
            "updated_at": datetime.now().isoformat()
        }
    else:
        return {
            "completed": False,
            "next_page_url": current_page_url or next_page_url,
            "item_offset": current_item_offset,
            "items_posted": total_posted,
            "updated_at": datetime.now().isoformat()
        }