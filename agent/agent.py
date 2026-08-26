"""Agent class quản lý session/conversation memory."""
from agent.loop import run_agent_loop
from config.prompts import SYSTEM_PROMPT
from tools.dataset_manager import DatasetManager


class Agent:
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> dict:
        """Gửi câu hỏi mới, giữ nguyên lịch sử hội thoại cho câu hỏi tiếp theo."""
        self.messages.append({"role": "user", "content": question})
        result = run_agent_loop(self.messages, self.dataset_manager)
        return result

    def reset(self):
        """Xóa lịch sử hội thoại, bắt đầu session mới."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
