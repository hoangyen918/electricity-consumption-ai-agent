"""Agent class quản lý session/conversation memory."""
import time
from agent.loop import run_agent_loop
from config.prompts import SYSTEM_PROMPT
from config.settings import SETTINGS
from tools.dataset_manager import DatasetManager
from services.logger import new_request_id, log_interaction
from services.cost_tracker import calculate_chat_cost, calculate_embedding_cost


class Agent:
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> dict:
        request_id = new_request_id()
        start_time = time.time()

        self.messages.append({"role": "user", "content": question})
        result = run_agent_loop(self.messages, self.dataset_manager)

        total_latency_ms = (time.time() - start_time) * 1000

        chat_cost = sum(
            calculate_chat_cost(SETTINGS.chat_model, u["input_tokens"], u["output_tokens"])
            for u in result["usage_per_call"]
        )
        embedding_cost = sum(
            calculate_embedding_cost(
                SETTINGS.embedding_model,
                t["result"]["data"].get("embedding_tokens", 0),
            )
            for t in result["trace"]
            if t["tool_name"] == "search_knowledge_base" and t["result"]["ok"]
        )
        total_cost = chat_cost + embedding_cost

        log_interaction(
            request_id=request_id,
            question=question,
            model=SETTINGS.chat_model,
            result=result,
            total_latency_ms=total_latency_ms,
            total_cost_usd=total_cost,
        )

        result["request_id"] = request_id
        result["estimated_cost_usd"] = round(total_cost, 8)
        result["latency_ms"] = round(total_latency_ms, 1)
        return result

    def reset(self):
        """Xóa lịch sử hội thoại, bắt đầu session mới."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
