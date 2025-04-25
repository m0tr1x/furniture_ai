from typing import Dict, Any


class DialogManager:
    def __init__(self):
        self.context: Dict[str, Any] = {}

    def update_context(self, user_id: int, key: str, value: Any):
        if user_id not in self.context:
            self.context[user_id] = {}
        self.context[user_id][key] = value

    def get_context(self, user_id: int, key: str) -> Any:
        return self.context.get(user_id, {}).get(key)