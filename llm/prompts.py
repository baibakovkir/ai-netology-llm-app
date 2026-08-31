from typing import Dict, List


def build_messages(system_prompt: str, message: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

