import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class Serializable(Protocol):
    def to_dict(self) -> dict: ...


def write_json(obj: Serializable, path: Path) -> None:
    with path.open("w") as fp:
        json.dump(obj.to_dict(), fp)


def write_jsonl(objects: Iterable[Serializable], path: Path) -> None:
    with path.open("w") as fp:
        lines = [json.dumps(obj.to_dict()) + "\n" for obj in objects]
        fp.writelines(lines)
