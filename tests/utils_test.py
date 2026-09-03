import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from src.main import save_outputs
from src.main import parse_args


# 1. Que save_outputs crea carpetas si no existen.
# 2. Que crea el archivo JSON.
# 3. Que guarda los objetos Pydantic como diccionarios correctos.

class FakeOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


def test_save_outputs_creates_json_file(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "output" / "result.json"

    output_list = [
        FakeOutput(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    ]

    save_outputs(str(output_path), output_list)

    assert output_path.exists()
    with open(output_path) as file:
        saved_data = json.load(file)
    assert saved_data == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_save_outputs_with_empty_list(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "empty.json"

    save_outputs(str(output_path), [])

    with open(output_path) as file:
        saved_data = json.load(file)

    assert saved_data == []


def test_parse_args_uses_default_paths(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["program"]
    )

    args = parse_args()

    assert args.functions_definition == "data/input/functions_definition.json"
    assert args.input == "data/input/function_calling_tests.json"
    assert args.output == "data/output/function_calling_results.json"
