import json
import pytest
from pathlib import Path


from src.models import (
    InputPrompt,
    FunctionDefinition,
    FieldDefinition,
)

from src.parser import (
    get_prompts_list,
    get_functions_list,
)


def test_get_prompts_list(tmp_path: Path) -> None:
    test_data = [
        {
            "prompt": "What is the sum of 2 and 3?"
        },
        {
            "prompt": "What is the sum of 265 and 345?"
        },
    ]

    file_path = tmp_path / "test_prompt.json"
    with open(file_path, "w") as file:
        json.dump(test_data, file)

    result = get_prompts_list(str(file_path))

    assert len(result) == 2
    assert isinstance(result[0], InputPrompt)
    assert result[0].prompt == "What is the sum of 2 and 3?"
    assert result[1].prompt == "What is the sum of 265 and 345?"


def test_get_functions_list(tmp_path: Path) -> None:
    test_data = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum.",
            "parameters": {
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
            },
            "returns": {
                "type": "number"
            }
        },
        {
            "name": "fn_greet",
            "description": "Generate a greeting message for a person by name.",
            "parameters": {
                "name": {
                    "type": "string"
                }
            },
            "returns": {
                "type": "string"
            }
        },
    ]

    file_path = tmp_path / "test_functions.json"
    with open(file_path, "w") as file:
        json.dump(test_data, file)

    result = get_functions_list(str(file_path))

    assert len(result) == 2
    assert isinstance(result[0], FunctionDefinition)
    assert result[0].name == "fn_add_numbers"
    assert result[0].description == (
        "Add two numbers together and return their sum."
    )
    assert isinstance(result[1].returns, FieldDefinition)
    assert result[1].returns.type == "string"


def test_get_prompts_list_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_file.json"

    with pytest.raises(FileNotFoundError):
        get_prompts_list(str(file_path))


def test_get_prompts_list_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "broken_prompt.json"

    with open(file_path, "w") as file:
        file.write("[{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON"):
        get_prompts_list(str(file_path))


def test_get_prompts_list_missing_required_field(tmp_path: Path) -> None:
    test_data = [
        {
            "wrong_key": "What is the sum of 2 and 3?"
        }
    ]

    file_path = tmp_path / "invalid_prompt.json"

    with open(file_path, "w") as file:
        json.dump(test_data, file)

    with pytest.raises(ValueError, match="Invalid data schema"):
        get_prompts_list(str(file_path))


def test_get_functions_list_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_functions.json"

    with pytest.raises(FileNotFoundError):
        get_functions_list(str(file_path))


def test_get_functions_list_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "broken_functions.json"

    with open(file_path, "w") as file:
        file.write("[{ invalid json }")

    with pytest.raises(ValueError, match="Invalid JSON"):
        get_functions_list(str(file_path))


def test_get_functions_list_missing_required_field(tmp_path: Path) -> None:
    test_data = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            }
        }
    ]

    file_path = tmp_path / "invalid_functions.json"

    with open(file_path, "w") as file:
        json.dump(test_data, file)

    with pytest.raises(ValueError, match="Invalid data schema"):
        get_functions_list(str(file_path))
