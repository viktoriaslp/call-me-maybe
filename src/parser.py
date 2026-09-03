import json

from pydantic import BaseModel, ValidationError

from src.models import InputPrompt, FunctionDefinition


def load_json_as_models(
        file_name: str,
        model_class: type[BaseModel]
) -> list[BaseModel]:
    """Load a JSON file and validate each item as a Pydantic model.

    Args:
        file_name: Path to the JSON file.
        model_class: Pydantic model class used to validate each item.

    Returns:
        A list of validated Pydantic model instances.

    Raises:
        ValueError: If the JSON is invalid or the data schema is incorrect.
    """

    try:
        with open(file_name) as file:
            raw_data = json.load(file)

        return [
            model_class.model_validate(item)
            for item in raw_data
        ]

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {file_name}: "
            f"{error.msg} at line {error.lineno}, column {error.colno}"
        ) from error

    except ValidationError as error:
        raise ValueError(
            f"Invalid data schema in {file_name}"
        ) from error


def get_prompts_list(file_name: str) -> list[InputPrompt]:
    """Load input prompt  from a JSON file."""
    return load_json_as_models(
        file_name,
        InputPrompt
    )


def get_functions_list(file_name: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file."""
    return load_json_as_models(
        file_name,
        FunctionDefinition
    )
