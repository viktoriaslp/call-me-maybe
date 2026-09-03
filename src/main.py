"""Main application flow for generating function-calling results."""
import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.generator import get_output_from_prompt
from src.models import OutputResult
from src.parser import get_functions_list, get_prompts_list
from src.utils import parse_args, save_outputs


def main() -> None:
    """Run the full generation pipeline."""

    args = parse_args()
    try:
        function_definitions = get_functions_list(args.functions_definition)
        input_prompts = get_prompts_list(args.input)
    except (
        FileNotFoundError, PermissionError, UnicodeError, ValueError
    ) as error:
        print(error)
        return

    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)

    except Exception as error:
        print(f"Could not load LLM model: {error}")
        return

    results = []

    for input_prompt in input_prompts:
        raw_output = get_output_from_prompt(
            model,
            function_definitions,
            input_prompt.prompt,
            vocab_size,
            args.visualize,
        )
        results.append(OutputResult.model_validate_json(raw_output))

    save_outputs(args.output, results)


if __name__ == "__main__":
    main()
