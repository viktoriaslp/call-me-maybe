import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoder import DecoderMachine, State
from src.models import FunctionDefinition
from src.utils import build_prompt


def get_output_from_prompt(
        llm_model: Small_LLM_Model,
        func_definitions: list[FunctionDefinition],
        prompt: str,
        vocab_size: int,
        visualize: bool
) -> str:
    """Generate a JSON function call using constrained decoding.

    At each decoding step, only tokens that keep the output consistent with
    the function schema are considered valid.
    """

    decoder = DecoderMachine(
        prompt=prompt,
        func_definitions=func_definitions,
        vocab_size=vocab_size,
    )
    decoder._model = llm_model

    formatted_prompt = build_prompt(prompt, func_definitions)
    input_ids: list[int] = decoder._model.encode(formatted_prompt).tolist()[0]

    while decoder.current_state != State.DONE:
        allowed_ids = decoder.allowed_tokens()

        if allowed_ids.size == 0:
            decoder.current_state = State.DONE
            break

        elif allowed_ids.size == 1:
            next_token_id = int(allowed_ids[0])
        else:
            logits = np.array(
                decoder._model.get_logits_from_input_ids(input_ids),
                dtype=np.float32
            )
            masked_logits = np.full_like(logits, -np.inf)
            masked_logits[allowed_ids] = logits[allowed_ids]
            next_token_id = int(np.argmax(masked_logits))
            del masked_logits
            del logits

        del allowed_ids
        input_ids.append(next_token_id)
        decoded_token = decoder._model.decode([next_token_id])
        if visualize:
            print(decoded_token, end="", flush=True)
        decoder.current_json += decoded_token
        decoder.advance_state_if_needed(decoded_token)

    return decoder.current_json
