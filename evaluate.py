import logging
import re

logger = logging.getLogger(__name__)


class AnswerExtractionError(ValueError):
    """Raised when answer extraction fails."""

    pass


def extract_single_answer(output: str) -> str:
    """
    Extract a single letter answer (A-F) from model output.

    Args:
        output: Raw model output containing "Final Answer: X"

    Returns:
        Uppercase letter answer (A-F)

    Raises:
        AnswerExtractionError: If no valid answer can be extracted
    """
    match = re.search(r"Final Answer:\s*([A-F])", output.strip(), re.IGNORECASE)
    if not match:
        raise AnswerExtractionError("No answer found in model output")

    return match.group(1).upper()


def extract_multiple_answers(outputs: list[str]) -> list[str]:
    """
    Extract answers from multiple model outputs.

    Args:
        outputs: List of raw model outputs

    Returns:
        List of successfully extracted answers
    """
    valid_answers = []
    for output in outputs:
        try:
            answer = extract_single_answer(output)
            valid_answers.append(answer)
        except AnswerExtractionError:
            logger.warning(f"Failed to extract answer from model output: {output}")

    return valid_answers


def calculate_majority_vote(answers: list[str], correct_answer: str) -> bool:
    """
    Determine if the correct answer is the majority among given answers.

    Args:
        answers: List of extracted answers
        correct_answer: The known correct answer

    Returns:
        True if correct answer is majority, False otherwise
    """
    if not answers:
        return False

    return answers.count(correct_answer) > len(answers) / 2


def eval_majority_vote(model_outputs: list[str], answer: str) -> tuple[bool, list[str]]:
    """
    Evaluate if the majority of model outputs match the correct answer.

    Args:
        model_outputs: List of raw model outputs to evaluate
        answer: The correct answer to compare against

    Returns:
        Tuple of (bool indicating majority correct, list of extracted answers)

    Raises:
        ValueError: If no valid answers could be extracted
    """
    extracted_answers = extract_multiple_answers(model_outputs)

    if not extracted_answers:
        raise ValueError("Failed to extract any valid answers from model outputs")

    is_majority = calculate_majority_vote(extracted_answers, answer)

    return is_majority, extracted_answers
