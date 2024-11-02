from evaluate import eval_majority_vote
from model import OllamaModel
from utils import LogLevel

import json
import logging
from pathlib import Path
import time
from typing import TypedDict


class Question(TypedDict):
    """Type definition for a benchmark question."""

    question_id: int
    prompt: str
    answer: str


class EvaluationData(TypedDict):
    """Type definition for the complete benchmark dataset."""

    eval_data: list[Question]


class BenchmarkLogger:
    """Handle logging setup and management for benchmarking."""

    def __init__(self, log_level: LogLevel, silence_http: bool = True):
        self.log_dir = Path("logs")
        self.log_level = log_level.value
        self.silence_http = silence_http
        self._setup_log_directory()
        self.logger = self._initialize_logger()

    def _setup_log_directory(self) -> None:
        """Create logs directory if it doesn't exist."""

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_filename(self) -> str:
        """Generate timestamp-based log filename."""

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        return f"{timestamp}_bench.log"

    def _initialize_logger(self) -> logging.Logger:
        """Initialize and configure logger."""

        logging.basicConfig(
            filename=self.log_dir / self._get_log_filename(),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=self.log_level,
        )

        if self.silence_http:
            self._silence_http_loggers()

        return logging.getLogger(__name__)

    def _silence_http_loggers(self) -> None:
        """Silence HTTP-related logging."""

        for logger_name in ["httpx", "httpcore"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


class Benchmark:
    """Handle benchmark execution and evaluation."""

    def __init__(
        self,
        model: OllamaModel,
        benchmark_path: str,
        num_responses: int,
        logger: logging.Logger,
    ):
        self.model = model
        self.benchmark_path = Path(benchmark_path)
        self.num_responses = num_responses
        self.logger = logger
        self.benchmark_data = self._load_benchmark()

    def _load_benchmark(self) -> list[Question]:
        """
        Load benchmark questions from JSON file.

        Returns:
            List of benchmark questions

        Raises:
            FileNotFoundError: If benchmark file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """

        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")

        try:
            with open(self.benchmark_path, "r") as file:
                data = json.load(file)
                return data["eval_data"]
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse benchmark file {self.benchmark_path}: {str(e)}",
                e.doc,
                e.pos,
            )

    def run(self) -> float:
        """Run the complete benchmark and return final score."""

        total_score = sum(
            self._process_question(question) for question in self.benchmark_data
        )

        final_score = (total_score / len(self.benchmark_data)) * 100
        self.logger.info(f"Final Score: {final_score:.1f}%")
        print(f"Final Score: {final_score:.1f}%")

        return final_score

    def _process_question(self, question: Question) -> int:
        """Process a single question and return its score."""

        responses = self._collect_responses(question)

        return self._evaluate_responses(responses, question)

    def _collect_responses(self, question: Question) -> list[str]:
        """Collect multiple responses for a single question."""

        self.logger.info(f"Testing Question: {question['question_id']}")

        responses = []
        for r in range(self.num_responses):
            response = self.model.predict(question["prompt"])
            responses.append(response)
            self.logger.debug(f"Response {r + 1}:\n{response}")

        return responses

    def _evaluate_responses(self, responses: list[str], question: Question) -> int:
        """Evaluate responses and return score."""
        result, answers = eval_majority_vote(responses, question["answer"])

        self.logger.info(f"Majority Vote: {'PASS' if result else 'FAIL'}")
        self.logger.info(f"Correct Answer: {question['answer']}")
        self.logger.debug(f"Answers: {answers}")

        return 1 if result else 0
