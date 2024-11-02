import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LogLevel(str, Enum):
    """Enum for valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @staticmethod
    def from_string(level_str: str) -> "LogLevel":
        """Convert string to LogLevel enum."""
        try:
            return LogLevel[level_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level: {level_str}")


@dataclass
class BenchmarkArguments:
    """Configuration for benchmark execution."""

    model_name: str
    benchmark_path: Path
    log_level: LogLevel
    silence_http: bool
    num_responses: int
    temperature: float
    top_p: float
    max_tokens: int
    max_retries: int


class ArgumentParser:
    """Handles command line argument parsing."""

    @staticmethod
    def _parse_log_level(level_str: str) -> LogLevel:
        """Convert string argument to LogLevel enum."""
        try:
            return LogLevel.from_string(level_str)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            description="Benchmark LLM models using Ollama"
        )

        parser.add_argument(
            "--model_name",
            type=str,
            default="llama3.1",
            help="The model to benchmark. Must be available in local Ollama.",
        )

        parser.add_argument(
            "--benchmark_path",
            type=Path,
            default=Path("benchmarks/simple_bench_public.json"),
            help="Path to benchmark question set.",
        )

        parser.add_argument(
            "--log_level",
            type=ArgumentParser._parse_log_level,
            default=LogLevel.DEBUG,
            choices=list(LogLevel),
            help="Set logging verbosity level.",
        )

        parser.add_argument(
            "--silence_http",
            type=lambda x: str(x).lower() == "true",
            default=True,
            help="Silence HTTP-related logs.",
        )

        parser.add_argument(
            "--num_responses",
            type=int,
            default=5,
            help="Number of responses for majority vote.",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Temperature for generation (0.0-1.0).",
        )

        parser.add_argument(
            "--top_p",
            type=float,
            default=0.95,
            help="Top-p sampling parameter (0.0-1.0).",
        )

        parser.add_argument(
            "--max_tokens",
            type=int,
            default=2048,
            help="Maximum tokens to generate.",
        )

        parser.add_argument(
            "--max_retries",
            type=int,
            default=3,
            help="Maximum retry attempts on failure.",
        )

        return parser

    @classmethod
    def parse_args(cls) -> BenchmarkArguments:
        """
        Parse command line arguments into BenchmarkArguments.

        Returns:
            Parsed arguments as BenchmarkArguments dataclass
        """
        parser = cls.get_parser()
        args = parser.parse_args()

        return BenchmarkArguments(
            model_name=args.model_name,
            benchmark_path=args.benchmark_path,
            log_level=args.log_level,
            silence_http=args.silence_http,
            num_responses=args.num_responses,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
