from benchmark import Benchmark, BenchmarkLogger
from model import OllamaModel
from utils import ArgumentParser
import sys


if __name__ == "__main__":
    try:
        # Parse arguments and setup
        args = ArgumentParser.parse_args()
        benchmark_logger = BenchmarkLogger(args.log_level, args.silence_http)
        logger = benchmark_logger.logger

        # Initialize model
        model = OllamaModel(
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )

        # Log initial information
        logger.info(f"Testing model: {model.config.model_name}")
        logger.debug(f"Arguments: {args}")

        # Setup and run benchmark
        benchmarker = Benchmark(
            model=model,
            benchmark_path=args.benchmark_path,
            num_responses=args.num_responses,
            logger=logger,
        )

        # Run benchmark
        benchmarker.run()

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
    else:
        logger.info("Benchmark completed successfully")
        sys.exit(0)
