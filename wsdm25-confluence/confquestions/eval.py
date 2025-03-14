import asyncio
import json
from pathlib import Path
from typing import List, Optional, Literal
from openai import OpenAI
from pydantic import TypeAdapter
from ragonite.rag import RAG

from evaluation.batch_processors import RAGBatchProcessor
from evaluation.contextualization.evaluator import RAGEvaluator
from evaluation.model import Conversation


def load_benchmark_data(input_file: Path):
    with open(input_file, "r") as f:
        data = json.load(f)
    return TypeAdapter(List[Conversation]).validate_python(data)


def run_benchmark(
    input_file: Path,
    rag: RAG,
    output_path: Path,
    languages: Optional[List[Literal["en", "de"]]] = None,
    use_rephraser: Optional[bool] = True,
    use_judgement_batching: bool = True,
):
    output_file = output_path / "benchmark-final-output.json"

    async def main():
        data = load_benchmark_data(input_file)
        print(f"✅ Loaded {len(data)} conversations from {input_file}")

        processor = RAGBatchProcessor(
            client=OpenAI(),
            input_data=data,
            rag=rag,
            output_path=output_path,
            use_rephraser=use_rephraser,
            languages=languages,
        )

        # Process batches. This can take a while
        processed_data = await processor.run_rag_batch_processing()
        print(f"✅ Batch processing completed. Processed {len(processed_data)} conversations.")
        with open(output_file, "w") as file:
            json.dump([conv.dict() for conv in processed_data], file, indent=4)
        print(f"🏁 Results saved to: {output_file}\n")


        print("Now starting the evaluation..")
        # Evaluate responses and retrieval
        if not use_judgement_batching:
            evaluator = RAGEvaluator(rag=rag)
            evaluated_data = await evaluator.evaluate_conversations(processed_data)
            print(f"✅ Evaluation completed for {len(evaluated_data)} conversations.")

            with open(output_file, "w") as file:
                json.dump([conv.dict() for conv in evaluated_data], file, indent=4)
            print(f"🏁 Results saved to: {output_file}\n")
        else:
            print("Doing judgment evaluation in batch mode")
            evaluator = RAGEvaluator(rag=rag, client=OpenAI(), output_path=output_path)
            print("FEEDING data into valuator:")
            print(processed_data)
            evaluated_data = await evaluator.evaluate_conversations(processed_data, use_batch_api=True)  # Data written internally via callback

    asyncio.run(main())


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(run_benchmark, as_positional=False)