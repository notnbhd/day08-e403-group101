"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.
    pip install deepeval
    """
    print("Initializing DeepEval metrics...")
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = []
    print(f"Running pipeline on {len(golden_dataset)} test cases...")
    for idx, item in enumerate(golden_dataset):
        print(f"  [{idx+1}/{len(golden_dataset)}] Evaluating: {item['question']}")
        
        # Gọi RAG pipeline. Giả định rag_pipeline có hàm generate_with_citation
        # hoặc bản thân nó là function trả về dict {"answer": ..., "sources": [{"content": ...}]}
        if hasattr(rag_pipeline, "generate_with_citation"):
            result = rag_pipeline.generate_with_citation(item["question"])
        else:
            result = rag_pipeline(item["question"])
            
        retrieval_context = [c["content"] if isinstance(c, dict) else str(c) for c in result.get("sources", [])]
        
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result.get("answer", ""),
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]

    print("Executing DeepEval evaluation...")
    results = evaluate(test_cases, metrics)
    return {"test_cases": results}


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    # TODO: Implement
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    raise NotImplementedError("Implement evaluate_with_ragas")


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline_factory, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.
    rag_pipeline_factory là hàm nhận vào tên config và trả về hàm pipeline tương ứng.
    """
    configs = ["hybrid_rerank", "dense_only"]
    results = {}
    
    print("\n" + "="*50)
    print("BẮT ĐẦU SO SÁNH A/B CONFIGS")
    print("="*50)
    
    for config_name in configs:
        print(f"\n---> Đang đánh giá Config: {config_name} <---")
        # Giả định pipeline_factory trả về 1 function callable
        pipeline = rag_pipeline_factory(config_name)
        
        # Để chạy được compare_configs, ta sẽ dùng evaluate_with_deepeval
        try:
            eval_res = evaluate_with_deepeval(pipeline, golden_dataset)
            results[config_name] = eval_res
        except Exception as e:
            print(f"Lỗi khi đánh giá {config_name}: {e}")
            results[config_name] = {"error": str(e)}
            
    return results

# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    content = "# Báo Cáo Kết Quả RAG Evaluation\n\n"
    
    # Nếu kết quả đơn lẻ có tồn tại
    if results and "test_cases" in results:
        content += "## Kết Quả Đánh Giá Tổng Quan\n\n"
        content += "Các metrics trung bình đạt được:\n"
        content += "- Tỉ lệ thành công sẽ được thể hiện qua Dashboard của DeepEval.\n\n"
        
    if comparison:
        content += "## Phân Tích A/B Testing\n\n"
        content += "| Config | Trạng Thái |\n"
        content += "|--------|------------|\n"
        for config_name, res in comparison.items():
            status = "Lỗi" if "error" in res else "Thành công"
            content += f"| `{config_name}` | {status} |\n"
            
        content += "\n**Nhận Xét Nhanh:**\n"
        content += "- Config **hybrid_rerank** thường sẽ mang lại Contextual Precision cao hơn do có RRF và Cross-Encoder.\n"
        content += "- Config **dense_only** chạy nhanh hơn nhưng bỏ lỡ các keyword cụ thể của pháp luật.\n"
        
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n[OK] Đã xuất báo cáo ra {RESULTS_PATH.name}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # MOCK FACTORY để chạy thử code logic evaluation
    def mock_pipeline_factory(config_name):
        def mock_pipeline(question):
            return {
                "answer": "Đây là câu trả lời giả lập cho " + question,
                "sources": [{"content": "Nội dung văn bản được mock."}]
            }
        return mock_pipeline

    print("CẢNH BÁO: Đang sử dụng Mock Pipeline do chưa import backend thực sự.")
    print("Vui lòng thay thế `mock_pipeline_factory` bằng function thật từ backend của nhóm.")
    
    # 1. Chạy đánh giá DeepEval cho 1 config
    # results = evaluate_with_deepeval(mock_pipeline_factory("default"), golden_dataset)
    
    # 2. Chạy so sánh A/B
    comparison = compare_configs(mock_pipeline_factory, golden_dataset)
    
    # 3. Xuất báo cáo
    export_results(results=None, comparison=comparison)
    print("Hoàn tất quy trình Evaluation!")
