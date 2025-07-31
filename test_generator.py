import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.dataset_preparation.qa_generator import EVQAGenerator

# Test GPT-2
print("Testing GPT-2 generation...")
gpt_generator = EVQAGenerator(model_name="gpt2-medium", use_gpt=True)
test_text = "CCS and CHAdeMO connectors support fast charging up to 350 kW."
qa_pairs = gpt_generator._gpt_generate_qa(test_text, num_questions=2)
print("GPT-2 Output:", qa_pairs)

# Test rule-based
print("\nTesting rule-based generation...")
rule_generator = EVQAGenerator(use_gpt=False)
test_text = "This station has CCS (50 kW) and CHAdeMO (100 kW)."
qa_pairs = rule_generator._rule_based_pdf_qa(test_text)
print("Rule-based Output:", qa_pairs)