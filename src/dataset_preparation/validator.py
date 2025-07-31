import re
from typing import Dict, List

class QAValidator:
    MIN_QUESTION_LEN = 10
    MIN_ANSWER_LEN = 5
    
    @classmethod
    def validate_single(cls, qa: Dict) -> Dict:
        errors = []
        
        # Key checks
        for key in ['question', 'answer', 'context']:
            if key not in qa:
                errors.append(f"Missing key: {key}")
                
        # Content checks
        if 'question' in qa:
            if len(qa['question']) < cls.MIN_QUESTION_LEN:
                errors.append(f"Question too short (<{cls.MIN_QUESTION_LEN} chars)")
            if not qa['question'].endswith('?'):
                errors.append("Question should end with '?'")
                
        if 'answer' in qa:
            if len(qa['answer']) < cls.MIN_ANSWER_LEN:
                errors.append(f"Answer too short (<{cls.MIN_ANSWER_LEN} chars)")
            if qa['answer'].lower() == 'unknown':
                errors.append("Answer is 'Unknown'")
            
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "source": qa.get('source')
        }
    
    @classmethod
    def validate_batch(cls, qa_list: List[Dict]) -> Dict:
        results = [cls.validate_single(qa) for qa in qa_list]
        valid_count = sum(1 for r in results if r['valid'])
        
        return {
            "valid_pairs": valid_count,
            "invalid_pairs": len(qa_list) - valid_count,
            "error_breakdown": {
                err: sum(1 for r in results if err in r['errors'])
                for r in results for err in r['errors']
            }
        }