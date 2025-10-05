from typing import List, Dict
from transformers import pipeline

class HallucinationDetector:
    """
    Simple Hallucination Detector using NLI models (e.g., facebook/bart-large-mnli).
    - Extract claims (basic sentence split)
    - Verify claims against evidence docs using NLI
    - Compute hallucination rate
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.nli_pipeline = pipeline("text-classification", model=model_name)

    def extract_claims(self, text: str) -> List[str]:
        """Naive sentence-based claim extraction"""
        return [c.strip() for c in text.split(".") if c.strip()]

    def verify_claim(self, claim: str, evidence: str) -> bool:
        """Check if a claim is supported by given evidence"""
        result = self.nli_pipeline(f"{claim} </s></s> {evidence}")
        return result[0]["label"].lower() == "entailment"

    def verify_claim_multi(self, claim: str, evidence_docs: List[str]) -> bool:
        """A claim is supported if any evidence doc entails it"""
        return any(self.verify_claim(claim, e) for e in evidence_docs)

    def compute_hallucination_rate(self, text: str, evidence_docs: List[str]) -> Dict[str, float]:
        claims = self.extract_claims(text)
        if not claims:
            return {"total_claims": 0, "unsupported_claims": 0, "hallucination_rate": 0.0}

        unsupported = sum(not self.verify_claim_multi(c, evidence_docs) for c in claims)
        return {
            "total_claims": len(claims),
            "unsupported_claims": unsupported,
            "hallucination_rate": unsupported / len(claims),
        }
