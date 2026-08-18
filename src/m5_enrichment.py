from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, create_llm_client


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _chat(messages: list[dict], max_tokens: int) -> str:
    """Call the configured OpenRouter model and return plain response text."""
    client = create_llm_client()
    if client is None:
        return ""
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json_object(value: str) -> dict:
    """Parse a JSON object, accepting common Markdown fenced responses."""
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _extractive_summary(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    statements = [
        statement.strip().rstrip(".?!")
        for statement in re.split(r"[.!?\n]+", text)
        if len(statement.strip()) > 10
    ]
    return [f"{statement}?" for statement in statements[:max(n_questions, 0)]]


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    categories = {
        "it": ("mật khẩu", "vpn", "malware", "bảo mật", "cntt"),
        "finance": ("lương", "chi phí", "tạm ứng", "thanh toán", "vnđ"),
        "hr": ("nhân viên", "nghỉ phép", "thử việc", "đào tạo"),
    }
    category = next(
        (name for name, keywords in categories.items() if any(word in lowered for word in keywords)),
        "policy",
    )
    return {"topic": "general", "entities": [], "category": category, "language": "vi"}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENROUTER_API_KEY:
        try:
            result = _chat([
                {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                {"role": "user", "content": text},
            ], max_tokens=150)
            if result:
                return result
        except Exception as error:
            print(f"  ⚠️  OpenRouter summarize failed: {error}")
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    if OPENROUTER_API_KEY:
        try:
            result = _chat([
                {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên một dòng."},
                {"role": "user", "content": text},
            ], max_tokens=200)
            questions = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                for line in result.splitlines()
                if line.strip()
            ]
            if questions:
                return questions[:n_questions]
        except Exception as error:
            print(f"  ⚠️  OpenRouter HyQA failed: {error}")
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENROUTER_API_KEY:
        try:
            context = _chat([
                {"role": "system", "content": "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về một câu."},
                {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
            ], max_tokens=80)
            if context:
                return f"{context}\n\n{text}"
        except Exception as error:
            print(f"  ⚠️  OpenRouter contextual failed: {error}")
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENROUTER_API_KEY:
        try:
            result = _chat([
                {"role": "system", "content": 'Trích xuất metadata và chỉ trả về JSON hợp lệ: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                {"role": "user", "content": text},
            ], max_tokens=150)
            metadata = _parse_json_object(result)
            if metadata:
                return metadata
        except Exception as error:
            print(f"  ⚠️  OpenRouter metadata failed: {error}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENROUTER_API_KEY:
        try:
            result = _chat([
                {"role": "system", "content": """Phân tích đoạn văn và chỉ trả về một JSON object hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ], max_tokens=400)
            parsed = _parse_json_object(result)
            if parsed:
                return parsed
        except Exception as error:
            print(f"  ⚠️  OpenRouter combined enrichment failed: {error}")

    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": f"Trích từ {source}." if source else "",
        "metadata": _fallback_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            # Original evidence wins on collisions: an LLM must never rewrite
            # source/version identifiers used to resolve policy conflicts.
            auto_metadata={**auto_meta, **chunk.get("metadata", {})},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
