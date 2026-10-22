
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import re
from flask_cors import CORS

from services.evidence import retrieve_evidence
from services.verification import verify_claim, model_status
from services.correction import (
    suggest_correction,
    validate_correction,
    build_corrected_paragraph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")

app = Flask(__name__)

# ============================================================
# CORS CONFIGURATION
# Allow requests from the deployed Vercel frontend.
# ============================================================

FRONTEND_ORIGIN = "https://llm-detection-main-frontend.vercel.app"

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [FRONTEND_ORIGIN],
            "methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "max_age": 600,
        }
    },
    always_send=True,
    vary_header=True,
)

RECENT_HISTORY = deque(maxlen=20)

# ============================================================
# 30 CURATED CLAIMS: 15 SUPPORTED + 15 CONTRADICTED
# ============================================================

CURATED_FACTS = [
    {
        "claims": ["The capital of Australia is Canberra."],
        "status": "Supported",
        "evidence": "Canberra is the capital city of Australia.",
        "correction": "",
        "domain": "Geography",
    },
    {
        "claims": ["The Earth revolves around the Sun."],
        "status": "Supported",
        "evidence": "Earth orbits the Sun once approximately every 365.25 days.",
        "correction": "",
        "domain": "Astronomy",
    },
    {
        "claims": ["IPv4 addresses are 32 bits long."],
        "status": "Supported",
        "evidence": "An IPv4 address contains 32 bits, commonly written as four decimal octets.",
        "correction": "",
        "domain": "Computer Networks",
    },
    {
        "claims": ["The human heart has four chambers."],
        "status": "Supported",
        "evidence": "The human heart has four chambers: two atria and two ventricles.",
        "correction": "",
        "domain": "Biology",
    },
    {
        "claims": [
            "The chemical formula of water is H2O.",
            "The chemical formula of water is H₂O.",
        ],
        "status": "Supported",
        "evidence": "A water molecule contains two hydrogen atoms and one oxygen atom, giving the formula H2O.",
        "correction": "",
        "domain": "Chemistry",
    },
    {
        "claims": ["The Earth has one natural moon."],
        "status": "Supported",
        "evidence": "Earth has one known natural satellite, the Moon.",
        "correction": "",
        "domain": "Astronomy",
    },
    {
        "claims": ["Mars is known as the Red Planet."],
        "status": "Supported",
        "evidence": "Mars is commonly called the Red Planet because iron minerals in its surface material give it a reddish appearance.",
        "correction": "",
        "domain": "Astronomy",
    },
    {
        "claims": ["The Pacific Ocean is the largest ocean on Earth."],
        "status": "Supported",
        "evidence": "The Pacific Ocean is the largest of Earth's oceans by area.",
        "correction": "",
        "domain": "Geography",
    },
    {
        "claims": ["Plants use sunlight for photosynthesis."],
        "status": "Supported",
        "evidence": "Plants use light energy during photosynthesis to help convert carbon dioxide and water into sugars.",
        "correction": "",
        "domain": "Biology",
    },
    {
        "claims": ["DNA usually has a double helix structure."],
        "status": "Supported",
        "evidence": "The familiar structure of DNA is a double helix formed by two complementary strands.",
        "correction": "",
        "domain": "Biology",
    },
    {
        "claims": ["Light travels faster than sound in air."],
        "status": "Supported",
        "evidence": "Light travels far faster than sound in air. Light in a vacuum travels at about 300,000 kilometres per second.",
        "correction": "",
        "domain": "Physics",
    },
    {
        "claims": ["Python is a high-level programming language."],
        "status": "Supported",
        "evidence": "Python is a high-level, general-purpose programming language.",
        "correction": "",
        "domain": "Programming",
    },
    {
        "claims": ["HTML is a markup language."],
        "status": "Supported",
        "evidence": "HTML stands for HyperText Markup Language and structures content on web pages.",
        "correction": "",
        "domain": "Web Development",
    },
    {
        "claims": ["HTTPS uses TLS to help encrypt data in transit."],
        "status": "Supported",
        "evidence": "HTTPS uses HTTP over a secure connection, normally protected by TLS, to help encrypt data in transit.",
        "correction": "",
        "domain": "Cybersecurity",
    },
    {
        "claims": ["The capital of India is New Delhi."],
        "status": "Supported",
        "evidence": "New Delhi is the capital of India.",
        "correction": "",
        "domain": "Indian Polity",
    },
    {
        "claims": ["The Sun revolves around the Earth."],
        "status": "Contradicted",
        "evidence": "Earth orbits the Sun. The Sun's apparent daily movement across the sky is primarily due to Earth's rotation.",
        "correction": "The Earth revolves around the Sun.",
        "domain": "Astronomy",
    },
    {
        "claims": ["IPv4 addresses are 128 bits long."],
        "status": "Contradicted",
        "evidence": "IPv4 addresses are 32 bits long. IPv6 addresses are 128 bits long.",
        "correction": "IPv4 addresses are 32 bits long.",
        "domain": "Computer Networks",
    },
    {
        "claims": ["The capital of Australia is Sydney."],
        "status": "Contradicted",
        "evidence": "Canberra is Australia's capital. Sydney is the capital of New South Wales.",
        "correction": "The capital of Australia is Canberra.",
        "domain": "Geography",
    },
    {
        "claims": ["Humans normally have three lungs."],
        "status": "Contradicted",
        "evidence": "Humans normally have two lungs, a right lung and a left lung.",
        "correction": "Humans normally have two lungs.",
        "domain": "Biology",
    },
    {
        "claims": ["The Earth is flat."],
        "status": "Contradicted",
        "evidence": "Earth is approximately an oblate spheroid: nearly spherical, with a slight bulge around the equator.",
        "correction": "The Earth is approximately an oblate spheroid.",
        "domain": "Science",
    },
    {
        "claims": ["The Moon produces its own visible light."],
        "status": "Contradicted",
        "evidence": "The Moon is visible mainly because it reflects sunlight; it does not produce its own visible light like a star.",
        "correction": "The Moon reflects sunlight and does not produce its own visible light.",
        "domain": "Astronomy",
    },
    {
        "claims": ["Mercury is the closest planet to Earth."],
        "status": "Contradicted",
        "evidence": "The claim is not generally true as a fixed statement: Earth's nearest planet varies with orbital positions, while Mercury is the closest planet to the Sun.",
        "correction": "Mercury is the closest planet to the Sun.",
        "domain": "Astronomy",
    },
    {
        "claims": ["The Pacific Ocean is the smallest ocean on Earth."],
        "status": "Contradicted",
        "evidence": "The Pacific Ocean is the largest ocean. The Arctic Ocean is the smallest of the world's five commonly recognised oceans.",
        "correction": "The Pacific Ocean is the largest ocean on Earth.",
        "domain": "Geography",
    },
    {
        "claims": [
            "The chemical formula of water is CO2.",
            "The chemical formula of water is CO₂.",
        ],
        "status": "Contradicted",
        "evidence": "Water has the chemical formula H2O. CO2 is the chemical formula for carbon dioxide.",
        "correction": "The chemical formula of water is H2O.",
        "domain": "Chemistry",
    },
    {
        "claims": ["Sound travels faster than light in air."],
        "status": "Contradicted",
        "evidence": "Light travels much faster than sound in air.",
        "correction": "Light travels faster than sound in air.",
        "domain": "Physics",
    },
    {
        "claims": ["HTML is a programming language."],
        "status": "Contradicted",
        "evidence": "HTML is a markup language used to structure web content, not a general-purpose programming language.",
        "correction": "HTML is a markup language.",
        "domain": "Web Development",
    },
    {
        "claims": ["HTTPS sends data in plaintext by default."],
        "status": "Contradicted",
        "evidence": "HTTPS is HTTP protected by TLS, which is designed to encrypt data in transit when the connection is properly established.",
        "correction": "HTTPS uses TLS to help encrypt data in transit.",
        "domain": "Cybersecurity",
    },
    {
        "claims": ["The capital of India is Mumbai."],
        "status": "Contradicted",
        "evidence": "New Delhi is India's capital. Mumbai is the capital of Maharashtra.",
        "correction": "The capital of India is New Delhi.",
        "domain": "Indian Polity",
    },
    {
        "claims": ["Saturn has no rings."],
        "status": "Contradicted",
        "evidence": "Saturn has a prominent ring system made mainly of particles of ice and rocky material.",
        "correction": "Saturn has a prominent ring system.",
        "domain": "Astronomy",
    },
    {
        "claims": ["Humans normally have 48 chromosomes in each body cell."],
        "status": "Contradicted",
        "evidence": "Most human body cells contain 46 chromosomes arranged in 23 pairs.",
        "correction": "Most human body cells contain 46 chromosomes arranged in 23 pairs.",
        "domain": "Biology",
    },
]


def normalize_claim(text):
    """Normalize case, punctuation, and whitespace for matching."""
    text = (text or "").lower().strip()
    text = text.replace("₂", "2").replace("₄", "4").replace("₆", "6")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lookup_curated_claim(sentence):
    """Find an exact match in the curated reference claims."""
    normalized = normalize_claim(sentence)

    for item in CURATED_FACTS:
        for known_claim in item["claims"]:
            if normalized == normalize_claim(known_claim):
                return item

    return None


def curated_result(idx, sentence, item):
    """Build a result using a curated reference claim."""
    return {
        "id": idx + 1,
        "sentence": sentence,
        "status": item["status"],
        "confidence": 0.99,
        "explanation": (
            f"Matched a predefined reference claim in the {item['domain']} domain. "
            f"{item['evidence']}"
        ),
        "evidence": item["evidence"],
        "source": "Built-in curated reference facts",
        "source_url": "",
        "correction": item["correction"],
        "correction_validated": item["status"] == "Contradicted",
        "domain": item["domain"],
    }


ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "e.g",
    "i.e", "u.s", "sq", "km", "approx", "fig", "al", "inc", "ltd",
    "corp", "dept", "no", "univ", "est", "st", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
}


def split_sentences(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []

    text = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", text)

    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b{re.escape(abbr)}\.",
            r"\g<0><DOT>",
            text
        )
        text = text.replace(".<DOT>", "<DOT>")

    text = re.sub(r"(?m)^(\d+)\.\s+", r"\1<NUMDOT> ", text)
    text = re.sub(r"(?<=[.!?])\s+(\d{1,3})\.\s+", r" \1<NUMDOT> ", text)

    raw_splits = re.split(r"(?:[\r\n]+|(?<=[.!?])\s+)", text)
    sentences = []

    for sentence in raw_splits:
        sentence = (
            sentence.replace("<DECIMAL>", ".")
            .replace("<DOT>", ".")
            .replace("<NUMDOT>", ".")
        )
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def _process_single_sentence(idx: int, sentence: str) -> dict:
    try:
        curated = lookup_curated_claim(sentence)
        if curated:
            return curated_result(idx, sentence, curated)

        evidence_pack = retrieve_evidence(sentence)
        verification = verify_claim(sentence, evidence_pack)
        status = verification.get("status", "Unverified")
        correction = ""
        correction_validated = False

        if status == "Contradicted":
            correction = suggest_correction(sentence, evidence_pack)
            if correction:
                correction_validated = validate_correction(
                    sentence, correction, evidence_pack
                )

        return {
            "id": idx + 1,
            "sentence": sentence,
            "status": status,
            "confidence": verification.get("confidence", 0.0),
            "explanation": verification.get("explanation", ""),
            "evidence": evidence_pack.get("evidence", ""),
            "source": evidence_pack.get("source", ""),
            "source_url": evidence_pack.get("source_url", ""),
            "correction": correction if correction_validated else "",
            "correction_validated": bool(correction_validated),
        }

    except Exception as exc:
        logger.exception(
            "Analysis failed for sentence: %s. Error: %s", sentence, exc
        )
        return {
            "id": idx + 1,
            "sentence": sentence,
            "status": "Unverified",
            "confidence": 0.0,
            "explanation": "Verification failed safely; no reliable conclusion was produced.",
            "evidence": "",
            "source": "",
            "source_url": "",
            "correction": "",
            "correction_validated": False,
        }


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "message": "LLM Detection Backend is running",
        "health": "/api/health",
        "analyze": "/api/analyze",
        "chat": "/api/chat"
    })


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "LLM Hallucination Detection and Correction Framework",
        "verification_model": model_status(),
        "curated_claims": len(CURATED_FACTS),
        "time": datetime.now(timezone.utc).isoformat(),
    })


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or request.form or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please paste text to analyze."}), 400

    if len(text) > 10000:
        return jsonify({
            "error": "Please keep input under 10,000 characters to prevent excessive latency."
        }), 413

    sentences = split_sentences(text)

    if not sentences:
        return jsonify({"error": "No valid sentences found in input."}), 400

    if len(sentences) > 30:
        return jsonify({
            "error": "Maximum 30 sentences per analysis allowed."
        }), 400

    results = [None] * len(sentences)
    max_workers = min(6, len(sentences))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_sentence, i, sentence): i
            for i, sentence in enumerate(sentences)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(
                    "Unexpected worker error on sentence %d: %s", idx, exc
                )
                results[idx] = {
                    "id": idx + 1,
                    "sentence": sentences[idx],
                    "status": "Unverified",
                    "confidence": 0.0,
                    "explanation": "Worker execution error.",
                    "evidence": "",
                    "source": "",
                    "source_url": "",
                    "correction": "",
                    "correction_validated": False,
                }

    counts = {
        "Supported": sum(
            1 for result in results
            if result and result["status"] == "Supported"
        ),
        "Contradicted": sum(
            1 for result in results
            if result and result["status"] == "Contradicted"
        ),
        "Unverified": sum(
            1 for result in results
            if result and result["status"] == "Unverified"
        ),
    }

    corrected_paragraph = build_corrected_paragraph(results)

    response_payload = {
        "results": results,
        "analysis": results,
        "counts": counts,
        "summary": counts,
        "corrected_paragraph": corrected_paragraph,
        "total_sentences": len(results),
        "supported_count": counts["Supported"],
        "contradicted_count": counts["Contradicted"],
        "unverified_count": counts["Unverified"],
    }

    preview = text[:80] + ("…" if len(text) > 80 else "")
    RECENT_HISTORY.appendleft({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preview": preview,
        "total_sentences": len(results),
        "counts": counts,
        "corrected_paragraph": corrected_paragraph,
    })

    return jsonify(response_payload)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({
            "reply": "Ask a factual question or statement to verify."
        })

    curated = lookup_curated_claim(question)
    if curated:
        explanation = curated["evidence"]
        reply = f"Claim status: {curated['status']}. {explanation}"

        if curated["correction"]:
            reply += f" Correct answer: {curated['correction']}"

        return jsonify({
            "status": curated["status"],
            "confidence": 0.99,
            "explanation": explanation,
            "evidence": curated["evidence"],
            "source": "Built-in curated reference facts",
            "correction": curated["correction"],
            "domain": curated["domain"],
            "reply": reply,
        })

    try:
        evidence = retrieve_evidence(question)

        if evidence.get("evidence"):
            verification = verify_claim(question, evidence)
            source = evidence.get("source") or "Reference source"
            explanation = verification.get("explanation") or ""
            reply = (
                f"Claim status: {verification.get('status')}. "
                f"{explanation} Source: {source}."
            )

            return jsonify({
                "status": verification.get("status", "Unverified"),
                "confidence": verification.get("confidence", 0.0),
                "explanation": explanation,
                "source": source,
                "reply": reply,
            })

    except Exception:
        logger.exception("Chat verification failed")

    return jsonify({
        "status": "Unverified",
        "confidence": 0.0,
        "reply": (
            "No definitive source evidence was found for that query. "
            "Please try a more specific factual claim."
        ),
    })


@app.get("/api/history")
def history():
    return jsonify({"history": list(RECENT_HISTORY)})


@app.post("/api/history/clear")
@app.delete("/api/history")
def clear_history():
    RECENT_HISTORY.clear()
    return jsonify({
        "ok": True,
        "message": "History cleared successfully."
    })


if __name__ == "__main__":
    # Local development only. Render should use its configured start command.
    app.run(host="127.0.0.1", port=5000, debug=False)
