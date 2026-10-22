
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
