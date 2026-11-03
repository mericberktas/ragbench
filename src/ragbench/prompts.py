"""Prompts (Turkish). Kept in one place so the exact wording is part of the recorded method."""

GEN_SYSTEM_RAG = (
    "Sen bir soru-cevap asistanısın. Yalnızca verilen bağlam parçalarındaki bilgilere dayanarak soruyu Türkçe, "
    "kısa ve doğrudan yanıtla. Bağlamda cevap yoksa yalnızca \"Bilmiyorum.\" yaz. Bağlam dışından bilgi ekleme."
)
GEN_SYSTEM_NONE = (
    "Sen bir soru-cevap asistanısın. Soruyu kendi bilginle Türkçe, kısa ve doğrudan yanıtla. "
    "Emin değilsen yalnızca \"Bilmiyorum.\" yaz."
)


def generation_messages(question: str, context: str | None) -> list[dict]:
    if context is None:
        return [{"role": "system", "content": GEN_SYSTEM_NONE}, {"role": "user", "content": f"Soru: {question}"}]
    return [
        {"role": "system", "content": GEN_SYSTEM_RAG},
        {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"},
    ]


JUDGE_SYSTEM = (
    "Bir değerlendirme yardımcısısın. Sana bir soru, bir referans cevap ve bir aday cevap verilir. "
    "Aday cevabın referans cevapla anlamca uyuşup uyuşmadığına karar ver. Yalnızca referans cevaba göre "
    "değerlendir, kendi bilgini kullanma. Yazım biçimi, tarih biçimi ya da ek ayrıntı farkları önemsizdir; "
    "referansla çelişen ya da referanstaki temel bilgiyi vermeyen cevap yanlıştır. Aday \"Bilmiyorum\" "
    "diyorsa ya da cevap vermiyorsa \"abstain\" de. Çıktın yalnızca şu biçimde bir JSON olsun: "
    "{\"verdict\": \"correct\" | \"incorrect\" | \"abstain\", \"reason\": \"<en fazla 15 kelimelik bir cümle>\"}"
)


def judge_messages(question: str, reference: str, candidate: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Soru: {question}\n\nReferans cevap: {reference}\n\nAday cevap: {candidate}"},
    ]
