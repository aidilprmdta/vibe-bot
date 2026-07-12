#!/usr/bin/env python3
"""
Multi-Agent Router
===================
1 "Project Manager / Router" agent menerima task dari user, mengklasifikasikan
task itu ke salah satu dari 5 role, lalu mendispatch ke model gratis yang
paling cocok untuk role tersebut (OpenRouter atau Google AI Studio).

Setup:
    pip install requests
    export OPENROUTER_API_KEY="sk-or-..."
    export GEMINI_API_KEY="AIza..."

Jalankan:
    python agent_router.py "Buatkan aku wireframe untuk halaman login"
    # atau tanpa argumen untuk mode interaktif:
    python agent_router.py
"""

import os
import sys
import re
import ast
import json
import time
import datetime
import sqlite3
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "agent_log.jsonl")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.db")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_code")

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Ekstensi file yang dianggap teks dan aman dibaca langsung sebagai konteks.
# File di luar daftar ini (gambar, PDF, biner, dll) belum didukung.
READABLE_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".csv",
    ".html", ".css", ".yaml", ".yml", ".sql", ".log", ".xml", ".ini", ".cfg",
}

# Batas karakter isi file yang diikutsertakan sebagai konteks, biar tidak
# boros token / gampang kena rate limit TPM.
MAX_FILE_CONTEXT_CHARS = 4000

# Berapa banyak pasangan (user, assistant) terakhir yang diingat per role.
# Dibatasi supaya nggak boros token & nggak gampang kena rate limit TPM.
MEMORY_MAX_TURNS = 10

# ---------------------------------------------------------------------------
# Konfigurasi role -> provider & model.
# `fallback` dipakai kalau model utama kena rate limit (HTTP 429) atau error.
# Daftar model gratis berubah dari waktu ke waktu — cek ulang di:
#   https://openrouter.ai/models (filter Price: Free)
#   https://aistudio.google.com
# ---------------------------------------------------------------------------
ROLE_CONFIG = {
    "project_manager": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback": ["openrouter/free"],
        "system": "Kamu adalah Project Manager yang membagi tugas dan membuat ringkasan rencana kerja yang jelas dan terstruktur.",
        "memory_enabled": True,
        "team_aware": True,
    },
    "designer": {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "fallback": ["gemini-3.1-flash-lite"],
        "system": (
            "Kamu adalah UI/UX Designer.\n\n"
            "UNTUK PERTANYAAN UMUM/STRATEGI (struktur, hierarki, rekomendasi komponen, saran): "
            "jawab dengan bahasa SEDERHANA, bullet point atau daftar bernomor, mudah dibaca orang "
            "non-teknis. JANGAN bikin diagram ASCII/wireframe teks yang rumit (kotak-kotak dengan "
            "garis |, +, -). Hindari kalimat panjang bertele-tele, langsung ke poin.\n\n"
            "UNTUK PERMINTAAN MOCKUP/VISUAL/TAMPILAN/PROTOTYPE (ada kata seperti 'mockup', 'tampilan', "
            "'visual', 'prototype', 'contoh halaman', 'desainkan'): buat FILE HTML LENGKAP dan siap "
            "dibuka langsung di browser, dalam SATU code block ```html ... ```, dengan ketentuan:\n"
            "- Sertakan <!DOCTYPE html> lengkap, styling inline di dalam <style> atau pakai Tailwind "
            "lewat <script src=\"https://cdn.tailwindcss.com\"></script>\n"
            "- Layout, warna, tipografi, dan komponen harus benar-benar terlihat visual (bukan kotak "
            "placeholder kosong) — buat senyata dan serapi mungkin seolah mockup produk sungguhan\n"
            "- Pastikan responsive dasar (enak dilihat di layar HP)\n"
            "- Beri penjelasan SINGKAT 1-2 kalimat di luar code block saja, JANGAN ulangi detail "
            "desain dalam bentuk teks panjang — biarkan kode HTML-nya yang jadi buktinya."
        ),
        "memory_enabled": True,
        "code_output_enabled": True,
    },
    "programmer": {
        "provider": "openrouter",
        "model": "qwen/qwen3-coder:free",
        "fallback": ["openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free"],
        "system": "Kamu adalah Software Engineer. Tulis kode yang bersih, benar, dan beri penjelasan singkat.",
        "memory_enabled": True,
        "self_check_enabled": True,
        "code_output_enabled": True,
    },
    "copywriter": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback": ["openrouter/free"],
        "system": "Kamu adalah Copywriter kreatif. Gunakan bahasa yang luwes, menarik, dan sesuai target audiens.",
        "memory_enabled": True,
    },
    "analyst": {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "fallback": ["gemini-3.1-flash-lite"],
        "system": "Kamu adalah Business/Data Analyst. Berikan analisis berbasis logika, angka, dan insight yang actionable.",
        "memory_enabled": True,
        "web_search_enabled": True,
    },
}

ROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
ROUTER_FALLBACK = ["openrouter/free"]

MAX_RETRIES = 3
# Retry lebih sedikit khusus untuk percobaan dengan search grounding, karena
# kegagalannya biasanya bukan rate limit sementara (lebih sering soal billing
# account yang belum aktif) -> lebih cepat pindah ke fallback tanpa search.
SEARCH_MAX_RETRIES = 1
BACKOFF_BASE = 2  # detik


# ---------------------------------------------------------------------------
# Provider callers
# ---------------------------------------------------------------------------
def call_openrouter(model: str, system: str, history: list) -> str:
    """history: list of {"role": "user"/"assistant", "content": str}, pesan terbaru di akhir."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY belum di-set.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + history,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError(f"OpenRouter rate limit untuk model {model}")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_gemini(model: str, system: str, history: list, enable_search: bool = False) -> str:
    """history: list of {"role": "user"/"assistant", "content": str}, pesan terbaru di akhir.
    enable_search=True mengaktifkan Google Search grounding bawaan Gemini, supaya
    jawaban bisa berbasis data terkini dari web, bukan cuma pengetahuan model."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY belum di-set.")

    # Gemini pakai role "model" bukan "assistant"
    gemini_contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in history
    ]
    url = GEMINI_URL_TMPL.format(model=model, key=GEMINI_API_KEY)
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": gemini_contents,
    }
    if enable_search:
        payload["tools"] = [{"google_search": {}}]

    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError(f"Gemini rate limit untuk model {model}")
    resp.raise_for_status()
    data = resp.json()
    candidate = data["candidates"][0]
    text = candidate["content"]["parts"][0]["text"]

    if enable_search:
        sources = _extract_gemini_sources(candidate)
        if sources:
            text += "\n\nSumber:\n" + "\n".join(sources)

    return text


def _extract_gemini_sources(candidate: dict) -> list:
    """Ambil daftar sumber (judul + link) dari groundingMetadata hasil Google Search
    grounding Gemini. Return list string siap ditampilkan, bisa kosong kalau
    Gemini tidak menyertakan grounding (misalnya kalau ternyata tidak perlu cari web)."""
    grounding = candidate.get("groundingMetadata", {})
    chunks = grounding.get("groundingChunks", [])
    sources = []
    for c in chunks:
        web = c.get("web", {})
        title = web.get("title")
        uri = web.get("uri")
        if title and uri:
            sources.append(f"- {title}: {uri}")
    return sources


class RateLimitError(Exception):
    pass


def call_with_fallback(provider: str, models: list, system: str, history: list, enable_search: bool = False) -> tuple:
    """Coba model utama, kalau rate limit / error, coba fallback satu-satu.
    Tiap model juga di-retry dengan exponential backoff sebelum pindah ke fallback berikutnya.
    enable_search hanya berlaku untuk provider gemini (diabaikan untuk openrouter).

    Kalau enable_search=True dan SEMUA percobaan dengan search gagal (misal karena
    Google Search grounding butuh billing account yang belum aktif), sebelum pindah
    ke model fallback berikutnya, dicoba SEKALI LAGI di model yang sama tapi TANPA
    search — jadi analyst tetap bisa jawab (tanpa data web terkini) daripada gagal total.
    Percobaan search-enabled dibatasi lebih sedikit retry (SEARCH_MAX_RETRIES) karena
    kegagalan grounding biasanya bukan rate limit sementara (lebih sering soal billing
    belum aktif), jadi retry panjang cuma buang waktu.

    Return: (hasil_teks, nama_model_yang_berhasil)"""
    last_error = None
    for model in models:
        retries = SEARCH_MAX_RETRIES if (provider == "gemini" and enable_search) else MAX_RETRIES
        for attempt in range(retries):
            try:
                if provider == "openrouter":
                    result = call_openrouter(model, system, history)
                else:
                    result = call_gemini(model, system, history, enable_search=enable_search)
                return result, model
            except RateLimitError as e:
                last_error = e
                if attempt < retries - 1:  # jangan sleep kalau ini percobaan terakhir
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [rate limit] {model}, retry dalam {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f"  [rate limit] {model}, tidak ada retry tersisa.", file=sys.stderr)
            except Exception as e:
                last_error = e
                print(f"  [error] {model}: {e}", file=sys.stderr)
                break  # error non-rate-limit -> langsung coba model fallback berikutnya

        # Fallback tanpa search: cuma relevan untuk gemini + enable_search=True
        if provider == "gemini" and enable_search:
            print(f"  [fallback] {model} gagal dengan search, coba tanpa search...", file=sys.stderr)
            try:
                result = call_gemini(model, system, history, enable_search=False)
                result += (
                    "\n\n_(Catatan: web search grounding tidak tersedia saat ini — kemungkinan "
                    "billing account belum aktif di project AI Studio, atau kuota grounding habis. "
                    "Jawaban di atas berdasarkan pengetahuan model, bukan pencarian web langsung.)_"
                )
                return result, model
            except RateLimitError as e:
                last_error = e
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Semua model gagal untuk provider={provider}. Error terakhir: {last_error}")


# ---------------------------------------------------------------------------
# Memory (percakapan multi-turn) — per role DAN per user, disimpan persisten
# di SQLite (agent_memory.db) supaya tidak hilang saat script/bot di-restart.
# Hanya aktif kalau memory_enabled=True. user_id default "default" dipakai CLI
# (satu sesi lokal); bot Telegram/Discord mengirim chat_id sebagai user_id
# supaya tiap orang punya histori sendiri-sendiri.
# ---------------------------------------------------------------------------
DEFAULT_USER_ID = "default"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            user_id TEXT NOT NULL,
            msg_role TEXT NOT NULL,   -- 'user' atau 'assistant'
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_role_user ON memory(role, user_id)")
    conn.commit()
    conn.close()


init_db()  # pastikan tabel ada begitu module di-import


def _insert_message(role: str, user_id: str, msg_role: str, content: str) -> None:
    conn = _get_db()
    conn.execute(
        "INSERT INTO memory (role, user_id, msg_role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (role, user_id, msg_role, content, datetime.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def _fetch_history(role: str, user_id: str, max_turns: int) -> list:
    conn = _get_db()
    rows = conn.execute(
        "SELECT msg_role, content FROM memory WHERE role=? AND user_id=? ORDER BY id DESC LIMIT ?",
        (role, user_id, max_turns * 2),
    ).fetchall()
    conn.close()
    rows = list(reversed(rows))  # DESC -> kronologis (lama ke baru)
    return [{"role": r["msg_role"], "content": r["content"]} for r in rows]


def get_history(role: str, new_user_message: str, user_id: str = DEFAULT_USER_ID) -> list:
    cfg = ROLE_CONFIG[role]
    if not cfg.get("memory_enabled", False):
        return [{"role": "user", "content": new_user_message}]

    _insert_message(role, user_id, "user", new_user_message)
    return _fetch_history(role, user_id, MEMORY_MAX_TURNS)


def save_assistant_reply(role: str, reply: str, user_id: str = DEFAULT_USER_ID) -> None:
    cfg = ROLE_CONFIG[role]
    if cfg.get("memory_enabled", False):
        _insert_message(role, user_id, "assistant", reply)


def reset_memory(role: str = None, user_id: str = None) -> None:
    """Reset memory. Kombinasi:
    - role=None, user_id=None -> hapus SEMUA memory (semua role, semua user)
    - role=None, user_id="X"  -> hapus semua role milik user X
    - role="Y", user_id=None  -> hapus role Y milik SEMUA user
    - role="Y", user_id="X"   -> hapus role Y milik user X saja
    """
    query = "DELETE FROM memory WHERE 1=1"
    params = []
    if role is not None:
        query += " AND role=?"
        params.append(role)
    if user_id is not None:
        query += " AND user_id=?"
        params.append(user_id)
    conn = _get_db()
    conn.execute(query, params)
    conn.commit()
    conn.close()


# Berapa karakter maksimum tiap cuplikan hasil kerja role lain yang ditampilkan ke PM
TEAM_CONTEXT_SNIPPET_LEN = 300
# Berapa hasil terakhir per role yang diikutkan
TEAM_CONTEXT_ENTRIES_PER_ROLE = 1

ROLE_LABELS = {
    "designer": "UI/UX Designer",
    "programmer": "Programmer",
    "copywriter": "Copywriter",
    "analyst": "Business/Data Analyst",
    "project_manager": "Project Manager",
}


def build_team_context(exclude_role: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Kumpulkan hasil kerja terakhir dari role lain (selain exclude_role) MILIK
    USER YANG SAMA, supaya konteks tim tidak bocor antar user yang berbeda saat
    bot dipakai banyak orang sekaligus."""
    conn = _get_db()
    rows = conn.execute(
        """
        SELECT role, content FROM memory
        WHERE user_id=? AND role != ? AND msg_role='assistant'
        ORDER BY id DESC
        """,
        (user_id, exclude_role),
    ).fetchall()
    conn.close()

    seen_count = {}
    picked = []
    for r in rows:  # rows terurut dari yang paling baru
        c = seen_count.get(r["role"], 0)
        if c < TEAM_CONTEXT_ENTRIES_PER_ROLE:
            picked.append(r)
            seen_count[r["role"]] = c + 1
    picked.reverse()  # jadi kronologis lagi

    blocks = []
    for r in picked:
        label = ROLE_LABELS.get(r["role"], r["role"])
        snippet = r["content"][:TEAM_CONTEXT_SNIPPET_LEN]
        if len(r["content"]) > TEAM_CONTEXT_SNIPPET_LEN:
            snippet += "..."
        blocks.append(f"[{label}] {snippet}")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Export laporan (Markdown) dari seluruh riwayat kerja tim untuk seorang user
# ---------------------------------------------------------------------------
def generate_report(user_id: str = DEFAULT_USER_ID) -> str:
    """Susun laporan Markdown dari seluruh riwayat percakapan (semua role) milik
    user_id tertentu, urut dari role PM -> Designer -> Programmer -> Copywriter
    -> Analyst, dan tiap role diurutkan kronologis (lama ke baru)."""
    lines = [
        "# Laporan Progres Proyek",
        f"_User: {user_id} | Dibuat: {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
    ]

    conn = _get_db()
    ada_isi = False
    for role in ROLE_CONFIG:
        rows = conn.execute(
            "SELECT msg_role, content, created_at FROM memory WHERE role=? AND user_id=? ORDER BY id ASC",
            (role, user_id),
        ).fetchall()
        if not rows:
            continue
        ada_isi = True
        label = ROLE_LABELS.get(role, role)
        lines.append(f"## {label}")
        lines.append("")
        for r in rows:
            if r["msg_role"] == "user":
                lines.append(f"**Task** _{r['created_at']}_: {r['content']}")
            else:
                lines.append(f"**Hasil:**\n\n{r['content']}")
            lines.append("")
        lines.append("---")
        lines.append("")
    conn.close()

    if not ada_isi:
        lines.append("_Belum ada riwayat kerja tersimpan untuk user ini._")

    return "\n".join(lines)


def export_report(user_id: str = DEFAULT_USER_ID) -> str:
    """Generate laporan dan simpan ke file .md di folder reports/. Return path file."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    content = generate_report(user_id)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user_id = re.sub(r"[^\w-]", "_", str(user_id))
    filename = f"report_{safe_user_id}_{ts}.md"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_interaction(role: str, model_used: str, user_message: str, result: str, error: str = None) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "model_used": model_used,
        "input": user_message,
        "output": result,
        "error": error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Router / PM Agent
# ---------------------------------------------------------------------------
def classify_task(user_message: str) -> str:
    """Router agent menentukan role mana yang paling cocok untuk task ini."""
    roles = list(ROLE_CONFIG.keys())
    system = (
        "Kamu adalah router yang mengklasifikasikan permintaan user ke SATU dari role berikut: "
        f"{', '.join(roles)}. "
        "Balas HANYA dengan satu kata nama role, tanpa penjelasan apa pun."
    )
    result, _ = call_with_fallback(
        "openrouter", [ROUTER_MODEL] + ROUTER_FALLBACK, system,
        [{"role": "user", "content": user_message}],
    )
    result = result.strip().lower().strip(".")

    # fallback pencocokan sederhana kalau model tidak menjawab persis
    for role in roles:
        if role in result:
            return role
    return "project_manager"  # default aman kalau klasifikasi tidak jelas


# ---------------------------------------------------------------------------
# Self-check kode (khusus role dengan self_check_enabled=True, misal programmer)
# ---------------------------------------------------------------------------
MAX_SELF_CHECK_RETRIES = 2

CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Bahasa yang di-treat sebagai Python untuk keperluan syntax check.
# Kalau label bahasa kosong (model kadang lupa nulis ```python), tetap dicoba
# di-parse sebagai Python — kalau bukan Python beneran, ast.parse akan gagal
# dan hasilnya cuma diabaikan (bukan dianggap error blocking).
PYTHON_LABELS = {"python", "py", ""}


def extract_code_blocks(text: str) -> list:
    """Ambil semua code block ```lang\n...\n``` dari teks. Return list of (lang, code)."""
    return [(lang.lower(), code) for lang, code in CODE_BLOCK_RE.findall(text)]


def _lint_python(code: str) -> list:
    """Lint pakai pyflakes kalau ter-install; kalau tidak ada, dilewati saja
    (self-check tetap jalan, cuma tanpa lint tambahan)."""
    try:
        import io
        from pyflakes.api import check as pyflakes_check
        from pyflakes.reporter import Reporter

        out, err = io.StringIO(), io.StringIO()
        pyflakes_check(code, "<generated>", Reporter(out, err))
        messages = [line for line in out.getvalue().splitlines() if line.strip()]
        return messages
    except ImportError:
        return []  # pyflakes tidak ter-install, skip lint (opsional: pip install pyflakes)


def self_check_code(result_text: str) -> tuple:
    """Cek semua code block Python di dalam hasil: syntax error (wajib, pakai
    ast.parse) + lint warning (opsional, pakai pyflakes kalau ada).
    Return: (is_ok: bool, issues: list of str). is_ok=False hanya kalau ada
    SYNTAX ERROR (lint warning tidak menggagalkan self-check, cuma dilaporkan)."""
    blocks = extract_code_blocks(result_text)
    issues = []
    has_syntax_error = False

    for i, (lang, code) in enumerate(blocks, start=1):
        if lang not in PYTHON_LABELS or not code.strip():
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Kalau label eksplisit "python"/"py", ini beneran error.
            # Kalau label kosong dan gagal parse, kemungkinan besar bukan kode
            # Python (misalnya JS/HTML tanpa label) -> jangan dianggap error.
            if lang in ("python", "py"):
                has_syntax_error = True
                issues.append(f"Blok kode #{i}: SyntaxError - {e.msg} (baris {e.lineno})")
            continue

        lint_msgs = _lint_python(code)
        for m in lint_msgs:
            issues.append(f"Blok kode #{i} (lint): {m}")

    return (not has_syntax_error), issues


def dispatch(role: str, user_message: str, user_id: str = DEFAULT_USER_ID) -> tuple:
    cfg = ROLE_CONFIG[role]
    history = get_history(role, user_message, user_id)

    system = cfg["system"]
    if cfg.get("team_aware"):
        team_context = build_team_context(exclude_role=role, user_id=user_id)
        if team_context:
            system = (
                f"{system}\n\n"
                "Berikut progres/hasil kerja terbaru dari anggota tim lain "
                "(pakai sebagai konteks kalau relevan, jangan diulang mentah-mentah):\n"
                f"{team_context}"
            )

    result, model_used = call_with_fallback(
        cfg["provider"], [cfg["model"]] + cfg["fallback"], system, history,
        enable_search=cfg.get("web_search_enabled", False),
    )

    if cfg.get("self_check_enabled"):
        result, model_used = _run_self_check_loop(cfg, system, history, result, model_used)

    save_assistant_reply(role, result, user_id)
    return result, model_used


def _run_self_check_loop(cfg: dict, system: str, history: list, result: str, model_used: str) -> tuple:
    """Jalankan self-check kode, dan kalau ada syntax error, minta model
    memperbaikinya (maks MAX_SELF_CHECK_RETRIES kali) sebelum dikembalikan.
    History yang dipakai untuk retry TIDAK disimpan permanen ke memory —
    hanya hasil akhir yang lolos (atau hasil terakhir kalau tetap gagal)."""
    ok, issues = self_check_code(result)
    attempt = 0
    while not ok and attempt < MAX_SELF_CHECK_RETRIES:
        attempt += 1
        fix_prompt = (
            "Kode yang barusan kamu buat punya masalah berikut:\n"
            + "\n".join(issues)
            + "\n\nTolong perbaiki dan kirim ulang kode LENGKAP-nya (jangan cuma potongan)."
        )
        retry_history = history + [
            {"role": "assistant", "content": result},
            {"role": "user", "content": fix_prompt},
        ]
        print(f"  [self-check] percobaan perbaikan #{attempt}: {issues}", file=sys.stderr)
        result, model_used = call_with_fallback(
            cfg["provider"], [cfg["model"]] + cfg["fallback"], system, retry_history,
        )
        ok, issues = self_check_code(result)

    if not ok:
        warning = (
            "\n\n⚠️ Catatan self-check: kode di atas masih mengandung syntax error "
            f"setelah {attempt}x percobaan perbaikan otomatis:\n" + "\n".join(issues)
        )
        result += warning
    elif issues:
        # Lolos syntax check, tapi ada lint warning (non-blocking) -> tetap dilaporkan
        result += "\n\nCatatan lint (non-blocking):\n" + "\n".join(issues)

    return result, model_used


LANG_EXT_MAP = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "jsx": "jsx", "tsx": "tsx",
    "html": "html", "css": "css",
    "json": "json",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp",
    "go": "go", "rust": "rs", "ruby": "rb", "php": "php",
    "sql": "sql", "yaml": "yml", "yml": "yml",
}


def save_code_files(result_text: str, user_id: str = DEFAULT_USER_ID) -> list:
    """Ekstrak semua code block dari hasil dan simpan masing-masing sebagai file
    terpisah di generated_code/, dengan ekstensi sesuai bahasa yang dilabel.
    Return: list path file yang berhasil disimpan (bisa kosong kalau tidak ada code block)."""
    blocks = extract_code_blocks(result_text)
    if not blocks:
        return []

    os.makedirs(CODE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user_id = re.sub(r"[^\w-]", "_", str(user_id))

    paths = []
    for i, (lang, code) in enumerate(blocks, start=1):
        if not code.strip():
            continue
        ext = LANG_EXT_MAP.get(lang, "txt")
        filename = f"code_{safe_user_id}_{ts}_{i}.{ext}"
        path = os.path.join(CODE_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        paths.append(path)
    return paths


def read_file_content(filepath: str, max_chars: int = MAX_FILE_CONTEXT_CHARS) -> str:
    """Baca isi file teks dengan aman. Return string berisi isi file (dipotong
    kalau kepanjangan), atau pesan error yang jelas kalau file tidak bisa dibaca
    (format tidak didukung, terlalu besar untuk dibuka, dll)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in READABLE_TEXT_EXTENSIONS:
        return (
            f"[Tidak bisa membaca isi file '{os.path.basename(filepath)}': "
            f"format '{ext or '(tanpa ekstensi)'}' belum didukung. "
            f"Format yang didukung: {', '.join(sorted(READABLE_TEXT_EXTENSIONS))}]"
        )
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"[Gagal membaca file '{os.path.basename(filepath)}': {e}]"

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n...[dipotong, total {len(content)} karakter]"
    return content


def process_message(user_message: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Proses satu pesan: klasifikasi role -> dispatch -> log. Return teks hasil
    (sudah diformat) supaya bisa dipakai oleh CLI. Untuk interface yang bisa kirim
    file (misal bot), pakai process_message_with_files() sebagai gantinya."""
    text, _ = process_message_with_files(user_message, user_id)
    return text


def process_message_with_files(
    user_message: str,
    user_id: str = DEFAULT_USER_ID,
    attached_file_path: str = None,
) -> tuple:
    """Sama seperti process_message, tapi juga mengembalikan list path file kode
    yang di-generate (kalau role-nya programmer dan hasilnya mengandung code block).
    Kalau attached_file_path diisi, isi file itu dibaca dan disisipkan sebagai
    konteks tambahan sebelum di-dispatch ke role yang sesuai.
    Return: (teks_hasil, list_path_file_kode)"""
    effective_message = user_message
    if attached_file_path:
        file_content = read_file_content(attached_file_path)
        filename = os.path.basename(attached_file_path)
        effective_message = (
            f"{user_message}\n\n"
            f"[File terlampir: {filename}]\n"
            f'"""\n{file_content}\n"""'
        )

    role = classify_task(effective_message)
    try:
        result, model_used = dispatch(role, effective_message, user_id)
        log_interaction(role, model_used, effective_message, result)

        code_files = []
        if ROLE_CONFIG[role].get("code_output_enabled"):
            code_files = save_code_files(result, user_id)

        text = f"[{role} | {model_used}]\n{result}"
        if code_files:
            text += "\n\nFile kode tersimpan:\n" + "\n".join(f"- {p}" for p in code_files)
        return text, code_files
    except Exception as e:
        log_interaction(role, "N/A", effective_message, "", error=str(e))
        return f"Gagal memproses task ({role}): {e}", []


def handle_task(user_message: str) -> None:
    """Versi CLI dari process_message: print langsung ke terminal."""
    print(f"\n>> Task: {user_message}")
    print(process_message(user_message))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        handle_task(" ".join(sys.argv[1:]))
    else:
        print("Mode interaktif. Ketik task-mu, 'reset' untuk hapus memory, 'export' untuk simpan laporan, atau 'exit' untuk keluar.")
        while True:
            try:
                msg = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            cmd = msg.strip().lower()
            if cmd in ("exit", "quit"):
                break
            if cmd == "reset":
                reset_memory()
                print(">> Semua memory role sudah direset.")
                continue
            if cmd == "export":
                path = export_report()
                print(f">> Laporan disimpan di: {path}")
                continue
            if msg.strip():
                handle_task(msg)