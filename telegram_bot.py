#!/usr/bin/env python3
"""
Telegram Bot untuk Multi-Agent Router
=======================================
Menjembatani chat Telegram ke agent_router.py, jadi lo bisa kirim task dari HP
dan otomatis di-routing ke role yang sesuai (PM/Designer/Programmer/Copywriter/Analyst).

Setup:
    1. Chat @BotFather di Telegram -> /newbot -> ikuti instruksi -> dapat token
    2. pip install requests
    3. export TELEGRAM_BOT_TOKEN="123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
       export OPENROUTER_API_KEY="sk-or-xxxx"
       export GEMINI_API_KEY="AIzaSyxxxx"

Jalankan:
    python telegram_bot.py

Catatan: pakai long-polling sederhana (getUpdates), cukup buat pemakaian personal/tim
kecil. Untuk skala besar / production, pertimbangkan webhook + hosting (Railway,
Render, dll) supaya bot tetap jalan 24/7 tanpa perlu laptop menyala terus.
"""

import os
import sys
import time
import requests

import agent_router as ar

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

POLL_TIMEOUT = 30       # detik, long-polling ke Telegram
TELEGRAM_MAX_LEN = 4000  # batas aman per pesan Telegram (limit asli 4096)


def get_updates(offset: int = None) -> list:
    params = {"timeout": POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(f"{API_URL}/getUpdates", params=params, timeout=POLL_TIMEOUT + 10)
    resp.raise_for_status()
    return resp.json().get("result", [])


def download_telegram_file(file_id: str, original_filename: str) -> str:
    """Download file yang dikirim user ke Telegram, simpan ke ar.UPLOADS_DIR.
    Return path lokal file yang sudah di-download."""
    resp = requests.get(f"{API_URL}/getFile", params={"file_id": file_id}, timeout=30)
    resp.raise_for_status()
    file_path = resp.json()["result"]["file_path"]

    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    file_resp = requests.get(file_url, timeout=60)
    file_resp.raise_for_status()

    os.makedirs(ar.UPLOADS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}_{original_filename}"
    local_path = os.path.join(ar.UPLOADS_DIR, safe_name)
    with open(local_path, "wb") as f:
        f.write(file_resp.content)
    return local_path


def send_message(chat_id: int, text: str) -> None:
    """Kirim pesan, dipecah otomatis kalau lebih panjang dari limit Telegram."""
    if not text:
        text = "(kosong)"
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        chunk = text[i:i + TELEGRAM_MAX_LEN]
        try:
            requests.post(
                f"{API_URL}/sendMessage",
                json={"chat_id": chat_id, "text": chunk},
                timeout=30,
            )
        except Exception as e:
            print(f"[error kirim pesan] {e}", file=sys.stderr)


def send_document(chat_id: int, filepath: str, caption: str = None) -> None:
    """Upload & kirim file (dipakai untuk /export)."""
    try:
        with open(filepath, "rb") as f:
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
            requests.post(
                f"{API_URL}/sendDocument",
                data=data,
                files={"document": f},
                timeout=60,
            )
    except Exception as e:
        print(f"[error kirim dokumen] {e}", file=sys.stderr)
        send_message(chat_id, f"Gagal mengirim laporan: {e}")


HELP_TEXT = (
    "Halo! Kirim task apa aja dalam bahasa natural, nanti otomatis di-routing ke "
    "agent yang sesuai:\n"
    "- Project Manager\n- UI/UX Designer\n- Programmer\n- Copywriter\n- Business/Data Analyst\n\n"
    "Kirim file (.txt, .md, .py, .js, .json, .csv, dll) dengan caption berisi "
    "instruksi (mis. \"analisis data ini\") untuk memakai isi file sebagai konteks.\n\n"
    "Perintah:\n"
    "/reset - hapus semua memory percakapan\n"
    "/export - unduh laporan progres proyek (Markdown)\n"
    "/help - tampilkan pesan ini"
)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum di-set. Lihat instruksi di bagian atas file ini.")

    print("Bot Telegram jalan... (Ctrl+C untuk stop)")
    offset = None
    while True:
        try:
            updates = get_updates(offset)
        except Exception as e:
            print(f"[error polling] {e}", file=sys.stderr)
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message:
                continue

            chat_id = message["chat"]["id"]

            # --- Pesan berupa file/dokumen ---
            document = message.get("document")
            if document:
                caption = (message.get("caption") or "").strip()
                filename = document.get("file_name", "file_tanpa_nama")
                print(f">> [{chat_id}] <file: {filename}> {caption}")

                try:
                    local_path = download_telegram_file(document["file_id"], filename)
                except Exception as e:
                    send_message(chat_id, f"Gagal mengunduh file: {e}")
                    continue

                task_text = caption or "Tolong analisis/proses isi file ini."
                send_message(chat_id, f"File '{filename}' diterima, memproses...")
                try:
                    reply, code_files = ar.process_message_with_files(
                        task_text, user_id=str(chat_id), attached_file_path=local_path,
                    )
                except Exception as e:
                    reply, code_files = f"Terjadi error tak terduga: {e}", []
                send_message(chat_id, reply)
                for path in code_files:
                    send_document(chat_id, path)
                continue

            # --- Pesan teks biasa ---
            if "text" not in message:
                continue

            text = message["text"].strip()
            print(f">> [{chat_id}] {text}")

            if text.lower() in ("/reset", "reset"):
                ar.reset_memory(user_id=str(chat_id))
                send_message(chat_id, "Memory kamu sudah direset.")
                continue

            if text.lower() in ("/export", "export"):
                try:
                    path = ar.export_report(user_id=str(chat_id))
                    send_document(chat_id, path, caption="Laporan progres proyek kamu.")
                except Exception as e:
                    send_message(chat_id, f"Gagal membuat laporan: {e}")
                continue

            if text.lower() in ("/start", "/help"):
                send_message(chat_id, HELP_TEXT)
                continue

            try:
                reply, code_files = ar.process_message_with_files(text, user_id=str(chat_id))
            except Exception as e:
                reply, code_files = f"Terjadi error tak terduga: {e}", []

            send_message(chat_id, reply)
            for path in code_files:
                send_document(chat_id, path)


if __name__ == "__main__":
    main()