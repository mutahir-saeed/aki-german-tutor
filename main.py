"""
Aki – A1/A2 German Language Tutor (Voice)
FastAPI backend: browser WebSocket ↔ Gemini Live API.
Features: live transcription, inline translation, VAD tuning.
"""

import asyncio
import base64
import json
import os
import traceback

# Load .env file if present (for local development)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

# ── Configuration ────────────────────────────────────────────────────────────
# SECURITY: API key must be set via environment variable, never hardcoded.
# Set GEMINI_API_KEY in your environment or .env file.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_ID = "gemini-3.1-flash-live-preview"
TRANSLATE_MODEL = "gemini-2.0-flash"  # fast model for subtitle translation

SYSTEM_PROMPT = (
    "You are Aki, a strict but warm German language tutor for absolute beginners. "
    "FOLLOW THESE RULES WITHOUT EXCEPTION:\n\n"
    
    "RULE 1 – LANGUAGE: You MUST speak in German at least 80% of the time. "
    "Use ONLY A1/A2 vocabulary. Maximum 8 words per sentence. "
    "Use Present Tense (Präsens) only.\n\n"
    
    "RULE 2 – PACING: Speak SLOWLY and CLEARLY. Pause between sentences. "
    "You are talking to a complete beginner who needs time to understand. "
    "WAIT for the student to finish speaking before you respond. "
    "NEVER rush. NEVER talk over them.\n\n"
    
    "RULE 3 – CORRECTION IS MANDATORY: This is your MOST IMPORTANT job. "
    "Every single time the student says ANYTHING in German, you MUST check "
    "for mistakes. If there is ANY grammatical error, wrong word order, wrong "
    "article (der/die/das), wrong verb conjugation, or pronunciation issue:\n"
    "  a) FIRST say the CORRECT German sentence slowly\n"
    "  b) THEN briefly explain the mistake in English (1 sentence max)\n"
    "  c) THEN continue the conversation in German\n"
    "Example: Student says 'Ich möchte ein Kaffee' → You say: "
    "'Ich möchte einEN Kaffee. — Kaffee is masculine, so we say einen. "
    "Sehr gut! Möchtest du Milch?'\n\n"
    
    "RULE 4 – ENGLISH INPUT: When the student speaks English, this is NORMAL. "
    "Do NOT ignore it. ALWAYS respond:\n"
    "  a) Briefly acknowledge in English (max 5 words)\n"
    "  b) Give the German translation\n"
    "  c) Continue in simple German\n"
    "Example: Student says 'How do I say thank you?' → You say: "
    "'Good question! Danke means thank you. Danke schön! "
    "Möchtest du noch etwas?'\n\n"
    
    "RULE 5 – BREVITY: Keep responses under 3 short sentences. "
    "Ask exactly ONE simple question at the end.\n\n"
    
    "RULE 6 – SCENARIO: Start as a friendly barista in a Munich café. "
    "Greet the student simply and ask what they want to order. "
    "Keep the roleplay grounded in daily life."
)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Aki – German Tutor")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Translation helper ───────────────────────────────────────────────────────
async def translate_text(client, text: str) -> str:
    """Translate a short German/mixed text to English using Gemini text API."""
    if not text or not text.strip():
        return ""
    try:
        response = await client.aio.models.generate_content(
            model=TRANSLATE_MODEL,
            contents=(
                "Translate the following to English. "
                "Return ONLY the English translation, nothing else. "
                "If it's already English, return it as-is.\n\n"
                f"{text}"
            ),
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"[translate] Error: {e}")
        return ""


# ── WebSocket bridge ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[server] Browser WebSocket connected")

    client = genai.Client(api_key=GEMINI_API_KEY)

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"
                )
            )
        ),
        # Enable transcription for subtitles
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Tune VAD: wait longer before responding so student can finish
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=1000,  # Wait 1 second of silence before responding
            )
        ),
    )

    try:
        async with client.aio.live.connect(
            model=MODEL_ID, config=config
        ) as session:
            print("[server] Gemini Live session opened")

            stop_event = asyncio.Event()

            async def send_json(data: dict):
                try:
                    await ws.send_text(json.dumps(data))
                except (WebSocketDisconnect, Exception):
                    stop_event.set()

            # Background translation: fire-and-forget for subtitle translations
            async def send_with_translation(speaker: str, text: str):
                """Send original text immediately, then translate and send."""
                # Send original text right away
                await send_json({
                    "type": "transcript",
                    "speaker": speaker,
                    "text": text,
                })
                # Fire off translation in background
                asyncio.create_task(
                    _translate_and_send(speaker, text)
                )

            async def _translate_and_send(speaker: str, text: str):
                """Translate text and send as a separate message."""
                try:
                    translation = await translate_text(client, text)
                    if translation and translation.strip():
                        await send_json({
                            "type": "translation",
                            "speaker": speaker,
                            "text": translation,
                        })
                except Exception:
                    pass  # Don't crash on translation failures

            # ── Browser → Gemini ─────────────────────────────────────────
            async def forward_browser_audio():
                try:
                    while not stop_event.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                ws.receive(), timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            continue
                        except WebSocketDisconnect:
                            print("[server] Browser disconnected (send-side)")
                            stop_event.set()
                            return

                        msg_type = msg.get("type", "")
                        if msg_type == "websocket.disconnect":
                            print("[server] Browser sent disconnect frame")
                            stop_event.set()
                            return
                        elif msg_type == "websocket.receive":
                            data = msg.get("bytes")
                            if data:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=data,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )
                except Exception as exc:
                    print(f"[server] forward_browser_audio error: {exc}")
                    traceback.print_exc()
                    stop_event.set()

            # ── Gemini → Browser ─────────────────────────────────────────
            async def forward_gemini_audio():
                try:
                    while not stop_event.is_set():
                        try:
                            async for response in session.receive():
                                if stop_event.is_set():
                                    return

                                sc = getattr(response, "server_content", None)

                                # ── Transcription ────────────────────
                                if sc:
                                    # User's speech
                                    itr = getattr(sc, "input_transcription", None)
                                    if itr and getattr(itr, "text", None):
                                        await send_with_translation(
                                            "user", itr.text
                                        )

                                    # Aki's speech
                                    otr = getattr(sc, "output_transcription", None)
                                    if otr and getattr(otr, "text", None):
                                        await send_with_translation(
                                            "aki", otr.text
                                        )

                                # ── Audio data ───────────────────────
                                audio_data = None

                                if hasattr(response, "data") and response.data:
                                    raw = response.data
                                    if isinstance(raw, str):
                                        audio_data = base64.b64decode(raw)
                                    elif isinstance(raw, bytes):
                                        audio_data = raw
                                elif (
                                    sc
                                    and getattr(sc, "model_turn", None)
                                    and sc.model_turn.parts
                                ):
                                    for part in sc.model_turn.parts:
                                        inline = getattr(part, "inline_data", None)
                                        if inline and getattr(inline, "data", None):
                                            raw = inline.data
                                            if isinstance(raw, str):
                                                audio_data = base64.b64decode(raw)
                                            elif isinstance(raw, bytes):
                                                audio_data = raw
                                            break

                                if audio_data:
                                    try:
                                        await ws.send_bytes(audio_data)
                                    except WebSocketDisconnect:
                                        stop_event.set()
                                        return

                        except StopAsyncIteration:
                            break
                except Exception as exc:
                    print(f"[server] forward_gemini_audio error: {exc}")
                    traceback.print_exc()
                    stop_event.set()

            await asyncio.gather(
                forward_browser_audio(),
                forward_gemini_audio(),
            )

    except WebSocketDisconnect:
        print("[server] Browser WebSocket closed")
    except Exception as exc:
        print(f"[server] Session error: {exc}")
        traceback.print_exc()
    finally:
        print("[server] Cleaning up session")
