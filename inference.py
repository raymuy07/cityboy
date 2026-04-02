"""
Live listener for the NUC.
Monitors @tzevaadomm for pre-alert trigger images, runs the trained
MobileNetV3-Small model, and sends the prediction via a Telegram bot.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto

load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALERT_CHANNEL = os.getenv("TELEGRAM_CHANNEL", "")

CHANNEL = "@tzevaadomm"
SESSION_NAME = "cityboy_listener"
TRIGGER_CAPTION = "מבזק פיקוד העורף - התרעה מקדימה"
TARGET_LOCATION = "תל אביב - מרכז העיר"

WEIGHTS = {
    "v1": Path("model_weights.pth"),
    "v3": Path("model_weights_v3.pth"),
}
IMG_SIZE = 224
THRESHOLD = 0.5
V3_FLIP_THRESHOLD = 0.35

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_models: dict[str, nn.Module] = {}

BUILDERS: dict[str, callable] = {}


def _build_v1() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def _build_v3() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(256, 1),
    )
    return model


BUILDERS = {"v1": _build_v1, "v3": _build_v3}


def load_model(version: str) -> nn.Module:
    model = BUILDERS[version]()
    model.load_state_dict(torch.load(WEIGHTS[version], map_location="cpu"))
    model.eval()
    return model


def get_model(version: str = "v1") -> nn.Module:
    if version not in _models:
        _models[version] = load_model(version)
    return _models[version]


def _infer(image_tensor: torch.Tensor, version: str) -> float:
    """Run a single model forward pass and return the sigmoid probability."""
    model = get_model(version)
    with torch.no_grad():
        logit = model(image_tensor).squeeze()
    return torch.sigmoid(logit).item()


def predict(image_path: str | Path) -> dict:
    """Two-layer conditional inference: V1 (gatekeeper) then V3 (validator).

    Returns a dict with ``is_tel_aviv`` (bool) and ``probability`` (0-100 %).
    """
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)

    v1_prob = _infer(tensor, "v1")

    if v1_prob < THRESHOLD:
        return {
            "probability": round(v1_prob * 100, 2),
            "is_tel_aviv": False,
            "layer": "v1",
        }

    v3_prob = _infer(tensor, "v3")

    if v3_prob > 0.62:
        v3_prob = min(v3_prob * 1.2, 1.0)

    if v3_prob >= THRESHOLD:
        return {
            "probability": round(v3_prob * 100, 2),
            "is_tel_aviv": True,
            "layer": "v3-agree",
        }

    if v3_prob < V3_FLIP_THRESHOLD:
        return {
            "probability": round(v3_prob * 100, 2),
            "is_tel_aviv": False,
            "layer": "v3-flip",
        }

    blended = 0.6 * v1_prob + 0.4 * v3_prob
    return {
        "probability": round(blended * 100, 2),
        "is_tel_aviv": True,
        "layer": "v3-uncertain",
    }


# ---------------------------------------------------------------------------
# Optional: OpenVINO for faster CPU inference on Intel NUC
# ---------------------------------------------------------------------------

def export_to_openvino(
    version: str = "v1",
    output_dir: Path = Path("openvino_model"),
) -> None:
    import openvino as ov

    model = load_model(version)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    ov_model = ov.convert_model(model, example_input=dummy_input)

    output_dir.mkdir(exist_ok=True)
    ov.save_model(ov_model, output_dir / "model.xml")
    print(f"OpenVINO IR saved to {output_dir.resolve()}")


def predict_openvino(
    image_path: str | Path,
    model_dir: Path = Path("openvino_model"),
) -> dict:
    import numpy as np
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(model_dir / "model.xml", "CPU")
    infer_request = compiled.create_infer_request()

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).numpy()

    result = infer_request.infer({0: tensor})
    logit = list(result.values())[0].flatten()[0]

    prob = 1.0 / (1.0 + np.exp(-logit))
    return {
        "probability": round(float(prob) * 100, 2),
        "is_tel_aviv": float(prob) > THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Legacy subscriber management (kept for migration broadcast only)
# ---------------------------------------------------------------------------

SUBSCRIBERS_FILE = Path("subscribers.json")


def load_subscribers() -> set[int]:
    if SUBSCRIBERS_FILE.exists():
        data = json.loads(SUBSCRIBERS_FILE.read_text(encoding="utf-8"))
        return set(data)
    return set()


# ---------------------------------------------------------------------------
# Telegram listener
# ---------------------------------------------------------------------------

async def run_listener() -> None:
    user_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    bot_client = TelegramClient("cityboy_bot", API_ID, API_HASH)

    await user_client.start()
    await bot_client.start(bot_token=BOT_TOKEN)

    print(f"Listening on {CHANNEL} for trigger images ...")
    print(f"Posting alerts to {ALERT_CHANNEL}")

    # --- Bot /start redirects users to the channel ---

    @bot_client.on(events.NewMessage(pattern="/start"))
    async def on_start(event):
        await event.respond(
            "\u200F<b>חברה אנחנו עוברים לערוץ!</b>\n\n"
            f"\u200F<b>https://t.me/cityboy_alerts</b>\n\n"
            "\u200Fמעכשיו כל ההתרעות יפורסמו שם.\n"
            "\u200Fתודה אהובים ",
            parse_mode="html",
        )

    # --- Channel listener for trigger images ---

    DEBOUNCE_SECONDS = 10
    ALERT_COOLDOWN_SECONDS = 300

    sent_message_id: int | None = None
    last_alert_time: float = 0
    last_prediction: dict | None = None
    confirmation_sent: bool = False
    pending_task: asyncio.Task | None = None

    def _extract_title(caption: str) -> str:
        first_line = caption.strip().split("\n")[0]
        return first_line

    def _contains_dan(caption: str) -> bool:
        return bool(re.search(r"\bדן\b", caption))

    def _is_same_alert() -> bool:
        return (time.time() - last_alert_time) < ALERT_COOLDOWN_SECONDS

    async def _process_trigger(message) -> None:
        nonlocal last_alert_time, sent_message_id, last_prediction, confirmation_sent
        caption = message.message or ""
        title = _extract_title(caption)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await user_client.download_media(message, file=tmp_path)
            result = predict(tmp_path)

            prob = result["probability"]
            is_ta = result["is_tel_aviv"]
            layer = result.get("layer", "?")
            emoji = "🔴" if is_ta else "🟢"

            print(f"  Probability: {prob}% | Tel Aviv: {is_ta} | Layer: {layer}")

            notification = (
                f"<b>{title}</b>\n\n"
                f"{emoji} {emoji} {emoji}\n"
                f"סיכוי לאזעקה בתל אביב: <b>{prob}%</b>\n"
                f"מסקנה: <b>{'סיכוי גבוה לאזעקה!' if is_ta else 'יש מצב הכל טוב'}</b>\n\n"
                f"<b>תזכורת: המידע מבוסס על אלגוריתם למידת מכונה ואינו מהווה תחליף להתרעה רשמית, יש להישמע אך ורק להוראות פיקוד העורף!</b>"
            )

            if sent_message_id is not None:
                try:
                    await bot_client.delete_messages(ALERT_CHANNEL, sent_message_id)
                except Exception:
                    pass

            sent = await bot_client.send_file(
                ALERT_CHANNEL, tmp_path,
                caption=notification, parse_mode="html",
            )
            sent_message_id = sent.id
            print(f"  📤 Posted to {ALERT_CHANNEL}")

            last_alert_time = time.time()
            last_prediction = result
            confirmation_sent = False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def _debounced_process(message) -> None:
        await asyncio.sleep(DEBOUNCE_SECONDS)
        await _process_trigger(message)

    def _schedule_trigger(message) -> None:
        nonlocal pending_task, sent_message_id, last_alert_time

        if not _is_same_alert():
            sent_message_id = None

        last_alert_time = time.time()

        if pending_task and not pending_task.done():
            pending_task.cancel()
            print("  ⏳ Debounce reset — waiting for photo to stabilize ...")

        pending_task = asyncio.create_task(_debounced_process(message))

    @user_client.on(events.NewMessage(chats=CHANNEL))
    async def on_new_trigger(event):
        message = event.message
        caption = message.message or ""

        if not (isinstance(message.media, MessageMediaPhoto) and TRIGGER_CAPTION in caption):
            return

        if not _contains_dan(caption):
            print(f"\n⏭️  Trigger (msg {message.id}) — 'דן' not in regions, skipping.")
            return

        same = "same alert" if _is_same_alert() else "new alert"
        print(f"\n⚡ Trigger detected (msg {message.id}) [{same}]")
        _schedule_trigger(message)

    @user_client.on(events.MessageEdited(chats=CHANNEL))
    async def on_edit_trigger(event):
        message = event.message
        caption = message.message or ""

        if not (isinstance(message.media, MessageMediaPhoto) and TRIGGER_CAPTION in caption):
            return

        if not _contains_dan(caption):
            return

        print(f"\n🔄 Trigger edited (msg {message.id})")
        _schedule_trigger(message)

    # --- Confirmation: TA found instantly, or event-end as fallback ---

    EVENT_END_CAPTION = "עדכון פיקוד העורף - סיום אירוע"

    async def _send_confirmation(ta_found: bool) -> None:
        nonlocal confirmation_sent
        if confirmation_sent or last_prediction is None:
            return
        confirmation_sent = True

        we_predicted_ta = last_prediction["is_tel_aviv"]
        prob = last_prediction["probability"]
        correct = ta_found == we_predicted_ta

        if correct:
            emoji = "✅"
            verdict = " :0 באמת הייתה אזעקה" if ta_found else "יאסס לא נכנסנו למקלט סתם, אבא אוהב."
        else:
            emoji = "❌"
            verdict = "שיט... היתה אזעקה בתל אביב" if ta_found else "רבי יקותיאל...שוב ירדנו סתם"

        confirmation_msg = (
            f"{emoji} <b>אימות תוצאה</b>\n\n"
            f"חיזוי: <b>{prob}%</b> ({'כן' if we_predicted_ta else 'לא'})\n"
            f"<b>{verdict}</b>"
        )

        print(f"\n{'✅' if correct else '❌'} Confirmation: predicted={we_predicted_ta}, actual={ta_found}")

        try:
            await bot_client.send_message(
                ALERT_CHANNEL, confirmation_msg, parse_mode="html",
            )
            print(f"  📤 Confirmation posted to {ALERT_CHANNEL}")
        except Exception as e:
            print(f"  Failed to post confirmation: {e}")

    @user_client.on(events.NewMessage(chats=CHANNEL))
    async def on_ta_found(event):
        """TA city list arrives mid-event → confirm immediately."""
        if confirmation_sent or last_prediction is None:
            return

        text = event.message.message or ""

        if TRIGGER_CAPTION in text:
            return

        if TARGET_LOCATION not in text:
            return

        print(f"\n🎯 Found '{TARGET_LOCATION}' in msg {event.message.id}!")
        await _send_confirmation(ta_found=True)

    @user_client.on(events.NewMessage(chats=CHANNEL))
    async def on_event_end(event):
        """Event-end message arrives → TA was NOT hit, confirm now."""
        if confirmation_sent or last_prediction is None:
            return

        text = event.message.message or ""

        if EVENT_END_CAPTION not in text:
            return

        if not _contains_dan(text):
            print(f"\n⏭️  Event end (msg {event.message.id}) — 'דן' not in regions, skipping.")
            return

        print(f"\n🏁 Event ended (msg {event.message.id}) — TA not found, confirming.")
        await _send_confirmation(ta_found=False)

    print("Bot is ready. Waiting for events ...")
    await asyncio.gather(
        user_client.run_until_disconnected(),
        bot_client.run_until_disconnected(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CityBoy — NUC inference & listener")
    parser.add_argument("--listen", action="store_true", help="Start live Telegram listener")
    parser.add_argument("--predict", metavar="IMAGE", help="Run prediction on a single image")
    parser.add_argument("--v3", action="store_true", help="Use v3 model for OpenVINO export")
    parser.add_argument("--openvino", action="store_true", help="Use OpenVINO model")
    parser.add_argument("--export-openvino", action="store_true", help="Export to OpenVINO IR")
    args = parser.parse_args()

    version = "v3" if args.v3 else "v1"

    if args.export_openvino:
        export_to_openvino(version)
    elif args.listen:
        if not BOT_TOKEN:
            print("Set TELEGRAM_BOT_TOKEN in .env")
            return
        asyncio.run(run_listener())
    elif args.predict:
        if args.openvino:
            result = predict_openvino(args.predict)
        else:
            result = predict(args.predict)
        print(f"Probability: {result['probability']}%")
        print(f"Tel Aviv: {result['is_tel_aviv']}")
        print(f"Layer: {result.get('layer', 'openvino')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
