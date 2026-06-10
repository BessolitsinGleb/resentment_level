import os
import io
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from telegram import Update
from telegram.ext import (
   ApplicationBuilder,
   CommandHandler,
   MessageHandler,
   ContextTypes,
   filters,
   ConversationHandler,
)
import gdown
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
GDRIVE_FILE_ID   = os.getenv("GDDRIVE_FILE_ID")
MODEL_PATH       = "offence_model.pkl"
DEVICE           = "cpu"
WAITING_MESSAGE  = 1  

def download_model():
   if os.path.exists(MODEL_PATH):
       print(f"✓ Модель уже есть: {MODEL_PATH}")
       return

   print("Скачиваем модель с Google Drive...")
   url = GDRIVE_FILE_ID
   gdown.download(url, MODEL_PATH, quiet=False)
   print(f"✓ Модель скачана: {MODEL_PATH}")

class OffenceRegressor(nn.Module):
   def __init__(self, model_name: str, dropout: float = 0.1):
       super().__init__()
       self.backbone = AutoModel.from_pretrained(model_name)
       hidden = self.backbone.config.hidden_size
       self.regressor = nn.Sequential(
           nn.Dropout(dropout),
           nn.Linear(hidden, 256),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(256, 1),
           nn.Sigmoid(),
       )

   def forward(self, input_ids, attention_mask):
       out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
       cls = out.last_hidden_state[:, 0, :]
       return self.regressor(cls).squeeze(-1)


class CPUUnpickler(pickle.Unpickler):
   def find_class(self, module, name):
       if module == "torch.storage" and name == "_load_from_bytes":
           return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
       return super().find_class(module, name)


def load_model():
   with open(MODEL_PATH, "rb") as f:
       payload = CPUUnpickler(f).load()

   model_name = payload["model_config"]["model_name"]
   max_len    = payload["model_config"]["max_len"]

   model = OffenceRegressor(model_name)
   model.load_state_dict(payload["model_state"])
   model.to(DEVICE)
   model.eval()

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   print("✓ Модель загружена")
   return model, tokenizer, max_len


def predict(text: str) -> float:
   enc = TOKENIZER(
       text,
       max_length=MAX_LEN,
       padding="max_length",
       truncation=True,
       return_tensors="pt",
   )
   with torch.no_grad():
       raw = MODEL(
           enc["input_ids"].to(DEVICE),
           enc["attention_mask"].to(DEVICE),
       )
   return round(raw.item() * 9.0 + 1.0, 2)


def score_to_emoji(score: float) -> str:
   if score < 2:   return "😊"
   if score < 4:   return "😐"
   if score < 6:   return "😕"
   if score < 8:   return "😠"
   return              "🤬"


def score_to_bar(score: float) -> str:
   filled = int(score)
   empty  = 10 - filled
   return "█" * filled + "░" * empty

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
   await update.message.reply_text(
       "👋 Привет! Я анализирую степень обиды в тексте.\n\n"
       "Команды:\n"
       "/check — проверить насколько сильно на тебя обиделись\n"
       "/help  — помощь"
   )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
   await update.message.reply_text(
       "📖 Как пользоваться:\n\n"
       "1. Напиши /check\n"
       "2. Отправь сообщение которое хочешь проверить\n"
       "3. Получи оценку обиды от 1 до 10\n\n"
       "Шкала:\n"
       "1–2  😊 Не обижен\n"
       "3–4  😐 Слегка задет\n"
       "5–6  😕 Заметная обида\n"
       "7–8  😠 Сильная обида\n"
       "9–10 🤬 Очень сильная обида"
   )


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
   await update.message.reply_text(
       "📩 Отправь мне сообщение которое хочешь проверить.\n"
       "Для отмены напиши /cancel"
   )
   return WAITING_MESSAGE


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
   text  = update.message.text
   score = predict(text)
   emoji = score_to_emoji(score)
   bar   = score_to_bar(score)

   await update.message.reply_text(
       f"{emoji} Результат анализа:\n\n"
       f"💬 Сообщение: «{text}»\n\n"
       f"📊 Степень обиды: {score}/10\n"
       f"[{bar}]\n\n"
       f"{'Человек совсем не обижен 😊'        if score < 2 else ''}"
       f"{'Человек слегка задет 😐'            if 2 <= score < 4 else ''}"
       f"{'Заметная обида, стоит поговорить 😕' if 4 <= score < 6 else ''}"
       f"{'Человек сильно обижен 😠'            if 6 <= score < 8 else ''}"
       f"{'Очень сильная обида! 🤬'             if score >= 8 else ''}"
   )
   return ConversationHandler.END


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
   await update.message.reply_text("❌ Отменено.")
   return ConversationHandler.END

import asyncio

async def main():
    download_model()

    global MODEL, TOKENIZER, MAX_LEN
    MODEL, TOKENIZER, MAX_LEN = load_model()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("check", cmd_check)],
        states={
            WAITING_MESSAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
            ]
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(conv_handler)

    print("🤖 Бот запущен...")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        await asyncio.Event().wait()  # держим бота живым


if __name__ == "__main__":
    asyncio.run(main())