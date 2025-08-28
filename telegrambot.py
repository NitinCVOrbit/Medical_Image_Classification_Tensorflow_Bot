from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, CallbackContext, filters
)
from io import BytesIO
from PIL import Image
from Models import classifier  # Your TensorFlow-based classifier

# Token
TOKEN = ""

# Predefined Data Dictionary (TensorFlow models with .h5 format)
data = {
    1: {
        "class_names": ['NO BRAIN TUMOR', 'BRAIN TUMOR'],
        "weights_name": "brain_tumor.keras",
    },
    2: {
        "class_names": ["NORMAL", "PNEUMONIA"],
        "weights_name": "chest_xray.keras",
    }
}

# /start command handler: shows model selection buttons
async def start(update: Update, context: CallbackContext):
    buttons = []

    for i in data:
        model_classes = ', '.join(data[i]['class_names'])  # NO BRAIN TUMOR, BRAIN TUMOR
        button = [InlineKeyboardButton(f"Model {i}: {model_classes}", callback_data=str(i))]
        buttons.append(button)

    keyboard = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(
        "🧠 Please select the classification model you want to use:",
        reply_markup=keyboard
    )


# Button handler for model selection
async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    print("\n\n\n", query, "\n\n\n" , query.data, "\n\n\n")
    model_id = int(query.data)
    context.user_data["selected_model"] = model_id
  
    model_name = data[model_id]["weights_name"]

    await query.edit_message_text(
        f"✅ Model selected: *{model_name}*\n\n📸 Now send an image for classification.",
        parse_mode="Markdown"
    )

# Handle uploaded photo for classification
async def handle_photo(update: Update, context: CallbackContext):
    if "selected_model" not in context.user_data:
        await update.message.reply_text("⚠️ Please select a model first using /start before sending an image.")
        return

    model_id = context.user_data["selected_model"]
    model_data = data[model_id]

    file = await update.message.photo[-1].get_file()
    image_bytes = BytesIO(await file.download_as_bytearray())
    image = Image.open(image_bytes).convert("RGB")

    # TensorFlow-based classification
    label, score = classifier(
        image,
        model_data["weights_name"],
        model_data["class_names"]
    )

    # Return result
    await update.message.reply_text(
        f"📷 *Prediction:* `{label}`\n"
        f"🔢 *Confidence:* `{score:.2f}`\n"
        f"🧠 *Model Used:* `{model_data['weights_name']}`",
        parse_mode="Markdown"
    )

# /help command
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text("Use /start to select a model before uploading images.")

# Main function
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
