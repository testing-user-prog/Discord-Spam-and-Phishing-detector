# Discord Spam & Phishing Detector

A Discord bot that automatically detects and deletes spam messages using a machine learning classifier trained on a public Telegram spam/ham dataset. Flagged messages are removed instantly and the sender gets a warning reply.

## How It Works

1. On startup, the bot pulls the [`thehamkercat/telegram-spam-ham`](https://huggingface.co/datasets/thehamkercat/telegram-spam-ham) dataset from HuggingFace.
2. The dataset is cleaned (duplicates and empty rows dropped) and labels are mapped to `spam = 1`, `ham = 0`.
3. Message text is vectorized with **TF-IDF**, and a **Multinomial Naive Bayes** classifier is trained on a 90/10 train-test split.
4. Test accuracy is printed to the console after training.
5. The trained classifier is wrapped in a **Singleton** so the model is only trained once per bot process.
6. Every incoming Discord message is scored by the classifier. If the predicted spam probability exceeds a threshold (`0.5`), the message is deleted and the author is warned in-channel.

> **Note:** Despite the repo name, the current implementation classifies messages as spam vs. ham using text-based Naive Bayes — it does not yet do URL/domain-based phishing link analysis.

## Tech Stack

- [discord.py](https://pypi.org/project/discord.py/) — Discord bot framework
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorizer + Multinomial Naive Bayes
- [datasets](https://pypi.org/project/datasets/) (HuggingFace) — dataset loading
- [pandas](https://pandas.pydata.org/) — data handling
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment variable management

## Requirements

```
discord.py==2.7.1
python-dotenv==1.2.2
scikit-learn==1.8.0
datasets==5.0.0
pandas==3.0.3
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/testing-user-prog/Discord-Spam-and-Phishing-detector.git
cd Discord-Spam-and-Phishing-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a Discord bot & get your token

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. Under **Bot**, create a bot user and copy its token.
3. Enable the **Message Content Intent** and **Server Members Intent** under Privileged Gateway Intents (required — the bot reads message content to classify it).
4. Invite the bot to your server using the OAuth2 URL generator, granting at least `Read Messages`, `Send Messages`, and `Manage Messages` permissions.

### 4. Configure environment variables

Create a `.env` file in the project root:

```
TOKEN=your_discord_bot_token_here
```

### 5. Run the bot

```bash
python bot.py
```

On first run, the bot will download the training dataset from HuggingFace and train the classifier — this may take a moment. Once training finishes and the accuracy is printed, the bot logs in and starts monitoring messages.

## Configuration

- **Detection threshold:** Adjust `THRESHOLD` in the `SpamClassifier` class (`bot.py`) to make detection more or less strict. Default is `0.5`.
- **Command prefix:** The bot uses `!` as its command prefix by default (`commands.Bot(command_prefix="!", ...)`).

## Limitations

- Trained on a Telegram dataset, so accuracy on Discord-specific spam patterns (raid links, fake nitro, etc.) may vary.
- No dedicated phishing/malicious-URL checking (e.g., domain reputation, blacklists) — detection is purely text-classification based.
- Model retrains from scratch every time the bot restarts; there's no persisted/cached model file.
- False positives are possible — messages are deleted immediately with no review queue.

## Disclaimer

This bot deletes messages automatically based on model predictions. Test it in a private server before deploying to a live community, and monitor its behavior to catch false positives.

## License

No license specified yet. Add one (e.g., MIT) if you plan to share or accept contributions.
