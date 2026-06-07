import discord
from discord.ext import commands
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import os
import pandas as pd

load_dotenv()
TOKEN = os.getenv("TOKEN")


class SpamClassifier:
    _instance = None
    THRESHOLD = 0.5

    def __new__(cls):
        if cls._instance is None:
            print("No existing instance found. Creating new classifier...")
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        else:
            print("Returning existing classifier instance.")
        return cls._instance

    def _initialize(self):
        """Fetch Telegram spam dataset from HuggingFace, clean it, and train the model."""

        print("Fetching Telegram spam dataset from HuggingFace...")

        # Load Telegram spam/ham dataset directly from HuggingFace
        dataset = load_dataset("thehamkercat/telegram-spam-ham")

        # Convert to pandas dataframe
        df = dataset['train'].to_pandas()

        print(f"Dataset loaded — Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Remove duplicates and empty rows
        df = df.dropna()
        df = df.drop_duplicates()

        # Convert text labels to binary — spam=1, ham=0
        df['label'] = df['text_type'].map({'spam': 1, 'ham': 0})

        print(f"Label distribution:\n{df['label'].value_counts()}")

        # Features (message text) and Labels (0=ham, 1=spam)
        X = df['text']
        Y = df['label']

        # Split into training and testing sets (90% train, 10% test)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        # Convert text into numbers using TF-IDF
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train the Naive Bayes model
        self.model = MultinomialNB()
        self.model.fit(X_train_tfidf, Y_train)

        # Print testing accuracy
        test_predictions = self.model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(Y_test, test_predictions)
        print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

    def predict(self, message: str):
        message_tfidf = self.vectorizer.transform([message])
        proba = self.model.predict_proba(message_tfidf)
        print(proba)
        # Check if spam probability exceeds the threshold
        print(proba[0][1])
        if proba[0][1] > self.THRESHOLD:
            return 1
        return 0


# Create the single instance (or reuse if it already exists)
classifier = SpamClassifier()

# Create an intents object with default settings
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

# Create the bot instance
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    result = classifier.predict(message.content)
    print(f"Message: '{message.content}' → {result}")

    if result == 1:
        await message.delete()
        await message.channel.send(f"{message.author.mention} your message was flagged as spam!")
        return

    await bot.process_commands(message)


bot.run(TOKEN)