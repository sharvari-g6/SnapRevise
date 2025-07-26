# Imports
import spacy
import os   
from dotenv import load_dotenv
import json
from dotenv import load_dotenv, find_dotenv
from spacy.tokens import Span
from transformers import pipeline
import requests
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import yake
import random
import re

# Load environment variables
load_dotenv(find_dotenv())

API_KEY = os.getenv('API_KEY')
API_URL = os.getenv('API_URL')
HEADERS_str = os.getenv('HEADERS')
HEADERS = json.loads(HEADERS_str)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to summarize text using a transformer model
def summarize_text(text):
    payload = {"inputs": text, "parameters": {"max_length": 200, "min_length": 10, "do_sample": True , "top_k": 100 , "top_p": 0.95}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()[0]["summary_text"]

# Function to extract key phrases using RAKE
def extract_key_phrases(text):
  phrases = []
  rake = Rake(min_length = 2, sentence_tokenizer=nltk.tokenize.sent_tokenize,include_repeated_phrases = False, stopwords = nltk.corpus.stopwords.words('english'))
  rake.extract_keywords_from_text(text)
  phrases_with_scores = rake.get_ranked_phrases_with_scores()     # Getting ranked phrases with scores
  for score, phrase in phrases_with_scores:
    if score > 5.0 and is_good_phrase(phrase):
        phrases.append(phrase)                                    # Append only good phrases with a score above 5.0
  return phrases

# Function to check if a phrase is good for question generation
def is_good_phrase(phrase):
  doc = nlp(phrase)
  # Keep if phrase mostly nouns or adjectives
  for token in doc:
      if token.pos_ not in ['NOUN', 'PROPN', 'ADJ','VERB']:        # POS tagging to filter out non-noun/adjective phrases
          return False
  return True


# Function to find the senrtence containing a specific phrase
def find_sentence_with_phrase(text, phrase):
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    for sentence in sentences:
        if phrase in sentence:
            return sentence  # Return the first sentence containing the phrase
    return None  # If not found, return None

question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

# Function to highlight the answer in the context
def highlight_answer(context, answer):
    escaped_answer = re.escape(answer)
    highlighted_context = re.sub(escaped_answer, f"<hl>{answer}<hl>", context, count=1)
    return highlighted_context

# Function to generate questions based on the provided text, extracts key phrases, and highlights answers in the context.
def generate_questions(text):
    question_list = []
    answers = extract_key_phrases(text)
    contexts = []
    for answer in answers:
      temp = find_sentence_with_phrase(text, answer)
      if not temp:
        continue
      else:
        contexts.append(temp)
      context = highlight_answer(temp , answer)
      question = question_generator(context)
      question_list.append({
          "question": question[0]["generated_text"],
          "answer": answer,
          "context": temp
      })
    return question_list

