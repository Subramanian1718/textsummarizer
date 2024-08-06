import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_sentence(sentence):
    words = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def summarize_text(text, summary_length):
    sentences = sent_tokenize(text)
    processed_sentences = [preprocess_sentence(sentence) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    sentence_scores = tfidf_matrix.mean(axis=1).flatten().tolist()[0]
    ranked_sentences = [(score, index) for index, score in enumerate(sentence_scores)]
    ranked_sentences.sort(reverse=True, key=lambda x: x[0])
    selected_sentences_indices = [ranked_sentences[i][1] for i in range(summary_length)]
    selected_sentences_indices.sort()
    summary = ' '.join([sentences[index] for index in selected_sentences_indices])
    return summary

text = input("Enter the text to be summarized: ")
summary_length = int(input("Enter the number of sentences for the summary: "))
summary = summarize_text(text, summary_length)
print("\nSummary:")
print(summary)
