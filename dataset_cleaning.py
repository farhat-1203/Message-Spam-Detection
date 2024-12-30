# importing required libraries
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')

# reading the dataset
msg = pd.read_csv(r"dataset.csv", encoding='latin-1')
msg.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
msg.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

# mapping ham=0 and spam=1
msg['label'] = msg['label'].map({'ham': 0, 'spam': 1})

# dropping duplicate columns
msg = msg.drop_duplicates()

# data cleaning/preprocessing - removing punctuation and digits
def clean_text(text):
    return ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

msg['cleaned_text'] = msg['text'].apply(clean_text)
msg.drop(['text'], axis=1, inplace=True)

# tokenization and convert to lower case
msg['token'] = msg['cleaned_text'].apply(lambda text: re.split(r"\W+", text.lower()))

# stopwords removal
stopwords_list = stopwords.words('english')
msg['updated_token'] = msg['token'].apply(lambda tokens: [word for word in tokens if word not in stopwords_list])
msg.drop(['token'], axis=1, inplace=True)

# lemmatization
wordlem = WordNetLemmatizer()
msg['lem_text'] = msg['updated_token'].apply(lambda tokens: [wordlem.lemmatize(token) for token in tokens])
msg.drop(['updated_token'], axis=1, inplace=True)

# merging tokens into a string
msg['final_text'] = msg['lem_text'].apply(lambda tokens: " ".join(tokens))
msg.drop(['cleaned_text', 'lem_text'], axis=1, inplace=True)

# saving the cleaned dataset
msg.to_csv('Cleaned_Dataset.csv', index=False)
