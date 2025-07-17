from flask import Flask, render_template, request

app= Flask(__name__)

import gdown

url ="https://drive.google.com/file/d/1KOa-DYaLphPMb3fYEFF81NoWx82h5Aqa/view?usp=drive_link"
gdown.download(url, "imdb.csv", quiet=False)

import pandas as pd
import string

df=pd.read_csv("imdb.csv",header=0)

# reshuffling
df=df.sample(frac=1).reset_index(drop=True)

# text preprocessing
stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "don", "should", "now"
])
def preprocessing(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return tokens
# Apply preprocessing
df['cleaned'] = df['review'].apply(preprocessing)
# convertion to list
ls = set()  # for unique words

# Building vocabulary
for token in df['cleaned']:
    for word in token:
        ls.add(word)

ls = list(ls)  # Convert set to list
word_index = {word: i for i, word in enumerate(ls)}  # Dict: word and index

# converting positive to 1 and negative to 0 ,for easy access
def to_ind(token):
    if(token=="positive"):
        return 1
    else:
        return 0
def convert_ind(tokens):
    return to_ind(tokens)

df['label']=df['sentiment'].apply(convert_ind)

# training and testing part
split_index = int(1 * len(df))
train = df[:split_index]
test = df[split_index:]

from collections import defaultdict # its give 0 as default value to keys that are not present (basically avoiding key errors)

word_count_pos=defaultdict(int)
word_count_neg=defaultdict(int)

total_pos_words=0
total_neg_words=0

for _, row in train.iterrows():
    if row['label'] == 1:  # if review is positive
        for word in row['cleaned']:
            word_count_pos[word] += 1
            total_pos_words += 1
    else:                  # if review isnegative
        for word in row['cleaned']:
            word_count_neg[word] += 1
            total_neg_words += 1
import math

def predict(tokens):
    log_prob_pos=math.log(len(train[train['label'] == 1])/len(train))  #used log to make computation easier
    log_prob_neg=math.log(len(train[train['label'] == 0])/len(train))

    for word in tokens:
        # Adding Laplace smoothing (to avoid 0 probability)
        count_pos = word_count_pos.get(word, 0) + 1
        count_neg = word_count_neg.get(word, 0) + 1

        # Calculating P(word|class)
        prob_word_pos = count_pos / (total_pos_words + len(word_index))
        prob_word_neg = count_neg / (total_neg_words + len(word_index))

        # Adding to log probabilities
        log_prob_pos += math.log(prob_word_pos)
        log_prob_neg += math.log(prob_word_neg)

    #  prediction
    return 1 if log_prob_pos > log_prob_neg else 0


def predict_review(review):
    if predict(preprocessing(review)) == 1:
        return "Positive"
    else:
        return "Negative"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        review = request.form["review"]
        prediction = predict_review(review)
        return render_template("project.html", prediction=prediction)
    return render_template("project.html")

if __name__ == "__main__":
    app.run(debug=True)





