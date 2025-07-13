import pandas as pd
import string

df=pd.read_csv("IMDB.csv")
# displaying sample data
print("sample data :")
print(df.head())
#counting no of positive and negative reviews
print("\n Sentiment Counts :")
print(df['sentiment'].value_counts())

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
print(df[['review', 'cleaned']].head())

# convertion to list
ls = set()  # for unique words

# Building vocabulary
for token in df['cleaned']:
    for word in token:
        ls.add(word)

ls = list(ls)  # Convert set to list
word_index = {word: i for i, word in enumerate(ls)}  # Dict: word and index

print(f"Total vocabulary size: {len(ls)}")
print("Sample word_to_index:", list(word_index.items())[:5])

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
split_index = int(0.9 * len(df))
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

# calculating accuracy
correct = 0
total = len(test)
TP = TN = FP = FN = 0 # TP-true positive, TN-true negative, FN-false negative, FP-false positive
for i, row in test.iterrows():
    predicted = predict(row['cleaned'])
    actual = row['label']
    if predicted == 1 and actual == 1:
        TP += 1
        correct+=1
    elif predicted == 0 and actual == 0:
        TN += 1
        correct+=1
    elif predicted == 1 and actual == 0:
        FP += 1
    elif predicted == 0 and actual == 1:
        FN += 1
    if i % 1000 == 0:
        print(f"Processed {i} test reviews")

# calculating  precision, recall, F1 and accuracy
precision = TP/(TP + FP) if (TP + FP) != 0 else 0
recall = TP/(TP + FN) if (TP + FN) != 0 else 0
f1_score = 2*(precision * recall)/(precision + recall) if (precision + recall) != 0 else 0
accuracy = correct / total
print(f"\n Training Accuracy: {accuracy * 100:.4f}%")
print(f"\n Precision: {precision * 100:.4f}")
print(f" Recall: {recall * 100:.4f}")
print(f" F1 Score: {f1_score * 100:.4f}")

'''
s=input("please enter your review:")
if predict(preprocessing(s)) == 1:
    print("positive")
else:
    print("negative")
'''