# import text preprocessing libraries and timing library
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
ps = PorterStemmer()
import spacy
nlp_spacy = spacy.load('en_core_web_sm')
nlp_spacy.add_pipe(nlp_spacy.create_pipe('sentencizer'))
import time

# Obtain a publicly available text corpus
with open("51060", encoding="utf8", errors='ignore') as infile:
    # Remove double spaces, line breaks, and slashes
    corpus = infile.read().replace("  ", " ").replace("\n", " ").replace("\'", "")

#### NLTK
# nltk tokenization
start = time.time()
tokens_nltk = sent_tokenize(corpus)
tokens_nltk_time = time.time() - start
# nltk word tokenization
start = time.time()
words_nltk = word_tokenize(corpus)
words_nltk_time = time.time() - start
# nltk stemming
start = time.time()
stem_nltk = [ps.stem(word) for word in words_nltk]
stem_nltk_time = time.time() - start
# nltk pos tagging
start = time.time()
pos_nltk = pos_tag(words_nltk)
pos_nltk_time = time.time() - start

#### SPACY
doc = nlp_spacy(corpus)
# spacy tokenization
start = time.time()
tokens_spacy = [sent.string.strip() for sent in doc.sents]
tokens_spacy_time = time.time() - start
# spacy word tokenization
start = time.time()
words_spacy = [token.text for token in doc]
words_spacy_time = time.time() - start
# spacy stemming
# spacy does not contain stemming functionality
# https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/
# spacy pos tagging
start = time.time()
pos_spacy = [(token.text.strip(), token.pos_) for token in doc]
pos_spacy_time = time.time() - start


# Print the results
print("nltk times:")
print("- nltk sentence tokenization:  ", tokens_nltk_time)
print("- nltk word tokenization:      ", words_nltk_time)
print("- nltk stemming:               ", stem_nltk_time)
print("- nltk pos tagging:            ", pos_nltk_time)
print("spacy times:")
print("- spacy sentence tokenization: ", tokens_spacy_time)
print("- spacy word tokenization:     ", words_spacy_time)
print("- spacy stemming:              ", "N/A")
print("- spacy pos tagging:           ", pos_spacy_time)




