Dataset - https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv

libraries need to download extra

!pip install nltk
nltk.download('punkt')
nltk.download('stopwords')

Preprocessing - Code Explanation:

1)Text Cleaning:
-Uses regular expressions to remove non-alphanumeric and non-numeric characters from the text.
-Utilizes the contractions library to expand contractions and replace them with their full forms.
-Removes numeric digits from the text.
-Replaces the string " s " with a blank space.
-Converts the entire text to lowercase.

2)Stopword Removal:
-Tokenizes the text using the nltk.word_tokenize function to split it into individual words.
-Filters out words from the text to remove those that are present in the set of stopwords (common words that don't contribute much meaning to the analysis).

3)Text Normalization:
-Utilizes a stemmer (in this case, the SnowballStemmer) to reduce words to their base or root form.
-Applies the stemmer to each word in the tokenized text.
-Joins the normalized words back together into a text string.

Feature Extraction - 
The code snippet defines a TfidfVectorizer object named Tf with the following configuration:

max_features: The maximum number of features (words or character n-grams) to consider in the vocabulary. In this case, it is set to 20,000.
ngram_range: Specifies the range of n-grams to generate. The value (1, 3) indicates that both unigrams (individual characters) and n-grams of length 2 and 3 (character sequences) will be included.
analyzer: Specifies whether the vectorizer should treat the input as words or characters. Here, 'char' indicates that character-level n-grams will be used.
