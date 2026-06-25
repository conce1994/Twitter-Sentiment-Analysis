### Twitter Hate Speech and Sentiment Analysis Using Multiple Machine Learning Models

This project focused on detecting hateful and offensive content on the Twitter platform, including messages related to hate speech, sexism, and abusive language. For this purpose, a dataset obtained from Kaggle was used. The dataset consisted of two files: **train** and **test** datasets.

#### 1. Importing Libraries

Initially, several Python libraries were imported to support data processing, visualization, and model implementation:

* **NumPy**: Used for numerical operations and handling arrays efficiently.
* **Pandas**: Used to load, manipulate, and analyse structured datasets.
* **Matplotlib** and **Seaborn**: Used to create graphs and visualise patterns and distributions within the dataset.

#### 2. Reading the Dataset

The train and test files were loaded into Python using Pandas. Each dataset contained tweets and their associated labels indicating whether the message represented hate speech, sexism, offensive language, or normal content.

After loading the datasets, the labels for both training and testing datasets were inspected to understand class distribution and verify the dataset structure.

#### 3. Combining the Datasets

Both datasets were combined into a single dataset before preprocessing. Combining the datasets ensured that identical cleaning and transformation steps were applied consistently across all tweets.

#### 4. Data Cleaning and Preprocessing

Text preprocessing was performed to improve data quality and reduce noise.

The following cleaning operations were applied:

* Removal of **special characters and numbers** because they generally do not contribute meaningful semantic information.
* Removal of **hashtags (#)** to eliminate unnecessary symbols while preserving the text content.
* Removal of **short words (less than two characters)** because these often provide limited contextual information.
* Removal of **stopwords**, which are common words such as *the*, *is*, and *and* that usually carry little predictive value.
* Removal of **frequently occurring words** to reduce bias caused by overrepresented terms.

These preprocessing steps helped improve the quality of the input data for machine learning models.

#### 5. Tokenisation Using spaCy

After cleaning, **tokenisation** was performed using spaCy.

Tokenisation is the process of dividing text into smaller units called **tokens**. A token is usually a word, punctuation mark, or meaningful text element.

For example:

Original tweet:

> "Women should not work"

Tokens:

> ["Women", "should", "not", "work"]

Using spaCy:

```python
document = nlp(tweet)
tokens = [token.text for token in document]
```

Tokenisation allows the algorithm to process text in a structured format instead of raw sentences.

#### 6. Generating Document Embeddings Using spaCy

The spaCy model **en_core_web_lg** was downloaded and loaded.

```python
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
```

This model contains pre-trained word vectors.

Each tweet was converted into a **document embedding** using:

```python
document = nlp(tweets[0])
```

Document embeddings transform an entire tweet into a numerical vector representation.

A document embedding captures the semantic meaning of the sentence and represents it as hundreds of numerical features.

Example:

Tweet:

> "I love this product"

Embedding:

> [0.42, -0.15, 0.88, ...]

These vectors allow machine learning algorithms to understand relationships between words and sentence meaning.

#### 7. Token-to-Vector Conversion

After tokenisation, each token was converted into a numerical vector.

Token vectors represent individual words numerically.

Example:

Word:

> "happy"

Vector:

> [0.12, 0.44, -0.21, ...]

Words with similar meanings produce similar vectors.

This process converts textual data into mathematical features suitable for machine learning algorithms.

#### 8. Sentence-to-Vector Conversion Using `pipe`

To process multiple tweets efficiently, spaCy's `pipe()` method was used.

```python
documents = list(nlp.pipe(tweets))
```

Instead of processing tweets one by one, `pipe()` processes batches of tweets simultaneously.

Each sentence was converted into a document vector:

```python
vectors = [doc.vector for doc in documents]
```

This produced a numerical representation for every tweet.

#### 9. Defining Features (X) and Labels (Y)

The dataset was divided into:

* **X (features):** Numerical vectors generated from tweets.
* **Y (target labels):** Classification labels indicating hate speech categories.

Example:

```python
X = tweet_vectors
Y = labels
```

#### 10. Train–Validation Split

The dataset was split into training and validation sets.

* Training set → used to train the models.
* Validation set → used to evaluate performance on unseen data.

Example:

```python
X_train, X_val, y_train, y_val
```

#### 11. Prediction and Confusion Matrix

After training, predictions were generated on the validation dataset.

A **confusion matrix** was then used to evaluate classification performance.

The confusion matrix contains:

* **True Positives (TP):** Correctly identified hateful tweets.
* **False Positives (FP):** Non-hateful tweets incorrectly classified as hateful.
* **True Negatives (TN):** Correctly identified non-hateful tweets.
* **False Negatives (FN):** Hateful tweets incorrectly classified as non-hateful.

#### 12. Logistic Regression

Logistic Regression was used as a baseline classification model.

This algorithm estimates probabilities and assigns tweets to categories using a decision boundary.

Results:

* Validation Accuracy: **94.06%**
* True Positives: **269**
* False Positives: **166**
* True Negatives: **8750**
* False Negatives: **404**

#### 13. Support Vector Classifier (SVC)

A **Support Vector Classifier (SVC)** was implemented.

SVC works by identifying an optimal hyperplane that maximises separation between classes.

The model attempts to create the largest possible margin between hateful and non-hateful tweets.

Results:

* Validation Accuracy: **94.44%**
* True Positives: **221**
* False Positives: **81**
* True Negatives: **8835**
* False Negatives: **452**

SVC reduced false positives compared with Logistic Regression but increased false negatives.

#### 14. Neural Network Implementation Using Keras

A Neural Network was implemented using **Keras**.

The neural network learns complex relationships between tweet vectors and labels through multiple interconnected layers.

Typical architecture:

* Input layer → receives tweet vectors
* Hidden layers → extract patterns
* Output layer → predicts class probabilities

The model updates weights during training using backpropagation to minimise prediction error.

Results:

* Validation Accuracy: **96.09%**
* True Positives: **388**
* False Positives: **90**
* True Negatives: **8826**
* False Negatives: **285**

The Neural Network achieved the highest overall performance, producing the highest validation accuracy and the lowest number of false negatives among all evaluated models.

#### 15. Conclusion

Three machine learning approaches were evaluated for hate speech detection on Twitter. While Logistic Regression and Support Vector Classifier provided strong baseline performance, the Neural Network implemented using Keras achieved the best overall results, reaching **96.09% validation accuracy** and demonstrating superior capability in identifying hateful content while reducing missed detections.
