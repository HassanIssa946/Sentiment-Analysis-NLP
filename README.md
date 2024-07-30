# Sentiment Analysis NLP

This repository contains a sentiment analysis project using VADER (Valence Aware Dictionary and sEntiment Reasoner) and a pretrained RoBERTa model. The project demonstrates data reading, cleaning, sentiment analysis, and visualization.

## Project Overview

1. **Data Reading and Cleaning**: Load and preprocess the dataset containing textual reviews.
2. **Exploratory Data Analysis (EDA)**: Visualize the distribution of review scores.
3. **Sentiment Analysis with VADER**: Perform sentiment analysis using VADER and analyze the results.
4. **Named Entity Recognition (NER)**: Use NLTK for tokenization, part-of-speech tagging, and named entity recognition.
5. **Sentiment Analysis with RoBERTa**: Utilize a pretrained RoBERTa model for sentiment analysis and compare the results with VADER.
6. **Results Visualization**: Visualize the sentiment scores using Seaborn and Matplotlib.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Sentiment-Analysis-NLP.git
    ```
2. Change into the project directory:
    ```bash
    cd Sentiment-Analysis-NLP
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Read and Clean Data**: Load the dataset and clean the data.
    ```python
    df = pd.read_csv("roberta_scores.csv")
    print(df.shape)
    df.head()
    ```

2. **Exploratory Data Analysis**: Plot the distribution of review scores.
    ```python
    ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews', figsize=(10, 5))
    ax.set_xlabel('Review Stars')
    plt.show()
    ```

3. **Sentiment Analysis with VADER**: Perform sentiment analysis and visualize the results.
    ```python
    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)
    vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
    df_vaders = df.merge(vaders, how="right")
    ```

4. **Named Entity Recognition**: Use NLTK for tokenization and named entity recognition.
    ```python
    tokens = nltk.word_tokenize(example)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    entities.pprint()
    ```

5. **Sentiment Analysis with RoBERTa**: Use a pretrained RoBERTa model for sentiment analysis.
    ```python
    Model = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(Model)
    model = AutoModelForSequenceClassification.from_pretrained(Model)

    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        return scores_dict
    ```

6. **Results Visualization**: Visualize sentiment scores using Seaborn and Matplotlib.
    ```python
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    sns.barplot(data=df_vaders, x="Score", y="pos", ax=axs[0])
    sns.barplot(data=df_vaders, x="Score", y="neg", ax=axs[1])
    sns.barplot(data=df_vaders, x="Score", y="neu", ax=axs[2])
    axs[0].set_title("Positive")
    axs[1].set_title("Negative")
    axs[2].set_title("Neutral")
    plt.show()
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The `nltk` library for natural language processing tools.
- The `transformers` library by Hugging Face for the pretrained RoBERTa model.
- The `seaborn` and `matplotlib` libraries for data visualization.
- The `tqdm` library for progress bars.
- All other contributors and resources.

