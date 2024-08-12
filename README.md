a. How long did it take to solve the problem?

    The task of extracting text from links provided in an Excel sheet took approximately 10 hours. This was due to the challenges encountered, such as some links being broken or leading to web pages instead of PDFs. To handle the web pages, I used Beautiful Soup for data extraction. After processing, I was left with around 12,000 data points, which included duplicates and null values. Upon cleaning the data, I ended up with 915 usable data points. The subsequent preprocessing, including removing punctuation, special symbols, digits, tokenizing, and removing stopwords, was relatively straightforward and took less time.


b. Explain your solution:

    The solution was developed in several key steps:
        Data Extraction: I started by extracting text from the provided PDF URLs using PyMuPDF. For the links that redirected to web pages, I used Beautiful Soup to extract the relevant text data.
        Data Cleaning: After extraction, I cleaned the data by removing duplicates and handling null values, which left me with 915 data points.
        Text Preprocessing: I applied standard preprocessing techniques, such as removing punctuation, special symbols, digits, tokenizing the text, and removing stopwords to prepare the data for modeling.
        Text Vectorization: I vectorized the cleaned text using TF-IDF to convert the text data into numerical features that could be used for classification.
        Model Selection: I chose a Naive Bayes classifier for its simplicity and effectiveness in handling text classification tasks, especially with smaller datasets.
        Model Training and Prediction: The model was trained on the vectorized data, and predictions were made on the test set. Finally, I developed a Streamlit app to make the model accessible through a user-friendly interface.


c. Which model did you use and why?

I used a Naive Bayes classifier because it is simple, efficient, and generally performs well on text classification tasks, particularly with smaller datasets. Naive Bayes is also a good choice when the features (in this case, words) are assumed to be independent, which aligns well with the bag-of-words approach used in TF-IDF vectorization.

I also experimented with a Random Forest model. However, the data was not sufficient to generalize this more complex model, leading to overfitting. Despite the overfitting, the Random Forest model did produce significant results, but it wasn't as reliable as the Naive Bayes classifier for this specific dataset.

d. Any shortcomings and how can we improve the performance?

    The model’s performance could be affected by the limited size of the dataset (915 data points) and the simplicity of the model. The TF-IDF approach, while effective, may not capture more complex relationships in the text. To improve performance, we could:
        Expand the Dataset: Gathering more data points would help the model generalize better.
        Feature Engineering: Incorporating additional features, such as n-grams, part-of-speech tags, or domain-specific keywords, could improve classification.
        Model Upgradation: Experimenting with more sophisticated models like Random Forest, Support Vector Machines (SVM), or deep learning models like BERT could yield better performance.
        Hyperparameter Tuning: Fine-tuning the hyperparameters of the Naive Bayes model could also lead to performance improvements.
