
import streamlit as st  
import pickle
import time
import requests  # to download the model from Dropbox
from io import BytesIO
from textblob import TextBlob
import pandas as pd
import altair as alt  # for visualization
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # for Token Sentiment

# function to convert the sentiment result into a dataframe to visualize 
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# function to find the Token Sentiment
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

# function to download and load model from Dropbox
def download_model_from_dropbox(dropbox_url):
    try:
        # Download the model from Dropbox
        res = requests.get(dropbox_url, stream=True)
        res.raise_for_status()  # Raise an error if the download fails
        
        # Load the model into memory
        model_file = BytesIO()
        for chunk in res.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                model_file.write(chunk)
        
        model_file.seek(0)  # Go back to the start of the file
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error downloading or loading the model: {e}")
        return None

def main():
    # Dropbox link to the model file
    dropbox_url = 'https://www.dropbox.com/scl/fi/fqbltfd6y3qqxiej7c32v/twitter_sentiment.pkl?rlkey=agbisk8250r179pgbqvuppfrz&st=zhcczrw6&dl=1'
    
    # Load the model from Dropbox
    model = download_model_from_dropbox(dropbox_url)

    if model is None:
        st.stop()  # Stop execution if model isn't loaded

    st.title("Twitter Sentiment Analysis")
    
    raw_text = st.text_area("Enter Text Here")
    st.write(f"You entered: {raw_text}")
    
    submit_button = st.button('Analyze')

    # layout
    col1, col2 = st.columns(2)

    if submit_button:
        start = time.time()
        with col1:
            st.info("Results")

            # using trained Random Forest Classifier Model
            st.markdown("<p style='font-size:20px; color: green;'>Prediction using RFC model:</p>", unsafe_allow_html=True)
            
            prediction = model.predict([raw_text])
            end = time.time()
            st.write('Prediction time taken: ', round(end - start, 2), 'seconds')

            st.write("Sentiment:", prediction[0])

            # using TextBlob and VADER Sentiment
            st.markdown("<p style='font-size:20px; color: green;'>Prediction using TextBlob Library:</p>", unsafe_allow_html=True)
            
            sentiment = TextBlob(raw_text).sentiment
            st.write(sentiment)

            # adding emojis
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :smiley:")
            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :angry:")
            else:
                st.markdown("Sentiment:: Neutral 😐")

            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric'
            )
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Token Sentiment")
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)

if __name__ == '__main__':
    main()
