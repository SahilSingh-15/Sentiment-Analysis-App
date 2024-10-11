import streamlit as st  
import pickle
import time
from textblob import TextBlob
import pandas as pd
import altair as alt# for visualization
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #for Token Sentiment



#function to convert the sentiment result into a dataframe to visualize 
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

#function to find the Token Sentiment
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

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

def main():
    # load the model
    model = pickle.load(open('model/twitter_sentiment.pkl', 'rb'))
    
    st.title("Sentiment Analysis App")
    
    raw_text = st.text_area("Enter Text Here")
    st.write(f"You entered: {raw_text}")
    
    submit_button = st.button('Analyze')

    #layout
    col1,col2 = st.columns(2)

    if submit_button:
        start = time.time()
        with col1:
            
            st.info("Results")

            #using trained Random Forrest Classifier Model
            #st.write("Prediction using RFC model")
            st.markdown(
    "<p style='font-size:20px; color: green;'>Prediction using RFC model:</p>", 
    unsafe_allow_html=True
)
            prediction = model.predict([raw_text])
            end = time.time()
            st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

            #print(prediction[0])
            st.write("Sentiment:",prediction[0])


            #using TextBlob and VadarSentiment
            # st.write("Prediction using TextBlob Library")
            st.markdown(
    "<p style='font-size:20px; color: green;'>Prediction using TextBlob Library:</p>", 
    unsafe_allow_html=True
)

            
            sentiment=TextBlob(raw_text).sentiment
            st.write(sentiment)

            #adding emojies
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :smiley: ")

            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :angry: ")
            
            else:
                st.markdown("Sentiment:: Neutral 😐 ")

            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)    

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
            st.altair_chart(c,use_container_width=True)
            


        with col2:
            st.info("Token Sentiment")
            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)









if __name__ == '__main__':
    main()
