import json
import os
import csv
import requests
from textblob import TextBlob
from datetime import datetime
import time


# Initialize any necessary variables or data structures
session_data = {
    "historic_price_data_file": 'data/price/AAPL/AAPL.txt',
    "tweets_directory": 'data/tweet/AAPL',
    "results_file": 'strategy_2_predictions_llama-3-sauerkrautlm-70b-instruct.csv',
    "history_length": 6,
    "previous_close": None,
    "last_processed_date": None
}

def get_prediction(response):
    # Given byte string
    byte_string = response

    # Decode the byte string to a regular string
    decoded_string = byte_string.decode('utf-8')

    # Find the prediction
    # We know the prediction follows the text "Prediction: "
    prediction_prefix = "Prediction: "
    prediction = decoded_string.split(prediction_prefix)[-1].strip('.')

    # Print the prediction
    return prediction[0:4]

def get_context(response):
    # Given byte string
    byte_string = response

    # Decode the byte string to a regular string
    decoded_string = byte_string.decode('utf-8')

    # Find the prediction
    # We know the prediction follows the text "Prediction: "
    context_prefix = "Context:"
    context = decoded_string.split(context_prefix)[-1].strip('.')
    print(context)

    # Print the prediction
    return context


# Function to read historic price data from a file
def read_price_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    processed_data = preprocess_price_data(data)
    processed_data.reverse()  # Reverse the list to make it chronological
    return processed_data

# Function to read tweets from a file for a specific date
def read_tweets(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return preprocess_tweets(data)

# Function to preprocess historic price datax
def preprocess_price_data(data):
    lines = data.strip().split('\n')
    processed_data = []
    for line in lines:
        parts = line.split('\t')
        date = parts[0].strip()
        open_, high, low, close, adj_close, volume = map(float, parts[1:])
        processed_data.append({
            "Date": date,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume
        })
    return processed_data

# Function to preprocess tweets
def preprocess_tweets(data):
    tweets = data.strip().split('\n')
    processed_tweets = []
    for tweet in tweets:
        tweet_json = json.loads(tweet)
        text = ' '.join([word for word in tweet_json["text"] if not word.startswith(('http', '@', 'RT'))])
        processed_tweets.append({
            "Text": text,
            "Timestamp": tweet_json["created_at"]
        })
    return processed_tweets


# Function to generate the initial prompt for historic data analysis
def generate_initial_prompt(price_data, tweets):
    prompt = """
    We will follow an incremental learning strategy. Starting from the next prompt, I will provide you only with the data for the next trading day, which will be the previous trading day relative to the day for which you have to predict. You must maintain the context of all previous data and make predictions accordingly.

    Here is the historic price data for the last few trading days and recent tweets. Please analyze the data to predict stock price movement (up or down) for the next day. Perform the following analyses:

    1. Technical Analysis and Market Trend Assessment:
    - Examine the moving averages, volatility, volume trends, price momentum, and price deviation from moving averages to identify market trends and potential price movements. Look for any technical patterns that might suggest future price movements, including uptrends, downtrends, or sideways trends.

    2. Sentiment and Context Analysis:
    - Assess the sentiment of the recent tweets and track changes in sentiment polarity (positive, negative, neutral) over time. Determine the overall sentiment and analyze whether the context of these sentiments is beneficial or detrimental to the company.

    Historic Price Data (Last few Days):
    """
    for entry in price_data[:session_data["history_length"]]:  # Take the last 30 days
        prompt += f"Date: {entry['Date']}, Open: {entry['Open']}, High: {entry['High']}, Low: {entry['Low']}, Close: {entry['Close']}, Adj Close: {entry['Adj Close']}, Volume: {entry['Volume']}\n"

    prompt += "\nRelevant Tweets:\n"
    for tweet in tweets:  # Take the last 10 tweets
        prompt += f"Text: {tweet['Text']}, Timestamp: {tweet['Timestamp']}\n"

    prompt += "\nBased on the combined analysis of technical indicators, market trends, sentiment/context, and provided data, predict whether the stock price will go up or down for the next day. Give the response in this format:\nPrediction: Rise/Fall. Also provide the updated context for the next prompt in this format:\nContext: updated context."
    return prompt

# Function to generate the daily update prompt with a reminder line
def generate_daily_update_prompt(entry, tweets, context):
    prompt = f"""
    Reminder: We are following an incremental learning strategy and using the technical analysis and sentiment analysis for stock price prediction. Utilize the provided context from the previous prompt response, the last trading day historic data and associated tweets to predict whether the stock price will rise or fall for the next day. After making your prediction, update the maintained context with the information from today's data. This approach continues with each subsequent prompt.

    Context from previous prompt response: {context}

    Historic Price Data (Last Day):
    """
    prompt += f"Date: {entry['Date']}, Open: {entry['Open']}, High: {entry['High']}, Low: {entry['Low']}, Close: {entry['Close']}, Adj Close: {entry['Adj Close']}, Volume: {entry['Volume']}\n"

    prompt += "\nRelevant Tweets:\n"
    for tweet in tweets:
        prompt += f"Text: {tweet['Text']}, Timestamp: {tweet['Timestamp']}\n"

    prompt += "\nBased on the combined analysis of technical indicators, market trends, sentiment/context, and provided data, predict whether the stock price will go up or down for the next day. Give the response in this format:\nPrediction: Rise/Fall\nContext: updated context"
    return prompt

# Function to send request to GWDG AI Chatbot
def send_request(prompt):
    response = requests.post(
        "https://chat-ai.academiccloud.de/chat-ai-backend",
        headers={
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8,hi;q=0.7",
            "Connection": "keep-alive",
            "Content-type": "application/json",
            "Cookie": "mod_auth_openidc_session=567b6885-573b-4b3d-b0ce-9c3d9454ff91",
            "Host": "chat-ai.academiccloud.de",
            "Origin": "https://chat-ai.academiccloud.de",
            "Referer": "https://chat-ai.academiccloud.de/chat",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "macOS"
        },
        json={
            "model": "llama-3-sauerkrautlm-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a trading expert and market analyst"
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            "temperature": 0
        }
    )
    prediction = get_prediction(response.content)
    context = get_context(response.content)
    return prediction, context

# Function to calculate actual movement
def calculate_actual_movement(prev_close, curr_close):
    if float(curr_close) > float(prev_close):
        return 'Rise'
    else:
        return 'Fall'

# Function to process daily updates
def process_daily_update(date, next_close, entry, context):
    # Read and preprocess new day data
    new_day_file = f'{session_data["tweets_directory"]}/{date}'
    processed_new_day_data = read_price_data(session_data["historic_price_data_file"])

    current_close = entry['Close']

    start = datetime.strptime(session_data["last_processed_date"], "%Y-%m-%d")
    end = datetime.strptime(date, "%Y-%m-%d")

    tweets = []

    # Ensure start date is less than or equal to end date
    if start > end:
        # List to hold all dates
        date_list = []

        # Iterate from start to end date
        current_date = start
        while current_date <= end:
            try:
                tweets.extend(read_tweets(f'{session_data["tweets_directory"]}/{current_date.strftime("%Y-%m-%d")}'))
            except Exception:
                pass

    # Generate daily update prompt with reminder
    daily_prompt = generate_daily_update_prompt(entry, tweets, context)

    # Send daily update request
    prediction, context = send_request(daily_prompt)

    # Store actual movement and predicted result in CSV
    actual_movement = calculate_actual_movement(current_close, next_close)
    with open(session_data["results_file"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([date, actual_movement, prediction])# daily_response.split(": ")[1]])
    return context


# Main function to iterate through all dates from historic data file
def main():
    # Read first 30 days of historic price data and corresponding tweets for initial prompt
    price_data = read_price_data(session_data["historic_price_data_file"])
    tweets = []
    for entry in price_data[:session_data["history_length"]]:  # Take the last 30 days
        # Fetch tweets for each date within the last 30 days
        tweet_date = entry['Date']
        tweet_file_path = f'{session_data["tweets_directory"]}/{tweet_date}'
        try:
            day_tweets = read_tweets(tweet_file_path)
            tweets.extend(day_tweets)
        except Exception as e:

            # skip if not found
            pass
    initial_prompt = generate_initial_prompt(price_data, tweets)

    # print(initial_prompt)

    # Send initial prompt to OpenAI for training (optional step if incremental learning is supported)
    prediction, context = send_request(initial_prompt)

    # Initialize results file with headers
    with open(session_data["results_file"], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Actual Movement', 'Predicted Movement'])

    # update in csv for the actual and predicted
    next_close = price_data[session_data["history_length"]+1]['Close']
    current_close = price_data[session_data["history_length"]]['Close']

    session_data["last_processed_date"] = price_data[session_data["history_length"]]['Date']

    actual_movement = calculate_actual_movement(current_close, next_close)
    with open(session_data["results_file"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([price_data[session_data["history_length"]]['Date'], actual_movement, prediction])# daily_response.split(": ")[1]])

    count = 1

    # Process each date from the historic price data file
    for i, entry in enumerate(price_data[session_data["history_length"]+1:]):  # Start from the 31st day
        count += 1
        if count == 10:
            count = 0
            time.sleep(60)
        if len(price_data) <= session_data["history_length"] + i + 2:
            break
        next_close = price_data[session_data["history_length"]+i+2]['Close']
        context = process_daily_update(entry['Date'], next_close, entry, context)
        session_data["last_processed_date"] = price_data[session_data["history_length"]+i+1]['Date']
        # Update prev_close for the next iteration

# Execute main function
if __name__ == "__main__":
    main()


