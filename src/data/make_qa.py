
# import requests
# import base64
# from dotenv import dotenv_values

# import openai


# # Configuration
 
# DEPLOYMENT = "gpt-4"
# IMAGE_PATH = "/gscratch/bdata/mikeam/TSandLanguage/IMG_2907.jpeg"
# GPT4V_ENDPOINT = "https://bdata-gpt-vision.openai.azure.com/openai/deployments/gpt-vision/chat/completions?api-version=2023-07-01-preview"


# # Configuration
# GPT4V_KEY = dotenv_values(".env")["GPT4V_KEY"]
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
# headers = {
#     "Content-Type": "application/json",
#     "api-key": GPT4V_KEY,
# }

# # Payload for the request
# payload = {
#   "messages": [
#     {
#       "role": "system",
#       "content": [
#         {
#           "type": "text",
#           "text": "You are a helpful assistant."
#         }
#       ]
#     },
#     {
#             "role": "user", 
#             "content": [
# 	            {
# 	                "type": "text",
# 	                "text": "Describe this picture:"
# 	            },
# 	            {
# 	                "type": "image_url",
# 	                "image_url": {
#                         "url": encoded_image,
#                     }
#                 } 
#            ] 
#         }
#   ],
#   "temperature": 0.7,
#   "top_p": 0.95,
#   "max_tokens": 4000
# }

# # Send request
# try:
#     response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
#     response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
# except requests.RequestException as e:
#     raise SystemExit(f"Failed to make the request. Error: {e}")

# # Handle the response as needed (e.g., print or process)


import os
from dotenv import dotenv_values
import requests
import base64
import base64
from PIL import Image
import io
GPT4V_KEY = dotenv_values(".env")["GPT4V_KEY"]

# Configuration
IMAGE_PATH = "/gscratch/bdata/mikeam/TSandLanguage/image_720.png"
TEST_IMAGE_URL="https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?cs=srgb&dl=pexels-chevanon-photography-1108099.jpg&fm=jpg"
GPT4V_ENDPOINT = "https://bdata-gpt-vision.openai.azure.com/openai/deployments/gpt-vision/chat/completions?api-version=2023-12-01-preview"
HEADERS = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

DEFAULT_PROMPT = """\

Here is the code that was used to generate a time series and a picture of its visualization:

Characteristics:
1. The number of tickets sold would be high during the initial days after the movie's premiere as moviegoers rush to watch it first.
2. The sales will then gradually decline over the following months.
3. There would be a sudden spike in the number of tickets sold around the launch date of the movie's sequel, due to renewed public interest.
4. Weekend days will generally have higher ticket sales compared to weekdays due to people having more leisure time.
5. All values would be non-negative as ticket sales cannot be negative.

Code:
```python
import numpy as np
import pandas as pd

def generate_series():
    base_sales = np.random.normal(loc=5000, scale=500, size=180)
    base_sales = np.maximum(0, base_sales)  # Ensures ticket sales cannot be negative
    ticket_sales = pd.Series(base_sales)

    decay_factor = np.linspace(1, 0.4, 180) # Decay factor to simulate decrease in interest over time
    ticket_sales *= decay_factor

    sequel_release_day = 90  # Sequel releases on 90th day causing spike in sales
    sequel_spike = np.random.normal(loc=2000, scale=200, size=30)
    ticket_sales[sequel_release_day : sequel_release_day+30] += sequel_spike

    weekend_factor = 1.2  # Assumes 20% more sales on weekends
    ticket_sales[ticket_sales.index % 7 <= 1] *= weekend_factor

    return ticket_sales.to_numpy()
```

Here are examples of five types of questions that were generated for another, different time series.
I've included the code that corresponds to this series to help with the example.

Characteristics:
1. The number of calls is expected to be relatively stable before the product launch.
2. There is a significant increase in the number of calls after the product launch.
3. The increase might settle to a new normal level after a few days of the launch.
4. There could be possible weekly patterns where weekdays have more calls than weekends.
5. Outliers can be observed on days with extraordinarily high call volumes due to specific customer issues or product bugs.

Code:
```python
import numpy as np
import random
from scipy.signal import savgol_filter

def generate_series():
    # Base level calls (before product launch)
    base_calls = np.random.poisson(200, 14)

    # Spike due to product launch + gradual fall-off/burn-in
    spike = [np.random.poisson(x) for x in np.linspace(500, 200, 16)]

    # Putting both together
    calls = np.concatenate((base_calls, spike))

    # Adding weekly effect (weekdays are busier)
    for i in range(len(calls)):
        if i % 7 < 5:  # Make Mon - Fri busier
            calls[i] += np.random.poisson(50)
        else:  # Make weekends quieter
            calls[i] -= np.random.poisson(50)

    # Adding a few random spikes/outliers
    for _ in range(10):
        calls[random.randint(0, 29)] += np.random.poisson(100)

    # Smooth the values to make it more realistic
    calls = savgol_filter(calls, 5, 3)

    return calls
```


*** Start of expected output ****

Thinking step by step, here are some questions that I would ask about this time series:

"Counterfactual Reasoning":
Question:How would the total number of calls in this month change if the product launch was delayed by three days?
To answer this question, I would look at the time series and see how the number of calls changed when the product was launched.
Final answer: The total number of calls would not change by much, since the peak introduced by the shift would still be within the one month window

"Explanation":
Question: What could have led the company to launch the product when it did?
Looking at the time series, it seems like the product was launched in the middle of the month. It's possible
that previous research found that middle of the month launches performed best. 
Final answer: It's hard to say for sure, but one explanation is that a market survey found that products launched in the middle of the month performed best. The company might have also decided to launch in January because many businesses are closed for the December holidays, and so deals would have moved slower.

"Argumentation":
Question: Why might I not need to hire more call center staff?
Let's look at the trend of the number of calls. The number of calls has returned to the baseline. This suggests that the product launch did not lead to a permanent increase in call volume.
Final answer: One reason to not hire more staff is that the call volume seems to have returned to the baseline.

"Analogical Reasoning":
Question: What are some other product launches that were similar?
I'll start by describing the general trend. The number of calls was relatively low before the product launch, and then spiked on the day of the launch. The number of calls then gradually decreased over the next few weeks. This is similar to the launch of Healthcare.gov in 2013, which was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues.
Final answer: Healthcare.gov: The initial launch of the U.S. government's health insurance marketplace website in 2013 was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues.

"Fact Checking":
Question: Tell me if the following statement is true, and explain the reasoning behind your answer: The product launch led to a 10x increase in call volume.
From the image, I can tell that the number of sales peakjed at ~600 and prior to that peak had a mean of ~200 calls per day. This means that the statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.
Final answer: The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.


```json
{
  "Counterfactual Reasoning": {
    "Question": "How would the total number of calls in this month change if the product launch was delayed by three days?",
    "Answer": "The total number of calls would not change by much, since the peak introduced by the shift would still be within the one month window"
  },
  "Explanation": {
    "Question": "What could have led the company to launch the product when it did?",
    "Answer": "It's hard to say for sure, but one explanation is that a market survey found that products launched in the middle of the month performed best. The company might have also decided to launch in January because many businesses are closed for the December holidays, and so deals would have moved slower."
  },
  "Argumentation": {
    "Question": "Why might I not need to hire more call center staff?",
    "Answer": "One reason to not hire more staff is that the call volume seems to have returned to the baseline."
  },
  "Analogical Reasoning": {
    "Question": "What are some other product launches that were similar?",
    "Answer": "Healthcare.gov: The initial launch of the U.S. government's health insurance marketplace website in 2013 was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues."
  },
  "Fact Checking": {
    "Question": "Tell me if the following statement is true, and explain the reasoning behind your answer: The product launch led to a 10x increase in call volume.",
    "Answer": "The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day."
  }
}
```
*** End of expected output ***

Can you please generate similar questions for the time series I provided? 
The following instructions are very important:

1. It is very important that the questions can't be answered without looking at the time series.
2. Your questions and answers should not explicitly reference the code that was used to generate the time series,
    or the picture of the time series. Simply integrate the information.
3. Be definitive about your answers and use the image and code to help answer them. For example,  the response:
    ""To verify the accuracy of this statement, one would need to examine the actual ticket sales data before and after the sequel release day to see if there was indeed a 100% increase from one day to the next.""
    is not as good as:
    ""The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.""
4. Make sure to output the results as a JSON file, as above 

"""


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG") # You can change JPEG to a different format if needed
        img_str = base64.b64encode(buffered.getvalue())
        return img_str
    


# Payload for the request
def get_payload(prompt, image_path):
    b64_image = image_to_base64(image_path)
    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are an AI assistant that helps people find information."
            }
        ]
        },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image.decode()}",
                        }
                    } 
            ] 
            }
    ],
    "temperature": 0.3,
    # "top_p": 0.95,
    "max_tokens": 4000
    }
    return payload

def response_to_text(response):
    return response.json()["choices"][0]["message"]["content"]

def get_response(prompt, image_path):
    payload = get_payload(prompt, image_path)

    try:
        response = requests.post(GPT4V_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    print(response.json())
    print(response_to_text(response))


if __name__ == "__main__":
    prompt = DEFAULT_PROMPT
    print(prompt)
    get_response(prompt, IMAGE_PATH)