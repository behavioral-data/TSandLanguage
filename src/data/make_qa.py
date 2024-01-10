import os
from dotenv import dotenv_values
import requests
import base64
import base64
from PIL import Image
import io
from tqdm import tqdm
import json

from src.utils import read_jsonl

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
<CHARACTERISTICS>

Code:
<CODE>

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

"Counterfactual Reasoning": Imagining what the time series would look like if the setting were slightly different.
Question:How would the total number of calls in this month change if the product launch was delayed by three days?
To answer this question, I would look at the time series and see how the number of calls changed when the product was launched.
Final answer: The total number of calls would not change by much, since the peak introduced by the shift would still be within the one month window

"Explanation": Imagine a scenario that plausibly explains how the time series was generated.
Question: What could have led the company to launch the product when it did?
Looking at the time series, it seems like the product was launched in the middle of the month. It's possible
that previous research found that middle of the month launches performed best. 
Final answer: It's hard to say for sure, but one explanation is that a market survey found that products launched in the middle of the month performed best. The company might have also decided to launch in January because many businesses are closed for the December holidays, and so deals would have moved slower.

"Argumentation": Use properties of the time series to support an argument.
Question: Why might I not need to hire more call center staff?
Let's look at the trend of the number of calls. The number of calls has returned to the baseline. This suggests that the product launch did not lead to a permanent increase in call volume.
Final answer: One reason to not hire more staff is that the call volume seems to have returned to the baseline.

"Analogical Reasoning": How does this time series relate to others?
Question: What are some other product launches that were similar?
I'll start by describing the general trend. The number of calls was relatively low before the product launch, and then spiked on the day of the launch. The number of calls then gradually decreased over the next few weeks. This is similar to the launch of Healthcare.gov in 2013, which was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues.
Final answer: Healthcare.gov: The initial launch of the U.S. government's health insurance marketplace website in 2013 was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues.

"Fact Checking":
Question: Tell me if the following statement is true, and explain the reasoning behind your answer: The product launch led to a 10x increase in call volume.
From the image, I can tell that the number of sales peakjed at ~600 and prior to that peak had a mean of ~200 calls per day. This means that the statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.
Final answer: The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.


In each of the following, the correct answer is the first option. The other options are incorrect, but plausible.

```json
[
    {"category : "counterfactual_reasoning",
     "question": "How would the total number of calls in this month change if the product launch was delayed by three days?",
     "options": 
        ["The total number of calls would not change by much, since the peak introduced by the shift would still be within the one month window",
        "The total number of calls would increase by a lot, since the peak introduced by the shift would still be within the one month window",
        "The total number of calls would decrease by a lot, since the peak introduced by the shift would still be within the one month window",
        "The total number of calls would not change by much, since the peak introduced by the shift would be outside the one month window",
        ]
     "label_index": 0 
  },
  {
    "category": "explanation",
    "question": "What could have led the company to launch the product when it did?",
    "options": 
        ["It's hard to say for sure, but one explanation is that a market survey found that products launched in the middle of the month performed best. The company might have also decided to launch in January because many businesses are closed for the December holidays, and so deals would have moved slower.",
         "When products launch at the end of the month, they tend to perform better because the sales people are trying to make their quotas.",
         "The company might have decided to launch in September because deals move slowly during the summer months.",
         "The company might have decided to launch in January because many businesses are just returning from the December holidays, and so deals would have moved slowly.",
        ],
    "label_index": 0
  },
  {
    "category": "argumentation",
    "question": "Why might I not need to hire more call center staff?",
    "options": [
        "One reason to not hire more staff is that the call volume seems to have returned to the baseline.",
        "You should hire more staff because calls hit their limit.",
        "You should hire more staff because calls are increasing.",
        "You should hire more staff because calls are decreasing.",
        ],
    "label_index": 0
  }
  {
    "category": "analogical_reasoning",
    "question": "What are some other product launches that were similar?",
    "options": [
       "Healthcare.gov: The initial launch of the U.S. government's health insurance marketplace website in 2013 was plagued by technical problems that prevented many users from signing up for healthcare coverage. The site's launch led to an overwhelming number of calls to support centers and required significant efforts to resolve the technical issues.",
       "The launch of the Tesla Model 3 in 2017 went perfectly smoothly, with no technical issues and no increase in call volume.",
       "Pizza Hut launched a new pizza in 2018, and the launch caused such a large increase in popularity that call volume never returned to normal.",
       "The premier of the movie 'My Goofy Cat and Me' in 1980 was a disaster. The movie was so bad that there were zero calls to the theater."
    ],
    "label_index": 0
  },
  {
    "category": "fact_checking",
    "question": "Tell me if the following statement is true, and explain the reasoning behind your answer: The product launch led to a 10x increase in call volume.",
    "options": [
        "The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~200 calls per day.",
        "The statement is true, because the number of calls peaked at ~600 and prior to that peak had a mean of ~60 calls per day.",
        "The satement is true because calls were zero the whole way through.",
        "The statement is false, because the number of calls peaked at ~600 and prior to that peak had a mean of ~60 calls per day.",
        ],
    "label_index": 0
  }
}
```
*** End of expected output ***

Can you please generate similar questions for the time series I provided? 
The following instructions are VERY IMPORTANT to follow. EACH QUESTION MUST MATCH THESE CRITERIA:

1. It is very important that the questions can't be answered without looking at the time series. You should
   avoid describing the time series in a question. For example, the question:
   "" Why did the time series decrease?"" is not as good as:
   "" Why did the time series change"" or "" What could have caused the change in the time series?""
   because the second two do not reference the specific change in the time series.

    As another example, the question:
    "What factors could contribute to a repetitive pattern of higher traffic on weekends?
        is not as good as:
    "What factors could contribute to the pattern of traffic?        
    or 
    "Can you think of similar scenarios where external events significantly increase customer traffic?"
    is not as good as:
    "Can you think of similar scenarios where external events have the same impact on traffic?"

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
        raise Exception(f"Failed to make the request. Error: {e}")
    
    return response_to_text(response)


def get_prompt(scenario):
    characteristics = scenario["characteristics"]
    code = scenario["generator"]

    prompt = DEFAULT_PROMPT.replace("<CHARACTERISTICS>", characteristics)
    prompt = prompt.replace("<CODE>", code)
    return prompt

def extract_json_from_response_text(response_text):
    """Find the JSON in the response text, which is delimited by the following:
    '''json
    <JSON>
    '''
    """
    # print(response_text)
    json_start = response_text.find("```json\n")
    json_end = response_text.find("```", json_start + 1)
    json_text = response_text[json_start + 8 : json_end]
    print(json_text)
    return json_text


if __name__ == "__main__":
    scenarios = read_jsonl("data/processed/ts2desc/v2.jsonl")
    output_dir = "data/processed/mikes_qa_prompt_v2.jsonl"

    for scenario in tqdm(scenarios):
        try:
            uuid = scenario["uuid"]
            image_path = f"/gscratch/bdata/mikeam/TSandLanguage/data/processed/to_retool/figs_600_400/{uuid}.png"
            prompt = get_prompt(scenario)
            # print(prompt)
            response = get_response(prompt, image_path)
            json_text = extract_json_from_response_text(response)
            
            with open(output_dir, "a") as f:
                # print(response)
                json_repr = json.loads(json_text)
                for question in json_repr:
                    question["uuid"] = uuid
                    f.write(json.dumps(question))
                    f.write("\n")

            print(f"Finished writing {uuid}")
        except Exception as e:
            print(f"Failed on {uuid} with error {e}")
            continue

