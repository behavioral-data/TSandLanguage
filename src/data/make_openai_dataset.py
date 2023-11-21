import argparse
import json
import concurrent.futures
from textwrap import wrap
from dotenv import dotenv_values
import openai
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time

"""
This script generates the Synthetic Scenarios dataset.
"""

# # Set your OpenAI API key from your .env file
# openai.api_key = dotenv_values(".env")["OPENAI_API_KEY"]

# For Azure keys:

openai.api_base = "https://ts-language-oai.openai.azure.com/"
openai.api_key = dotenv_values(".env")["OPENAI_API_KEY_AZURE"]
openai.api_version = "2023-09-01-preview"
openai.api_type = "azure"
DEPLOYMENT = "gpt-4"


PROMPT = """
1.   Describe a scenario that might produce a time series.   This scenario should include an 
    external event and how it might influence the reading. Be sure to describe the sample rate 
    of the time series and the duration over which it is sampled. The description should be 
    less than 100 words in length. Delimit this description with the XML tag <description>. 

    The time series must be less than 1000 observations in length, be a single variable,
    have no values greater than 1e6, and have no missing values. 

    Also add a summary of the description, no more than 25 words in length with the tag 
    <description_short>. 
    
    Also add a summary, no more than three words in length with the tag <description_tiny>.
    
    The scenario should be as different as possible from any of the following: [<previous_descriptions>]

2.  You will generate a list of up to five characteristics of this specific time series, 
    including patterns that you might expect to see in the series and how external events 
    might cause distribution shifts in the data generating process. Delimit these characteristics
    with the XML tag  <characteristics>.

3.  You will write a numpy  function called `generate_series`  that takes no arguments and
    outputs a time series that matches the description. 
    All parameters from the data generating process should be drawn from reasonable 
    distributions. The function must return a single numpy array. 
    Place this code inside a python markdown block and delimit your code with 
    the XML tag <generator>. Do not call the function, simply define it. You should also make sure
    that the scale of time series is realistic. For example, a time series of a quantity 
    like stock price should never be less than zero. 
    

4   Return a json string, delimited by the tag <metadata> that contains the units of 
    the time series and the timestamps corresponding to the first and last values.
    Remember that in JSON format datetimes must be passed as strings. Also include a 
    string that relects the frequency of the time series.

Here is an example of a complete response: 
<description> *your description* </description> 
<description_short> *your description* </description_short>
<description_tiny> *your description* </description_tiny>
<characteristics> *your characteristics* </characteristics>
<generator> 
    ```python
    def generate_series():
        # your code here
        return x
    ```
</generator>
<metadata>        
        {
        "start": x,
        "end": y,
        "units": z,
        "frequency" : freq
        } 
</metadata>
"""

RESPONSE_TAGS = [
    "description",
    "description_short",
    "description_tiny",
    "characteristics",
    "generator",
    "metadata",
]


def parse_response_for_xml_contents(response, *tags):
    xml_contents = {}
    for tag in tags:
        xml_contents[tag] = response.choices[0]["message"]["content"].split(f"<{tag}>")[1].split(f"</{tag}>")[0]
    return xml_contents


def call_generated_function(fn_code, fn_name):
    fn_dict = {}
    exec(fn_code, fn_dict)
    return fn_dict[fn_name]()


def generated_series_okay(series):
    if isinstance(series, list):
        series = np.array(series)
    if series.ndim != 1:
        return False
    if len(series) > 1500:
        return False
    if not np.issubdtype(series.dtype, np.number):
        return False
    if np.isnan(series).any():
        return False
    if np.isinf(series).any():
        return False
    if np.iscomplex(series).any():
        return False
    if np.max(series) > 1e9:
        return False

    return True


def filter_generated_series(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                response = json.loads(line)
                if 'series' in response and generated_series_okay(np.array(response['series'])):
                    outfile.write(json.dumps(response) + "\n")
            except Exception as e:
                print(f"Failed to process line: {line}\nError: {e}")

def make_api_call_for_scenario(prompt, model="gpt-4"):
    while True:
        response = openai.ChatCompletion.create(
                model=model,
                deployment_id=DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
                temperature=1,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        try:
            resp = parse_response_for_xml_contents(response, *RESPONSE_TAGS)
            resp["metadata"] = json.loads(resp["metadata"])
        except (KeyError, json.decoder.JSONDecodeError, IndexError) as e:
            print("Failed to parse response")
            continue

        try:
            code = resp["generator"].replace("```python", "").replace("```", "")
            series = call_generated_function(code, "generate_series")
        except Exception as e:
            print("Failed to generate series")
            print(e)
            continue

        if not generated_series_okay(series):
            print("Generated series does not match criteria")
            continue

        resp["series"] = series.tolist()
        return resp
    

def generate_ts(n_series, model="gpt-4", output_file="output.json", batch_size=5):
    def task(formatted_prompt):
        return make_api_call_for_scenario(formatted_prompt, model=model)

    results = []
    with open(output_file, "a") as outfile:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while len(results) < n_series:
                futures = []
                current_batch_size = min(batch_size, n_series - len(results))
                for _ in range(current_batch_size):
                    sampled_results = np.random.choice(results, size=min(20, len(results)), replace=False)
                    formatted_previous = ", ".join([r["description_tiny"] for r in sampled_results])
                    formatted_prompt = PROMPT.replace("<previous_descriptions>", formatted_previous)
                    futures.append(executor.submit(task, formatted_prompt))

                for future in tqdm(concurrent.futures.as_completed(futures), total=current_batch_size):
                    result = future.result()
                    outfile.write(json.dumps(result) + "\n")
                    results.append(result)

def main():
    parser = argparse.ArgumentParser(description="Generate time series data using GPT models.")
    parser.add_argument("sample_num", type=int, help="Number of time series samples to generate")
    parser.add_argument("output_file", type=str, help="Output file to write results")
    parser.add_argument("model_name", type=str, help="Name of the GPT model to use", default="gpt-4")
    parser.add_argument("--filter", action="store_true", help="Filter generated series and save to output file")

    args = parser.parse_args()

    if args.filter:
        filter_generated_series(args.output_file, "filtered_output.json")
    else:
        generate_ts(args.sample_num, args.model_name, args.output_file)


if __name__ == "__main__":
    main()
