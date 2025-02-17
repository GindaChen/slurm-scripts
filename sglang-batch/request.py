import openai
import argparse
import time
import os
def perform_request(client, messages, model):
    s_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Lower temperature for more focused responses
        max_tokens=20000,  # Reasonable length for a concise response
        top_p=1,  # Slightly higher for better fluency
        n=1,  # Single response is usually more stable
        seed=20242,  # Keep for reproducibility
    )
    e_time = time.time()
    print("Time taken for request: ", e_time - s_time)
    return response.choices[0].message.content
def main(args):
    print("Arguments: ", args)
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    message = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hey"},
    ]
    response = perform_request(client, message, args.model)
    print(response)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    args = argparser.parse_args()
    main(args)
