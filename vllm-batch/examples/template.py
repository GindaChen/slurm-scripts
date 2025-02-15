# BatchRequestInput
import json
from vllm.entrypoints.openai.protocol import BatchRequestInput


def prepare_requests_default(input_file: str):
    requests = []
    with open(input_file, encoding="utf-8") as f:
        data = f.read()
        data = data.strip().split("\n")
    
    for request_json in data:
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)
        requests.append(request)
    return requests



def get_requests(input_file: str):
    requests = []
    with open(input_file, "r") as f:
        data = f.read()
        data = data.split("\n")
    
    for idx, line in enumerate(data):
        if not line:
            continue
        
        request_dict = dict(
            custom_id=f"request_{idx}",
            method="POST",
            url="/v1/chat/completions",
            body=dict(
                logprobs=True,
                top_logprobs=2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": line}
                ],
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                stream=False,
                temperature=0.7,
                top_p=1,
                max_tokens=128,
            )
        )
        request_json = json.dumps(request_dict)
        request = BatchRequestInput.model_validate_json(request_json)
        requests.append(request)
    return requests

if __name__ == "__main__":
    requests_1 = prepare_requests_default("examples/input.jsonl")
    requests_2 = get_requests("examples/input.txt")
    print(requests_1[0])
    print(requests_2[0])
    breakpoint()