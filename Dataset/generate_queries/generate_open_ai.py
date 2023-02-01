
import os
import openai
import json

from torch.utils.data import Dataset, DataLoader
from dataset import DPRDataset

from datetime import datetime

from tqdm import tqdm

import time

API_KEY = "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"
openai.api_key = API_KEY

def generate_open_ai(shard_id, input_path, output_path, prompt_info, batch_size):
    ds = DPRDataset(input_path)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=False)

    print(f"\nStarting shard {shard_id}")

    current_time = datetime.now()
    prev_time = current_time
    print(f"current time {current_time}")

    print(f"Output file: {output_path}")
    examples = 1

    output_dataset = []
    errors = 0
    for batch in tqdm(dl):
        outputs = batch

        current_information = [[outputs["passage"][i]] for i in range(len(outputs["passage"]))]

        for step in range(prompt_info["steps"]):
            prompt = prompt_info[str(step)]["prompt"]
            next_step = prompt_info[str(step)]["next_step"]

            outputs["prompt"] = []
            for i in range(len(outputs["passage"])):
                # Create the prompt
                outputs["prompt"].append(prompt.format(*current_information[i]) + next_step)
   
            for i, text_to_send in enumerate(outputs["prompt"]):
                try:
                    open_ai_output = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=text_to_send,
                        max_tokens=32,
                    )
                except:
                    print("Error, sleeping 5 minutes and trying to recover ...")
                    time.sleep(5 * 60)
                    open_ai_output = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=text_to_send,
                        max_tokens=32,
                    )
            
                if len(open_ai_output["choices"]) == 0:
                    current_information[i].append("")
                    errors += 1
                else:
                    current_information[i].append(open_ai_output["choices"][0]["text"].strip())
    

            for i in range(len(outputs["prompt"])):
                output_dataset.append({
                    "dataset": "instruct-gpt",
                    "question": current_information[i][-1],
                    "answers": ["empty"],
                    "positive_ctxs": [
                        {
                            "title": "",
                            "text": outputs["passage"][i],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": str(outputs["passage_id"][i].item())
                        }
                    ],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": []
                })
        print("Sleeping 1 minute. OpenAI limit 60 examples per minute")
        time.sleep(60)
        examples += 1
    
    with open(output_path, "w") as f:
        json.dump(output_dataset, f)

    final_time = datetime.now()
    print(final_time)

    diff = final_time - current_time

    minutes = diff.total_seconds() / 60
    print(f'Total difference in minutes: {minutes}\n')
    print(f"End shard {shard_id}")

    return minutes, errors
"""
OPENAI Example:
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "\n\nWhat color models can Color Hex be used to obtain information for?"
    }
  ],
  "created": 1672145003,
  "id": "cmpl-6S3adT7qDR1TA7RTs77xcEQ5Sfpp7",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 15,
    "prompt_tokens": 47,
    "total_tokens": 62
  }
}
"""