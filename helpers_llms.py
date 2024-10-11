import os, sys
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))

sys.path.append(WORKDIR)

from private_obfuscation.paths import WORKDIR, DATADIR

hf_model_ids = {
    'mamba': 'state-spaces/mamba-130m-hf',
    'mambaxl': 'state-spaces/mamba-2.8b-hf',
    'gpt2': 'openai-community/gpt2',
    'gpt2xl': 'openai-community/gpt2-xl',
    'llama3': 'meta-llama/Llama-3.2-1B',
    'falcon': 'tiiuae/falcon-7b',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'internlm': 'internlm/internlm2_5-7b-chat',
}


def use_chatgpt(
        personality="You are a helpful assistant that translates English to French. Translate the user sentence.",
        questions=["What is the capital of France?", "What is the capital of Germany?"],
):
    os.path.join(WORKDIR, 'all_stuff.py', )

    from all_stuff import api_key_openai

    # os.environ["OPENAI_API_KEY"] = api_key_openai

    from langchain_openai import ChatOpenAI

    api_model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(
        model=api_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key_openai,
    )

    answers = []
    print("Generating responses with ChatGPT...")
    for question in tqdm(questions):
        messages = [
            ("system", personality),
            ("human", question),
        ]
        ai_msg = llm.invoke(messages).content
        print(ai_msg)
        answers.append(ai_msg)

    pairs = {question: answer for question, answer in zip(questions, answers)}
    return pairs


def get_model(model_id, device, more_kwargs={}, tkn_kwargs={}):
    path = os.path.join(DATADIR, model_id.replace('/', '-'))
    if not os.path.exists(path):
        print('Downloading model')
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, truncation_side='left',
                                                  **tkn_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **more_kwargs)

        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
    else:
        print('Loading model')
        tokenizer = AutoTokenizer.from_pretrained(path, truncation_side='left', **tkn_kwargs)
        model = AutoModelForCausalLM.from_pretrained(path, **more_kwargs)

    return model.to(device), tokenizer


def use_huggingface(
        personality="You are a helpful assistant that translates English to French. Translate the user sentence.",
        questions=["What is the capital of France?", "What is the capital of Germany?"],
        model_id='mambaxl'
):
    model_id = hf_model_ids[model_id]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = get_model(model_id, device)

    # inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")
    # outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    # reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # print(reply)

    answers = []
    print(f"Generating responses with {model_id}...")
    for question in tqdm(questions):
        print('-' * 50)
        full_question = f"{personality} {question}. Answer:".replace('..', '.').replace('?.', '.')

        inputs = tokenizer(full_question, return_tensors="pt")
        print(inputs)
        if 'mamba' in model_id:
            del inputs['attention_mask']
        print(inputs)
        outputs = model.generate(**inputs, do_sample=True, top_p=0.2, max_new_tokens=40)
        reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print('full_question:', full_question)
        print('reply:        ', reply)
        answers.append(reply)

    pairs = {question: answer for question, answer in zip(questions, answers)}
    return pairs


if __name__ == '__main__':
    use_huggingface()
