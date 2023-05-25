import sys
import os
import argparse
import asyncio
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
from langchain.chains.dialogue_answering import *
from langchain.llms import OpenAI


async def dispatch(args: Namespace):
    args_dict = vars(args)

    if not os.path.isfile(args.dialogue_path):
        raise FileNotFoundError(f'Invalid dialogue file path for demo mode: "{args.dialogue_path}"')
    llm = OpenAI(temperature=0)
    dialogue_instance = DialogueWithSharedMemoryChains(zero_shot_react_llm=llm, ask_llm=llm, params=args_dict)

    dialogue_instance.agent_chain.run(input="What did David say before, summarize it")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='langchina-Dialogue',
                                     description='langchina-Dialogue')
    parser.add_argument('--dialogue-path', default='', type=str, help='dialogue-path')
    parser.add_argument('--embedding-model', default='', type=str, help='embedding-model')
    args = parser.parse_args(['--dialogue-path', '/home/dmeck/Downloads/log.txt',
                              '--embedding-mode', '/media/checkpoint/text2vec-large-chinese/'])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dispatch(args))
