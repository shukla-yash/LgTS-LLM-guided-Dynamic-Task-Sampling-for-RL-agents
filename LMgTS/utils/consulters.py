import openai
import pickle
import replicate

from abc import abstractmethod, abstractproperty

from .gpt_util import *
from .util import *

llama_model_to_url = {'llama-2-70b-chat': 'meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
                      'llama-2-13b-chat': 'meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d',
                      'llama-2-7b-chat': 'meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e'
                      }




class LLMException(Exception):
    pass


class ConsulterException(Exception):
    pass


class LLMConsulter:
    gpt_model = 'gpt-3.5-turbo'
    openai_key = None
    # prompt = None
    max_consult_time = 10
    messages = []

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_one_path(self):
        pass


class GptConsulter(LLMConsulter):

    def __init__(self, openai_key, max_consult_time,
                 prompt_file='LMgTS/prompts/minigrid_prompt.txt', gpt_model='gpt-3.5-turbo'):
        self.openai_key = openai_key
        openai.api_key = self.openai_key
        self.max_consult_time = max_consult_time
        with open(prompt_file, 'r') as f:
            prompt = f.read()
        # print(prompt)
        self.messages.append(
            {'role': 'user', 'content': prompt}
        )
        assert (gpt_model in gpt_model_list, 'Unknown GPT model.')
        self.gpt_model = gpt_model
        self.consult_time = 0
        self.response = None
        # self.current_choice_id = -1
        # self.response = openai.ChatCompletion.create(
        #     model=self.gpt_model,
        #     messages=self.messages
        # )
        # print(string_to_list(self.response['choices'][self.current_choice_id]['message']['content']))
        # print(self.response)

    def get_one_path(self) -> list:
        # self.current_choice_id += 1
        if self.response is None:
            self.response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=self.messages
            )
        print('Getting a new path...')
        response_content = string_to_lists(self.response['choices'][0]['message']['content'])
        print('Current response: {}'.format(response_content))
        print('Got a new path!')
        return response_content

    def regenerate_response(self, exception):
        # print('Received an exception. Regenerating responses.')
        print('------------------------------------------------------------------')
        print('Regenerating responses...')
        self.consult_time += 1
        if self.consult_time > self.max_consult_time:
            print('Reached maximum consult time!!!')
            raise ConsulterException('Already generated {} failed responses with current prompt. Try something new.'.format(self.max_consult_time))
        self.messages.append(
            {
                'role': 'assistant',
                'content': self.response['choices'][0]['message']['content']
            }
        )
        self.messages.append(
            {
                'role': 'user',
                'content': exception
            }
        )
        self.response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=self.messages
        )
        # self.current_choice_id = -1
        print('Response regenerated.')
        print('------------------------------------------------------------------\n')

    def save_response(self, file):
        print('------------------------------------------------------------------')
        print('Saving response to {}...'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print('Response saved')
        print('------------------------------------------------------------------\n')

    @classmethod
    def load_response(cls, file):
        print('------------------------------------------------------------------')
        print('Loading response from {}...'.format(file))
        with open(file, 'rb') as f:
            ins = pickle.load(f)
            print('Response loaded')
            print('------------------------------------------------------------------\n')
        return ins 
    
class LLamaConsulter(LLMConsulter):
    def __init__(self, max_consult_time,
                 prompt_file='LMgTS/prompts/minigrid_prompt.txt', gpt_model='llama-2-70b-chat'):
        self.max_consult_time = max_consult_time
        with open(prompt_file, 'r') as f:
            prompt = f.read()
        # print(prompt)
        self.messages.append(
            {'role': 'user', 'content': prompt}
        ) 
        self.prompt = prompt
        assert (gpt_model in gpt_model_list, 'Unknown LLM model.')
        self.gpt_model = gpt_model
        self.consult_time = 0
        self.response = None 
        self.temperature = 0.5
        self.seed = 1000

    def get_one_path(self) -> list:
        # self.current_choice_id += 1
        if self.response is None:
            self.response = replicate.run(
            llama_model_to_url[self.gpt_model], 
            min_new_tokens=1000,
            max_new_tokens=5000, 
            temperature=self.temperature,
            seed=self.seed,
            input={'prompt': self.prompt},
            top_p=1
        ) 
        print('------------------------------------------------------------------')
        print('Getting a new path from LLM...')
        response_content = string_to_lists(self.get_response_content())
        print('------------------------------------------------------------------')
        # print('Got a new path!')
        return response_content 
    
    def get_response_content(self):
        content = ''
        num_tokens = 0
        for item in self.response:
            content += item 
            num_tokens += 1
            print(item, end='')
        print('Total tokens: ', num_tokens)
        return content

    def regenerate_response(self, exception):
        # print('Received an exception. Regenerating responses.')
        # print('-----------------------------------------------------------------')
        print('Regenerating responses...')
        self.consult_time += 1
        self.temperature += 0.1
        self.seed += 1000
        if self.consult_time > self.max_consult_time:
            print('WARNING: LLM Consulter Reached maximum consult time')
            raise ConsulterException('Already generated {} failed responses with current prompt. Try something new.'.format(self.max_consult_time))
        self.response = replicate.run(
            llama_model_to_url[self.gpt_model], 
            temperature=self.temperature, 
            seed=self.seed,
            input={'prompt': self.prompt}, 
            max_new_tokens=5000,
            min_new_tokens=1000
        )
        print('Regenerated response') 
        # print('------------------------------------------------------------------\n')

    def save_response(self, file):
        print('------------------------------------------------------------------')
        print('Saving response to {}...'.format(file))
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print('------------------------------------------------------------------\n') 

    @classmethod
    def load_response(cls, file):
        print('-------------------------------------------------------------------') 
        print('Loading response from {}...'.format(file))
        with open(file, 'rb') as f:
            ins = pickle.load(f)
        print('-------------------------------------------------------------------\n')

        return ins 