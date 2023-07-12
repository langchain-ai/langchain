from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import http.client
import json
import ssl
import requests
import time


class NIBittensorLLM(LLM):
    
    system: Optional[str]

    @property
    def _llm_type(self) -> str:
        return "NIBittensorLLM"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:

        """
        "Wrapper around the bittensor top miner models. Its built by Neural Internet.

        
        Parameters:
          system: the system prompt that you want your model to basically act
          API_KEY: API_KEY for validator endpoint server

        Example:
            .. code-block:: python

                from langchain.llms import NIBittensorLLM 
                llm = NIBittensorLLM(system="Your task is to give response based on user prompt")
        """
        
        if self.system is None:  # Check if system parameter is None
            system_prompt = "Your task is to provide accurate response based on user prompt"
        else:
            system_prompt = self.system
            
        payload = json.dumps({
                "top_n": 10,
                "messages": [{"role": "system", "content": system_prompt} ,{"role": "user", "content": prompt}]
        })

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer ytfljFmQaCs4sC7BFsCgoKt4jE_oah4ql4vjLQKaRjodVEewpRctT72NoyDdzVh7',
            'Endpoint-Version': '2023-05-19'
        }


        context = ssl.create_default_context()
        context.check_hostname = True
        # context.verify_mode = ssl.CERT_REQUIRED,

        conn = http.client.HTTPSConnection("6860-65-108-32-175.ngrok-free.app",context=context)
        
        reply = 'Sorry, Currently I am unable to respond please try again later'
        for i in range(0,3):
          time.sleep(3)
          conn.request("POST", "/chat", payload, headers)
          response = conn.getresponse()
          resp_str = response.read().decode("utf-8").replace('\n', '').replace('\t', '')
          #print(f"Response from localhost URL endpoint: {resp_str}")
          #print()
          json_resp = json.loads(resp_str)
          if 'choices' in json_resp.keys():
              choice = json_resp["choices"][0]
              reply = choice["message"]["content"]
              break
          else:
            time.sleep(8)
        conn.close()
        
        return reply
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"system": self.system}
