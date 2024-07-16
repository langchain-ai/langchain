import http.client
import json
import ssl
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


class NIBittensorLLM(LLM):
    """NIBittensor LLMs

    NIBittensorLLM is created by Neural Internet (https://neuralinternet.ai/),
    powered by Bittensor, a decentralized network full of different AI models.

    To analyze API_KEYS and logs of your usage visit
        https://api.neuralinternet.ai/api-keys
        https://api.neuralinternet.ai/logs

    Example:
        .. code-block:: python

            from langchain_community.llms import NIBittensorLLM
            llm = NIBittensorLLM()
    """

    system_prompt: Optional[str]
    """Provide system prompt that you want to supply it to model before every prompt"""

    top_responses: Optional[int] = 0
    """Provide top_responses to get Top N miner responses on one request.May get delayed
        Don't use in Production"""

    @property
    def _llm_type(self) -> str:
        return "NIBittensorLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Wrapper around the bittensor top miner models. Its built by Neural Internet.

        Call the Neural Internet's BTVEP Server and return the output.

        Parameters (optional):
            system_prompt(str): A system prompt defining how your model should respond.
            top_responses(int): Total top miner responses to retrieve from Bittensor
                protocol.

        Return:
            The generated response(s).

        Example:
            .. code-block:: python

                from langchain_community.llms import NIBittensorLLM
                llm = NIBittensorLLM(system_prompt="Act like you are programmer with \
                5+ years of experience.")
        """

        # Creating HTTPS connection with SSL
        context = ssl.create_default_context()
        context.check_hostname = True
        conn = http.client.HTTPSConnection("test.neuralinternet.ai", context=context)

        # Sanitizing User Input before passing to API.
        if isinstance(self.top_responses, int):
            top_n = min(100, self.top_responses)
        else:
            top_n = 0

        default_prompt = "You are an assistant which is created by Neural Internet(NI) \
            in decentralized network named as a Bittensor."
        if self.system_prompt is None:
            system_prompt = (
                default_prompt
                + " Your task is to provide accurate response based on user prompt"
            )
        else:
            system_prompt = default_prompt + str(self.system_prompt)

        # Retrieving API KEY to pass into header of each request
        conn.request("GET", "/admin/api-keys/")
        api_key_response = conn.getresponse()
        api_keys_data = (
            api_key_response.read().decode("utf-8").replace("\n", "").replace("\t", "")
        )
        api_keys_json = json.loads(api_keys_data)
        api_key = api_keys_json[0]["api_key"]

        # Creating Header and getting top benchmark miner uids
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Endpoint-Version": "2023-05-19",
        }
        conn.request("GET", "/top_miner_uids", headers=headers)
        miner_response = conn.getresponse()
        miner_data = (
            miner_response.read().decode("utf-8").replace("\n", "").replace("\t", "")
        )
        uids = json.loads(miner_data)

        # Condition for benchmark miner response
        if isinstance(uids, list) and uids and not top_n:
            for uid in uids:
                try:
                    payload = json.dumps(
                        {
                            "uids": [uid],
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt},
                            ],
                        }
                    )

                    conn.request("POST", "/chat", payload, headers)
                    init_response = conn.getresponse()
                    init_data = (
                        init_response.read()
                        .decode("utf-8")
                        .replace("\n", "")
                        .replace("\t", "")
                    )
                    init_json = json.loads(init_data)
                    if "choices" not in init_json:
                        continue
                    reply = init_json["choices"][0]["message"]["content"]
                    conn.close()
                    return reply
                except Exception:
                    continue

        # For top miner based on bittensor response
        try:
            payload = json.dumps(
                {
                    "top_n": top_n,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
            )

            conn.request("POST", "/chat", payload, headers)
            response = conn.getresponse()
            utf_string = (
                response.read().decode("utf-8").replace("\n", "").replace("\t", "")
            )
            if top_n:
                conn.close()
                return utf_string
            json_resp = json.loads(utf_string)
            reply = json_resp["choices"][0]["message"]["content"]
            conn.close()
            return reply
        except Exception as e:
            conn.request("GET", f"/error_msg?e={e}&p={prompt}", headers=headers)
            return "Sorry I am unable to provide response now, Please try again later."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "system_prompt": self.system_prompt,
            "top_responses": self.top_responses,
        }
