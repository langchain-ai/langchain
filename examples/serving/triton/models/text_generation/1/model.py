# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np

import triton_python_backend_utils as pb_utils
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, GenerationConfig, plugins

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        output_re_config = pb_utils.get_output_config_by_name(model_config, "request_id")
        self.output_re_dtype = pb_utils.triton_string_to_numpy(output_re_config["data_type"])

        for plugin_name, plugin_config in plugins.items():
            if plugin_name == 'retrieval':
                plugin_config['enable'] = True
                plugin_config["args"] = {"input_path": "/rag_files/docs","persist_directory": "/rag_files/persist"}
        self.config = PipelineConfig()
        self.chatbot = build_chatbot(self.config)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        output_re_dtype = self.output_re_dtype
        chatbot = self.chatbot

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get request_id
            request_id = pb_utils.get_input_tensor_by_name(request, "request_id")
            request_id = request_id.as_numpy()
            request_id_text = request_id[0].decode("utf-8")
            print(f"request_id: {request_id_text}")

            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "prompt")
            in_0 = in_0.as_numpy()
            text = in_0[0].decode("utf-8")
            print(f"input prompt: {text}")

            # Get kb_id
            kb_id = pb_utils.get_input_tensor_by_name(request, "kb_id")
            kb_id = kb_id.as_numpy()
            kb_id = kb_id[0].decode("utf-8")
            print(f"kb_id: {kb_id}")

            # load/reload localdb according to kb_id
            RETRIEVAL_FILE_PATH = "/rag_files/"
            if kb_id == 'default':
                persist_dir = RETRIEVAL_FILE_PATH+"persist"
            else:
                persist_dir = RETRIEVAL_FILE_PATH+kb_id+'/persist_dir'
            if not os.path.exists(persist_dir):
                raise Exception(f"Knowledge base id [{kb_id}] does not exist, please check again.")

            # reload retrieval instance with specific knowledge base
            if kb_id != 'default':
                print("[askdoc - chat] starting to reload local db...")
                instance = plugins['retrieval']["instance"]
                instance.reload_localdb(local_persist_dir = persist_dir)

            config = GenerationConfig(max_new_tokens=512)
            out_0 = chatbot.predict(query=text, origin_query=text, config=config)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_0 = np.array(out_0)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(output0_dtype))

            # output: request_id
            out_request_id = np.array(request_id_text)
            out_request_id_tensor = pb_utils.Tensor("request_id", out_request_id.astype(output_re_dtype))

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_request_id_tensor]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
