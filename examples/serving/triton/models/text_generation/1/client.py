
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import argparse
import json
import sys
import warnings

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="text_generation",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default="Tell me about Intel Xeon Scalable Processors.",
        help="Prompt string",
    )
    parser.add_argument(
        "--request_id",
        type=str,
        required=False,
        default="111",
        help="Your request ID",
    )
    parser.add_argument(
        "--kb_id",
        type=str,
        required=False,
        default="default",
        help="The knowledge base ID",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    args = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(args.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if args.verbose:
        print(json.dumps(triton_client.get_model_config(args.model_name), indent=4))

    inputs = []
    inputs.append(httpclient.InferInput('prompt', [1], "BYTES"))
    inputs.append(httpclient.InferInput('request_id', [1], "BYTES"))
    inputs.append(httpclient.InferInput('kb_id', [1], "BYTES"))
    input_data0 = args.prompt
    input_data0 = np.array([input_data0.encode("utf-8")],dtype=np.object_)
    inputs[0].set_data_from_numpy(input_data0)
    input_data1 = args.request_id
    input_data1 = np.array([input_data1.encode("utf-8")],dtype=np.object_)
    inputs[1].set_data_from_numpy(input_data1)
    input_data2 = args.kb_id
    input_data2 = np.array([input_data2.encode("utf-8")],dtype=np.object_)
    inputs[2].set_data_from_numpy(input_data0)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('request_id', binary_data=False))

    results = triton_client.infer(model_name=args.model_name, inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT0')

    print("input prompt:",input_data0)
    print("input reqeuest id:",input_data1)
    print("input kb_id:",input_data2)
    print("output:",output_data0)

