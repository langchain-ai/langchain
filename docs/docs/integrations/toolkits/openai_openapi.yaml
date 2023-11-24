openapi: 3.0.0
info:
  title: OpenAI API
  description: The OpenAI REST API. Please see https://platform.openai.com/docs/api-reference for more details.
  version: "2.0.0"
  termsOfService: https://openai.com/policies/terms-of-use
  contact:
    name: OpenAI Support
    url: https://help.openai.com/
  license:
    name: MIT
    url: https://github.com/openai/openai-openapi/blob/master/LICENSE
servers:
  - url: https://api.openai.com/v1
tags:
  - name: Assistants
    description: Build Assistants that can call models and use tools.
  - name: Audio
    description: Learn how to turn audio into text or text into audio.
  - name: Chat
    description: Given a list of messages comprising a conversation, the model will return a response.
  - name: Completions
    description: Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
  - name: Embeddings
    description: Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
  - name: Fine-tuning
    description: Manage fine-tuning jobs to tailor a model to your specific training data.
  - name: Files
    description: Files are used to upload documents that can be used with features like Assistants and Fine-tuning.
  - name: Images
    description: Given a prompt and/or an input image, the model will generate a new image.
  - name: Models
    description: List and describe the various models available in the API.
  - name: Moderations
    description: Given a input text, outputs if the model classifies it as violating OpenAI's content policy.
  - name: Fine-tunes
    description: Manage legacy fine-tuning jobs to tailor a model to your specific training data.
  - name: Edits
    description: Given a prompt and an instruction, the model will return an edited version of the prompt.
paths:
  # Note: When adding an endpoint, make sure you also add it in the `groups` section, in the end of this file,
  # under the appropriate group
  /chat/completions:
    post:
      operationId: createChatCompletion
      tags:
        - Chat
      summary: Creates a model response for the given chat conversation.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateChatCompletionRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateChatCompletionResponse"

      x-oaiMeta:
        name: Create chat completion
        group: chat
        returns: |
          Returns a [chat completion](/docs/api-reference/chat/object) object, or a streamed sequence of [chat completion chunk](/docs/api-reference/chat/streaming) objects if the request is streamed.
        path: create
        examples:
          - title: Default
            request:
              curl: |
                curl https://api.openai.com/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "model": "VAR_model_id",
                    "messages": [
                      {
                        "role": "system",
                        "content": "You are a helpful assistant."
                      },
                      {
                        "role": "user",
                        "content": "Hello!"
                      }
                    ]
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                completion = client.chat.completions.create(
                  model="VAR_model_id",
                  messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                  ]
                )

                print(completion.choices[0].message)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const completion = await openai.chat.completions.create({
                    messages: [{ role: "system", content: "You are a helpful assistant." }],
                    model: "VAR_model_id",
                  });

                  console.log(completion.choices[0]);
                }

                main();
            response: &chat_completion_example |
              {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                  "index": 0,
                  "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                  },
                  "finish_reason": "stop"
                }],
                "usage": {
                  "prompt_tokens": 9,
                  "completion_tokens": 12,
                  "total_tokens": 21
                }
              }
          - title: Image input
            request:
              curl: |
                curl https://api.openai.com/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "model": "gpt-4-vision-preview",
                    "messages": [
                      {
                        "role": "user",
                        "content": [
                          {
                            "type": "text",
                            "text": "What’s in this image?"
                          },
                          {
                            "type": "image_url",
                            "image_url": {
                              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            }
                          }
                        ]
                      }
                    ],
                    "max_tokens": 300
                  }'
              python: |
                from openai import OpenAI

                client = OpenAI()

                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )

                print(response.choices[0])
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const response = await openai.chat.completions.create({
                    model: "gpt-4-vision-preview",
                    messages: [
                      {
                        role: "user",
                        content: [
                          { type: "text", text: "What’s in this image?" },
                          {
                            type: "image_url",
                            image_url:
                              "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                          },
                        ],
                      },
                    ],
                  });
                  console.log(response.choices[0]);
                }
                main();
            response: &chat_completion_image_example |
              {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                  "index": 0,
                  "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                  },
                  "finish_reason": "stop"
                }],
                "usage": {
                  "prompt_tokens": 9,
                  "completion_tokens": 12,
                  "total_tokens": 21
                }
              }
          - title: Streaming
            request:
              curl: |
                curl https://api.openai.com/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "model": "VAR_model_id",
                    "messages": [
                      {
                        "role": "system",
                        "content": "You are a helpful assistant."
                      },
                      {
                        "role": "user",
                        "content": "Hello!"
                      }
                    ],
                    "stream": true
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                completion = client.chat.completions.create(
                  model="VAR_model_id",
                  messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                  ],
                  stream=True
                )

                for chunk in completion:
                  print(chunk.choices[0].delta)

              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const completion = await openai.chat.completions.create({
                    model: "VAR_model_id",
                    messages: [
                      {"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": "Hello!"}
                    ],
                    stream: true,
                  });

                  for await (const chunk of completion) {
                    console.log(chunk.choices[0].delta.content);
                  }
                }

                main();
            response: &chat_completion_chunk_example |
              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

              ....

              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":" today"},"finish_reason":null}]}

              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"?"},"finish_reason":null}]}

              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
          - title: Function calling
            request:
              curl: |
                curl https://api.openai.com/v1/chat/completions \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -d '{
                  "model": "gpt-3.5-turbo",
                  "messages": [
                    {
                      "role": "user",
                      "content": "What is the weather like in Boston?"
                    }
                  ],
                  "functions": [
                    {
                      "name": "get_current_weather",
                      "description": "Get the current weather in a given location",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                          },
                          "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                          }
                        },
                        "required": ["location"]
                      }
                    }
                  ],
                  "function_call": "auto"
                }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                functions = [
                  {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                  }
                ]
                messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
                completion = client.chat.completions.create(
                  model="VAR_model_id",
                  messages=messages,
                  functions=functions,
                  function_call="auto"
                )

                print(completion)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const messages = [{"role": "user", "content": "What's the weather like in Boston today?"}];
                  const functions = [
                      {
                          "name": "get_current_weather",
                          "description": "Get the current weather in a given location",
                          "parameters": {
                              "type": "object",
                              "properties": {
                                  "location": {
                                      "type": "string",
                                      "description": "The city and state, e.g. San Francisco, CA",
                                  },
                                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                              },
                              "required": ["location"],
                          },
                      }
                  ];

                  const response = await openai.chat.completions.create({
                      model: "gpt-3.5-turbo",
                      messages: messages,
                      functions: functions,
                      function_call: "auto",  // auto is default, but we'll be explicit
                  });

                  console.log(response);
                }

                main();
            response: &chat_completion_function_example |
              {
                "choices": [
                  {
                    "finish_reason": "function_call",
                    "index": 0,
                    "message": {
                      "content": null,
                      "function_call": {
                        "arguments": "{\n  \"location\": \"Boston, MA\"\n}",
                        "name": "get_current_weather"
                      },
                      "role": "assistant"
                    }
                  }
                ],
                "created": 1694028367,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "object": "chat.completion",
                "usage": {
                  "completion_tokens": 18,
                  "prompt_tokens": 82,
                  "total_tokens": 100
                }
              }
  /completions:
    post:
      operationId: createCompletion
      tags:
        - Completions
      summary: Creates a completion for the provided prompt and parameters.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateCompletionRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateCompletionResponse"
      x-oaiMeta:
        name: Create completion
        returns: |
          Returns a [completion](/docs/api-reference/completions/object) object, or a sequence of completion objects if the request is streamed.
        legacy: true
        examples:
          - title: No streaming
            request:
              curl: |
                curl https://api.openai.com/v1/completions \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "model": "VAR_model_id",
                    "prompt": "Say this is a test",
                    "max_tokens": 7,
                    "temperature": 0
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                client.completions.create(
                  model="VAR_model_id",
                  prompt="Say this is a test",
                  max_tokens=7,
                  temperature=0
                )
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const completion = await openai.completions.create({
                    model: "VAR_model_id",
                    prompt: "Say this is a test.",
                    max_tokens: 7,
                    temperature: 0,
                  });

                  console.log(completion);
                }
                main();
            response: |
              {
                "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
                "object": "text_completion",
                "created": 1589478378,
                "model": "VAR_model_id",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                  {
                    "text": "\n\nThis is indeed a test",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "length"
                  }
                ],
                "usage": {
                  "prompt_tokens": 5,
                  "completion_tokens": 7,
                  "total_tokens": 12
                }
              }
          - title: Streaming
            request:
              curl: |
                curl https://api.openai.com/v1/completions \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "model": "VAR_model_id",
                    "prompt": "Say this is a test",
                    "max_tokens": 7,
                    "temperature": 0,
                    "stream": true
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                for chunk in client.completions.create(
                  model="VAR_model_id",
                  prompt="Say this is a test",
                  max_tokens=7,
                  temperature=0,
                  stream=True
                ):
                  print(chunk.choices[0].text)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const stream = await openai.completions.create({
                    model: "VAR_model_id",
                    prompt: "Say this is a test.",
                    stream: true,
                  });

                  for await (const chunk of stream) {
                    console.log(chunk.choices[0].text)
                  }
                }
                main();
            response: |
              {
                "id": "cmpl-7iA7iJjj8V2zOkCGvWF2hAkDWBQZe",
                "object": "text_completion",
                "created": 1690759702,
                "choices": [
                  {
                    "text": "This",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                  }
                ],
                "model": "gpt-3.5-turbo-instruct"
                "system_fingerprint": "fp_44709d6fcb",
              }
  /edits:
    post:
      operationId: createEdit
      deprecated: true
      tags:
        - Edits
      summary: Creates a new edit for the provided input, instruction, and parameters.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateEditRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateEditResponse"
      x-oaiMeta:
        name: Create edit
        returns: |
          Returns an [edit](/docs/api-reference/edits/object) object.
        group: edits
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/edits \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -d '{
                  "model": "VAR_model_id",
                  "input": "What day of the wek is it?",
                  "instruction": "Fix the spelling mistakes"
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.edits.create(
                model="VAR_model_id",
                input="What day of the wek is it?",
                instruction="Fix the spelling mistakes"
              )
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const edit = await openai.edits.create({
                  model: "VAR_model_id",
                  input: "What day of the wek is it?",
                  instruction: "Fix the spelling mistakes.",
                });

                console.log(edit);
              }

              main();
          response: &edit_example |
            {
              "object": "edit",
              "created": 1589478378,
              "choices": [
                {
                  "text": "What day of the week is it?",
                  "index": 0,
                }
              ],
              "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 32,
                "total_tokens": 57
              }
            }

  /images/generations:
    post:
      operationId: createImage
      tags:
        - Images
      summary: Creates an image given a prompt.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateImageRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ImagesResponse"
      x-oaiMeta:
        name: Create image
        returns: Returns a list of [image](/docs/api-reference/images/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/images/generations \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -d '{
                  "model": "dall-e-3",
                  "prompt": "A cute baby sea otter",
                  "n": 1,
                  "size": "1024x1024"
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.images.generate(
                model="dall-e-3",
                prompt="A cute baby sea otter",
                n=1,
                size="1024x1024"
              )
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const image = await openai.images.generate({ model: "dall-e-3", prompt: "A cute baby sea otter" });

                console.log(image.data);
              }
              main();
          response: |
            {
              "created": 1589478378,
              "data": [
                {
                  "url": "https://..."
                },
                {
                  "url": "https://..."
                }
              ]
            }
  /images/edits:
    post:
      operationId: createImageEdit
      tags:
        - Images
      summary: Creates an edited or extended image given an original image and a prompt.
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: "#/components/schemas/CreateImageEditRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ImagesResponse"
      x-oaiMeta:
        name: Create image edit
        returns: Returns a list of [image](/docs/api-reference/images/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/images/edits \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -F image="@otter.png" \
                -F mask="@mask.png" \
                -F prompt="A cute baby sea otter wearing a beret" \
                -F n=2 \
                -F size="1024x1024"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.images.edit(
                image=open("otter.png", "rb"),
                mask=open("mask.png", "rb"),
                prompt="A cute baby sea otter wearing a beret",
                n=2,
                size="1024x1024"
              )
            node.js: |-
              import fs from "fs";
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const image = await openai.images.edit({
                  image: fs.createReadStream("otter.png"),
                  mask: fs.createReadStream("mask.png"),
                  prompt: "A cute baby sea otter wearing a beret",
                });

                console.log(image.data);
              }
              main();
          response: |
            {
              "created": 1589478378,
              "data": [
                {
                  "url": "https://..."
                },
                {
                  "url": "https://..."
                }
              ]
            }
  /images/variations:
    post:
      operationId: createImageVariation
      tags:
        - Images
      summary: Creates a variation of a given image.
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: "#/components/schemas/CreateImageVariationRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ImagesResponse"
      x-oaiMeta:
        name: Create image variation
        returns: Returns a list of [image](/docs/api-reference/images/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/images/variations \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -F image="@otter.png" \
                -F n=2 \
                -F size="1024x1024"
            python: |
              from openai import OpenAI
              client = OpenAI()

              response = client.images.create_variation(
                image=open("image_edit_original.png", "rb"),
                n=2,
                size="1024x1024"
              )
            node.js: |-
              import fs from "fs";
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const image = await openai.images.createVariation({
                  image: fs.createReadStream("otter.png"),
                });

                console.log(image.data);
              }
              main();
          response: |
            {
              "created": 1589478378,
              "data": [
                {
                  "url": "https://..."
                },
                {
                  "url": "https://..."
                }
              ]
            }

  /embeddings:
    post:
      operationId: createEmbedding
      tags:
        - Embeddings
      summary: Creates an embedding vector representing the input text.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateEmbeddingRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateEmbeddingResponse"
      x-oaiMeta:
        name: Create embeddings
        returns: A list of [embedding](/docs/api-reference/embeddings/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/embeddings \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "Content-Type: application/json" \
                -d '{
                  "input": "The food was delicious and the waiter...",
                  "model": "text-embedding-ada-002",
                  "encoding_format": "float"
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.embeddings.create(
                model="text-embedding-ada-002",
                input="The food was delicious and the waiter...",
                encoding_format="float"
              )
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const embedding = await openai.embeddings.create({
                  model: "text-embedding-ada-002",
                  input: "The quick brown fox jumped over the lazy dog",
                  encoding_format: "float",
                });

                console.log(embedding);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "object": "embedding",
                  "embedding": [
                    0.0023064255,
                    -0.009327292,
                    .... (1536 floats total for ada-002)
                    -0.0028842222,
                  ],
                  "index": 0
                }
              ],
              "model": "text-embedding-ada-002",
              "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
              }
            }

  /audio/speech:
    post:
      operationId: createSpeech
      tags:
        - Audio
      summary: Generates audio from the input text.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateSpeechRequest"
      responses:
        "200":
          description: OK
          headers:
            Transfer-Encoding:
              schema:
                type: string
              description: chunked
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
      x-oaiMeta:
        name: Create speech
        returns: The audio file content.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/audio/speech \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "Content-Type: application/json" \
                -d '{
                  "model": "tts-1",
                  "input": "The quick brown fox jumped over the lazy dog.",
                  "voice": "alloy"
                }' \
                --output speech.mp3
            python: |
              from pathlib import Path
              import openai

              speech_file_path = Path(__file__).parent / "speech.mp3"
              response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="The quick brown fox jumped over the lazy dog."
              )
              response.stream_to_file(speech_file_path)
            node: |
              import fs from "fs";
              import path from "path";
              import OpenAI from "openai";

              const openai = new OpenAI();

              const speechFile = path.resolve("./speech.mp3");

              async function main() {
                const mp3 = await openai.audio.speech.create({
                  model: "tts-1",
                  voice: "alloy",
                  input: "Today is a wonderful day to build something people love!",
                });
                console.log(speechFile);
                const buffer = Buffer.from(await mp3.arrayBuffer());
                await fs.promises.writeFile(speechFile, buffer);
              }
              main();
  /audio/transcriptions:
    post:
      operationId: createTranscription
      tags:
        - Audio
      summary: Transcribes audio into the input language.
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: "#/components/schemas/CreateTranscriptionRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateTranscriptionResponse"
      x-oaiMeta:
        name: Create transcription
        returns: The transcribed text.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/audio/transcriptions \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "Content-Type: multipart/form-data" \
                -F file="@/path/to/file/audio.mp3" \
                -F model="whisper-1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              audio_file = open("speech.mp3", "rb")
              transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
              )
            node: |
              import fs from "fs";
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const transcription = await openai.audio.transcriptions.create({
                  file: fs.createReadStream("audio.mp3"),
                  model: "whisper-1",
                });

                console.log(transcription.text);
              }
              main();
          response: |
            {
              "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
            }
  /audio/translations:
    post:
      operationId: createTranslation
      tags:
        - Audio
      summary: Translates audio into English.
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: "#/components/schemas/CreateTranslationRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateTranslationResponse"
      x-oaiMeta:
        name: Create translation
        returns: The translated text.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/audio/translations \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "Content-Type: multipart/form-data" \
                -F file="@/path/to/file/german.m4a" \
                -F model="whisper-1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              audio_file = open("speech.mp3", "rb")
              transcript = client.audio.translations.create(
                model="whisper-1", 
                file=audio_file
              )
            node: |
              const { Configuration, OpenAIApi } = require("openai");
              const configuration = new Configuration({
                apiKey: process.env.OPENAI_API_KEY,
              });
              const openai = new OpenAIApi(configuration);
              const resp = await openai.createTranslation(
                fs.createReadStream("audio.mp3"),
                "whisper-1"
              );
          response: |
            {
              "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
            }

  /files:
    get:
      operationId: listFiles
      tags:
        - Files
      summary: Returns a list of files that belong to the user's organization.
      parameters:
        - in: query
          name: purpose
          required: false
          schema:
            type: string
          description: Only return files with the given purpose.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListFilesResponse"
      x-oaiMeta:
        name: List files
        returns: A list of [File](/docs/api-reference/files/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/files \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.files.list()
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const list = await openai.files.list();

                for await (const file of list) {
                  console.log(file);
                }
              }

              main();
          response: |
            {
              "data": [
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 175,
                  "created_at": 1613677385,
                  "filename": "salesOverview.pdf",
                  "purpose": "assistants",
                },
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 140,
                  "created_at": 1613779121,
                  "filename": "puppy.jsonl",
                  "purpose": "fine-tune",
                }
              ],
              "object": "list"
            }
    post:
      operationId: createFile
      tags:
        - Files
      summary: |
        Upload a file that can be used across various endpoints/features. The size of all the files uploaded by one organization can be up to 100 GB.

        The size of individual files for can be a maximum of 512MB. See the [Assistants Tools guide](/docs/assistants/tools) to learn more about the types of files supported. The Fine-tuning API only supports `.jsonl` files.

        Please [contact us](https://help.openai.com/) if you need to increase these storage limits.
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              $ref: "#/components/schemas/CreateFileRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/OpenAIFile"
      x-oaiMeta:
        name: Upload file
        returns: The uploaded [File](/docs/api-reference/files/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/files \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -F purpose="fine-tune" \
                -F file="@mydata.jsonl"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.files.create(
                file=open("mydata.jsonl", "rb"),
                purpose="fine-tune"
              )
            node.js: |-
              import fs from "fs";
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const file = await openai.files.create({
                  file: fs.createReadStream("mydata.jsonl"),
                  purpose: "fine-tune",
                });

                console.log(file);
              }

              main();
          response: |
            {
              "id": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
              "object": "file",
              "bytes": 120000,
              "created_at": 1677610602,
              "filename": "mydata.jsonl",
              "purpose": "fine-tune",
            }
  /files/{file_id}:
    delete:
      operationId: deleteFile
      tags:
        - Files
      summary: Delete a file.
      parameters:
        - in: path
          name: file_id
          required: true
          schema:
            type: string
          description: The ID of the file to use for this request.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DeleteFileResponse"
      x-oaiMeta:
        name: Delete file
        returns: Deletion status.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/files/file-abc123 \
                -X DELETE \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.files.delete("file-oaG6vwLtV3v3mWpvxexWDKxq")
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const file = await openai.files.del("file-abc123");

                console.log(file);
              }

              main();
          response: |
            {
              "id": "file-abc123",
              "object": "file",
              "deleted": true
            }
    get:
      operationId: retrieveFile
      tags:
        - Files
      summary: Returns information about a specific file.
      parameters:
        - in: path
          name: file_id
          required: true
          schema:
            type: string
          description: The ID of the file to use for this request.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/OpenAIFile"
      x-oaiMeta:
        name: Retrieve file
        returns: The [File](/docs/api-reference/files/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/files/file-BK7bzQj3FfZFXr7DbL6xJwfo \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.files.retrieve("file-BK7bzQj3FfZFXr7DbL6xJwfo")
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const file = await openai.files.retrieve("file-BK7bzQj3FfZFXr7DbL6xJwfo");

                console.log(file);
              }

              main();
          response: |
            {
              "id": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
              "object": "file",
              "bytes": 120000,
              "created_at": 1677610602,
              "filename": "mydata.jsonl",
              "purpose": "fine-tune",
            }
  /files/{file_id}/content:
    get:
      operationId: downloadFile
      tags:
        - Files
      summary: Returns the contents of the specified file.
      parameters:
        - in: path
          name: file_id
          required: true
          schema:
            type: string
          description: The ID of the file to use for this request.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                type: string
      x-oaiMeta:
        name: Retrieve file content
        returns: The file content.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/files/file-BK7bzQj3FfZFXr7DbL6xJwfo/content \
                -H "Authorization: Bearer $OPENAI_API_KEY" > file.jsonl
            python: |
              from openai import OpenAI
              client = OpenAI()

              content = client.files.retrieve_content("file-BK7bzQj3FfZFXr7DbL6xJwfo")
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const file = await openai.files.retrieveContent("file-BK7bzQj3FfZFXr7DbL6xJwfo");

                console.log(file);
              }

              main();

  /fine_tuning/jobs:
    post:
      operationId: createFineTuningJob
      tags:
        - Fine-tuning
      summary: |
        Creates a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.

        [Learn more about fine-tuning](/docs/guides/fine-tuning)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateFineTuningJobRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTuningJob"
      x-oaiMeta:
        name: Create fine-tuning job
        returns: A [fine-tuning.job](/docs/api-reference/fine-tuning/object) object.
        examples:
          - title: No hyperparameters
            request:
              curl: |
                curl https://api.openai.com/v1/fine_tuning/jobs \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "training_file": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
                    "model": "gpt-3.5-turbo"
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                client.fine_tuning.jobs.create(
                  training_file="file-abc123", 
                  model="gpt-3.5-turbo"
                )
              node.js: |
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const fineTune = await openai.fineTuning.jobs.create({
                    training_file: "file-abc123"
                  });

                  console.log(fineTune);
                }

                main();
            response: |
              {
                "object": "fine_tuning.job",
                "id": "ftjob-abc123",
                "model": "gpt-3.5-turbo-0613",
                "created_at": 1614807352,
                "fine_tuned_model": null,
                "organization_id": "org-123",
                "result_files": [],
                "status": "queued",
                "validation_file": null,
                "training_file": "file-abc123",
              }
          - title: Hyperparameters
            request:
              curl: |
                curl https://api.openai.com/v1/fine_tuning/jobs \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "training_file": "file-abc123",
                    "model": "gpt-3.5-turbo",
                    "hyperparameters": {
                      "n_epochs": 2
                    }
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                client.fine_tuning.jobs.create(
                  training_file="file-abc123", 
                  model="gpt-3.5-turbo", 
                  hyperparameters={
                    "n_epochs":2
                  }
                )
              node.js: |
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const fineTune = await openai.fineTuning.jobs.create({
                    training_file: "file-abc123",
                    model: "gpt-3.5-turbo",
                    hyperparameters: { n_epochs: 2 }
                  });

                  console.log(fineTune);
                }

                main();
            response: |
              {
                "object": "fine_tuning.job",
                "id": "ftjob-abc123",
                "model": "gpt-3.5-turbo-0613",
                "created_at": 1614807352,
                "fine_tuned_model": null,
                "organization_id": "org-123",
                "result_files": [],
                "status": "queued",
                "validation_file": null,
                "training_file": "file-abc123",
                "hyperparameters": {"n_epochs": 2},
              }
          - title: Validation file
            request:
              curl: |
                curl https://api.openai.com/v1/fine_tuning/jobs \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -d '{
                    "training_file": "file-abc123",
                    "validation_file": "file-abc123",
                    "model": "gpt-3.5-turbo"
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                client.fine_tuning.jobs.create(
                  training_file="file-abc123", 
                  validation_file="file-def456", 
                  model="gpt-3.5-turbo"
                )
              node.js: |
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const fineTune = await openai.fineTuning.jobs.create({
                    training_file: "file-abc123",
                    validation_file: "file-abc123"
                  });

                  console.log(fineTune);
                }

                main();
            response: |
              {
                "object": "fine_tuning.job",
                "id": "ftjob-abc123",
                "model": "gpt-3.5-turbo-0613",
                "created_at": 1614807352,
                "fine_tuned_model": null,
                "organization_id": "org-123",
                "result_files": [],
                "status": "queued",
                "validation_file": "file-abc123",
                "training_file": "file-abc123",
              }
    get:
      operationId: listPaginatedFineTuningJobs
      tags:
        - Fine-tuning
      summary: |
        List your organization's fine-tuning jobs
      parameters:
        - name: after
          in: query
          description: Identifier for the last job from the previous pagination request.
          required: false
          schema:
            type: string
        - name: limit
          in: query
          description: Number of fine-tuning jobs to retrieve.
          required: false
          schema:
            type: integer
            default: 20
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListPaginatedFineTuningJobsResponse"
      x-oaiMeta:
        name: List fine-tuning jobs
        returns: A list of paginated [fine-tuning job](/docs/api-reference/fine-tuning/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine_tuning/jobs?limit=2 \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.fine_tuning.jobs.list()
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const list = await openai.fineTuning.jobs.list();

                for await (const fineTune of list) {
                  console.log(fineTune);
                }
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "object": "fine_tuning.job.event",
                  "id": "ft-event-TjX0lMfOniCZX64t9PUQT5hn",
                  "created_at": 1689813489,
                  "level": "warn",
                  "message": "Fine tuning process stopping due to job cancellation",
                  "data": null,
                  "type": "message"
                },
                { ... },
                { ... }
              ], "has_more": true
            }
  /fine_tuning/jobs/{fine_tuning_job_id}:
    get:
      operationId: retrieveFineTuningJob
      tags:
        - Fine-tuning
      summary: |
        Get info about a fine-tuning job.

        [Learn more about fine-tuning](/docs/guides/fine-tuning)
      parameters:
        - in: path
          name: fine_tuning_job_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tuning job.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTuningJob"
      x-oaiMeta:
        name: Retrieve fine-tuning job
        returns: The [fine-tuning](/docs/api-reference/fine-tunes/object) object with the given ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine_tuning/jobs/ft-AF1WoRqd3aJAHsqc9NY7iL8F \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.fine_tuning.jobs.retrieve("ftjob-abc123")
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTuning.jobs.retrieve("ftjob-abc123");

                console.log(fineTune);
              }

              main();
          response: &fine_tuning_example |
            {
              "object": "fine_tuning.job",
              "id": "ftjob-abc123",
              "model": "davinci-002",
              "created_at": 1692661014,
              "finished_at": 1692661190,
              "fine_tuned_model": "ft:davinci-002:my-org:custom_suffix:7q8mpxmy",
              "organization_id": "org-123",
              "result_files": [
                  "file-abc123"
              ],
              "status": "succeeded",
              "validation_file": null,
              "training_file": "file-abc123",
              "hyperparameters": {
                  "n_epochs": 4,
              },
              "trained_tokens": 5768
            }
  /fine_tuning/jobs/{fine_tuning_job_id}/events:
    get:
      operationId: listFineTuningEvents
      tags:
        - Fine-tuning
      summary: |
        Get status updates for a fine-tuning job.
      parameters:
        - in: path
          name: fine_tuning_job_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tuning job to get events for.
        - name: after
          in: query
          description: Identifier for the last event from the previous pagination request.
          required: false
          schema:
            type: string
        - name: limit
          in: query
          description: Number of events to retrieve.
          required: false
          schema:
            type: integer
            default: 20
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListFineTuningJobEventsResponse"
      x-oaiMeta:
        name: List fine-tuning events
        returns: A list of fine-tuning event objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123/events \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.fine_tuning.jobs.list_events(
                fine_tuning_job_id="ftjob-abc123", 
                limit=2
              )
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const list = await openai.fineTuning.list_events(id="ftjob-abc123", limit=2);

                for await (const fineTune of list) {
                  console.log(fineTune);
                }
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "object": "fine_tuning.job.event",
                  "id": "ft-event-ddTJfwuMVpfLXseO0Am0Gqjm",
                  "created_at": 1692407401,
                  "level": "info",
                  "message": "Fine tuning job successfully completed",
                  "data": null,
                  "type": "message"
                },
                {
                  "object": "fine_tuning.job.event",
                  "id": "ft-event-tyiGuB72evQncpH87xe505Sv",
                  "created_at": 1692407400,
                  "level": "info",
                  "message": "New fine-tuned model created: ft:gpt-3.5-turbo:openai::7p4lURel",
                  "data": null,
                  "type": "message"
                }
              ],
              "has_more": true
            }
  /fine_tuning/jobs/{fine_tuning_job_id}/cancel:
    post:
      operationId: cancelFineTuningJob
      tags:
        - Fine-tuning
      summary: |
        Immediately cancel a fine-tune job.
      parameters:
        - in: path
          name: fine_tuning_job_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tuning job to cancel.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTuningJob"
      x-oaiMeta:
        name: Cancel fine-tuning
        returns: The cancelled [fine-tuning](/docs/api-reference/fine-tuning/object) object.
        examples:
          request:
            curl: |
              curl -X POST https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123/cancel \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.fine_tuning.jobs.cancel("ftjob-abc123")
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTuning.jobs.cancel("ftjob-abc123");

                console.log(fineTune);
              }
              main();
          response: |
            {
              "object": "fine_tuning.job",
              "id": "ftjob-abc123",
              "model": "gpt-3.5-turbo-0613",
              "created_at": 1689376978,
              "fine_tuned_model": null,
              "organization_id": "org-123",
              "result_files": [],
              "hyperparameters": {
                "n_epochs":  "auto"
              },
              "status": "cancelled",
              "validation_file": "file-abc123",
              "training_file": "file-abc123"
            }

  /fine-tunes:
    post:
      operationId: createFineTune
      deprecated: true
      tags:
        - Fine-tunes
      summary: |
        Creates a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.

        [Learn more about fine-tuning](/docs/guides/legacy-fine-tuning)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateFineTuneRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTune"
      x-oaiMeta:
        name: Create fine-tune
        returns: A [fine-tune](/docs/api-reference/fine-tunes/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine-tunes \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -d '{
                  "training_file": "file-abc123"
                }'
            python: |
              # deprecated
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTunes.create({
                  training_file: "file-abc123"
                });

                console.log(fineTune);
              }

              main();
          response: |
            {
              "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
              "object": "fine-tune",
              "model": "curie",
              "created_at": 1614807352,
              "events": [
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807352,
                  "level": "info",
                  "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
                }
              ],
              "fine_tuned_model": null,
              "hyperparams": {
                "batch_size": 4,
                "learning_rate_multiplier": 0.1,
                "n_epochs": 4,
                "prompt_loss_weight": 0.1,
              },
              "organization_id": "org-123",
              "result_files": [],
              "status": "pending",
              "validation_files": [],
              "training_files": [
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 1547276,
                  "created_at": 1610062281,
                  "filename": "my-data-train.jsonl",
                  "purpose": "fine-tune-results"
                }
              ],
              "updated_at": 1614807352,
            }
    get:
      operationId: listFineTunes
      deprecated: true
      tags:
        - Fine-tunes
      summary: |
        List your organization's fine-tuning jobs
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListFineTunesResponse"
      x-oaiMeta:
        name: List fine-tunes
        returns: A list of [fine-tune](/docs/api-reference/fine-tunes/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine-tunes \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              # deprecated
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const list = await openai.fineTunes.list();

                for await (const fineTune of list) {
                  console.log(fineTune);
                }
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
                  "object": "fine-tune",
                  "model": "curie",
                  "created_at": 1614807352,
                  "fine_tuned_model": null,
                  "hyperparams": { ... },
                  "organization_id": "org-123",
                  "result_files": [],
                  "status": "pending",
                  "validation_files": [],
                  "training_files": [ { ... } ],
                  "updated_at": 1614807352,
                },
                { ... },
                { ... }
              ]
            }
  /fine-tunes/{fine_tune_id}:
    get:
      operationId: retrieveFineTune
      deprecated: true
      tags:
        - Fine-tunes
      summary: |
        Gets info about the fine-tune job.

        [Learn more about fine-tuning](/docs/guides/legacy-fine-tuning)
      parameters:
        - in: path
          name: fine_tune_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tune job
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTune"
      x-oaiMeta:
        name: Retrieve fine-tune
        returns: The [fine-tune](/docs/api-reference/fine-tunes/object) object with the given ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine-tunes/ft-AF1WoRqd3aJAHsqc9NY7iL8F \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              # deprecated
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTunes.retrieve("ft-AF1WoRqd3aJAHsqc9NY7iL8F");

                console.log(fineTune);
              }

              main();
          response: &fine_tune_example |
            {
              "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
              "object": "fine-tune",
              "model": "curie",
              "created_at": 1614807352,
              "events": [
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807352,
                  "level": "info",
                  "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807356,
                  "level": "info",
                  "message": "Job started."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807861,
                  "level": "info",
                  "message": "Uploaded snapshot: curie:ft-acmeco-2021-03-03-21-44-20."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807864,
                  "level": "info",
                  "message": "Uploaded result files: file-abc123."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807864,
                  "level": "info",
                  "message": "Job succeeded."
                }
              ],
              "fine_tuned_model": "curie:ft-acmeco-2021-03-03-21-44-20",
              "hyperparams": {
                "batch_size": 4,
                "learning_rate_multiplier": 0.1,
                "n_epochs": 4,
                "prompt_loss_weight": 0.1,
              },
              "organization_id": "org-123",
              "result_files": [
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 81509,
                  "created_at": 1614807863,
                  "filename": "compiled_results.csv",
                  "purpose": "fine-tune-results"
                }
              ],
              "status": "succeeded",
              "validation_files": [],
              "training_files": [
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 1547276,
                  "created_at": 1610062281,
                  "filename": "my-data-train.jsonl",
                  "purpose": "fine-tune"
                }
              ],
              "updated_at": 1614807865,
            }
  /fine-tunes/{fine_tune_id}/cancel:
    post:
      operationId: cancelFineTune
      deprecated: true
      tags:
        - Fine-tunes
      summary: |
        Immediately cancel a fine-tune job.
      parameters:
        - in: path
          name: fine_tune_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tune job to cancel
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/FineTune"
      x-oaiMeta:
        name: Cancel fine-tune
        returns: The cancelled [fine-tune](/docs/api-reference/fine-tunes/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine-tunes/ft-AF1WoRqd3aJAHsqc9NY7iL8F/cancel \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              # deprecated
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTunes.cancel("ft-AF1WoRqd3aJAHsqc9NY7iL8F");

                console.log(fineTune);
              }
              main();
          response: |
            {
              "id": "ft-xhrpBbvVUzYGo8oUO1FY4nI7",
              "object": "fine-tune",
              "model": "curie",
              "created_at": 1614807770,
              "events": [ { ... } ],
              "fine_tuned_model": null,
              "hyperparams": { ... },
              "organization_id": "org-123",
              "result_files": [],
              "status": "cancelled",
              "validation_files": [],
              "training_files": [
                {
                  "id": "file-abc123",
                  "object": "file",
                  "bytes": 1547276,
                  "created_at": 1610062281,
                  "filename": "my-data-train.jsonl",
                  "purpose": "fine-tune"
                }
              ],
              "updated_at": 1614807789,
            }
  /fine-tunes/{fine_tune_id}/events:
    get:
      operationId: listFineTuneEvents
      deprecated: true
      tags:
        - Fine-tunes
      summary: |
        Get fine-grained status updates for a fine-tune job.
      parameters:
        - in: path
          name: fine_tune_id
          required: true
          schema:
            type: string
            example: ft-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the fine-tune job to get events for.
        - in: query
          name: stream
          required: false
          schema:
            type: boolean
            default: false
          description: |
            Whether to stream events for the fine-tune job. If set to true,
            events will be sent as data-only
            [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
            as they become available. The stream will terminate with a
            `data: [DONE]` message when the job is finished (succeeded, cancelled,
            or failed).

            If set to false, only events generated so far will be returned.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListFineTuneEventsResponse"
      x-oaiMeta:
        name: List fine-tune events
        returns: A list of fine-tune event objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/fine-tunes/ft-AF1WoRqd3aJAHsqc9NY7iL8F/events \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              # deprecated
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const fineTune = await openai.fineTunes.listEvents("ft-AF1WoRqd3aJAHsqc9NY7iL8F");

                console.log(fineTune);
              }
              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807352,
                  "level": "info",
                  "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807356,
                  "level": "info",
                  "message": "Job started."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807861,
                  "level": "info",
                  "message": "Uploaded snapshot: curie:ft-acmeco-2021-03-03-21-44-20."
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807864,
                  "level": "info",
                  "message": "Uploaded result files: file-abc123"
                },
                {
                  "object": "fine-tune-event",
                  "created_at": 1614807864,
                  "level": "info",
                  "message": "Job succeeded."
                }
              ]
            }

  /models:
    get:
      operationId: listModels
      tags:
        - Models
      summary: Lists the currently available models, and provides basic information about each one such as the owner and availability.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListModelsResponse"
      x-oaiMeta:
        name: List models
        returns: A list of [model](/docs/api-reference/models/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/models \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.models.list()
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const list = await openai.models.list();

                for await (const model of list) {
                  console.log(model);
                }
              }
              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "model-id-0",
                  "object": "model",
                  "created": 1686935002,
                  "owned_by": "organization-owner"
                },
                {
                  "id": "model-id-1",
                  "object": "model",
                  "created": 1686935002,
                  "owned_by": "organization-owner",
                },
                {
                  "id": "model-id-2",
                  "object": "model",
                  "created": 1686935002,
                  "owned_by": "openai"
                },
              ],
              "object": "list"
            }
  /models/{model}:
    get:
      operationId: retrieveModel
      tags:
        - Models
      summary: Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
      parameters:
        - in: path
          name: model
          required: true
          schema:
            type: string
            # ideally this will be an actual ID, so this will always work from browser
            example: gpt-3.5-turbo
          description: The ID of the model to use for this request
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Model"
      x-oaiMeta:
        name: Retrieve model
        returns: The [model](/docs/api-reference/models/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/models/VAR_model_id \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.models.retrieve("VAR_model_id")
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const model = await openai.models.retrieve("gpt-3.5-turbo");

                console.log(model);
              }

              main();
          response: &retrieve_model_response |
            {
              "id": "VAR_model_id",
              "object": "model",
              "created": 1686935002,
              "owned_by": "openai"
            }
    delete:
      operationId: deleteModel
      tags:
        - Models
      summary: Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.
      parameters:
        - in: path
          name: model
          required: true
          schema:
            type: string
            example: ft:gpt-3.5-turbo:acemeco:suffix:abc123
          description: The model to delete
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DeleteModelResponse"
      x-oaiMeta:
        name: Delete fine-tune model
        returns: Deletion status.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/models/ft:gpt-3.5-turbo:acemeco:suffix:abc123 \
                -X DELETE \
                -H "Authorization: Bearer $OPENAI_API_KEY"
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const model = await openai.models.del("ft:gpt-3.5-turbo:acemeco:suffix:abc123");

                console.log(model);
              }
              main();
          response: |
            {
              "id": "ft:gpt-3.5-turbo:acemeco:suffix:abc123",
              "object": "model",
              "deleted": true
            }

  /moderations:
    post:
      operationId: createModeration
      tags:
        - Moderations
      summary: Classifies if text violates OpenAI's Content Policy
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateModerationRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CreateModerationResponse"
      x-oaiMeta:
        name: Create moderation
        returns: A [moderation](/docs/api-reference/moderations/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/moderations \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -d '{
                  "input": "I want to kill them."
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              client.moderations.create(input="I want to kill them.")
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const moderation = await openai.moderations.create({ input: "I want to kill them." });

                console.log(moderation);
              }
              main();
          response: &moderation_example |
            {
              "id": "modr-XXXXX",
              "model": "text-moderation-005",
              "results": [
                {
                  "flagged": true,
                  "categories": {
                    "sexual": false,
                    "hate": false,
                    "harassment": false,
                    "self-harm": false,
                    "sexual/minors": false,
                    "hate/threatening": false,
                    "violence/graphic": false,
                    "self-harm/intent": false,
                    "self-harm/instructions": false,
                    "harassment/threatening": true,
                    "violence": true,
                  },
                  "category_scores": {
                    "sexual": 1.2282071e-06,
                    "hate": 0.010696256,
                    "harassment": 0.29842457,
                    "self-harm": 1.5236925e-08,
                    "sexual/minors": 5.7246268e-08,
                    "hate/threatening": 0.0060676364,
                    "violence/graphic": 4.435014e-06,
                    "self-harm/intent": 8.098441e-10,
                    "self-harm/instructions": 2.8498655e-11,
                    "harassment/threatening": 0.63055265,
                    "violence": 0.99011886,
                  }
                }
              ]
            }

  # Assistants
  /assistants:
    get:
      operationId: listAssistants
      tags:
        - Assistants
      summary: Returns a list of assistants.
      parameters:
        - name: limit
          in: query
          description: &pagination_limit_param_description |
            A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: &pagination_order_param_description |
            Sort order by the `created_at` timestamp of the objects. `asc` for ascending order and `desc` for descending order.
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: &pagination_after_param_description |
            A cursor for use in pagination. `after` is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after=obj_foo in order to fetch the next page of the list.
          schema:
            type: string
        - name: before
          in: query
          description: &pagination_before_param_description |
            A cursor for use in pagination. `before` is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous page of the list.
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListAssistantsResponse"
      x-oaiMeta:
        name: List assistants
        beta: true
        returns: A list of [assistant](/docs/api-reference/assistants/object) objects.
        examples:
          request:
            curl: |
              curl "https://api.openai.com/v1/assistants?order=desc&limit=20" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              my_assistants = client.beta.assistants.list(
                  order="desc",
                  limit="20",
              )
              print(my_assistants.data)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const myAssistants = await openai.beta.assistants.list({
                  order: "desc",
                  limit: "20",
                });

                console.log(myAssistants.data);
              }

              main();
          response: &list_assistants_example |
            {
              "object": "list",
              "data": [
                {
                  "id": "asst_abc123",
                  "object": "assistant",
                  "created_at": 1698982736,
                  "name": "Coding Tutor",
                  "description": null,
                  "model": "gpt-4",
                  "instructions": "You are a helpful assistant designed to make me better at coding!",
                  "tools": [],
                  "file_ids": [],
                  "metadata": {}
                },
                {
                  "id": "asst_abc456",
                  "object": "assistant",
                  "created_at": 1698982718,
                  "name": "My Assistant",
                  "description": null,
                  "model": "gpt-4",
                  "instructions": "You are a helpful assistant designed to make me better at coding!",
                  "tools": [],
                  "file_ids": [],
                  "metadata": {}
                },
                {
                  "id": "asst_abc789",
                  "object": "assistant",
                  "created_at": 1698982643,
                  "name": null,
                  "description": null,
                  "model": "gpt-4",
                  "instructions": null,
                  "tools": [],
                  "file_ids": [],
                  "metadata": {}
                }
              ],
              "first_id": "asst_abc123",
              "last_id": "asst_abc789",
              "has_more": false
            }
    post:
      operationId: createAssistant
      tags:
        - Assistants
      summary: Create an assistant with a model and instructions.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateAssistantRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AssistantObject"
      x-oaiMeta:
        name: Create assistant
        beta: true
        returns: An [assistant](/docs/api-reference/assistants/object) object.
        examples:
          - title: Code Interpreter
            request:
              curl: |
                curl "https://api.openai.com/v1/assistants" \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -H "OpenAI-Beta: assistants=v1" \
                  -d '{
                    "instructions": "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
                    "name": "Math Tutor"
                    "tools": [{"type": "code_interpreter"}],
                    "model": "gpt-4"
                  }'

              python: |
                from openai import OpenAI
                client = OpenAI()

                my_assistant = client.beta.assistants.create(
                    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
                    name="Math Tutor",
                    tools=[{"type": "code_interpreter"}],
                    model="gpt-4",
                )
                print(my_assistant)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const myAssistant = await openai.beta.assistants.create({
                    instructions:
                      "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
                    name: "Math Tutor",
                    tools: [{ type: "code_interpreter" }],
                    model: "gpt-4",
                  });

                  console.log(myAssistant);
                }

                main();
            response: &create_assistants_example |
              {
                "id": "asst_abc123",
                "object": "assistant",
                "created_at": 1698984975,
                "name": "Math Tutor",
                "description": null,
                "model": "gpt-4",
                "instructions": "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
                "tools": [
                  {
                    "type": "code_interpreter"
                  }
                ],
                "file_ids": [],
                "metadata": {}
              }
          - title: Files
            request:
              curl: |
                curl https://api.openai.com/v1/assistants \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -H "OpenAI-Beta: assistants=v1" \
                  -d '{
                    "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies.",
                    "tools": [{"type": "retrieval"}],
                    "model": "gpt-4",
                    "file_ids": ["file-abc123"]
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                my_assistant = client.beta.assistants.create(
                    instructions="You are an HR bot, and you have access to files to answer employee questions about company policies.",
                    name="HR Helper",
                    tools=[{"type": "retrieval"}],
                    model="gpt-4",
                    file_ids=["file-abc123"],
                )
                print(my_assistant)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const myAssistant = await openai.beta.assistants.create({
                    instructions:
                      "You are an HR bot, and you have access to files to answer employee questions about company policies.",
                    name: "HR Helper",
                    tools: [{ type: "retrieval" }],
                    model: "gpt-4",
                    file_ids: ["file-abc123"],
                  });

                  console.log(myAssistant);
                }

                main();
            response: |
              {
                "id": "asst_abc123",
                "object": "assistant",
                "created_at": 1699009403,
                "name": "HR Helper",
                "description": null,
                "model": "gpt-4",
                "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies.",
                "tools": [
                  {
                    "type": "retrieval"
                  }
                ],
                "file_ids": [
                  "file-abc123"
                ],
                "metadata": {}
              }

  /assistants/{assistant_id}:
    get:
      operationId: getAssistant
      tags:
        - Assistants
      summary: Retrieves an assistant.
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
          description: The ID of the assistant to retrieve.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AssistantObject"
      x-oaiMeta:
        name: Retrieve assistant
        beta: true
        returns: The [assistant](/docs/api-reference/assistants/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              my_assistant = client.beta.assistants.retrieve("asst_abc123")
              print(my_assistant)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const myAssistant = await openai.beta.assistants.retrieve(
                  "asst_abc123"
                );

                console.log(myAssistant);
              }

              main();
          response: |
            {
              "id": "asst_abc123",
              "object": "assistant",
              "created_at": 1699009709,
              "name": "HR Helper",
              "description": null,
              "model": "gpt-4",
              "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies.",
              "tools": [
                {
                  "type": "retrieval"
                }
              ],
              "file_ids": [
                "file-abc123"
              ],
              "metadata": {}
            }
    post:
      operationId: modifyAssistant
      tags:
        - Assistant
      summary: Modifies an assistant.
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
          description: The ID of the assistant to modify.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ModifyAssistantRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AssistantObject"
      x-oaiMeta:
        name: Modify assistant
        beta: true
        returns: The modified [assistant](/docs/api-reference/assistants/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -d '{
                    "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
                    "tools": [{"type": "retrieval"}],
                    "model": "gpt-4",
                    "file_ids": ["file-abc123", "file-abc456"]
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              my_updated_assistant = client.beta.assistants.update(
                "asst_abc123",
                instructions="You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
                name="HR Helper",
                tools=[{"type": "retrieval"}],
                model="gpt-4",
                file_ids=["file-abc123", "file-abc456"],
              )

              print(my_updated_assistant)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const myUpdatedAssistant = await openai.beta.assistants.update(
                  "asst_abc123",
                  {
                    instructions:
                      "You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
                    name: "HR Helper",
                    tools: [{ type: "retrieval" }],
                    model: "gpt-4",
                    file_ids: [
                      "file-abc123",
                      "file-abc456",
                    ],
                  }
                );

                console.log(myUpdatedAssistant);
              }

              main();
          response: |
            {
              "id": "asst_abc123",
              "object": "assistant",
              "created_at": 1699009709,
              "name": "HR Helper",
              "description": null,
              "model": "gpt-4",
              "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
              "tools": [
                {
                  "type": "retrieval"
                }
              ],
              "file_ids": [
                "file-abc123",
                "file-abc456"
              ],
              "metadata": {}
            }
    delete:
      operationId: deleteAssistant
      tags:
        - Assistants
      summary: Delete an assistant.
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
          description: The ID of the assistant to delete.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DeleteAssistantResponse"
      x-oaiMeta:
        name: Delete assistant
        beta: true
        returns: Deletion status
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -X DELETE
            python: |
              from openai import OpenAI
              client = OpenAI()

              response = client.beta.assistants.delete("asst_QLoItBbqwyAJEzlTy4y9kOMM")
              print(response)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const response = await openai.beta.assistants.del("asst_QLoItBbqwyAJEzlTy4y9kOMM");

                console.log(response);
              }
              main();
          response: |
            {
              "id": "asst_abc123",
              "object": "assistant.deleted",
              "deleted": true
            }

  /threads:
    post:
      operationId: createThread
      tags:
        - Assistants
      summary: Create a thread.
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateThreadRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ThreadObject"
      x-oaiMeta:
        name: Create thread
        beta: true
        returns: A [thread](/docs/api-reference/threads) object.
        examples:
          - title: Empty
            request:
              curl: |
                curl https://api.openai.com/v1/threads \
                  -H "Content-Type: application/json" \
                  -H "Authorization: Bearer $OPENAI_API_KEY" \
                  -H "OpenAI-Beta: assistants=v1" \
                  -d ''
              python: |
                from openai import OpenAI
                client = OpenAI()

                empty_thread = client.beta.threads.create()
                print(empty_thread)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const emptyThread = await openai.beta.threads.create();

                  console.log(emptyThread);
                }

                main();
            response: |
              {
                "id": "thread_abc123",
                "object": "thread",
                "created_at": 1699012949,
                "metadata": {}
              }
          - title: Messages
            request:
              curl: |
                curl https://api.openai.com/v1/threads \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -d '{
                    "messages": [{
                      "role": "user",
                      "content": "Hello, what is AI?",
                      "file_ids": ["file-abc123"]
                    }, {
                      "role": "user",
                      "content": "How does AI work? Explain it in simple terms."
                    }]
                  }'
              python: |
                from openai import OpenAI
                client = OpenAI()

                message_thread = client.beta.threads.create(
                  messages=[
                    {
                      "role": "user",
                      "content": "Hello, what is AI?",
                      "file_ids": ["file-abc123"],
                    },
                    {
                      "role": "user",
                      "content": "How does AI work? Explain it in simple terms."
                    },
                  ]
                )

                print(message_thread)
              node.js: |-
                import OpenAI from "openai";

                const openai = new OpenAI();

                async function main() {
                  const messageThread = await openai.beta.threads.create({
                    messages: [
                      {
                        role: "user",
                        content: "Hello, what is AI?",
                        file_ids: ["file-abc123"],
                      },
                      {
                        role: "user",
                        content: "How does AI work? Explain it in simple terms.",
                      },
                    ],
                  });

                  console.log(messageThread);
                }

                main();
            response: |
              {
                id: 'thread_abc123',
                object: 'thread',
                created_at: 1699014083,
                metadata: {}
              }

  /threads/{thread_id}:
    get:
      operationId: getThread
      tags:
        - Assistants
      summary: Retrieves a thread.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to retrieve.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ThreadObject"
      x-oaiMeta:
        name: Retrieve thread
        beta: true
        returns: The [thread](/docs/api-reference/threads/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              my_thread = client.beta.threads.retrieve("thread_abc123")
              print(my_thread)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const myThread = await openai.beta.threads.retrieve(
                  "thread_abc123"
                );

                console.log(myThread);
              }

              main();
          response: |
            {
              "id": "thread_abc123",
              "object": "thread",
              "created_at": 1699014083,
              "metadata": {}
            }
    post:
      operationId: modifyThread
      tags:
        - Assistants
      summary: Modifies a thread.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to modify. Only the `metadata` can be modified.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ModifyThreadRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ThreadObject"
      x-oaiMeta:
        name: Modify thread
        beta: true
        returns: The modified [thread](/docs/api-reference/threads/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -d '{
                    "metadata": {
                      "modified": "true",
                      "user": "abc123"
                    }
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              my_updated_thread = client.beta.threads.update(
                "thread_abc123", 
                metadata={
                  "modified": "true", 
                  "user": "abc123"
                }
              )
              print(my_updated_thread)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const updatedThread = await openai.beta.threads.update(
                  "thread_abc123",
                  {
                    metadata: { modified: "true", user: "abc123" },
                  }
                );

                console.log(updatedThread);
              }

              main();
          response: |
            {
              "id": "thread_abc123",
              "object": "thread",
              "created_at": 1699014083,
              "metadata": {
                "modified": "true",
                "user": "abc123"
              }
            }
    delete:
      operationId: deleteThread
      tags:
        - Assistants
      summary: Delete a thread.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to delete.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DeleteThreadResponse"
      x-oaiMeta:
        name: Delete thread
        beta: true
        returns: Deletion status
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -X DELETE
            python: |
              from openai import OpenAI
              client = OpenAI()

              response = client.beta.threads.delete("thread_abc123")
              print(response)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const response = await openai.beta.threads.del("thread_abc123");

                console.log(response);
              }
              main();
          response: |
            {
              "id": "thread_abc123",
              "object": "thread.deleted",
              "deleted": true
            }

  /threads/{thread_id}/messages:
    get:
      operationId: listMessages
      tags:
        - Assistants
      summary: Returns a list of messages for a given thread.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) the messages belong to.
        - name: limit
          in: query
          description: *pagination_limit_param_description
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: *pagination_order_param_description
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: *pagination_after_param_description
          schema:
            type: string
        - name: before
          in: query
          description: *pagination_before_param_description
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListMessagesResponse"
      x-oaiMeta:
        name: List messages
        beta: true
        returns: A list of [message](/docs/api-reference/messages) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123/messages \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              thread_messages = client.beta.threads.messages.list("thread_1OWaSqVIxJdy3KYnJLbXEWhy")
              print(thread_messages.data)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const threadMessages = await openai.beta.threads.messages.list(
                  "thread_1OWaSqVIxJdy3KYnJLbXEWhy"
                );

                console.log(threadMessages.data);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "msg_abc123",
                  "object": "thread.message",
                  "created_at": 1699016383,
                  "thread_id": "thread_abc123",
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": {
                        "value": "How does AI work? Explain it in simple terms.",
                        "annotations": []
                      }
                    }
                  ],
                  "file_ids": [],
                  "assistant_id": null,
                  "run_id": null,
                  "metadata": {}
                },
                {
                  "id": "msg_abc456",
                  "object": "thread.message",
                  "created_at": 1699016383,
                  "thread_id": "thread_abc123",
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": {
                        "value": "Hello, what is AI?",
                        "annotations": []
                      }
                    }
                  ],
                  "file_ids": [
                    "file-abc123"
                  ],
                  "assistant_id": null,
                  "run_id": null,
                  "metadata": {}
                }
              ],
              "first_id": "msg_abc123",
              "last_id": "msg_abc456",
              "has_more": false
            }
    post:
      operationId: createMessage
      tags:
        - Assistants
      summary: Create a message.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) to create a message for.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateMessageRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/MessageObject"
      x-oaiMeta:
        name: Create message
        beta: true
        returns: A [message](/docs/api-reference/messages/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123/messages \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -d '{
                    "role": "user",
                    "content": "How does AI work? Explain it in simple terms."
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              thread_message = client.beta.threads.messages.create(
                "thread_abc123",
                role="user",
                content="How does AI work? Explain it in simple terms.",
              )
              print(thread_message)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const threadMessages = await openai.beta.threads.messages.create(
                  "thread_abc123",
                  { role: "user", content: "How does AI work? Explain it in simple terms." }
                );

                console.log(threadMessages);
              }

              main();
          response: |
            {
              "id": "msg_abc123",
              "object": "thread.message",
              "created_at": 1699017614,
              "thread_id": "thread_abc123",
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": {
                    "value": "How does AI work? Explain it in simple terms.",
                    "annotations": []
                  }
                }
              ],
              "file_ids": [],
              "assistant_id": null,
              "run_id": null,
              "metadata": {}
            }

  /threads/{thread_id}/messages/{message_id}:
    get:
      operationId: getMessage
      tags:
        - Assistants
      summary: Retrieve a message.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) to which this message belongs.
        - in: path
          name: message_id
          required: true
          schema:
            type: string
          description: The ID of the message to retrieve.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/MessageObject"
      x-oaiMeta:
        name: Retrieve message
        beta: true
        returns: The [message](/docs/api-reference/threads/messages/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123/messages/msg_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1"
            python: |
              from openai import OpenAI
              client = OpenAI()

              message = client.beta.threads.messages.retrieve(
                message_id="msg_abc123",
                thread_id="thread_abc123",
              )
              print(message)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const message = await openai.beta.threads.messages.retrieve(
                  "thread_abc123",
                  "msg_abc123"
                );

                console.log(message);
              }

              main();
          response: |
            {
              "id": "msg_abc123",
              "object": "thread.message",
              "created_at": 1699017614,
              "thread_id": "thread_abc123",
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": {
                    "value": "How does AI work? Explain it in simple terms.",
                    "annotations": []
                  }
                }
              ],
              "file_ids": [],
              "assistant_id": null,
              "run_id": null,
              "metadata": {}
            }
    post:
      operationId: modifyMessage
      tags:
        - Assistants
      summary: Modifies a message.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to which this message belongs.
        - in: path
          name: message_id
          required: true
          schema:
            type: string
          description: The ID of the message to modify.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ModifyMessageRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/MessageObject"
      x-oaiMeta:
        name: Modify message
        beta: true
        returns: The modified [message](/docs/api-reference/threads/messages/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_abc123/messages/msg_abc123 \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $OPENAI_API_KEY" \
                -H "OpenAI-Beta: assistants=v1" \
                -d '{
                    "metadata": {
                      "modified": "true",
                      "user": "abc123"
                    }
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              message = client.beta.threads.messages.update(
                message_id="msg_abc12",
                thread_id="thread_abc123",
                metadata={
                  "modified": "true",
                  "user": "abc123",
                },
              )
              print(message)
            node.js: |-
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const message = await openai.beta.threads.messages.update(
                  "thread_abc123",
                  "msg_abc123",
                  {
                    metadata: {
                      modified: "true",
                      user: "abc123",
                    },
                  }
                }'
          response: |
            {
              "id": "msg_abc123",
              "object": "thread.message",
              "created_at": 1699017614,
              "thread_id": "thread_abc123",
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": {
                    "value": "How does AI work? Explain it in simple terms.",
                    "annotations": []
                  }
                }
              ],
              "file_ids": [],
              "assistant_id": null,
              "run_id": null,
              "metadata": {
                "modified": "true",
                "user": "abc123"
              }
            }

  /threads/runs:
    post:
      operationId: createThreadAndRun
      tags:
        - Assistants
      summary: Create a thread and run it in one request.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateThreadAndRunRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Create thread and run
        beta: true
        returns: A [run](/docs/api-reference/runs/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/runs \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1' \
                -d '{
                    "assistant_id": "asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
                    "thread": {
                      "messages": [
                        {"role": "user", "content": "Explain deep learning to a 5 year old."}
                      ]
                    }
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.create_and_run(
                assistant_id="asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
                thread={
                  "messages": [
                    {"role": "user", "content": "Explain deep learning to a 5 year old."}
                  ]
                }
              )
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.createAndRun({
                  assistant_id: "asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
                  thread: {
                    messages: [
                      { role: "user", content: "Explain deep learning to a 5 year old." },
                    ],
                  },
                });

                console.log(run);
              }

              main();
          response: |
            {
              "id": "run_3Qudf05GGhCleEg9ggwfJQih",
              "object": "thread.run",
              "created_at": 1699076792,
              "assistant_id": "asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
              "thread_id": "thread_Ec3eKZcWI00WDZRC7FZci8hP",
              "status": "queued",
              "started_at": null,
              "expires_at": 1699077392,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": null,
              "last_error": null,
              "model": "gpt-4",
              "instructions": "You are a helpful assistant.",
              "tools": [],
              "file_ids": [],
              "metadata": {}
            }

  /threads/{thread_id}/runs:
    get:
      operationId: listRuns
      tags:
        - Assistants
      summary: Returns a list of runs belonging to a thread.
      parameters:
        - name: thread_id
          in: path
          required: true
          schema:
            type: string
          description: The ID of the thread the run belongs to.
        - name: limit
          in: query
          description: *pagination_limit_param_description
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: *pagination_order_param_description
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: *pagination_after_param_description
          schema:
            type: string
        - name: before
          in: query
          description: *pagination_before_param_description
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListRunsResponse"
      x-oaiMeta:
        name: List runs
        beta: true
        returns: A list of [run](/docs/api-reference/runs/object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              runs = client.beta.threads.runs.list(
                "thread_BDDwIqM4KgHibXX3mqmN3Lgs"
              )
              print(runs)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const runs = await openai.beta.threads.runs.list(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs"
                );

                console.log(runs);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "run_5pyUEwhaPk11vCKiDneUWXXY",
                  "object": "thread.run",
                  "created_at": 1699075072,
                  "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
                  "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "status": "completed",
                  "started_at": 1699075072,
                  "expires_at": null,
                  "cancelled_at": null,
                  "failed_at": null,
                  "completed_at": 1699075073,
                  "last_error": null,
                  "model": "gpt-3.5-turbo",
                  "instructions": null,
                  "tools": [
                    {
                      "type": "code_interpreter"
                    }
                  ],
                  "file_ids": [
                    "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                    "file-dEWwUbt2UGHp3v0e0DpCzemP"
                  ],
                  "metadata": {}
                },
                {
                  "id": "run_UWvV94U0FQYiT2rlbBrdEVmC",
                  "object": "thread.run",
                  "created_at": 1699063290,
                  "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
                  "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "status": "completed",
                  "started_at": 1699063290,
                  "expires_at": null,
                  "cancelled_at": null,
                  "failed_at": null,
                  "completed_at": 1699063291,
                  "last_error": null,
                  "model": "gpt-3.5-turbo",
                  "instructions": null,
                  "tools": [
                    {
                      "type": "code_interpreter"
                    }
                  ],
                  "file_ids": [
                    "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                    "file-dEWwUbt2UGHp3v0e0DpCzemP"
                  ],
                  "metadata": {}
                }
              ],
              "first_id": "run_5pyUEwhaPk11vCKiDneUWXXY",
              "last_id": "run_UWvV94U0FQYiT2rlbBrdEVmC",
              "has_more": false
            }
    post:
      operationId: createRun
      tags:
        - Assistants
      summary: Create a run.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to run.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateRunRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Create run
        beta: true
        returns: A [run](/docs/api-reference/runs/object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1' \
                -d '{
                  "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ"
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.runs.create(
                thread_id="thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                assistant_id="asst_nGl00s4xa9zmVY6Fvuvz9wwQ"
              )
              print(run)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.runs.create(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  { assistant_id: "asst_nGl00s4xa9zmVY6Fvuvz9wwQ" }
                );

                console.log(run);
              }

              main();
          response: &run_object_example |
            {
              "id": "run_UWvV94U0FQYiT2rlbBrdEVmC",
              "object": "thread.run",
              "created_at": 1699063290,
              "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
              "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
              "status": "queued",
              "started_at": 1699063290,
              "expires_at": null,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": 1699063291,
              "last_error": null,
              "model": "gpt-4",
              "instructions": null,
              "tools": [
                {
                  "type": "code_interpreter"
                }
              ],
              "file_ids": [
                "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                "file-dEWwUbt2UGHp3v0e0DpCzemP"
              ],
              "metadata": {}
            }

  /threads/{thread_id}/runs/{run_id}:
    get:
      operationId: getRun
      tags:
        - Assistants
      summary: Retrieves a run.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) that was run.
        - in: path
          name: run_id
          required: true
          schema:
            type: string
          description: The ID of the run to retrieve.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Retrieve run
        beta: true
        returns: The [run](/docs/api-reference/runs/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs/run_5pyUEwhaPk11vCKiDneUWXXY \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.runs.retrieve(
                thread_id="thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                run_id="run_5pyUEwhaPk11vCKiDneUWXXY"
              )
              print(run)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.runs.retrieve(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "run_5pyUEwhaPk11vCKiDneUWXXY"
                );

                console.log(run);
              }

              main();
          response: |
            {
              "id": "run_5pyUEwhaPk11vCKiDneUWXXY",
              "object": "thread.run",
              "created_at": 1699075072,
              "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
              "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
              "status": "completed",
              "started_at": 1699075072,
              "expires_at": null,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": 1699075073,
              "last_error": null,
              "model": "gpt-3.5-turbo",
              "instructions": null,
              "tools": [
                {
                  "type": "code_interpreter"
                }
              ],
              "file_ids": [
                "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                "file-dEWwUbt2UGHp3v0e0DpCzemP"
              ],
              "metadata": {}
            }
    post:
      operationId: modifyRun
      tags:
        - Assistants
      summary: Modifies a run.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) that was run.
        - in: path
          name: run_id
          required: true
          schema:
            type: string
          description: The ID of the run to modify.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ModifyRunRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Modify run
        beta: true
        returns: The modified [run](/docs/api-reference/runs/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs/run_5pyUEwhaPk11vCKiDneUWXXY \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1' \
                -d '{
                  "metadata": {
                    "user_id": "user_zmVY6FvuBDDwIqM4KgH"
                  }
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.runs.update(
                thread_id="thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                run_id="run_5pyUEwhaPk11vCKiDneUWXXY",
                metadata={"user_id": "user_zmVY6FvuBDDwIqM4KgH"},
              )
              print(run)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.runs.update(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "run_5pyUEwhaPk11vCKiDneUWXXY",
                  {
                    metadata: {
                      user_id: "user_zmVY6FvuBDDwIqM4KgH",
                    },
                  }
                );

                console.log(run);
              }

              main();
          response: |
            {
              "id": "run_5pyUEwhaPk11vCKiDneUWXXY",
              "object": "thread.run",
              "created_at": 1699075072,
              "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
              "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
              "status": "completed",
              "started_at": 1699075072,
              "expires_at": null,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": 1699075073,
              "last_error": null,
              "model": "gpt-3.5-turbo",
              "instructions": null,
              "tools": [
                {
                  "type": "code_interpreter"
                }
              ],
              "file_ids": [
                "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                "file-dEWwUbt2UGHp3v0e0DpCzemP"
              ],
              "metadata": {
                "user_id": "user_zmVY6FvuBDDwIqM4KgH"
              }
            }

  /threads/{thread_id}/runs/{run_id}/submit_tool_outputs:
    post:
      operationId: submitToolOuputsToRun
      tags:
        - Assistants
      summary: |
        When a run has the `status: "requires_action"` and `required_action.type` is `submit_tool_outputs`, this endpoint can be used to submit the outputs from the tool calls once they're all completed. All outputs must be submitted in a single request.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the [thread](/docs/api-reference/threads) to which this run belongs.
        - in: path
          name: run_id
          required: true
          schema:
            type: string
          description: The ID of the run that requires the tool output submission.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/SubmitToolOutputsRunRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Submit tool outputs to run
        beta: true
        returns: The modified [run](/docs/api-reference/runs/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_EdR8UvCDJ035LFEJZMt3AxCd/runs/run_PHLyHQYIQn4F7JrSXslEYWwh/submit_tool_outputs \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1' \
                -d '{
                  "tool_outputs": [
                    {
                      "tool_call_id": "call_MbELIQcB72cq35Yzo2MRw5qs",
                      "output": "28C"
                    }
                  ]
                }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.runs.submit_tool_outputs(
                thread_id="thread_EdR8UvCDJ035LFEJZMt3AxCd",
                run_id="run_PHLyHQYIQn4F7JrSXslEYWwh",
                tool_outputs=[
                  {
                    "tool_call_id": "call_MbELIQcB72cq35Yzo2MRw5qs",
                    "output": "28C"
                  }
                ]
              )
              print(run)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.runs.submitToolOutputs(
                  "thread_EdR8UvCDJ035LFEJZMt3AxCd",
                  "run_PHLyHQYIQn4F7JrSXslEYWwh",
                  {
                    tool_outputs: [
                      {
                        tool_call_id: "call_MbELIQcB72cq35Yzo2MRw5qs",
                        output: "28C",
                      },
                    ],
                  }
                );

                console.log(run);
              }

              main();
          response: |
            {
              "id": "run_PHLyHQYIQn4F7JrSXslEYWwh",
              "object": "thread.run",
              "created_at": 1699075592,
              "assistant_id": "asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
              "thread_id": "thread_EdR8UvCDJ035LFEJZMt3AxCd",
              "status": "queued",
              "started_at": 1699075592,
              "expires_at": 1699076192,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": null,
              "last_error": null,
              "model": "gpt-4",
              "instructions": "You tell the weather.",
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_weather",
                    "description": "Determine weather in my location",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "location": {
                          "type": "string",
                          "description": "The city and state e.g. San Francisco, CA"
                        },
                        "unit": {
                          "type": "string",
                          "enum": [
                            "c",
                            "f"
                          ]
                        }
                      },
                      "required": [
                        "location"
                      ]
                    }
                  }
                }
              ],
              "file_ids": [],
              "metadata": {}
            }

  /threads/{thread_id}/runs/{run_id}/cancel:
    post:
      operationId: cancelRun
      tags:
        - Assistants
      summary: Cancels a run that is `in_progress`.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to which this run belongs.
        - in: path
          name: run_id
          required: true
          schema:
            type: string
          description: The ID of the run to cancel.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunObject"
      x-oaiMeta:
        name: Cancel a run
        beta: true
        returns: The modified [run](/docs/api-reference/runs/object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_1cjnJPXj8MFiqTx58jU9TivC/runs/run_BeRGmpGt2wb1VI22ZRniOkrR/cancel \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'OpenAI-Beta: assistants=v1' \
                -X POST
            python: |
              from openai import OpenAI
              client = OpenAI()

              run = client.beta.threads.runs.cancel(
                thread_id="thread_1cjnJPXj8MFiqTx58jU9TivC",
                run_id="run_BeRGmpGt2wb1VI22ZRniOkrR"
              )
              print(run)
            node.js: |
              import OpenAI from "openai";

              const openai = new OpenAI();

              async function main() {
                const run = await openai.beta.threads.runs.cancel(
                  "thread_1cjnJPXj8MFiqTx58jU9TivC",
                  "run_BeRGmpGt2wb1VI22ZRniOkrR"
                );

                console.log(run);
              }

              main();
          response: |
            {
              "id": "run_BeRGmpGt2wb1VI22ZRniOkrR",
              "object": "thread.run",
              "created_at": 1699076126,
              "assistant_id": "asst_IgmpQTah3ZfPHCVZjTqAY8Kv",
              "thread_id": "thread_1cjnJPXj8MFiqTx58jU9TivC",
              "status": "cancelling",
              "started_at": 1699076126,
              "expires_at": 1699076726,
              "cancelled_at": null,
              "failed_at": null,
              "completed_at": null,
              "last_error": null,
              "model": "gpt-4",
              "instructions": "You summarize books.",
              "tools": [
                {
                  "type": "retrieval"
                }
              ],
              "file_ids": [],
              "metadata": {}
            }

  /threads/{thread_id}/runs/{run_id}/steps:
    get:
      operationId: listRunSteps
      tags:
        - Assistants
      summary: Returns a list of run steps belonging to a run.
      parameters:
        - name: thread_id
          in: path
          required: true
          schema:
            type: string
          description: The ID of the thread the run and run steps belong to.
        - name: run_id
          in: path
          required: true
          schema:
            type: string
          description: The ID of the run the run steps belong to.
        - name: limit
          in: query
          description: *pagination_limit_param_description
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: *pagination_order_param_description
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: *pagination_after_param_description
          schema:
            type: string
        - name: before
          in: query
          description: *pagination_before_param_description
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListRunStepsResponse"
      x-oaiMeta:
        name: List run steps
        beta: true
        returns: A list of [run step](/docs/api-reference/runs/step-object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs/run_UWvV94U0FQYiT2rlbBrdEVmC/steps \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run_steps = client.beta.threads.runs.steps.list(
                  thread_id="thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  run_id="run_UWvV94U0FQYiT2rlbBrdEVmC"
              )
              print(run_steps)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const runStep = await openai.beta.threads.runs.steps.list(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "run_UWvV94U0FQYiT2rlbBrdEVmC"
                );
                console.log(runStep);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "step_QyjyrsVsysd7F4K894BZHG97",
                  "object": "thread.run.step",
                  "created_at": 1699063291,
                  "run_id": "run_UWvV94U0FQYiT2rlbBrdEVmC",
                  "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
                  "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "type": "message_creation",
                  "status": "completed",
                  "cancelled_at": null,
                  "completed_at": 1699063291,
                  "expired_at": null,
                  "failed_at": null,
                  "last_error": null,
                  "step_details": {
                    "type": "message_creation",
                    "message_creation": {
                      "message_id": "msg_6YmiCRmMbbE6FALYNePPHqwm"
                    }
                  }
                }
              ],
              "first_id": "step_QyjyrsVsysd7F4K894BZHG97",
              "last_id": "step_QyjyrsVsysd7F4K894BZHG97",
              "has_more": false
            }

  /threads/{thread_id}/runs/{run_id}/steps/{step_id}:
    get:
      operationId: getRunStep
      tags:
        - Assistants
      summary: Retrieves a run step.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
          description: The ID of the thread to which the run and run step belongs.
        - in: path
          name: run_id
          required: true
          schema:
            type: string
          description: The ID of the run to which the run step belongs.
        - in: path
          name: step_id
          required: true
          schema:
            type: string
          description: The ID of the run step to retrieve.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/RunStepObject"
      x-oaiMeta:
        name: Retrieve run step
        beta: true
        returns: The [run step](/docs/api-reference/runs/step-object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_BDDwIqM4KgHibXX3mqmN3Lgs/runs/run_UWvV94U0FQYiT2rlbBrdEVmC/steps/step_QyjyrsVsysd7F4K894BZHG97 \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              run_step = client.beta.threads.runs.steps.retrieve(
                  thread_id="thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  run_id="run_UWvV94U0FQYiT2rlbBrdEVmC",
                  step_id="step_QyjyrsVsysd7F4K894BZHG97"
              )
              print(run_step)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const runStep = await openai.beta.threads.runs.steps.retrieve(
                  "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
                  "run_UWvV94U0FQYiT2rlbBrdEVmC",
                  "step_QyjyrsVsysd7F4K894BZHG97"
                );
                console.log(runStep);
              }

              main();
          response: &run_step_object_example |
            {
              "id": "step_QyjyrsVsysd7F4K894BZHG97",
              "object": "thread.run.step",
              "created_at": 1699063291,
              "run_id": "run_UWvV94U0FQYiT2rlbBrdEVmC",
              "assistant_id": "asst_nGl00s4xa9zmVY6Fvuvz9wwQ",
              "thread_id": "thread_BDDwIqM4KgHibXX3mqmN3Lgs",
              "type": "message_creation",
              "status": "completed",
              "cancelled_at": null,
              "completed_at": 1699063291,
              "expired_at": null,
              "failed_at": null,
              "last_error": null,
              "step_details": {
                "type": "message_creation",
                "message_creation": {
                  "message_id": "msg_6YmiCRmMbbE6FALYNePPHqwm"
                }
              }
            }

  /assistants/{assistant_id}/files:
    get:
      operationId: listAssistantFiles
      tags:
        - Assistants
      summary: Returns a list of assistant files.
      parameters:
        - name: assistant_id
          in: path
          description: The ID of the assistant the file belongs to.
          required: true
          schema:
            type: string
        - name: limit
          in: query
          description: *pagination_limit_param_description
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: *pagination_order_param_description
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: *pagination_after_param_description
          schema:
            type: string
        - name: before
          in: query
          description: *pagination_before_param_description
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListAssistantFilesResponse"
      x-oaiMeta:
        name: List assistant files
        beta: true
        returns: A list of [assistant file](/docs/api-reference/assistants/file-object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_DUGk5I7sK0FpKeijvrO30z9J/files \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              assistant_files = client.beta.assistants.files.list(
                assistant_id="asst_DUGk5I7sK0FpKeijvrO30z9J"
              )
              print(assistant_files)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const assistantFiles = await openai.beta.assistants.files.list(
                  "asst_FBOFvAOHhwEWMghbMGseaPGQ"
                );
                console.log(assistantFiles);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
                  "object": "assistant.file",
                  "created_at": 1699060412,
                  "assistant_id": "asst_DUGk5I7sK0FpKeijvrO30z9J"
                },
                {
                  "id": "file-9F1ex49ipEnKzyLUNnCA0Yzx",
                  "object": "assistant.file",
                  "created_at": 1699060412,
                  "assistant_id": "asst_DUGk5I7sK0FpKeijvrO30z9J"
                }
              ],
              "first_id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
              "last_id": "file-9F1ex49ipEnKzyLUNnCA0Yzx",
              "has_more": false
            }
    post:
      operationId: createAssistantFile
      tags:
        - Assistants
      summary: Create an assistant file by attaching a [File](/docs/api-reference/files) to an [assistant](/docs/api-reference/assistants).
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
            example: file-AF1WoRqd3aJAHsqc9NY7iL8F
          description: |
            The ID of the assistant for which to create a File.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateAssistantFileRequest"
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AssistantFileObject"
      x-oaiMeta:
        name: Create assistant file
        beta: true
        returns: An [assistant file](/docs/api-reference/assistants/file-object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_FBOFvAOHhwEWMghbMGseaPGQ/files \
                  -H 'Authorization: Bearer $OPENAI_API_KEY"' \
                  -H 'Content-Type: application/json' \
                  -H 'OpenAI-Beta: assistants=v1' \
                  -d '{
                    "file_id": "file-wB6RM6wHdA49HfS2DJ9fEyrH"
                  }'
            python: |
              from openai import OpenAI
              client = OpenAI()

              assistant_file = client.beta.assistants.files.create(
                assistant_id="asst_FBOFvAOHhwEWMghbMGseaPGQ", 
                file_id="file-wB6RM6wHdA49HfS2DJ9fEyrH"
              )
              print(assistant_file)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const myAssistantFile = await openai.beta.assistants.files.create(
                  "asst_FBOFvAOHhwEWMghbMGseaPGQ", 
                  { 
                    file_id: "file-wB6RM6wHdA49HfS2DJ9fEyrH"
                  }
                );
                console.log(myAssistantFile);
              }

              main();
          response: &assistant_file_object |
            {
              "id": "file-wB6RM6wHdA49HfS2DJ9fEyrH",
              "object": "assistant.file",
              "created_at": 1699055364,
              "assistant_id": "asst_FBOFvAOHhwEWMghbMGseaPGQ"
            }

  /assistants/{assistant_id}/files/{file_id}:
    get:
      operationId: getAssistantFile
      tags:
        - Assistants
      summary: Retrieves an AssistantFile.
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
          description: The ID of the assistant who the file belongs to.
        - in: path
          name: file_id
          required: true
          schema:
            type: string
          description: The ID of the file we're getting.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AssistantFileObject"
      x-oaiMeta:
        name: Retrieve assistant file
        beta: true
        returns: The [assistant file](/docs/api-reference/assistants/file-object) object matching the specified ID.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_FBOFvAOHhwEWMghbMGseaPGQ/files/file-wB6RM6wHdA49HfS2DJ9fEyrH \
                -H 'Authorization: Bearer $OPENAI_API_KEY"' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              assistant_file = client.beta.assistants.files.retrieve(
                assistant_id="asst_FBOFvAOHhwEWMghbMGseaPGQ",
                file_id="file-wB6RM6wHdA49HfS2DJ9fEyrH"
              )
              print(assistant_file)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const myAssistantFile = await openai.beta.assistants.files.retrieve(
                  "asst_FBOFvAOHhwEWMghbMGseaPGQ",
                  "file-wB6RM6wHdA49HfS2DJ9fEyrH"
                );
                console.log(myAssistantFile);
              }

              main();
          response: *assistant_file_object
    delete:
      operationId: deleteAssistantFile
      tags:
        - Assistants
      summary: Delete an assistant file.
      parameters:
        - in: path
          name: assistant_id
          required: true
          schema:
            type: string
          description: The ID of the assistant that the file belongs to.
        - in: path
          name: file_id
          required: true
          schema:
            type: string
          description: The ID of the file to delete.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/DeleteAssistantFileResponse"
      x-oaiMeta:
        name: Delete assistant file
        beta: true
        returns: Deletion status
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/assistants/asst_DUGk5I7sK0FpKeijvrO30z9J/files/file-9F1ex49ipEnKzyLUNnCA0Yzx \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1' \
                -X DELETE
            python: |
              from openai import OpenAI
              client = OpenAI()

              deleted_assistant_file = client.beta.assistants.files.delete(
                  assistant_id="asst_DUGk5I7sK0FpKeijvrO30z9J",
                  file_id="file-dEWwUbt2UGHp3v0e0DpCzemP"
              )
              print(deleted_assistant_file)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const deletedAssistantFile = await openai.beta.assistants.files.del(
                  "asst_FBOFvAOHhwEWMghbMGseaPGQ",
                  "file-wB6RM6wHdA49HfS2DJ9fEyrH"
                );
                console.log(deletedAssistantFile);
              }

              main();
          response: |
            {
              id: "file-BK7bzQj3FfZFXr7DbL6xJwfo",
              object: "assistant.file.deleted",
              deleted: true
            }

  /threads/{thread_id}/messages/{message_id}/files:
    get:
      operationId: listMessageFiles
      tags:
        - Assistants
      summary: Returns a list of message files.
      parameters:
        - name: thread_id
          in: path
          description: The ID of the thread that the message and files belong to.
          required: true
          schema:
            type: string
        - name: message_id
          in: path
          description: The ID of the message that the files belongs to.
          required: true
          schema:
            type: string
        - name: limit
          in: query
          description: *pagination_limit_param_description
          required: false
          schema:
            type: integer
            default: 20
        - name: order
          in: query
          description: *pagination_order_param_description
          schema:
            type: string
            default: desc
            enum: ["asc", "desc"]
        - name: after
          in: query
          description: *pagination_after_param_description
          schema:
            type: string
        - name: before
          in: query
          description: *pagination_before_param_description
          schema:
            type: string
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListMessageFilesResponse"
      x-oaiMeta:
        name: List message files
        beta: true
        returns: A list of [message file](/docs/api-reference/messages/file-object) objects.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_RGUhOuO9b2nrktrmsQ2uSR6I/messages/msg_q3XhbGmMzsqEFa81gMLBDAVU/files \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              message_files = client.beta.threads.messages.files.list(
                thread_id="thread_RGUhOuO9b2nrktrmsQ2uSR6I",
                message_id="msg_q3XhbGmMzsqEFa81gMLBDAVU"
              )
              print(message_files)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const messageFiles = await openai.beta.threads.messages.files.list(
                  "thread_RGUhOuO9b2nrktrmsQ2uSR6I",
                  "msg_q3XhbGmMzsqEFa81gMLBDAVU"
                );
                console.log(messageFiles);
              }

              main();
          response: |
            {
              "object": "list",
              "data": [
                {
                  "id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
                  "object": "thread.message.file",
                  "created_at": 1699061776,
                  "message_id": "msg_q3XhbGmMzsqEFa81gMLBDAVU"
                },
                {
                  "id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
                  "object": "thread.message.file",
                  "created_at": 1699061776,
                  "message_id": "msg_q3XhbGmMzsqEFa81gMLBDAVU"
                }
              ],
              "first_id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
              "last_id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
              "has_more": false
            }

  /threads/{thread_id}/messages/{message_id}/files/{file_id}:
    get:
      operationId: getMessageFile
      tags:
        - Assistants
      summary: Retrieves a message file.
      parameters:
        - in: path
          name: thread_id
          required: true
          schema:
            type: string
            example: thread_AF1WoRqd3aJAHsqc9NY7iL8F
          description: The ID of the thread to which the message and File belong.
        - in: path
          name: message_id
          required: true
          schema:
            type: string
            example: msg_AF1WoRqd3aJAHsqc9NY7iL8F
          description: The ID of the message the file belongs to.
        - in: path
          name: file_id
          required: true
          schema:
            type: string
            example: file-AF1WoRqd3aJAHsqc9NY7iL8F
          description: The ID of the file being retrieved.
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/MessageFileObject"
      x-oaiMeta:
        name: Retrieve message file
        beta: true
        returns: The [message file](/docs/api-reference/messages/file-object) object.
        examples:
          request:
            curl: |
              curl https://api.openai.com/v1/threads/thread_RGUhOuO9b2nrktrmsQ2uSR6I/messages/msg_q3XhbGmMzsqEFa81gMLBDAVU/files/file-dEWwUbt2UGHp3v0e0DpCzemP \
                -H 'Authorization: Bearer $OPENAI_API_KEY' \
                -H 'Content-Type: application/json' \
                -H 'OpenAI-Beta: assistants=v1'
            python: |
              from openai import OpenAI
              client = OpenAI()

              message_files = client.beta.threads.messages.files.retrieve(
                  thread_id="thread_RGUhOuO9b2nrktrmsQ2uSR6I",
                  message_id="msg_q3XhbGmMzsqEFa81gMLBDAVU",
                  file_id="file-dEWwUbt2UGHp3v0e0DpCzemP"
              )
              print(message_files)
            node.js: |
              import OpenAI from "openai";
              const openai = new OpenAI();

              async function main() {
                const messageFile = await openai.beta.threads.messages.files.retrieve(
                  "thread_RGUhOuO9b2nrktrmsQ2uSR6I",
                  "msg_q3XhbGmMzsqEFa81gMLBDAVU",
                  "file-dEWwUbt2UGHp3v0e0DpCzemP"
                );
                console.log(messageFile);
              }

              main();
          response: |
            {
              "id": "file-dEWwUbt2UGHp3v0e0DpCzemP",
              "object": "thread.message.file",
              "created_at": 1699061776,
              "message_id": "msg_q3XhbGmMzsqEFa81gMLBDAVU"
            }

components:
  securitySchemes:
    ApiKeyAuth:
      type: http
      scheme: "bearer"

  schemas:
    Error:
      type: object
      properties:
        code:
          type: string
          nullable: true
        message:
          type: string
          nullable: false
        param:
          type: string
          nullable: true
        type:
          type: string
          nullable: false
      required:
        - type
        - message
        - param
        - code
    ErrorResponse:
      type: object
      properties:
        error:
          $ref: "#/components/schemas/Error"
      required:
        - error

    ListModelsResponse:
      type: object
      properties:
        object:
          type: string
          enum: [list]
        data:
          type: array
          items:
            $ref: "#/components/schemas/Model"
      required:
        - object
        - data
    DeleteModelResponse:
      type: object
      properties:
        id:
          type: string
        deleted:
          type: boolean
        object:
          type: string
      required:
        - id
        - object
        - deleted

    CreateCompletionRequest:
      type: object
      properties:
        model:
          description: &model_description |
            ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them.
          anyOf:
            - type: string
            - type: string
              enum:
                [
                  "babbage-002",
                  "davinci-002",
                  "gpt-3.5-turbo-instruct",
                  "text-davinci-003",
                  "text-davinci-002",
                  "text-davinci-001",
                  "code-davinci-002",
                  "text-curie-001",
                  "text-babbage-001",
                  "text-ada-001",
                ]
          x-oaiTypeLabel: string
        prompt:
          description: &completions_prompt_description |
            The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.

            Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.
          default: "<|endoftext|>"
          nullable: true
          oneOf:
            - type: string
              default: ""
              example: "This is a test."
            - type: array
              items:
                type: string
                default: ""
                example: "This is a test."
            - type: array
              minItems: 1
              items:
                type: integer
              example: "[1212, 318, 257, 1332, 13]"
            - type: array
              minItems: 1
              items:
                type: array
                minItems: 1
                items:
                  type: integer
              example: "[[1212, 318, 257, 1332, 13]]"
        best_of:
          type: integer
          default: 1
          minimum: 0
          maximum: 20
          nullable: true
          description: &completions_best_of_description |
            Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.

            When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.

            **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        echo:
          type: boolean
          default: false
          nullable: true
          description: &completions_echo_description >
            Echo back the prompt in addition to the completion
        frequency_penalty:
          type: number
          default: 0
          minimum: -2
          maximum: 2
          nullable: true
          description: &completions_frequency_penalty_description |
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

            [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details)
        logit_bias: &completions_logit_bias
          type: object
          x-oaiTypeLabel: map
          default: null
          nullable: true
          additionalProperties:
            type: integer
          description: &completions_logit_bias_description |
            Modify the likelihood of specified tokens appearing in the completion.

            Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.

            As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token from being generated.
        logprobs: &completions_logprobs_configuration
          type: integer
          minimum: 0
          maximum: 5
          default: null
          nullable: true
          description: &completions_logprobs_description |
            Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.

            The maximum value for `logprobs` is 5.
        max_tokens:
          type: integer
          minimum: 0
          default: 16
          example: 16
          nullable: true
          description: &completions_max_tokens_description |
            The maximum number of [tokens](/tokenizer) to generate in the completion.

            The token count of your prompt plus `max_tokens` cannot exceed the model's context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens.
        n:
          type: integer
          minimum: 1
          maximum: 128
          default: 1
          example: 1
          nullable: true
          description: &completions_completions_description |
            How many completions to generate for each prompt.

            **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        presence_penalty:
          type: number
          default: 0
          minimum: -2
          maximum: 2
          nullable: true
          description: &completions_presence_penalty_description |
            Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.

            [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details)
        seed: &completions_seed_param
          type: integer
          minimum: -9223372036854775808
          maximum: 9223372036854775807
          nullable: true
          description: |
            If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result.

            Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend.
        stop:
          description: &completions_stop_description >
            Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
          default: null
          nullable: true
          oneOf:
            - type: string
              default: <|endoftext|>
              example: "\n"
              nullable: true
            - type: array
              minItems: 1
              maxItems: 4
              items:
                type: string
                example: '["\n"]'
        stream:
          description: >
            Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
            as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
          type: boolean
          nullable: true
          default: false
        suffix:
          description: The suffix that comes after a completion of inserted text.
          default: null
          nullable: true
          type: string
          example: "test."
        temperature:
          type: number
          minimum: 0
          maximum: 2
          default: 1
          example: 1
          nullable: true
          description: &completions_temperature_description |
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

            We generally recommend altering this or `top_p` but not both.
        top_p:
          type: number
          minimum: 0
          maximum: 1
          default: 1
          example: 1
          nullable: true
          description: &completions_top_p_description |
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

            We generally recommend altering this or `temperature` but not both.
        user: &end_user_param_configuration
          type: string
          example: user-1234
          description: |
            A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids).
      required:
        - model
        - prompt

    CreateCompletionResponse:
      type: object
      description: |
        Represents a completion response from the API. Note: both the streamed and non-streamed response objects share the same shape (unlike the chat endpoint).
      properties:
        id:
          type: string
          description: A unique identifier for the completion.
        choices:
          type: array
          description: The list of completion choices the model generated for the input prompt.
          items:
            type: object
            required:
              - finish_reason
              - index
              - logprobs
              - text
            properties:
              finish_reason:
                type: string
                description: &completion_finish_reason_description |
                  The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,
                  `length` if the maximum number of tokens specified in the request was reached,
                  or `content_filter` if content was omitted due to a flag from our content filters.
                enum: ["stop", "length", "content_filter"]
              index:
                type: integer
              logprobs:
                type: object
                nullable: true
                properties:
                  text_offset:
                    type: array
                    items:
                      type: integer
                  token_logprobs:
                    type: array
                    items:
                      type: number
                  tokens:
                    type: array
                    items:
                      type: string
                  top_logprobs:
                    type: array
                    items:
                      type: object
                      additionalProperties:
                        type: number
              text:
                type: string
        created:
          type: integer
          description: The Unix timestamp (in seconds) of when the completion was created.
        model:
          type: string
          description: The model used for completion.
        system_fingerprint:
          type: string
          description: |
            This fingerprint represents the backend configuration that the model runs with.

            Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
        object:
          type: string
          description: The object type, which is always "text_completion"
          enum: [text_completion]
        usage:
          $ref: "#/components/schemas/CompletionUsage"
      required:
        - id
        - object
        - created
        - model
        - choices
      x-oaiMeta:
        name: The completion object
        legacy: true
        example: |
          {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "gpt-3.5-turbo",
            "choices": [
              {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": null,
                "finish_reason": "length"
              }
            ],
            "usage": {
              "prompt_tokens": 5,
              "completion_tokens": 7,
              "total_tokens": 12
            }
          }

    ChatCompletionRequestMessageContentPart:
      oneOf:
        - $ref: "#/components/schemas/ChatCompletionRequestMessageContentPartText"
        - $ref: "#/components/schemas/ChatCompletionRequestMessageContentPartImage"
      x-oaiExpandable: true

    ChatCompletionRequestMessageContentPartImage:
      type: object
      title: Image content part
      properties:
        type:
          type: string
          enum: ["image_url"]
          description: The type of the content part.
        image_url:
          type: object
          properties:
            url:
              type: string
              description: Either a URL of the image or the base64 encoded image data.
              format: uri
            detail:
              type: string
              description: Specifies the detail level of the image.
              enum: ["auto", "low", "high"]
              default: "auto"
          required:
            - url
      required:
        - type
        - image_url

    ChatCompletionRequestMessageContentPartText:
      type: object
      title: Text content part
      properties:
        type:
          type: string
          enum: ["text"]
          description: The type of the content part.
        text:
          type: string
          description: The text content.
      required:
        - type
        - text

    ChatCompletionRequestMessage:
      oneOf:
        - $ref: "#/components/schemas/ChatCompletionRequestSystemMessage"
        - $ref: "#/components/schemas/ChatCompletionRequestUserMessage"
        - $ref: "#/components/schemas/ChatCompletionRequestAssistantMessage"
        - $ref: "#/components/schemas/ChatCompletionRequestToolMessage"
        - $ref: "#/components/schemas/ChatCompletionRequestFunctionMessage"
      x-oaiExpandable: true

    ChatCompletionRequestSystemMessage:
      type: object
      title: System message
      properties:
        content:
          nullable: true
          description: The contents of the system message.
          type: string
        role:
          type: string
          enum: ["system"]
          description: The role of the messages author, in this case `system`.
      required:
        - content
        - role

    ChatCompletionRequestUserMessage:
      type: object
      title: User message
      properties:
        content:
          nullable: true
          description: |
            The contents of the user message.
          oneOf:
            - type: string
              description: The text contents of the message.
              title: Text content
            - type: array
              description: An array of content parts with a defined type, each can be of type `text` or `image_url` when passing in images. You can pass multiple images by adding multiple `image_url` content parts. Image input is only supported when using the `gpt-4-visual-preview` model.
              title: Array of content parts
              items:
                $ref: "#/components/schemas/ChatCompletionRequestMessageContentPart"
              minItems: 1
        role:
          type: string
          enum: ["user"]
          description: The role of the messages author, in this case `user`.
      required:
        - content
        - role

    ChatCompletionRequestAssistantMessage:
      type: object
      title: Assistant message
      properties:
        content:
          nullable: true
          type: string
          description: |
            The contents of the assistant message.
        role:
          type: string
          enum: ["assistant"]
          description: The role of the messages author, in this case `assistant`.
        tool_calls:
          $ref: "#/components/schemas/ChatCompletionMessageToolCalls"
        function_call:
          type: object
          deprecated: true
          description: "Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model."
          properties:
            arguments:
              type: string
              description: The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
            name:
              type: string
              description: The name of the function to call.
          required:
            - arguments
            - name
      required:
        - content
        - role

    ChatCompletionRequestToolMessage:
      type: object
      title: Tool message
      properties:
        role:
          type: string
          enum: ["tool"]
          description: The role of the messages author, in this case `tool`.
        content:
          nullable: true
          type: string
          description: The contents of the tool message.
        tool_call_id:
          type: string
          description: Tool call that this message is responding to.
      required:
        - role
        - content
        - tool_call_id

    ChatCompletionRequestFunctionMessage:
      type: object
      title: Function message
      deprecated: true
      properties:
        role:
          type: string
          enum: ["function"]
          description: The role of the messages author, in this case `function`.
        content:
          type: string
          nullable: true
          description: The return value from the function call, to return to the model.
        name:
          type: string
          description: The name of the function to call.
      required:
        - role
        - name
        - content

    FunctionParameters:
      type: object
      description: "The parameters the functions accepts, described as a JSON Schema object. See the [guide](/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.\n\nTo describe a function that accepts no parameters, provide the value `{\"type\": \"object\", \"properties\": {}}`."
      additionalProperties: true

    ChatCompletionFunctions:
      type: object
      deprecated: true
      properties:
        description:
          type: string
          description: A description of what the function does, used by the model to choose when and how to call the function.
        name:
          type: string
          description: The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        parameters:
          $ref: "#/components/schemas/FunctionParameters"
      required:
        - name
        - parameters

    ChatCompletionFunctionCallOption:
      type: object
      description: >
        Specifying a particular function via `{"name": "my_function"}` forces the model to call that function.
      properties:
        name:
          type: string
          description: The name of the function to call.
      required:
        - name

    ChatCompletionTool:
      type: object
      properties:
        type:
          type: string
          enum: ["function"]
          description: The type of the tool. Currently, only `function` is supported.
        function:
          $ref: "#/components/schemas/FunctionObject"
      required:
        - type
        - function

    FunctionObject:
      type: object
      properties:
        description:
          type: string
          description: A description of what the function does, used by the model to choose when and how to call the function.
        name:
          type: string
          description: The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        parameters:
          $ref: "#/components/schemas/FunctionParameters"
      required:
        - name
        - parameters

    ChatCompletionToolChoiceOption:
      description: |
        Controls which (if any) function is called by the model.
        `none` means the model will not call a function and instead generates a message.
        `auto` means the model can pick between generating a message or calling a function.
        Specifying a particular function via `{"type: "function", "function": {"name": "my_function"}}` forces the model to call that function.

        `none` is the default when no functions are present. `auto` is the default if functions are present.
      oneOf:
        - type: string
          description: >
            `none` means the model will not call a function and instead generates a message.
            `auto` means the model can pick between generating a message or calling a function.
          enum: [none, auto]
        - $ref: "#/components/schemas/ChatCompletionNamedToolChoice"
      x-oaiExpandable: true

    ChatCompletionNamedToolChoice:
      type: object
      description: Specifies a tool the model should use. Use to force the model to call a specific function.
      properties:
        type:
          type: string
          enum: ["function"]
          description: The type of the tool. Currently, only `function` is supported.
        function:
          type: object
          properties:
            name:
              type: string
              description: The name of the function to call.
          required:
            - name

    ChatCompletionMessageToolCalls:
      type: array
      description: The tool calls generated by the model, such as function calls.
      items:
        $ref: "#/components/schemas/ChatCompletionMessageToolCall"

    ChatCompletionMessageToolCall:
      type: object
      properties:
        # TODO: index included when streaming
        id:
          type: string
          description: The ID of the tool call.
        type:
          type: string
          enum: ["function"]
          description: The type of the tool. Currently, only `function` is supported.
        function:
          type: object
          description: The function that the model called.
          properties:
            name:
              type: string
              description: The name of the function to call.
            arguments:
              type: string
              description: The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
          required:
            - name
            - arguments
      required:
        - id
        - type
        - function

    ChatCompletionMessageToolCallChunk:
      type: object
      properties:
        index:
          type: integer
        id:
          type: string
          description: The ID of the tool call.
        type:
          type: string
          enum: ["function"]
          description: The type of the tool. Currently, only `function` is supported.
        function:
          type: object
          properties:
            name:
              type: string
              description: The name of the function to call.
            arguments:
              type: string
              description: The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
      required:
        - index

    # Note, this isn't referenced anywhere, but is kept as a convenience to record all possible roles in one place.
    ChatCompletionRole:
      type: string
      description: The role of the author of a message
      enum:
        - system
        - user
        - assistant
        - tool
        - function

    ChatCompletionResponseMessage:
      type: object
      description: A chat completion message generated by the model.
      properties:
        content:
          type: string
          description: The contents of the message.
          nullable: true
        tool_calls:
          $ref: "#/components/schemas/ChatCompletionMessageToolCalls"
        role:
          type: string
          enum: ["assistant"]
          description: The role of the author of this message.
        function_call:
          type: object
          deprecated: true
          description: "Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model."
          properties:
            arguments:
              type: string
              description: The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
            name:
              type: string
              description: The name of the function to call.
          required:
            - name
            - arguments
      required:
        - role
        - content

    ChatCompletionStreamResponseDelta:
      type: object
      description: A chat completion delta generated by streamed model responses.
      properties:
        content:
          type: string
          description: The contents of the chunk message.
          nullable: true
        function_call:
          deprecated: true
          type: object
          description: "Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model."
          properties:
            arguments:
              type: string
              description: The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
            name:
              type: string
              description: The name of the function to call.
        tool_calls:
          type: array
          items:
            $ref: "#/components/schemas/ChatCompletionMessageToolCallChunk"
        role:
          type: string
          enum: ["system", "user", "assistant", "tool"]
          description: The role of the author of this message.

    CreateChatCompletionRequest:
      type: object
      properties:
        messages:
          description: A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
          type: array
          minItems: 1
          items:
            $ref: "#/components/schemas/ChatCompletionRequestMessage"
        model:
          description: ID of the model to use. See the [model endpoint compatibility](/docs/models/model-endpoint-compatibility) table for details on which models work with the Chat API.
          example: "gpt-3.5-turbo"
          anyOf:
            - type: string
            - type: string
              enum:
                [
                  "gpt-4-1106-preview",
                  "gpt-4-vision-preview",
                  "gpt-4",
                  "gpt-4-0314",
                  "gpt-4-0613",
                  "gpt-4-32k",
                  "gpt-4-32k-0314",
                  "gpt-4-32k-0613",
                  "gpt-3.5-turbo-1106",
                  "gpt-3.5-turbo",
                  "gpt-3.5-turbo-16k",
                  "gpt-3.5-turbo-0301",
                  "gpt-3.5-turbo-0613",
                  "gpt-3.5-turbo-16k-0613",
                ]
          x-oaiTypeLabel: string
        frequency_penalty:
          type: number
          default: 0
          minimum: -2
          maximum: 2
          nullable: true
          description: *completions_frequency_penalty_description
        logit_bias:
          type: object
          x-oaiTypeLabel: map
          default: null
          nullable: true
          additionalProperties:
            type: integer
          description: |
            Modify the likelihood of specified tokens appearing in the completion.

            Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        max_tokens:
          description: |
            The maximum number of [tokens](/tokenizer) to generate in the chat completion.

            The total length of input tokens and generated tokens is limited by the model's context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens.
          default: inf
          type: integer
          nullable: true
        n:
          type: integer
          minimum: 1
          maximum: 128
          default: 1
          example: 1
          nullable: true
          description: How many chat completion choices to generate for each input message.
        presence_penalty:
          type: number
          default: 0
          minimum: -2
          maximum: 2
          nullable: true
          description: *completions_presence_penalty_description
        response_format:
          type: object
          description: |
            An object specifying the format that the model must output. 

            Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the message the model generates is valid JSON.

            **Important:** when using JSON mode, you **must** also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in increased latency and appearance of a "stuck" request. Also note that the message content may be partially cut off if `finish_reason="length"`, which indicates the generation exceeded `max_tokens` or the conversation exceeded the max context length.
          properties:
            type:
              type: string
              enum: ["text", "json_object"]
              example: "json_object"
              default: "text"
              description: Must be one of `text` or `json_object`.
        seed:
          type: integer
          minimum: -9223372036854775808
          maximum: 9223372036854775807
          nullable: true
          description: |
            This feature is in Beta. 
            If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result.
            Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend.
          x-oaiMeta:
            beta: true
        stop:
          description: |
            Up to 4 sequences where the API will stop generating further tokens.
          default: null
          oneOf:
            - type: string
              nullable: true
            - type: array
              minItems: 1
              maxItems: 4
              items:
                type: string
        stream:
          description: >
            If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
            as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
          type: boolean
          nullable: true
          default: false
        temperature:
          type: number
          minimum: 0
          maximum: 2
          default: 1
          example: 1
          nullable: true
          description: *completions_temperature_description
        top_p:
          type: number
          minimum: 0
          maximum: 1
          default: 1
          example: 1
          nullable: true
          description: *completions_top_p_description
        tools:
          type: array
          description: >
            A list of tools the model may call. Currently, only functions are supported as a tool.
            Use this to provide a list of functions the model may generate JSON inputs for.
          items:
            $ref: "#/components/schemas/ChatCompletionTool"
        tool_choice:
          $ref: "#/components/schemas/ChatCompletionToolChoiceOption"
        user: *end_user_param_configuration
        function_call:
          deprecated: true
          description: |
            Deprecated in favor of `tool_choice`.

            Controls which (if any) function is called by the model.
            `none` means the model will not call a function and instead generates a message.
            `auto` means the model can pick between generating a message or calling a function.
            Specifying a particular function via `{"name": "my_function"}` forces the model to call that function.

            `none` is the default when no functions are present. `auto`` is the default if functions are present.
          oneOf:
            - type: string
              description: >
                `none` means the model will not call a function and instead generates a message.
                `auto` means the model can pick between generating a message or calling a function.
              enum: [none, auto]
            - $ref: "#/components/schemas/ChatCompletionFunctionCallOption"
          x-oaiExpandable: true
        functions:
          deprecated: true
          description: |
            Deprecated in favor of `tools`.

            A list of functions the model may generate JSON inputs for.
          type: array
          minItems: 1
          maxItems: 128
          items:
            $ref: "#/components/schemas/ChatCompletionFunctions"

      required:
        - model
        - messages

    CreateChatCompletionResponse:
      type: object
      description: Represents a chat completion response returned by model, based on the provided input.
      properties:
        id:
          type: string
          description: A unique identifier for the chat completion.
        choices:
          type: array
          description: A list of chat completion choices. Can be more than one if `n` is greater than 1.
          items:
            type: object
            required:
              - finish_reason
              - index
              - message
            properties:
              finish_reason:
                type: string
                description: &chat_completion_finish_reason_description |
                  The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,
                  `length` if the maximum number of tokens specified in the request was reached,
                  `content_filter` if content was omitted due to a flag from our content filters,
                  `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function.
                enum:
                  [
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                  ]
              index:
                type: integer
                description: The index of the choice in the list of choices.
              message:
                $ref: "#/components/schemas/ChatCompletionResponseMessage"
        created:
          type: integer
          description: The Unix timestamp (in seconds) of when the chat completion was created.
        model:
          type: string
          description: The model used for the chat completion.
        system_fingerprint:
          type: string
          description: |
            This fingerprint represents the backend configuration that the model runs with.

            Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
        object:
          type: string
          description: The object type, which is always `chat.completion`.
          enum: [chat.completion]
        usage:
          $ref: "#/components/schemas/CompletionUsage"
      required:
        - choices
        - created
        - id
        - model
        - object
      x-oaiMeta:
        name: The chat completion object
        group: chat
        example: *chat_completion_example

    CreateChatCompletionFunctionResponse:
      type: object
      description: Represents a chat completion response returned by model, based on the provided input.
      properties:
        id:
          type: string
          description: A unique identifier for the chat completion.
        choices:
          type: array
          description: A list of chat completion choices. Can be more than one if `n` is greater than 1.
          items:
            type: object
            required:
              - finish_reason
              - index
              - message
            properties:
              finish_reason:
                type: string
                description:
                  &chat_completion_function_finish_reason_description |
                  The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, or `function_call` if the model called a function.
                enum: ["stop", "length", "function_call", "content_filter"]
              index:
                type: integer
                description: The index of the choice in the list of choices.
              message:
                $ref: "#/components/schemas/ChatCompletionResponseMessage"
        created:
          type: integer
          description: The Unix timestamp (in seconds) of when the chat completion was created.
        model:
          type: string
          description: The model used for the chat completion.
        system_fingerprint:
          type: string
          description: |
            This fingerprint represents the backend configuration that the model runs with.

            Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
        object:
          type: string
          description: The object type, which is always `chat.completion`.
          enum: [chat.completion]
        usage:
          $ref: "#/components/schemas/CompletionUsage"
      required:
        - choices
        - created
        - id
        - model
        - object
      x-oaiMeta:
        name: The chat completion object
        group: chat
        example: *chat_completion_function_example

    ListPaginatedFineTuningJobsResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/FineTuningJob"
        has_more:
          type: boolean
        object:
          type: string
          enum: [list]
      required:
        - object
        - data
        - has_more

    CreateChatCompletionStreamResponse:
      type: object
      description: Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
      properties:
        id:
          type: string
          description: A unique identifier for the chat completion. Each chunk has the same ID.
        choices:
          type: array
          description: A list of chat completion choices. Can be more than one if `n` is greater than 1.
          items:
            type: object
            required:
              - delta
              - finish_reason
              - index
            properties:
              delta:
                $ref: "#/components/schemas/ChatCompletionStreamResponseDelta"
              finish_reason:
                type: string
                description: *chat_completion_finish_reason_description
                enum:
                  [
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                  ]
                nullable: true
              index:
                type: integer
                description: The index of the choice in the list of choices.
        created:
          type: integer
          description: The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same timestamp.
        model:
          type: string
          description: The model to generate the completion.
        system_fingerprint:
          type: string
          description: |
            This fingerprint represents the backend configuration that the model runs with.
            Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
        object:
          type: string
          description: The object type, which is always `chat.completion.chunk`.
          enum: [chat.completion.chunk]
      required:
        - choices
        - created
        - id
        - model
        - object
      x-oaiMeta:
        name: The chat completion chunk object
        group: chat
        example: *chat_completion_chunk_example

    CreateChatCompletionImageResponse:
      type: object
      description: Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
      x-oaiMeta:
        name: The chat completion chunk object
        group: chat
        example: *chat_completion_image_example

    CreateEditRequest:
      type: object
      properties:
        instruction:
          description: The instruction that tells the model how to edit the prompt.
          type: string
          example: "Fix the spelling mistakes."
        model:
          description: ID of the model to use. You can use the `text-davinci-edit-001` or `code-davinci-edit-001` model with this endpoint.
          example: "text-davinci-edit-001"
          anyOf:
            - type: string
            - type: string
              enum: ["text-davinci-edit-001", "code-davinci-edit-001"]
          x-oaiTypeLabel: string
        input:
          description: The input text to use as a starting point for the edit.
          type: string
          default: ""
          nullable: true
          example: "What day of the wek is it?"
        n:
          type: integer
          minimum: 1
          maximum: 20
          default: 1
          example: 1
          nullable: true
          description: How many edits to generate for the input and instruction.
        temperature:
          type: number
          minimum: 0
          maximum: 2
          default: 1
          example: 1
          nullable: true
          description: *completions_temperature_description
        top_p:
          type: number
          minimum: 0
          maximum: 1
          default: 1
          example: 1
          nullable: true
          description: *completions_top_p_description
      required:
        - model
        - instruction

    CreateEditResponse:
      type: object
      title: Edit
      deprecated: true
      properties:
        choices:
          type: array
          description: A list of edit choices. Can be more than one if `n` is greater than 1.
          items:
            type: object
            required:
              - text
              - index
              - finish_reason
            properties:
              finish_reason:
                type: string
                description: *completion_finish_reason_description
                enum: ["stop", "length"]
              index:
                type: integer
                description: The index of the choice in the list of choices.
              text:
                type: string
                description: The edited result.
        object:
          type: string
          description: The object type, which is always `edit`.
          enum: [edit]
        created:
          type: integer
          description: The Unix timestamp (in seconds) of when the edit was created.
        usage:
          $ref: "#/components/schemas/CompletionUsage"
      required:
        - object
        - created
        - choices
        - usage
      x-oaiMeta:
        name: The edit object
        example: *edit_example

    CreateImageRequest:
      type: object
      properties:
        prompt:
          description: A text description of the desired image(s). The maximum length is 1000 characters for `dall-e-2` and 4000 characters for `dall-e-3`.
          type: string
          example: "A cute baby sea otter"
        model:
          anyOf:
            - type: string
            - type: string
              enum: ["dall-e-2", "dall-e-3"]
          x-oaiTypeLabel: string
          default: "dall-e-2"
          example: "dall-e-3"
          nullable: true
          description: The model to use for image generation.
        n: &images_n
          type: integer
          minimum: 1
          maximum: 10
          default: 1
          example: 1
          nullable: true
          description: The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.
        quality:
          type: string
          enum: ["standard", "hd"]
          default: "standard"
          example: "standard"
          description: The quality of the image that will be generated. `hd` creates images with finer details and greater consistency across the image. This param is only supported for `dall-e-3`.
        response_format: &images_response_format
          type: string
          enum: ["url", "b64_json"]
          default: "url"
          example: "url"
          nullable: true
          description: The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        size: &images_size
          type: string
          enum: ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
          default: "1024x1024"
          example: "1024x1024"
          nullable: true
          description: The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models.
        style:
          type: string
          enum: ["vivid", "natural"]
          default: "vivid"
          example: "vivid"
          nullable: true
          description: The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`.
        user: *end_user_param_configuration
      required:
        - prompt

    ImagesResponse:
      properties:
        created:
          type: integer
        data:
          type: array
          items:
            $ref: "#/components/schemas/Image"
      required:
        - created
        - data

    Image:
      type: object
      description: Represents the url or the content of an image generated by the OpenAI API.
      properties:
        b64_json:
          type: string
          description: The base64-encoded JSON of the generated image, if `response_format` is `b64_json`.
        url:
          type: string
          description: The URL of the generated image, if `response_format` is `url` (default).
        revised_prompt:
          type: string
          description: The prompt that was used to generate the image, if there was any revision to the prompt.
      x-oaiMeta:
        name: The image object
        example: |
          {
            "url": "...",
            "revised_prompt": "..."
          }

    CreateImageEditRequest:
      type: object
      properties:
        image:
          description: The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.
          type: string
          format: binary
        prompt:
          description: A text description of the desired image(s). The maximum length is 1000 characters.
          type: string
          example: "A cute baby sea otter wearing a beret"
        mask:
          description: An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`.
          type: string
          format: binary
        model:
          anyOf:
            - type: string
            - type: string
              enum: ["dall-e-2"]
          x-oaiTypeLabel: string
          default: "dall-e-2"
          example: "dall-e-2"
          nullable: true
          description: The model to use for image generation. Only `dall-e-2` is supported at this time.
        n:
          type: integer
          minimum: 1
          maximum: 10
          default: 1
          example: 1
          nullable: true
          description: The number of images to generate. Must be between 1 and 10.
        size: &dalle2_images_size
          type: string
          enum: ["256x256", "512x512", "1024x1024"]
          default: "1024x1024"
          example: "1024x1024"
          nullable: true
          description: The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024`.
        response_format: *images_response_format
        user: *end_user_param_configuration
      required:
        - prompt
        - image

    CreateImageVariationRequest:
      type: object
      properties:
        image:
          description: The image to use as the basis for the variation(s). Must be a valid PNG file, less than 4MB, and square.
          type: string
          format: binary
        model:
          anyOf:
            - type: string
            - type: string
              enum: ["dall-e-2"]
          x-oaiTypeLabel: string
          default: "dall-e-2"
          example: "dall-e-2"
          nullable: true
          description: The model to use for image generation. Only `dall-e-2` is supported at this time.
        n: *images_n
        response_format: *images_response_format
        size: *dalle2_images_size
        user: *end_user_param_configuration
      required:
        - image

    CreateModerationRequest:
      type: object
      properties:
        input:
          description: The input text to classify
          oneOf:
            - type: string
              default: ""
              example: "I want to kill them."
            - type: array
              items:
                type: string
                default: ""
                example: "I want to kill them."
        model:
          description: |
            Two content moderations models are available: `text-moderation-stable` and `text-moderation-latest`.

            The default is `text-moderation-latest` which will be automatically upgraded over time. This ensures you are always using our most accurate model. If you use `text-moderation-stable`, we will provide advanced notice before updating the model. Accuracy of `text-moderation-stable` may be slightly lower than for `text-moderation-latest`.
          nullable: false
          default: "text-moderation-latest"
          example: "text-moderation-stable"
          anyOf:
            - type: string
            - type: string
              enum: ["text-moderation-latest", "text-moderation-stable"]
          x-oaiTypeLabel: string
      required:
        - input

    CreateModerationResponse:
      type: object
      description: Represents policy compliance report by OpenAI's content moderation model against a given input.
      properties:
        id:
          type: string
          description: The unique identifier for the moderation request.
        model:
          type: string
          description: The model used to generate the moderation results.
        results:
          type: array
          description: A list of moderation objects.
          items:
            type: object
            properties:
              flagged:
                type: boolean
                description: Whether the content violates [OpenAI's usage policies](/policies/usage-policies).
              categories:
                type: object
                description: A list of the categories, and whether they are flagged or not.
                properties:
                  hate:
                    type: boolean
                    description: Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harrassment.
                  hate/threatening:
                    type: boolean
                    description: Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.
                  harassment:
                    type: boolean
                    description: Content that expresses, incites, or promotes harassing language towards any target.
                  harassment/threatening:
                    type: boolean
                    description: Harassment content that also includes violence or serious harm towards any target.
                  self-harm:
                    type: boolean
                    description: Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
                  self-harm/intent:
                    type: boolean
                    description: Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.
                  self-harm/instructions:
                    type: boolean
                    description: Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.
                  sexual:
                    type: boolean
                    description: Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).
                  sexual/minors:
                    type: boolean
                    description: Sexual content that includes an individual who is under 18 years old.
                  violence:
                    type: boolean
                    description: Content that depicts death, violence, or physical injury.
                  violence/graphic:
                    type: boolean
                    description: Content that depicts death, violence, or physical injury in graphic detail.
                required:
                  - hate
                  - hate/threatening
                  - harassment
                  - harassment/threatening
                  - self-harm
                  - self-harm/intent
                  - self-harm/instructions
                  - sexual
                  - sexual/minors
                  - violence
                  - violence/graphic
              category_scores:
                type: object
                description: A list of the categories along with their scores as predicted by model.
                properties:
                  hate:
                    type: number
                    description: The score for the category 'hate'.
                  hate/threatening:
                    type: number
                    description: The score for the category 'hate/threatening'.
                  harassment:
                    type: number
                    description: The score for the category 'harassment'.
                  harassment/threatening:
                    type: number
                    description: The score for the category 'harassment/threatening'.
                  self-harm:
                    type: number
                    description: The score for the category 'self-harm'.
                  self-harm/intent:
                    type: number
                    description: The score for the category 'self-harm/intent'.
                  self-harm/instructions:
                    type: number
                    description: The score for the category 'self-harm/instructions'.
                  sexual:
                    type: number
                    description: The score for the category 'sexual'.
                  sexual/minors:
                    type: number
                    description: The score for the category 'sexual/minors'.
                  violence:
                    type: number
                    description: The score for the category 'violence'.
                  violence/graphic:
                    type: number
                    description: The score for the category 'violence/graphic'.
                required:
                  - hate
                  - hate/threatening
                  - harassment
                  - harassment/threatening
                  - self-harm
                  - self-harm/intent
                  - self-harm/instructions
                  - sexual
                  - sexual/minors
                  - violence
                  - violence/graphic
            required:
              - flagged
              - categories
              - category_scores
      required:
        - id
        - model
        - results
      x-oaiMeta:
        name: The moderation object
        example: *moderation_example

    ListFilesResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/OpenAIFile"
        object:
          type: string
          enum: [list]
      required:
        - object
        - data

    CreateFileRequest:
      type: object
      additionalProperties: false
      properties:
        file:
          description: |
            The File object (not file name) to be uploaded.
          type: string
          format: binary
        purpose:
          description: |
            The intended purpose of the uploaded file.

            Use "fine-tune" for [Fine-tuning](/docs/api-reference/fine-tuning) and "assistants" for [Assistants](/docs/api-reference/assistants) and [Messages](/docs/api-reference/messages). This allows us to validate the format of the uploaded file is correct for fine-tuning.
          type: string
          enum: ["fine-tune", "assistants"]
      required:
        - file
        - purpose

    DeleteFileResponse:
      type: object
      properties:
        id:
          type: string
        object:
          type: string
          enum: [file]
        deleted:
          type: boolean
      required:
        - id
        - object
        - deleted

    CreateFineTuningJobRequest:
      type: object
      properties:
        model:
          description: |
            The name of the model to fine-tune. You can select one of the
            [supported models](/docs/guides/fine-tuning/what-models-can-be-fine-tuned).
          example: "gpt-3.5-turbo"
          anyOf:
            - type: string
            - type: string
              enum: ["babbage-002", "davinci-002", "gpt-3.5-turbo"]
          x-oaiTypeLabel: string
        training_file:
          description: |
            The ID of an uploaded file that contains training data.

            See [upload file](/docs/api-reference/files/upload) for how to upload a file.

            Your dataset must be formatted as a JSONL file. Additionally, you must upload your file with the purpose `fine-tune`.

            See the [fine-tuning guide](/docs/guides/fine-tuning) for more details.
          type: string
          example: "file-abc123"
        hyperparameters:
          type: object
          description: The hyperparameters used for the fine-tuning job.
          properties:
            batch_size:
              description: |
                Number of examples in each batch. A larger batch size means that model parameters
                are updated less frequently, but with lower variance.
              oneOf:
                - type: string
                  enum: [auto]
                - type: integer
                  minimum: 1
                  maximum: 256
              default: auto
            learning_rate_multiplier:
              description: |
                Scaling factor for the learning rate. A smaller learning rate may be useful to avoid
                overfitting.
              oneOf:
                - type: string
                  enum: [auto]
                - type: number
                  minimum: 0
                  exclusiveMinimum: true
              default: auto
            n_epochs:
              description: |
                The number of epochs to train the model for. An epoch refers to one full cycle 
                through the training dataset.
              oneOf:
                - type: string
                  enum: [auto]
                - type: integer
                  minimum: 1
                  maximum: 50
              default: auto
        suffix:
          description: |
            A string of up to 18 characters that will be added to your fine-tuned model name.

            For example, a `suffix` of "custom-model-name" would produce a model name like `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`.
          type: string
          minLength: 1
          maxLength: 40
          default: null
          nullable: true
        validation_file:
          description: |
            The ID of an uploaded file that contains validation data.

            If you provide this file, the data is used to generate validation
            metrics periodically during fine-tuning. These metrics can be viewed in
            the fine-tuning results file.
            The same data should not be present in both train and validation files.

            Your dataset must be formatted as a JSONL file. You must upload your file with the purpose `fine-tune`.

            See the [fine-tuning guide](/docs/guides/fine-tuning) for more details.
          type: string
          nullable: true
          example: "file-abc123"
      required:
        - model
        - training_file

    ListFineTuningJobEventsResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/FineTuningJobEvent"
        object:
          type: string
          enum: [list]
      required:
        - object
        - data

    CreateFineTuneRequest:
      type: object
      properties:
        training_file:
          description: |
            The ID of an uploaded file that contains training data.

            See [upload file](/docs/api-reference/files/upload) for how to upload a file.

            Your dataset must be formatted as a JSONL file, where each training
            example is a JSON object with the keys "prompt" and "completion".
            Additionally, you must upload your file with the purpose `fine-tune`.

            See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details.
          type: string
          example: "file-abc123"
        batch_size:
          description: |
            The batch size to use for training. The batch size is the number of
            training examples used to train a single forward and backward pass.

            By default, the batch size will be dynamically configured to be
            ~0.2% of the number of examples in the training set, capped at 256 -
            in general, we've found that larger batch sizes tend to work better
            for larger datasets.
          default: null
          type: integer
          nullable: true
        classification_betas:
          description: |
            If this is provided, we calculate F-beta scores at the specified
            beta values. The F-beta score is a generalization of F-1 score.
            This is only used for binary classification.

            With a beta of 1 (i.e. the F-1 score), precision and recall are
            given the same weight. A larger beta score puts more weight on
            recall and less on precision. A smaller beta score puts more weight
            on precision and less on recall.
          type: array
          items:
            type: number
          example: [0.6, 1, 1.5, 2]
          default: null
          nullable: true
        classification_n_classes:
          description: |
            The number of classes in a classification task.

            This parameter is required for multiclass classification.
          type: integer
          default: null
          nullable: true
        classification_positive_class:
          description: |
            The positive class in binary classification.

            This parameter is needed to generate precision, recall, and F1
            metrics when doing binary classification.
          type: string
          default: null
          nullable: true
        compute_classification_metrics:
          description: |
            If set, we calculate classification-specific metrics such as accuracy
            and F-1 score using the validation set at the end of every epoch.
            These metrics can be viewed in the [results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).

            In order to compute classification metrics, you must provide a
            `validation_file`. Additionally, you must
            specify `classification_n_classes` for multiclass classification or
            `classification_positive_class` for binary classification.
          type: boolean
          default: false
          nullable: true
        hyperparameters:
          type: object
          description: The hyperparameters used for the fine-tuning job.
          properties:
            n_epochs:
              description: |
                The number of epochs to train the model for. An epoch refers to one
                full cycle through the training dataset.
              oneOf:
                - type: string
                  enum: [auto]
                - type: integer
                  minimum: 1
                  maximum: 50
              default: auto
        learning_rate_multiplier:
          description: |
            The learning rate multiplier to use for training.
            The fine-tuning learning rate is the original learning rate used for
            pretraining multiplied by this value.

            By default, the learning rate multiplier is the 0.05, 0.1, or 0.2
            depending on final `batch_size` (larger learning rates tend to
            perform better with larger batch sizes). We recommend experimenting
            with values in the range 0.02 to 0.2 to see what produces the best
            results.
          default: null
          type: number
          nullable: true
        model:
          description: |
            The name of the base model to fine-tune. You can select one of "ada",
            "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21 and before 2023-08-22.
            To learn more about these models, see the
            [Models](/docs/models) documentation.
          default: "curie"
          example: "curie"
          nullable: true
          anyOf:
            - type: string
            - type: string
              enum: ["ada", "babbage", "curie", "davinci"]
          x-oaiTypeLabel: string
        prompt_loss_weight:
          description: |
            The weight to use for loss on the prompt tokens. This controls how
            much the model tries to learn to generate the prompt (as compared
            to the completion which always has a weight of 1.0), and can add
            a stabilizing effect to training when completions are short.

            If prompts are extremely long (relative to completions), it may make
            sense to reduce this weight so as to avoid over-prioritizing
            learning the prompt.
          default: 0.01
          type: number
          nullable: true
        suffix:
          description: |
            A string of up to 40 characters that will be added to your fine-tuned model name.

            For example, a `suffix` of "custom-model-name" would produce a model name like `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.
          type: string
          minLength: 1
          maxLength: 40
          default: null
          nullable: true
        validation_file:
          description: |
            The ID of an uploaded file that contains validation data.

            If you provide this file, the data is used to generate validation
            metrics periodically during fine-tuning. These metrics can be viewed in
            the [fine-tuning results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).
            Your train and validation data should be mutually exclusive.

            Your dataset must be formatted as a JSONL file, where each validation
            example is a JSON object with the keys "prompt" and "completion".
            Additionally, you must upload your file with the purpose `fine-tune`.

            See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details.
          type: string
          nullable: true
          example: "file-abc123"
      required:
        - training_file

    ListFineTunesResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/FineTune"
        object:
          type: string
          enum: [list]
      required:
        - object
        - data

    ListFineTuneEventsResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/FineTuneEvent"
        object:
          type: string
          enum: [list]
      required:
        - object
        - data

    CreateEmbeddingRequest:
      type: object
      additionalProperties: false
      properties:
        input:
          description: |
            Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for `text-embedding-ada-002`) and cannot be an empty string. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens.
          example: "The quick brown fox jumped over the lazy dog"
          oneOf:
            - type: string
              default: ""
              example: "This is a test."
            - type: array
              minItems: 1
              items:
                type: string
                default: ""
                example: "This is a test."
            - type: array
              minItems: 1
              items:
                type: integer
              example: "[1212, 318, 257, 1332, 13]"
            - type: array
              minItems: 1
              items:
                type: array
                minItems: 1
                items:
                  type: integer
              example: "[[1212, 318, 257, 1332, 13]]"
        model:
          description: *model_description
          example: "text-embedding-ada-002"
          anyOf:
            - type: string
            - type: string
              enum: ["text-embedding-ada-002"]
          x-oaiTypeLabel: string
        encoding_format:
          description: "The format to return the embeddings in. Can be either `float` or [`base64`](https://pypi.org/project/pybase64/)."
          example: "float"
          default: "float"
          type: string
          enum: ["float", "base64"]
        user: *end_user_param_configuration
      required:
        - model
        - input

    CreateEmbeddingResponse:
      type: object
      properties:
        data:
          type: array
          description: The list of embeddings generated by the model.
          items:
            $ref: "#/components/schemas/Embedding"
        model:
          type: string
          description: The name of the model used to generate the embedding.
        object:
          type: string
          description: The object type, which is always "list".
          enum: [list]
        usage:
          type: object
          description: The usage information for the request.
          properties:
            prompt_tokens:
              type: integer
              description: The number of tokens used by the prompt.
            total_tokens:
              type: integer
              description: The total number of tokens used by the request.
          required:
            - prompt_tokens
            - total_tokens
      required:
        - object
        - model
        - data
        - usage

    CreateTranscriptionRequest:
      type: object
      additionalProperties: false
      properties:
        file:
          description: |
            The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
          type: string
          x-oaiTypeLabel: file
          format: binary
        model:
          description: |
            ID of the model to use. Only `whisper-1` is currently available.
          example: whisper-1
          anyOf:
            - type: string
            - type: string
              enum: ["whisper-1"]
          x-oaiTypeLabel: string
        language:
          description: |
            The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
          type: string
        prompt:
          description: |
            An optional text to guide the model's style or continue a previous audio segment. The [prompt](/docs/guides/speech-to-text/prompting) should match the audio language.
          type: string
        response_format:
          description: |
            The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
          type: string
          enum:
            - json
            - text
            - srt
            - verbose_json
            - vtt
          default: json
        temperature:
          description: |
            The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit.
          type: number
          default: 0
      required:
        - file
        - model

    # Note: This does not currently support the non-default response format types.
    CreateTranscriptionResponse:
      type: object
      properties:
        text:
          type: string
      required:
        - text

    CreateTranslationRequest:
      type: object
      additionalProperties: false
      properties:
        file:
          description: |
            The audio file object (not file name) translate, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
          type: string
          x-oaiTypeLabel: file
          format: binary
        model:
          description: |
            ID of the model to use. Only `whisper-1` is currently available.
          example: whisper-1
          anyOf:
            - type: string
            - type: string
              enum: ["whisper-1"]
          x-oaiTypeLabel: string
        prompt:
          description: |
            An optional text to guide the model's style or continue a previous audio segment. The [prompt](/docs/guides/speech-to-text/prompting) should be in English.
          type: string
        response_format:
          description: |
            The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
          type: string
          default: json
        temperature:
          description: |
            The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit.
          type: number
          default: 0
      required:
        - file
        - model

    # Note: This does not currently support the non-default response format types.
    CreateTranslationResponse:
      type: object
      properties:
        text:
          type: string
      required:
        - text

    CreateSpeechRequest:
      type: object
      additionalProperties: false
      properties:
        model:
          description: |
            One of the available [TTS models](/docs/models/tts): `tts-1` or `tts-1-hd`
          anyOf:
            - type: string
            - type: string
              enum: ["tts-1", "tts-1-hd"]
          x-oaiTypeLabel: string
        input:
          type: string
          description: The text to generate audio for. The maximum length is 4096 characters.
          maxLength: 4096
        voice:
          description: The voice to use when generating the audio. Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.
          type: string
          enum: ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        response_format:
          description: "The format to audio in. Supported formats are `mp3`, `opus`, `aac`, and `flac`."
          default: "mp3"
          type: string
          enum: ["mp3", "opus", "aac", "flac"]
        speed:
          description: "The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default."
          type: number
          default: 1.0
          minimum: 0.25
          maximum: 4.0
      required:
        - model
        - input
        - voice

    Model:
      title: Model
      description: Describes an OpenAI model offering that can be used with the API.
      properties:
        id:
          type: string
          description: The model identifier, which can be referenced in the API endpoints.
        created:
          type: integer
          description: The Unix timestamp (in seconds) when the model was created.
        object:
          type: string
          description: The object type, which is always "model".
          enum: [model]
        owned_by:
          type: string
          description: The organization that owns the model.
      required:
        - id
        - object
        - created
        - owned_by
      x-oaiMeta:
        name: The model object
        example: *retrieve_model_response

    OpenAIFile:
      title: OpenAIFile
      description: The `File` object represents a document that has been uploaded to OpenAI.
      properties:
        id:
          type: string
          description: The file identifier, which can be referenced in the API endpoints.
        bytes:
          type: integer
          description: The size of the file, in bytes.
        created_at:
          type: integer
          description: The Unix timestamp (in seconds) for when the file was created.
        filename:
          type: string
          description: The name of the file.
        object:
          type: string
          description: The object type, which is always `file`.
          enum: ["file"]
        purpose:
          type: string
          description: The intended purpose of the file. Supported values are `fine-tune`, `fine-tune-results`, `assistants`, and `assistants_output`.
          enum:
            [
              "fine-tune",
              "fine-tune-results",
              "assistants",
              "assistants_output",
            ]
        status:
          type: string
          deprecated: true
          description: Deprecated. The current status of the file, which can be either `uploaded`, `processed`, or `error`.
          enum: ["uploaded", "processed", "error"]
        status_details:
          type: string
          deprecated: true
          description: Deprecated. For details on why a fine-tuning training file failed validation, see the `error` field on `fine_tuning.job`.
      required:
        - id
        - object
        - bytes
        - created_at
        - filename
        - purpose
        - status
      x-oaiMeta:
        name: The File object
        example: |
          {
            "id": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
            "object": "file",
            "bytes": 120000,
            "created_at": 1677610602,
            "filename": "salesOverview.pdf",
            "purpose": "assistants",
          }
    Embedding:
      type: object
      description: |
        Represents an embedding vector returned by embedding endpoint.
      properties:
        index:
          type: integer
          description: The index of the embedding in the list of embeddings.
        embedding:
          type: array
          description: |
            The embedding vector, which is a list of floats. The length of vector depends on the model as listed in the [embedding guide](/docs/guides/embeddings).
          items:
            type: number
        object:
          type: string
          description: The object type, which is always "embedding".
          enum: [embedding]
      required:
        - index
        - object
        - embedding
      x-oaiMeta:
        name: The embedding object
        example: |
          {
            "object": "embedding",
            "embedding": [
              0.0023064255,
              -0.009327292,
              .... (1536 floats total for ada-002)
              -0.0028842222,
            ],
            "index": 0
          }

    FineTuningJob:
      type: object
      title: FineTuningJob
      description: |
        The `fine_tuning.job` object represents a fine-tuning job that has been created through the API.
      properties:
        id:
          type: string
          description: The object identifier, which can be referenced in the API endpoints.
        created_at:
          type: integer
          description: The Unix timestamp (in seconds) for when the fine-tuning job was created.
        error:
          type: object
          nullable: true
          description: For fine-tuning jobs that have `failed`, this will contain more information on the cause of the failure.
          properties:
            code:
              type: string
              description: A machine-readable error code.
            message:
              type: string
              description: A human-readable error message.
            param:
              type: string
              description: The parameter that was invalid, usually `training_file` or `validation_file`. This field will be null if the failure was not parameter-specific.
              nullable: true
          required:
            - code
            - message
            - param
        fine_tuned_model:
          type: string
          nullable: true
          description: The name of the fine-tuned model that is being created. The value will be null if the fine-tuning job is still running.
        finished_at:
          type: integer
          nullable: true
          description: The Unix timestamp (in seconds) for when the fine-tuning job was finished. The value will be null if the fine-tuning job is still running.
        hyperparameters:
          type: object
          description: The hyperparameters used for the fine-tuning job. See the [fine-tuning guide](/docs/guides/fine-tuning) for more details.
          properties:
            n_epochs:
              oneOf:
                - type: string
                  enum: [auto]
                - type: integer
                  minimum: 1
                  maximum: 50
              default: auto
              description:
                The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.

                "auto" decides the optimal number of epochs based on the size of the dataset. If setting the number manually, we support any number between 1 and 50 epochs.
          required:
            - n_epochs
        model:
          type: string
          description: The base model that is being fine-tuned.
        object:
          type: string
          description: The object type, which is always "fine_tuning.job".
          enum: [fine_tuning.job]
        organization_id:
          type: string
          description: The organization that owns the fine-tuning job.
        result_files:
          type: array
          description: The compiled results file ID(s) for the fine-tuning job. You can retrieve the results with the [Files API](/docs/api-reference/files/retrieve-contents).
          items:
            type: string
            example: file-abc123
        status:
          type: string
          description: The current status of the fine-tuning job, which can be either `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
          enum:
            [
              "validating_files",
              "queued",
              "running",
              "succeeded",
              "failed",
              "cancelled",
            ]
        trained_tokens:
          type: integer
          nullable: true
          description: The total number of billable tokens processed by this fine-tuning job. The value will be null if the fine-tuning job is still running.
        training_file:
          type: string
          description: The file ID used for training. You can retrieve the training data with the [Files API](/docs/api-reference/files/retrieve-contents).
        validation_file:
          type: string
          nullable: true
          description: The file ID used for validation. You can retrieve the validation results with the [Files API](/docs/api-reference/files/retrieve-contents).
      required:
        - created_at
        - error
        - finished_at
        - fine_tuned_model
        - hyperparameters
        - id
        - model
        - object
        - organization_id
        - result_files
        - status
        - trained_tokens
        - training_file
        - validation_file
      x-oaiMeta:
        name: The fine-tuning job object
        example: *fine_tuning_example

    FineTuningJobEvent:
      type: object
      description: Fine-tuning job event object
      properties:
        id:
          type: string
        created_at:
          type: integer
        level:
          type: string
          enum: ["info", "warn", "error"]
        message:
          type: string
        object:
          type: string
          enum: [fine_tuning.job.event]
      required:
        - id
        - object
        - created_at
        - level
        - message
      x-oaiMeta:
        name: The fine-tuning job event object
        example: |
          {
            "object": "fine_tuning.job.event",
            "id": "ftevent-abc123"
            "created_at": 1677610602,
            "level": "info",
            "message": "Created fine-tuning job"
          }

    FineTune:
      type: object
      deprecated: true
      description: |
        The `FineTune` object represents a legacy fine-tune job that has been created through the API.
      properties:
        id:
          type: string
          description: The object identifier, which can be referenced in the API endpoints.
        created_at:
          type: integer
          description: The Unix timestamp (in seconds) for when the fine-tuning job was created.
        events:
          type: array
          description: The list of events that have been observed in the lifecycle of the FineTune job.
          items:
            $ref: "#/components/schemas/FineTuneEvent"
        fine_tuned_model:
          type: string
          nullable: true
          description: The name of the fine-tuned model that is being created.
        hyperparams:
          type: object
          description: The hyperparameters used for the fine-tuning job. See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/hyperparameters) for more details.
          properties:
            batch_size:
              type: integer
              description: |
                The batch size to use for training. The batch size is the number of
                training examples used to train a single forward and backward pass.
            classification_n_classes:
              type: integer
              description: |
                The number of classes to use for computing classification metrics.
            classification_positive_class:
              type: string
              description: |
                The positive class to use for computing classification metrics.
            compute_classification_metrics:
              type: boolean
              description: |
                The classification metrics to compute using the validation dataset at the end of every epoch.
            learning_rate_multiplier:
              type: number
              description: |
                The learning rate multiplier to use for training.
            n_epochs:
              type: integer
              description: |
                The number of epochs to train the model for. An epoch refers to one
                full cycle through the training dataset.
            prompt_loss_weight:
              type: number
              description: |
                The weight to use for loss on the prompt tokens.
          required:
            - batch_size
            - learning_rate_multiplier
            - n_epochs
            - prompt_loss_weight
        model:
          type: string
          description: The base model that is being fine-tuned.
        object:
          type: string
          description: The object type, which is always "fine-tune".
          enum: [fine-tune]
        organization_id:
          type: string
          description: The organization that owns the fine-tuning job.
        result_files:
          type: array
          description: The compiled results files for the fine-tuning job.
          items:
            $ref: "#/components/schemas/OpenAIFile"
        status:
          type: string
          description: The current status of the fine-tuning job, which can be either `created`, `running`, `succeeded`, `failed`, or `cancelled`.
        training_files:
          type: array
          description: The list of files used for training.
          items:
            $ref: "#/components/schemas/OpenAIFile"
        updated_at:
          type: integer
          description: The Unix timestamp (in seconds) for when the fine-tuning job was last updated.
        validation_files:
          type: array
          description: The list of files used for validation.
          items:
            $ref: "#/components/schemas/OpenAIFile"
      required:
        - created_at
        - fine_tuned_model
        - hyperparams
        - id
        - model
        - object
        - organization_id
        - result_files
        - status
        - training_files
        - updated_at
        - validation_files
      x-oaiMeta:
        name: The fine-tune object
        example: *fine_tune_example

    FineTuneEvent:
      type: object
      deprecated: true
      description: Fine-tune event object
      properties:
        created_at:
          type: integer
        level:
          type: string
        message:
          type: string
        object:
          type: string
          enum: [fine-tune-event]
      required:
        - object
        - created_at
        - level
        - message
      x-oaiMeta:
        name: The fine-tune event object
        example: |
          {
            "object": "fine-tune-event",
            "created_at": 1677610602,
            "level": "info",
            "message": "Created fine-tune job"
          }

    CompletionUsage:
      type: object
      description: Usage statistics for the completion request.
      properties:
        completion_tokens:
          type: integer
          description: Number of tokens in the generated completion.
        prompt_tokens:
          type: integer
          description: Number of tokens in the prompt.
        total_tokens:
          type: integer
          description: Total number of tokens used in the request (prompt + completion).
      required:
        - prompt_tokens
        - completion_tokens
        - total_tokens

    AssistantObject:
      type: object
      title: Assistant
      description: Represents an `assistant` that can call the model and use tools.
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `assistant`.
          type: string
          enum: [assistant]
        created_at:
          description: The Unix timestamp (in seconds) for when the assistant was created.
          type: integer
        name:
          description: &assistant_name_param_description |
            The name of the assistant. The maximum length is 256 characters.
          type: string
          maxLength: 256
          nullable: true
        description:
          description: &assistant_description_param_description |
            The description of the assistant. The maximum length is 512 characters.
          type: string
          maxLength: 512
          nullable: true
        model:
          description: *model_description
          type: string
        instructions:
          description: &assistant_instructions_param_description |
            The system instructions that the assistant uses. The maximum length is 32768 characters.
          type: string
          maxLength: 32768
          nullable: true
        tools:
          description: &assistant_tools_param_description |
            A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`.
          default: []
          type: array
          maxItems: 128
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
            x-oaiExpandable: true
        file_ids:
          description: &assistant_file_param_description |
            A list of [file](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order.
          default: []
          type: array
          maxItems: 20
          items:
            type: string
        metadata:
          description: &metadata_description |
            Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long.
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - id
        - object
        - created_at
        - name
        - description
        - model
        - instructions
        - tools
        - file_ids
        - metadata
      x-oaiMeta:
        name: The assistant object
        beta: true
        example: *create_assistants_example

    CreateAssistantRequest:
      type: object
      additionalProperties: false
      properties:
        model:
          description: *model_description
          anyOf:
            - type: string
        name:
          description: *assistant_name_param_description
          type: string
          nullable: true
          maxLength: 256
        description:
          description: *assistant_description_param_description
          type: string
          nullable: true
          maxLength: 512
        instructions:
          description: *assistant_instructions_param_description
          type: string
          nullable: true
          maxLength: 32768
        tools:
          description: *assistant_tools_param_description
          default: []
          type: array
          maxItems: 128
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
            x-oaiExpandable: true
        file_ids:
          description: *assistant_file_param_description
          default: []
          maxItems: 20
          type: array
          items:
            type: string
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - model

    ModifyAssistantRequest:
      type: object
      additionalProperties: false
      properties:
        model:
          description: *model_description
          anyOf:
            - type: string
        name:
          description: *assistant_name_param_description
          type: string
          nullable: true
          maxLength: 256
        description:
          description: *assistant_description_param_description
          type: string
          nullable: true
          maxLength: 512
        instructions:
          description: *assistant_instructions_param_description
          type: string
          nullable: true
          maxLength: 32768
        tools:
          description: *assistant_tools_param_description
          default: []
          type: array
          maxItems: 128
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
            x-oaiExpandable: true
        file_ids:
          description: |
            A list of [File](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. If a file was previosuly attached to the list but does not show up in the list, it will be deleted from the assistant.
          default: []
          type: array
          maxItems: 20
          items:
            type: string
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true

    DeleteAssistantResponse:
      type: object
      properties:
        id:
          type: string
        deleted:
          type: boolean
        object:
          type: string
          enum: [assistant.deleted]
      required:
        - id
        - object
        - deleted

    ListAssistantsResponse:
      type: object
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/AssistantObject"
        first_id:
          type: string
          example: "asst_hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "asst_QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - first_id
        - last_id
        - has_more
      x-oaiMeta:
        name: List assistants response object
        group: chat
        example: *list_assistants_example

    AssistantToolsCode:
      type: object
      title: Code interpreter tool
      properties:
        type:
          type: string
          description: "The type of tool being defined: `code_interpreter`"
          enum: ["code_interpreter"]
      required:
        - type

    AssistantToolsRetrieval:
      type: object
      title: Retrieval tool
      properties:
        type:
          type: string
          description: "The type of tool being defined: `retrieval`"
          enum: ["retrieval"]
      required:
        - type

    AssistantToolsFunction:
      type: object
      title: Function tool
      properties:
        type:
          type: string
          description: "The type of tool being defined: `function`"
          enum: ["function"]
        function:
          $ref: "#/components/schemas/FunctionObject"
      required:
        - type
        - function

    RunObject:
      type: object
      title: A run on a thread
      description: Represents an execution run on a [thread](/docs/api-reference/threads).
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `thread.run`.
          type: string
          enum: ["thread.run"]
        created_at:
          description: The Unix timestamp (in seconds) for when the run was created.
          type: integer
        thread_id:
          description: The ID of the [thread](/docs/api-reference/threads) that was executed on as a part of this run.
          type: string
        assistant_id:
          description: The ID of the [assistant](/docs/api-reference/assistants) used for execution of this run.
          type: string
        status:
          description: The status of the run, which can be either `queued`, `in_progress`, `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`, or `expired`.
          type: string
          enum:
            [
              "queued",
              "in_progress",
              "requires_action",
              "cancelling",
              "cancelled",
              "failed",
              "completed",
              "expired",
            ]
        required_action:
          type: object
          description: Details on the action required to continue the run. Will be `null` if no action is required.
          nullable: true
          properties:
            type:
              description: For now, this is always `submit_tool_outputs`.
              type: string
              enum: ["submit_tool_outputs"]
            submit_tool_outputs:
              type: object
              description: Details on the tool outputs needed for this run to continue.
              properties:
                tool_calls:
                  type: array
                  description: A list of the relevant tool calls.
                  items:
                    $ref: "#/components/schemas/RunToolCallObject"
              required:
                - tool_calls
          required:
            - type
            - submit_tool_outputs
        last_error:
          type: object
          description: The last error associated with this run. Will be `null` if there are no errors.
          nullable: true
          properties:
            code:
              type: string
              description: One of `server_error` or `rate_limit_exceeded`.
              enum: ["server_error", "rate_limit_exceeded"]
            message:
              type: string
              description: A human-readable description of the error.
          required:
            - code
            - message
        expires_at:
          description: The Unix timestamp (in seconds) for when the run will expire.
          type: integer
        started_at:
          description: The Unix timestamp (in seconds) for when the run was started.
          type: integer
          nullable: true
        cancelled_at:
          description: The Unix timestamp (in seconds) for when the run was cancelled.
          type: integer
          nullable: true
        failed_at:
          description: The Unix timestamp (in seconds) for when the run failed.
          type: integer
          nullable: true
        completed_at:
          description: The Unix timestamp (in seconds) for when the run was completed.
          type: integer
          nullable: true
        model:
          description: The model that the [assistant](/docs/api-reference/assistants) used for this run.
          type: string
        instructions:
          description: The instructions that the [assistant](/docs/api-reference/assistants) used for this run.
          type: string
        tools:
          description: The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.
          default: []
          type: array
          maxItems: 20
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
            x-oaiExpandable: true
        file_ids:
          description: The list of [File](/docs/api-reference/files) IDs the [assistant](/docs/api-reference/assistants) used for this run.
          default: []
          type: array
          items:
            type: string
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - id
        - object
        - created_at
        - thread_id
        - assistant_id
        - status
        - required_action
        - last_error
        - expires_at
        - started_at
        - cancelled_at
        - failed_at
        - completed_at
        - model
        - instructions
        - tools
        - file_ids
        - metadata
      x-oaiMeta:
        name: The run object
        beta: true
        example: |
          {
            "id": "run_example123",
            "object": "thread.run",
            "created_at": 1698107661,
            "assistant_id": "asst_gZ1aOomboBuYWPcXJx4vAYB0",
            "thread_id": "thread_adOpf7Jbb5Abymz0QbwxAh3c",
            "status": "completed",
            "started_at": 1699073476,
            "expires_at": null,
            "cancelled_at": null,
            "failed_at": null,
            "completed_at": 1699073498,
            "last_error": null,
            "model": "gpt-4",
            "instructions": null,
            "tools": [{"type": "retrieval"}, {"type": "code_interpreter"}],
            "file_ids": [],
            "metadata": {}
          }
    CreateRunRequest:
      type: object
      additionalProperties: false
      properties:
        assistant_id:
          description: The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.
          type: string
        model:
          description: The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
          type: string
          nullable: true
        instructions:
          description: Override the default system message of the assistant. This is useful for modifying the behavior on a per-run basis.
          type: string
          nullable: true
        tools:
          description: Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
          nullable: true
          type: array
          maxItems: 20
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
            x-oaiExpandable: true
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - assistant_id
    ListRunsResponse:
      type: object
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/RunObject"
        first_id:
          type: string
          example: "run_hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "run_QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - first_id
        - last_id
        - has_more
    ModifyRunRequest:
      type: object
      additionalProperties: false
      properties:
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
    SubmitToolOutputsRunRequest:
      type: object
      additionalProperties: false
      properties:
        tool_outputs:
          description: A list of tools for which the outputs are being submitted.
          type: array
          items:
            type: object
            properties:
              tool_call_id:
                type: string
                description: The ID of the tool call in the `required_action` object within the run object the output is being submitted for.
              output:
                type: string
                description: The output of the tool call to be submitted to continue the run.
      required:
        - tool_outputs
    RunToolCallObject:
      type: object
      description: Tool call objects
      properties:
        id:
          type: string
          description: The ID of the tool call. This ID must be referenced when you submit the tool outputs in using the [Submit tool outputs to run](/docs/api-reference/runs/submitToolOutputs) endpoint.
        type:
          type: string
          description: The type of tool call the output is required for. For now, this is always `function`.
          enum: ["function"]
        function:
          type: object
          description: The function definition.
          properties:
            name:
              type: string
              description: The name of the function.
            arguments:
              type: string
              description: The arguments that the model expects you to pass to the function.
          required:
            - name
            - arguments
      required:
        - id
        - type
        - function
    CreateThreadAndRunRequest:
      type: object
      additionalProperties: false
      properties:
        assistant_id:
          description: The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.
          type: string
        thread:
          $ref: "#/components/schemas/CreateThreadRequest"
          description: If no thread is provided, an empty thread will be created.
        model:
          description: The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
          type: string
          nullable: true
        instructions:
          description: Override the default system message of the assistant. This is useful for modifying the behavior on a per-run basis.
          type: string
          nullable: true
        tools:
          description: Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
          nullable: true
          type: array
          maxItems: 20
          items:
            oneOf:
              - $ref: "#/components/schemas/AssistantToolsCode"
              - $ref: "#/components/schemas/AssistantToolsRetrieval"
              - $ref: "#/components/schemas/AssistantToolsFunction"
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - assistant_id

    ThreadObject:
      type: object
      title: Thread
      description: Represents a thread that contains [messages](/docs/api-reference/messages).
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `thread`.
          type: string
          enum: ["thread"]
        created_at:
          description: The Unix timestamp (in seconds) for when the thread was created.
          type: integer
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - id
        - object
        - created_at
        - metadata
      x-oaiMeta:
        name: The thread object
        beta: true
        example: |
          {
            "id": "thread_abc123",
            "object": "thread",
            "created_at": 1698107661,
            "metadata": {}
          }

    CreateThreadRequest:
      type: object
      additionalProperties: false
      properties:
        messages:
          description: A list of [messages](/docs/api-reference/messages) to start the thread with.
          type: array
          items:
            $ref: "#/components/schemas/CreateMessageRequest"
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true

    ModifyThreadRequest:
      type: object
      additionalProperties: false
      properties:
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true

    DeleteThreadResponse:
      type: object
      properties:
        id:
          type: string
        deleted:
          type: boolean
        object:
          type: string
          enum: [thread.deleted]
      required:
        - id
        - object
        - deleted

    ListThreadsResponse:
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/ThreadObject"
        first_id:
          type: string
          example: "asst_hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "asst_QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - first_id
        - last_id
        - has_more

    MessageObject:
      type: object
      title: The message object
      description: Represents a message within a [thread](/docs/api-reference/threads).
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `thread.message`.
          type: string
          enum: ["thread.message"]
        created_at:
          description: The Unix timestamp (in seconds) for when the message was created.
          type: integer
        thread_id:
          description: The [thread](/docs/api-reference/threads) ID that this message belongs to.
          type: string
        role:
          description: The entity that produced the message. One of `user` or `assistant`.
          type: string
          enum: ["user", "assistant"]
        content:
          description: The content of the message in array of text and/or images.
          type: array
          items:
            oneOf:
              - $ref: "#/components/schemas/MessageContentImageFileObject"
              - $ref: "#/components/schemas/MessageContentTextObject"
            x-oaiExpandable: true
        assistant_id:
          description: If applicable, the ID of the [assistant](/docs/api-reference/assistants) that authored this message.
          type: string
          nullable: true
        run_id:
          description: If applicable, the ID of the [run](/docs/api-reference/runs) associated with the authoring of this message.
          type: string
          nullable: true
        file_ids:
          description: A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.
          default: []
          maxItems: 10
          type: array
          items:
            type: string
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - id
        - object
        - created_at
        - thread_id
        - role
        - content
        - assistant_id
        - run_id
        - file_ids
        - metadata
      x-oaiMeta:
        name: The message object
        beta: true
        example: |
          {
            "id": "msg_dKYDWyQvtjDBi3tudL1yWKDa",
            "object": "thread.message",
            "created_at": 1698983503,
            "thread_id": "thread_RGUhOuO9b2nrktrmsQ2uSR6I",
            "role": "assistant",
            "content": [
              {
                "type": "text",
                "text": {
                  "value": "Hi! How can I help you today?",
                  "annotations": []
                }
              }
            ],
            "file_ids": [],
            "assistant_id": "asst_ToSF7Gb04YMj8AMMm50ZLLtY",
            "run_id": "run_BjylUJgDqYK9bOhy4yjAiMrn",
            "metadata": {}
          }

    CreateMessageRequest:
      type: object
      additionalProperties: false
      required:
        - role
        - content
      properties:
        role:
          type: string
          enum: ["user"]
          description: The role of the entity that is creating the message. Currently only `user` is supported.
        content:
          type: string
          minLength: 1
          maxLength: 32768
          description: The content of the message.
        file_ids:
          description: A list of [File](/docs/api-reference/files) IDs that the message should use. There can be a maximum of 10 files attached to a message. Useful for tools like `retrieval` and `code_interpreter` that can access and use files.
          default: []
          type: array
          minItems: 1
          maxItems: 10
          items:
            type: string
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true

    ModifyMessageRequest:
      type: object
      additionalProperties: false
      properties:
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true

    DeleteMessageResponse:
      type: object
      properties:
        id:
          type: string
        deleted:
          type: boolean
        object:
          type: string
          enum: [thread.message.deleted]
      required:
        - id
        - object
        - deleted

    ListMessagesResponse:
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/MessageObject"
        first_id:
          type: string
          example: "msg_hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "msg_QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - first_id
        - last_id
        - has_more

    MessageContentImageFileObject:
      title: Image file
      type: object
      description: References an image [File](/docs/api-reference/files) in the content of a message.
      properties:
        type:
          description: Always `image_file`.
          type: string
          enum: ["image_file"]
        image_file:
          type: object
          properties:
            file_id:
              description: The [File](/docs/api-reference/files) ID of the image in the message content.
              type: string
          required:
            - file_id
      required:
        - type
        - image_file

    MessageContentTextObject:
      title: Text
      type: object
      description: The text content that is part of a message.
      properties:
        type:
          description: Always `text`.
          type: string
          enum: ["text"]
        text:
          type: object
          properties:
            value:
              description: The data that makes up the text.
              type: string
            annotations:
              type: array
              items:
                oneOf:
                  - $ref: "#/components/schemas/MessageContentTextAnnotationsFileCitationObject"
                  - $ref: "#/components/schemas/MessageContentTextAnnotationsFilePathObject"
                x-oaiExpandable: true
          required:
            - value
            - annotations
      required:
        - type
        - text

    MessageContentTextAnnotationsFileCitationObject:
      title: File citation
      type: object
      description: A citation within the message that points to a specific quote from a specific File associated with the assistant or the message. Generated when the assistant uses the "retrieval" tool to search files.
      properties:
        type:
          description: Always `file_citation`.
          type: string
          enum: ["file_citation"]
        text:
          description: The text in the message content that needs to be replaced.
          type: string
        file_citation:
          type: object
          properties:
            file_id:
              description: The ID of the specific File the citation is from.
              type: string
            quote:
              description: The specific quote in the file.
              type: string
          required:
            - file_id
            - quote
        start_index:
          type: integer
          minimum: 0
        end_index:
          type: integer
          minimum: 0
      required:
        - type
        - text
        - file_citation
        - start_index
        - end_index

    MessageContentTextAnnotationsFilePathObject:
      title: File path
      type: object
      description: A URL for the file that's generated when the assistant used the `code_interpreter` tool to generate a file.
      properties:
        type:
          description: Always `file_path`.
          type: string
          enum: ["file_path"]
        text:
          description: The text in the message content that needs to be replaced.
          type: string
        file_path:
          type: object
          properties:
            file_id:
              description: The ID of the file that was generated.
              type: string
          required:
            - file_id
        start_index:
          type: integer
          minimum: 0
        end_index:
          type: integer
          minimum: 0
      required:
        - type
        - text
        - file_path
        - start_index
        - end_index

    RunStepObject:
      type: object
      title: Run steps
      description: |
        Represents a step in execution of a run.
      properties:
        id:
          description: The identifier of the run step, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `thread.run.step``.
          type: string
          enum: ["thread.run.step"]
        created_at:
          description: The Unix timestamp (in seconds) for when the run step was created.
          type: integer
        assistant_id:
          description: The ID of the [assistant](/docs/api-reference/assistants) associated with the run step.
          type: string
        thread_id:
          description: The ID of the [thread](/docs/api-reference/threads) that was run.
          type: string
        run_id:
          description: The ID of the [run](/docs/api-reference/runs) that this run step is a part of.
          type: string
        type:
          description: The type of run step, which can be either `message_creation` or `tool_calls`.
          type: string
          enum: ["message_creation", "tool_calls"]
        status:
          description: The status of the run step, which can be either `in_progress`, `cancelled`, `failed`, `completed`, or `expired`.
          type: string
          enum: ["in_progress", "cancelled", "failed", "completed", "expired"]
        step_details:
          type: object
          description: The details of the run step.
          oneOf:
            - $ref: "#/components/schemas/RunStepDetailsMessageCreationObject"
            - $ref: "#/components/schemas/RunStepDetailsToolCallsObject"
          x-oaiExpandable: true
        last_error:
          type: object
          description: The last error associated with this run step. Will be `null` if there are no errors.
          nullable: true
          properties:
            code:
              type: string
              description: One of `server_error` or `rate_limit_exceeded`.
              enum: ["server_error", "rate_limit_exceeded"]
            message:
              type: string
              description: A human-readable description of the error.
          required:
            - code
            - message
        expired_at:
          description: The Unix timestamp (in seconds) for when the run step expired. A step is considered expired if the parent run is expired.
          type: integer
          nullable: true
        cancelled_at:
          description: The Unix timestamp (in seconds) for when the run step was cancelled.
          type: integer
          nullable: true
        failed_at:
          description: The Unix timestamp (in seconds) for when the run step failed.
          type: integer
          nullable: true
        completed_at:
          description: The Unix timestamp (in seconds) for when the run step completed.
          type: integer
          nullable: true
        metadata:
          description: *metadata_description
          type: object
          x-oaiTypeLabel: map
          nullable: true
      required:
        - id
        - object
        - created_at
        - assistant_id
        - thread_id
        - run_id
        - type
        - status
        - step_details
        - last_error
        - expired_at
        - cancelled_at
        - failed_at
        - completed_at
        - metadata
      x-oaiMeta:
        name: The run step object
        beta: true
        example: *run_step_object_example

    ListRunStepsResponse:
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/RunStepObject"
        first_id:
          type: string
          example: "step_hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "step_QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - first_id
        - last_id
        - has_more

    RunStepDetailsMessageCreationObject:
      title: Message creation
      type: object
      description: Details of the message creation by the run step.
      properties:
        type:
          description: Always `message_creation``.
          type: string
          enum: ["message_creation"]
        message_creation:
          type: object
          properties:
            message_id:
              type: string
              description: The ID of the message that was created by this run step.
          required:
            - message_id
      required:
        - type
        - message_creation

    RunStepDetailsToolCallsObject:
      title: Tool calls
      type: object
      description: Details of the tool call.
      properties:
        type:
          description: Always `tool_calls`.
          type: string
          enum: ["tool_calls"]
        tool_calls:
          type: array
          description: |
            An array of tool calls the run step was involved in. These can be associated with one of three types of tools: `code_interpreter`, `retrieval`, or `function`.
          items:
            type: object
            oneOf:
              - $ref: "#/components/schemas/RunStepDetailsToolCallsCodeObject"
              - $ref: "#/components/schemas/RunStepDetailsToolCallsRetrievalObject"
              - $ref: "#/components/schemas/RunStepDetailsToolCallsFunctionObject"
            x-oaiExpandable: true
      required:
        - type
        - tool_calls

    RunStepDetailsToolCallsCodeObject:
      title: Code interpreter tool call
      type: object
      description: Details of the Code Interpreter tool call the run step was involved in.
      properties:
        id:
          type: string
          description: The ID of the tool call.
        type:
          type: string
          description: The type of tool call. This is always going to be `code_interpreter` for this type of tool call.
          enum: ["code_interpreter"]
        code_interpreter:
          type: object
          description: The Code Interpreter tool call definition.
          required:
            - input
            - outputs
          properties:
            input:
              type: string
              description: The input to the Code Interpreter tool call.
            outputs:
              type: array
              description: The outputs from the Code Interpreter tool call. Code Interpreter can output one or more items, including text (`logs`) or images (`image`). Each of these are represented by a different object type.
              items:
                type: object
                oneOf:
                  - $ref: "#/components/schemas/RunStepDetailsToolCallsCodeOutputLogsObject"
                  - $ref: "#/components/schemas/RunStepDetailsToolCallsCodeOutputImageObject"
                x-oaiExpandable: true
      required:
        - id
        - type
        - code_interpreter

    RunStepDetailsToolCallsCodeOutputLogsObject:
      title: Code interpreter log output
      type: object
      description: Text output from the Code Interpreter tool call as part of a run step.
      properties:
        type:
          description: Always `logs`.
          type: string
          enum: ["logs"]
        logs:
          type: string
          description: The text output from the Code Interpreter tool call.
      required:
        - type
        - logs

    RunStepDetailsToolCallsCodeOutputImageObject:
      title: Code interpreter image output
      type: object
      properties:
        type:
          description: Always `image`.
          type: string
          enum: ["image"]
        image:
          type: object
          properties:
            file_id:
              description: The [file](/docs/api-reference/files) ID of the image.
              type: string
          required:
            - file_id
      required:
        - type
        - image

    RunStepDetailsToolCallsRetrievalObject:
      title: Retrieval tool call
      type: object
      properties:
        id:
          type: string
          description: The ID of the tool call object.
        type:
          type: string
          description: The type of tool call. This is always going to be `retrieval` for this type of tool call.
          enum: ["retrieval"]
        retrieval:
          type: object
          description: For now, this is always going to be an empty object.
          x-oaiTypeLabel: map
      required:
        - id
        - type
        - retrieval

    RunStepDetailsToolCallsFunctionObject:
      type: object
      title: Function tool call
      properties:
        id:
          type: string
          description: The ID of the tool call object.
        type:
          type: string
          description: The type of tool call. This is always going to be `function` for this type of tool call.
          enum: ["function"]
        function:
          type: object
          description: The definition of the function that was called.
          properties:
            name:
              type: string
              description: The name of the function.
            arguments:
              type: string
              description: The arguments passed to the function.
            output:
              type: string
              description: The output of the function. This will be `null` if the outputs have not been [submitted](/docs/api-reference/runs/submitToolOutputs) yet.
              nullable: true
          required:
            - name
            - arguments
            - output
      required:
        - id
        - type
        - function

    AssistantFileObject:
      type: object
      title: Assistant files
      description: A list of [Files](/docs/api-reference/files) attached to an `assistant`.
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `assistant.file`.
          type: string
          enum: [assistant.file]
        created_at:
          description: The Unix timestamp (in seconds) for when the assistant file was created.
          type: integer
        assistant_id:
          description: The assistant ID that the file is attached to.
          type: string
      required:
        - id
        - object
        - created_at
        - assistant_id
      x-oaiMeta:
        name: The assistant file object
        beta: true
        example: |
          {
            "id": "file-wB6RM6wHdA49HfS2DJ9fEyrH",
            "object": "assistant.file",
            "created_at": 1699055364,
            "assistant_id": "asst_FBOFvAOHhwEWMghbMGseaPGQ"
          }

    CreateAssistantFileRequest:
      type: object
      additionalProperties: false
      properties:
        file_id:
          description: A [File](/docs/api-reference/files) ID (with `purpose="assistants"`) that the assistant should use. Useful for tools like `retrieval` and `code_interpreter` that can access files.
          type: string
      required:
        - file_id

    DeleteAssistantFileResponse:
      type: object
      description: Deletes the association between the assistant and the file, but does not delete the [File](/docs/api-reference/files) object itself.
      properties:
        id:
          type: string
        deleted:
          type: boolean
        object:
          type: string
          enum: [assistant.file.deleted]
      required:
        - id
        - object
        - deleted
    ListAssistantFilesResponse:
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/AssistantFileObject"
        first_id:
          type: string
          example: "file-hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "file-QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - items
        - first_id
        - last_id
        - has_more

    MessageFileObject:
      type: object
      title: Message files
      description: A list of files attached to a `message`.
      properties:
        id:
          description: The identifier, which can be referenced in API endpoints.
          type: string
        object:
          description: The object type, which is always `thread.message.file`.
          type: string
          enum: ["thread.message.file"]
        created_at:
          description: The Unix timestamp (in seconds) for when the message file was created.
          type: integer
        message_id:
          description: The ID of the [message](/docs/api-reference/messages) that the [File](/docs/api-reference/files) is attached to.
          type: string
      required:
        - id
        - object
        - created_at
        - message_id
      x-oaiMeta:
        name: The message file object
        beta: true
        example: |
          {
            "id": "file-BK7bzQj3FfZFXr7DbL6xJwfo",
            "object": "thread.message.file",
            "created_at": 1698107661,
            "message_id": "message_QLoItBbqwyAJEzlTy4y9kOMM",
            "file_id": "file-BK7bzQj3FfZFXr7DbL6xJwfo"
          }

    ListMessageFilesResponse:
      properties:
        object:
          type: string
          example: "list"
        data:
          type: array
          items:
            $ref: "#/components/schemas/MessageFileObject"
        first_id:
          type: string
          example: "file-hLBK7PXBv5Lr2NQT7KLY0ag1"
        last_id:
          type: string
          example: "file-QLoItBbqwyAJEzlTy4y9kOMM"
        has_more:
          type: boolean
          example: false
      required:
        - object
        - data
        - items
        - first_id
        - last_id
        - has_more

security:
  - ApiKeyAuth: []
x-oaiMeta:
  groups:
    # > General Notes
    # The `groups` section is used to generate the API reference pages and navigation, in the same
    # order listed below. Additionally, each `group` can have a list of `sections`, each of which
    # will become a navigation subroute and subsection under the group. Each section has:
    #  - `type`: Currently, either an `endpoint` or `object`, depending on how the section needs to
    #            be rendered
    #  - `key`: The reference key that can be used to lookup the section definition
    #  - `path`: The path (url) of the section, which is used to generate the navigation link.
    #
    # > The `object` sections maps to a schema component and the following fields are read for rendering
    # - `x-oaiMeta.name`: The name of the object, which will become the section title
    # - `x-oaiMeta.example`: The example object, which will be used to generate the example sample (always JSON)
    # - `description`: The description of the object, which will be used to generate the section description
    #
    # > The `endpoint` section maps to an operation path and the following fields are read for rendering:
    # - `x-oaiMeta.name`: The name of the endpoint, which will become the section title
    # - `x-oaiMeta.examples`: The endpoint examples, which can be an object (meaning a single variation, most
    #                         endpoints, or an array of objects, meaning multiple variations, e.g. the
    #                         chat completion and completion endpoints, with streamed and non-streamed examples.
    # - `x-oaiMeta.returns`: text describing what the endpoint returns.
    # - `summary`: The summary of the endpoint, which will be used to generate the section description
    - id: audio
      title: Audio
      description: |
        Learn how to turn audio into text or text into audio.

        Related guide: [Speech to text](/docs/guides/speech-to-text)
      sections:
        - type: endpoint
          key: createSpeech
          path: createSpeech
        - type: endpoint
          key: createTranscription
          path: createTranscription
        - type: endpoint
          key: createTranslation
          path: createTranslation
    - id: chat
      title: Chat
      description: |
        Given a list of messages comprising a conversation, the model will return a response.

        Related guide: [Chat Completions](/docs/guides/gpt)
      sections:
        - type: object
          key: CreateChatCompletionResponse
          path: object
        - type: object
          key: CreateChatCompletionStreamResponse
          path: streaming
        - type: endpoint
          key: createChatCompletion
          path: create
    - id: completions
      title: Completions
      legacy: true
      description: |
        Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position. We recommend most users use our Chat Completions API. [Learn more](/docs/deprecations/2023-07-06-gpt-and-embeddings)

        Related guide: [Legacy Completions](/docs/guides/gpt/completions-api)
      sections:
        - type: object
          key: CreateCompletionResponse
          path: object
        - type: endpoint
          key: createCompletion
          path: create
    - id: embeddings
      title: Embeddings
      description: |
        Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.

        Related guide: [Embeddings](/docs/guides/embeddings)
      sections:
        - type: object
          key: Embedding
          path: object
        - type: endpoint
          key: createEmbedding
          path: create
    - id: fine-tuning
      title: Fine-tuning
      description: |
        Manage fine-tuning jobs to tailor a model to your specific training data.

        Related guide: [Fine-tune models](/docs/guides/fine-tuning)
      sections:
        - type: object
          key: FineTuningJob
          path: object
        - type: endpoint
          key: createFineTuningJob
          path: create
        - type: endpoint
          key: listPaginatedFineTuningJobs
          path: list
        - type: endpoint
          key: retrieveFineTuningJob
          path: retrieve
        - type: endpoint
          key: cancelFineTuningJob
          path: cancel
        - type: object
          key: FineTuningJobEvent
          path: event-object
        - type: endpoint
          key: listFineTuningEvents
          path: list-events
    - id: files
      title: Files
      description: |
        Files are used to upload documents that can be used with features like [Assistants](/docs/api-reference/assistants) and [Fine-tuning](/docs/api-reference/fine-tuning).
      sections:
        - type: object
          key: OpenAIFile
          path: object
        - type: endpoint
          key: listFiles
          path: list
        - type: endpoint
          key: createFile
          path: create
        - type: endpoint
          key: deleteFile
          path: delete
        - type: endpoint
          key: retrieveFile
          path: retrieve
        - type: endpoint
          key: downloadFile
          path: retrieve-contents
    - id: images
      title: Images
      description: |
        Given a prompt and/or an input image, the model will generate a new image.

        Related guide: [Image generation](/docs/guides/images)
      sections:
        - type: object
          key: Image
          path: object
        - type: endpoint
          key: createImage
          path: create
        - type: endpoint
          key: createImageEdit
          path: createEdit
        - type: endpoint
          key: createImageVariation
          path: createVariation
    - id: models
      title: Models
      description: |
        List and describe the various models available in the API. You can refer to the [Models](/docs/models) documentation to understand what models are available and the differences between them.
      sections:
        - type: object
          key: Model
          path: object
        - type: endpoint
          key: listModels
          path: list
        - type: endpoint
          key: retrieveModel
          path: retrieve
        - type: endpoint
          key: deleteModel
          path: delete
    - id: moderations
      title: Moderations
      description: |
        Given a input text, outputs if the model classifies it as violating OpenAI's content policy.

        Related guide: [Moderations](/docs/guides/moderation)
      sections:
        - type: object
          key: CreateModerationResponse
          path: object
        - type: endpoint
          key: createModeration
          path: create
    - id: assistants
      title: Assistants
      beta: true
      description: |
        Build assistants that can call models and use tools to perform tasks.

        [Get started with the Assistants API](/docs/assistants)
      sections:
        - type: object
          key: AssistantObject
          path: object
        - type: endpoint
          key: createAssistant
          path: createAssistant
        - type: endpoint
          key: getAssistant
          path: getAssistant
        - type: endpoint
          key: modifyAssistant
          path: modifyAssistant
        - type: endpoint
          key: deleteAssistant
          path: deleteAssistant
        - type: endpoint
          key: listAssistants
          path: listAssistants
        - type: object
          key: AssistantFileObject
          path: file-object
        - type: endpoint
          key: createAssistantFile
          path: createAssistantFile
        - type: endpoint
          key: getAssistantFile
          path: getAssistantFile
        - type: endpoint
          key: deleteAssistantFile
          path: deleteAssistantFile
        - type: endpoint
          key: listAssistantFiles
          path: listAssistantFiles
    - id: threads
      title: Threads
      beta: true
      description: |
        Create threads that assistants can interact with.

        Related guide: [Assistants](/docs/assistants/overview)
      sections:
        - type: object
          key: ThreadObject
          path: object
        - type: endpoint
          key: createThread
          path: createThread
        - type: endpoint
          key: getThread
          path: getThread
        - type: endpoint
          key: modifyThread
          path: modifyThread
        - type: endpoint
          key: deleteThread
          path: deleteThread
    - id: messages
      title: Messages
      beta: true
      description: |
        Create messages within threads

        Related guide: [Assistants](/docs/assistants/overview)
      sections:
        - type: object
          key: MessageObject
          path: object
        - type: endpoint
          key: createMessage
          path: createMessage
        - type: endpoint
          key: getMessage
          path: getMessage
        - type: endpoint
          key: modifyMessage
          path: modifyMessage
        - type: endpoint
          key: listMessages
          path: listMessages
        - type: object
          key: MessageFileObject
          path: file-object
        - type: endpoint
          key: getMessageFile
          path: getMessageFile
        - type: endpoint
          key: listMessageFiles
          path: listMessageFiles
    - id: runs
      title: Runs
      beta: true
      description: |
        Represents an execution run on a thread.

        Related guide: [Assistants](/docs/assistants/overview)
      sections:
        - type: object
          key: RunObject
          path: object
        - type: endpoint
          key: createRun
          path: createRun
        - type: endpoint
          key: getRun
          path: getRun
        - type: endpoint
          key: modifyRun
          path: modifyRun
        - type: endpoint
          key: listRuns
          path: listRuns
        - type: endpoint
          key: submitToolOuputsToRun
          path: submitToolOutputs
        - type: endpoint
          key: cancelRun
          path: cancelRun
        - type: endpoint
          key: createThreadAndRun
          path: createThreadAndRun
        - type: object
          key: RunStepObject
          path: step-object
        - type: endpoint
          key: getRunStep
          path: getRunStep
        - type: endpoint
          key: listRunSteps
          path: listRunSteps
    - id: fine-tunes
      title: Fine-tunes
      deprecated: true
      description: |
        Manage legacy fine-tuning jobs to tailor a model to your specific training data.

        We recommend transitioning to the updating [fine-tuning API](/docs/guides/fine-tuning)
      sections:
        - type: object
          key: FineTune
          path: object
        - type: endpoint
          key: createFineTune
          path: create
        - type: endpoint
          key: listFineTunes
          path: list
        - type: endpoint
          key: retrieveFineTune
          path: retrieve
        - type: endpoint
          key: cancelFineTune
          path: cancel
        - type: object
          key: FineTuneEvent
          path: event-object
        - type: endpoint
          key: listFineTuneEvents
          path: list-events
    - id: edits
      title: Edits
      deprecated: true
      description: |
        Given a prompt and an instruction, the model will return an edited version of the prompt.
      sections:
        - type: object
          key: CreateEditResponse
          path: object
        - type: endpoint
          key: createEdit
          path: create
