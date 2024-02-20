from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, cast

from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from langchain_experimental.recommenders.amazon_personalize import AmazonPersonalize

SUMMARIZE_PROMPT_QUERY = """
Summarize the recommended items for a user from the items list in tag <result> below.
Make correlation into the items in the list and provide a summary.
    <result>
        {result}
    </result>
"""

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["result"], template=SUMMARIZE_PROMPT_QUERY
)

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

# Input Key Names to be used
USER_ID_INPUT_KEY = "user_id"
ITEM_ID_INPUT_KEY = "item_id"
INPUT_LIST_INPUT_KEY = "input_list"
FILTER_ARN_INPUT_KEY = "filter_arn"
FILTER_VALUES_INPUT_KEY = "filter_values"
CONTEXT_INPUT_KEY = "context"
PROMOTIONS_INPUT_KEY = "promotions"
METADATA_COLUMNS_INPUT_KEY = "metadata_columns"
RESULT_OUTPUT_KEY = "result"


class AmazonPersonalizeChain(Chain):
    """Amazon Personalize Chain for retrieving recommendations
                        from Amazon Personalize, and summarizing
    the recommendations in natural language.
                    It will only return recommendations if return_direct=True.
    Can also be used in sequential chains for working with
                                                the output of Amazon Personalize.

    Example:
        .. code-block:: python

        chain = PersonalizeChain.from_llm(llm=agent_llm, client=personalize_lg,
                                        return_direct=True)\n
        response = chain.run({'user_id':'1'})\n
        response = chain.run({'user_id':'1', 'item_id':'234'})
    """

    client: AmazonPersonalize
    summarization_chain: LLMChain
    return_direct: bool = False
    return_intermediate_steps: bool = False
    is_ranking_recipe: bool = False

    @property
    def input_keys(self) -> List[str]:
        """This returns an empty list since not there are optional
        input_keys and none is required.

        :meta private:
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """Will always return result key.

        :meta private:
        """
        return [RESULT_OUTPUT_KEY]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        client: AmazonPersonalize,
        prompt_template: PromptTemplate = SUMMARIZE_PROMPT,
        is_ranking_recipe: bool = False,
        **kwargs: Any,
    ) -> AmazonPersonalizeChain:
        """Initializes the Personalize Chain with LLMAgent, Personalize Client,
                                        Prompts to be used

            Args:
                llm: BaseLanguageModel: The LLM to be used in the Chain
                client: AmazonPersonalize: The client created to support
                                            invoking AmazonPersonalize
                prompt_template: PromptTemplate: The prompt template which can be
                                invoked with the output from Amazon Personalize
                is_ranking_recipe: bool: default: False: specifies
                                if the trained recipe is USER_PERSONALIZED_RANKING

        Example:
            .. code-block:: python

                chain = PersonalizeChain.from_llm(llm=agent_llm,
                                client=personalize_lg, return_direct=True)\n
                response = chain.run({'user_id':'1'})\n
                response = chain.run({'user_id':'1', 'item_id':'234'})

                RANDOM_PROMPT_QUERY=" Summarize recommendations in {result}"
                chain = PersonalizeChain.from_llm(llm=agent_llm,
                        client=personalize_lg, prompt_template=PROMPT_TEMPLATE)\n
        """
        summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

        return cls(
            summarization_chain=summarization_chain,
            client=client,
            is_ranking_recipe=is_ranking_recipe,
            **kwargs,
        )

    def _call(
        self,
        inputs: Mapping[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Retrieves recommendations by invoking Amazon Personalize,
                        and invokes an LLM using the default/overridden
        prompt template with the output from Amazon Personalize

            Args:
                inputs: Mapping [str, Any] : Provide input identifiers in a map.
                                                For example - {'user_id','1'} or
                        {'user_id':'1', 'item_id':'123'}. You can also pass the
                                        filter_arn, filter_values as an
                        input.
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        user_id = inputs.get(USER_ID_INPUT_KEY)
        item_id = inputs.get(ITEM_ID_INPUT_KEY)
        input_list = inputs.get(INPUT_LIST_INPUT_KEY)
        filter_arn = inputs.get(FILTER_ARN_INPUT_KEY)
        filter_values = inputs.get(FILTER_VALUES_INPUT_KEY)
        promotions = inputs.get(PROMOTIONS_INPUT_KEY)
        context = inputs.get(CONTEXT_INPUT_KEY)
        metadata_columns = inputs.get(METADATA_COLUMNS_INPUT_KEY)

        intermediate_steps: List = []
        intermediate_steps.append({"Calling Amazon Personalize"})

        if self.is_ranking_recipe:
            response = self.client.get_personalized_ranking(
                user_id=str(user_id),
                input_list=cast(List[str], input_list),
                filter_arn=filter_arn,
                filter_values=filter_values,
                context=context,
                metadata_columns=metadata_columns,
            )
        else:
            response = self.client.get_recommendations(
                user_id=user_id,
                item_id=item_id,
                filter_arn=filter_arn,
                filter_values=filter_values,
                context=context,
                promotions=promotions,
                metadata_columns=metadata_columns,
            )

        _run_manager.on_text("Call to Amazon Personalize complete \n")

        if self.return_direct:
            final_result = response
        else:
            result = self.summarization_chain(
                {RESULT_OUTPUT_KEY: response}, callbacks=callbacks
            )
            final_result = result[self.summarization_chain.output_key]

        intermediate_steps.append({"context": response})
        chain_result: Dict[str, Any] = {RESULT_OUTPUT_KEY: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
        return chain_result

    @property
    def _chain_type(self) -> str:
        return "amazon_personalize_chain"
