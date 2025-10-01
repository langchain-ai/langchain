import React from 'react';
import clsx from 'clsx';
import {useLocation} from 'react-router-dom';

function LegacyBadge() {
  return (
    <span className="badge badge--secondary">LEGACY</span>
  );
}

export default function NotFoundContent({className}) {
  const location = useLocation();
  const pathname = location.pathname.endsWith('/') ? location.pathname : location.pathname + '/'; // Ensure the path matches the keys in suggestedLinks
  const {canonical, alternative} = suggestedLinks[pathname] || {};

  return (
    <main className={clsx('container margin-vert--xl', className)}>
      <div className="row">
        <div className="col col--6 col--offset-3">
          <h1 className="hero__title">
              {canonical ? 'Page Moved' : alternative ? 'Page Removed' : 'Page Not Found'}
          </h1>
          {
            canonical ? (
              <h3>You can find the new location <a href={canonical}>here</a>.</h3>
            ) : alternative ? (
              <p>The page you were looking for has been removed.</p>
            ) : (
              <p>We could not find what you were looking for.</p>
            )
          }
          {alternative && (
            <p>
              <details>
                <summary>Alternative pages</summary>
                  <ul>
                    {alternative.map((alt, index) => (
                      <li key={index}>
                        <a href={alt}>{alt}</a>{alt.startsWith('/v0.1/') && <>{' '}<LegacyBadge/></>}
                      </li>
                    ))}
                  </ul>
              </details>
            </p>
          )}
          <p>
              Please contact the owner of the site that linked you to the
              original URL and let them know their link {canonical ? 'has moved.' : alternative ? 'has been removed.' : 'is broken.'}
          </p>
        </div>
      </div>
    </main>
  );
}

const suggestedLinks = {
  "/docs/changelog/core/": {
    "canonical": "https://github.com/langchain-ai/langchain/releases?q=tag:%22langchain-core%3D%3D0%22&expanded=true",
    "alternative": [
      "/v0.1/docs/changelog/core/"
    ]
  },
  "/docs/changelog/langchain/": {
    "canonical": "https://github.com/langchain-ai/langchain/releases?q=tag:%22langchain%3D%3D0%22&expanded=true",
    "alternative": [
      "/v0.1/docs/changelog/langchain/"
    ]
  },
  "/docs/contributing/documentation/technical_logistics/": {
    "canonical": "/docs/contributing/how_to/documentation/",
    "alternative": [
      "/v0.1/docs/contributing/documentation/technical_logistics/"
    ]
  },
  "/docs/cookbook/": {
    "canonical": "/docs/tutorials/",
    "alternative": [
      "/v0.1/docs/cookbook/"
    ]
  },
  "/docs/expression_language/": {
    "canonical": "/docs/how_to/#langchain-expression-language-lcel",
    "alternative": [
      "/v0.1/docs/expression_language/"
    ]
  },
  "/docs/expression_language/cookbook/code_writing/": {
    "canonical": "https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/",
    "alternative": [
      "/v0.1/docs/expression_language/cookbook/code_writing/"
    ]
  },
  "/docs/expression_language/cookbook/multiple_chains/": {
    "canonical": "/docs/how_to/parallel/",
    "alternative": [
      "/v0.1/docs/expression_language/cookbook/multiple_chains/"
    ]
  },
  "/docs/expression_language/cookbook/prompt_llm_parser/": {
    "canonical": "/docs/tutorials/llm_chain/",
    "alternative": [
      "/v0.1/docs/expression_language/cookbook/prompt_llm_parser/"
    ]
  },
  "/docs/expression_language/cookbook/prompt_size/": {
    "canonical": "/docs/how_to/trim_messages/",
    "alternative": [
      "/v0.1/docs/expression_language/cookbook/prompt_size/"
    ]
  },
  "/docs/expression_language/get_started/": {
    "canonical": "/docs/how_to/sequence/",
    "alternative": [
      "/v0.1/docs/expression_language/get_started/"
    ]
  },
  "/docs/expression_language/how_to/decorator/": {
    "canonical": "/docs/how_to/functions/#the-convenience-chain-decorator",
    "alternative": [
      "/v0.1/docs/expression_language/how_to/decorator/"
    ]
  },
  "/docs/expression_language/how_to/inspect/": {
    "canonical": "/docs/how_to/inspect/",
    "alternative": [
      "/v0.1/docs/expression_language/how_to/inspect/"
    ]
  },
  "/docs/expression_language/how_to/message_history/": {
    "canonical": "/docs/how_to/message_history/",
    "alternative": [
      "/v0.1/docs/expression_language/how_to/message_history/"
    ]
  },
  "/docs/expression_language/how_to/routing/": {
    "canonical": "/docs/how_to/routing/",
    "alternative": [
      "/v0.1/docs/expression_language/how_to/routing/"
    ]
  },
  "/docs/expression_language/interface/": {
    "canonical": "/docs/how_to/lcel_cheatsheet/",
    "alternative": [
      "/v0.1/docs/expression_language/interface/"
    ]
  },
  "/docs/expression_language/primitives/": {
    "canonical": "/docs/how_to/#langchain-expression-language-lcel",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/"
    ]
  },
  "/docs/expression_language/primitives/assign/": {
    "canonical": "/docs/how_to/assign/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/assign/"
    ]
  },
  "/docs/expression_language/primitives/binding/": {
    "canonical": "/docs/how_to/binding/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/binding/"
    ]
  },
  "/docs/expression_language/primitives/configure/": {
    "canonical": "/docs/how_to/configure/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/configure/"
    ]
  },
  "/docs/expression_language/primitives/functions/": {
    "canonical": "/docs/how_to/functions/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/functions/"
    ]
  },
  "/docs/expression_language/primitives/parallel/": {
    "canonical": "/docs/how_to/parallel/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/parallel/"
    ]
  },
  "/docs/expression_language/primitives/passthrough/": {
    "canonical": "/docs/how_to/passthrough/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/passthrough/"
    ]
  },
  "/docs/expression_language/primitives/sequence/": {
    "canonical": "/docs/how_to/sequence/",
    "alternative": [
      "/v0.1/docs/expression_language/primitives/sequence/"
    ]
  },
  "/docs/expression_language/streaming/": {
    "canonical": "/docs/how_to/streaming/",
    "alternative": [
      "/v0.1/docs/expression_language/streaming/"
    ]
  },
  "/docs/expression_language/why/": {
    "canonical": "/docs/concepts/#langchain-expression-language-lcel",
    "alternative": [
      "/v0.1/docs/expression_language/why/"
    ]
  },
  "/docs/get_started/installation/": {
    "canonical": "/docs/tutorials/",
    "alternative": [
      "/v0.1/docs/get_started/installation/"
    ]
  },
  "/docs/get_started/introduction/": {
    "canonical": "/docs/tutorials/",
    "alternative": [
      "/v0.1/docs/get_started/introduction/"
    ]
  },
  "/docs/get_started/quickstart/": {
    "canonical": "/docs/tutorials/",
    "alternative": [
      "/v0.1/docs/get_started/quickstart/"
    ]
  },
  "/docs/guides/": {
    "canonical": "/docs/how_to/",
    "alternative": [
      "/v0.1/docs/guides/"
    ]
  },
  "/docs/guides/development/": {
    "canonical": "/docs/how_to/debugging/",
    "alternative": [
      "/v0.1/docs/guides/development/"
    ]
  },
  "/docs/guides/development/debugging/": {
    "canonical": "/docs/how_to/debugging/",
    "alternative": [
      "/v0.1/docs/guides/development/debugging/"
    ]
  },
  "/docs/guides/development/extending_langchain/": {
    "canonical": "/docs/how_to/#custom",
    "alternative": [
      "/v0.1/docs/guides/development/extending_langchain/"
    ]
  },
  "/docs/guides/development/local_llms/": {
    "canonical": "/docs/how_to/local_llms/",
    "alternative": [
      "/v0.1/docs/guides/development/local_llms/"
    ]
  },
  "/docs/guides/development/pydantic_compatibility/": {
    "canonical": "/docs/how_to/pydantic_compatibility/",
    "alternative": [
      "/v0.1/docs/guides/development/pydantic_compatibility/"
    ]
  },
  "/docs/guides/productionization/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/"
    ]
  },
  "/docs/guides/productionization/deployments/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/deployments/"
    ]
  },
  "/docs/guides/productionization/deployments/template_repos/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/deployments/template_repos/"
    ]
  },
  "/docs/guides/productionization/evaluation/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/"
    ]
  },
  "/docs/guides/productionization/evaluation/comparison/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/comparison/"
    ]
  },
  "/docs/guides/productionization/evaluation/comparison/custom/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/comparison/custom/"
    ]
  },
  "/docs/guides/productionization/evaluation/comparison/pairwise_embedding_distance/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/comparison/pairwise_embedding_distance/"
    ]
  },
  "/docs/guides/productionization/evaluation/comparison/pairwise_string/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/comparison/pairwise_string/"
    ]
  },
  "/docs/guides/productionization/evaluation/examples/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/examples/"
    ]
  },
  "/docs/guides/productionization/evaluation/examples/comparisons/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/examples/comparisons/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/criteria_eval_chain/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/criteria_eval_chain/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/custom/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/custom/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/embedding_distance/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/embedding_distance/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/exact_match/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/exact_match/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/json/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/json/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/regex_match/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/regex_match/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/scoring_eval_chain/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/scoring_eval_chain/"
    ]
  },
  "/docs/guides/productionization/evaluation/string/string_distance/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/string/string_distance/"
    ]
  },
  "/docs/guides/productionization/evaluation/trajectory/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/trajectory/"
    ]
  },
  "/docs/guides/productionization/evaluation/trajectory/custom/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/trajectory/custom/"
    ]
  },
  "/docs/guides/productionization/evaluation/trajectory/trajectory_eval/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/evaluation/trajectory/trajectory_eval/"
    ]
  },
  "/docs/guides/productionization/fallbacks/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/fallbacks/"
    ]
  },
  "/docs/guides/productionization/safety/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/"
    ]
  },
  "/docs/guides/productionization/safety/amazon_comprehend_chain/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/amazon_comprehend_chain/"
    ]
  },
  "/docs/guides/productionization/safety/constitutional_chain/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/constitutional_chain/"
    ]
  },
  "/docs/guides/productionization/safety/hugging_face_prompt_injection/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/hugging_face_prompt_injection/"
    ]
  },
  "/docs/guides/productionization/safety/layerup_security/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/layerup_security/"
    ]
  },
  "/docs/guides/productionization/safety/logical_fallacy_chain/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/logical_fallacy_chain/"
    ]
  },
  "/docs/guides/productionization/safety/moderation/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/moderation/"
    ]
  },
  "/docs/guides/productionization/safety/presidio_data_anonymization/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/"
    ]
  },
  "/docs/guides/productionization/safety/presidio_data_anonymization/multi_language/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/multi_language/"
    ]
  },
  "/docs/guides/productionization/safety/presidio_data_anonymization/qa_privacy_protection/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/qa_privacy_protection/"
    ]
  },
  "/docs/guides/productionization/safety/presidio_data_anonymization/reversible/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/reversible/"
    ]
  },
  "/docs/integrations/chat/ollama_functions/": {
    "canonical": "/docs/integrations/chat/ollama/",
    "alternative": [
      "/v0.1/docs/integrations/chat/ollama_functions/"
    ]
  },
  "/docs/integrations/document_loaders/notiondb/": {
    "canonical": "/docs/integrations/document_loaders/notion/",
    "alternative": [
      "/v0.1/docs/integrations/document_loaders/notiondb/"
    ]
  },
  "/docs/integrations/llms/llm_caching/": {
    "canonical": "/docs/how_to/llm_caching/",
    "alternative": [
      "/v0.1/docs/integrations/llms/llm_caching/"
    ]
  },
  "/docs/integrations/providers/vectara/vectara_summary/": {
    "canonical": "/docs/integrations/providers/vectara/",
    "alternative": [
      "/v0.1/docs/integrations/providers/vectara/vectara_summary/"
    ]
  },
  "/docs/integrations/text_embedding/nemo/": {
    "canonical": "/docs/integrations/text_embedding/nvidia_ai_endpoints/",
    "alternative": [
      "/v0.1/docs/integrations/text_embedding/nemo/"
    ]
  },
  "/docs/integrations/toolkits/": {
    "canonical": "/docs/integrations/tools/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/"
    ]
  },
  "/docs/integrations/toolkits/ainetwork/": {
    "canonical": "/docs/integrations/tools/ainetwork/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/ainetwork/"
    ]
  },
  "/docs/integrations/toolkits/airbyte_structured_qa/": {
    "canonical": "/docs/integrations/document_loaders/airbyte/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/airbyte_structured_qa/"
    ]
  },
  "/docs/integrations/toolkits/amadeus/": {
    "canonical": "/docs/integrations/tools/amadeus/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/amadeus/"
    ]
  },
  "/docs/integrations/toolkits/azure_ai_services/": {
    "canonical": "/docs/integrations/tools/azure_ai_services/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/azure_ai_services/"
    ]
  },
  "/docs/integrations/toolkits/azure_cognitive_services/": {
    "canonical": "/docs/integrations/tools/azure_cognitive_services/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/azure_cognitive_services/"
    ]
  },
  "/docs/integrations/toolkits/cassandra_database/": {
    "canonical": "/docs/integrations/tools/cassandra_database/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/cassandra_database/"
    ]
  },
  "/docs/integrations/toolkits/clickup/": {
    "canonical": "/docs/integrations/tools/clickup/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/clickup/"
    ]
  },
  "/docs/integrations/toolkits/cogniswitch/": {
    "canonical": "/docs/integrations/tools/cogniswitch/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/cogniswitch/"
    ]
  },
  "/docs/integrations/toolkits/connery/": {
    "canonical": "/docs/integrations/tools/connery/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/connery/"
    ]
  },
  "/docs/integrations/toolkits/csv/": {
    "canonical": "/docs/integrations/document_loaders/csv/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/csv/"
    ]
  },
  "/docs/integrations/toolkits/document_comparison_toolkit/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/integrations/toolkits/document_comparison_toolkit/"
    ]
  },
  "/docs/integrations/toolkits/github/": {
    "canonical": "/docs/integrations/tools/github/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/github/"
    ]
  },
  "/docs/integrations/toolkits/gitlab/": {
    "canonical": "/docs/integrations/tools/gitlab/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/gitlab/"
    ]
  },
  "/docs/integrations/toolkits/gmail/": {
    "canonical": "/docs/integrations/tools/gmail/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/gmail/"
    ]
  },
  "/docs/integrations/toolkits/jira/": {
    "canonical": "/docs/integrations/tools/jira/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/jira/"
    ]
  },
  "/docs/integrations/toolkits/json/": {
    "canonical": "/docs/integrations/tools/json/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/json/"
    ]
  },
  "/docs/integrations/toolkits/multion/": {
    "canonical": "/docs/integrations/tools/multion/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/multion/"
    ]
  },
  "/docs/integrations/toolkits/nasa/": {
    "canonical": "/docs/integrations/tools/nasa/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/nasa/"
    ]
  },
  "/docs/integrations/toolkits/office365/": {
    "canonical": "/docs/integrations/tools/office365/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/office365/"
    ]
  },
  "/docs/integrations/toolkits/openapi_nla/": {
    "canonical": "/docs/integrations/tools/openapi_nla/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/openapi_nla/"
    ]
  },
  "/docs/integrations/toolkits/openapi/": {
    "canonical": "/docs/integrations/tools/openapi/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/openapi/"
    ]
  },
  "/docs/integrations/toolkits/pandas/": {
    "canonical": "/docs/integrations/tools/pandas/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/pandas/"
    ]
  },
  "/docs/integrations/toolkits/playwright/": {
    "canonical": "/docs/integrations/tools/playwright/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/playwright/"
    ]
  },
  "/docs/integrations/toolkits/polygon/": {
    "canonical": "/docs/integrations/tools/polygon/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/polygon/"
    ]
  },
  "/docs/integrations/toolkits/powerbi/": {
    "canonical": "/docs/integrations/tools/powerbi/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/powerbi/"
    ]
  },
  "/docs/integrations/toolkits/python/": {
    "canonical": "/docs/integrations/tools/python/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/python/"
    ]
  },
  "/docs/integrations/toolkits/robocorp/": {
    "canonical": "/docs/integrations/tools/robocorp/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/robocorp/"
    ]
  },
  "/docs/integrations/toolkits/slack/": {
    "canonical": "/docs/integrations/tools/slack/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/slack/"
    ]
  },
  "/docs/integrations/toolkits/spark_sql/": {
    "canonical": "/docs/integrations/tools/spark_sql/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/spark_sql/"
    ]
  },
  "/docs/integrations/toolkits/spark/": {
    "canonical": "/docs/integrations/tools/spark_sql/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/spark/"
    ]
  },
  "/docs/integrations/toolkits/sql_database/": {
    "canonical": "/docs/integrations/tools/sql_database/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/sql_database/"
    ]
  },
  "/docs/integrations/toolkits/steam/": {
    "canonical": "/docs/integrations/tools/steam/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/steam/"
    ]
  },
  "/docs/integrations/toolkits/xorbits/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/integrations/toolkits/xorbits/"
    ]
  },
  "/docs/integrations/tools/apify/": {
    "canonical": "/docs/integrations/providers/apify/#utility",
    "alternative": [
      "/v0.1/docs/integrations/tools/apify/"
    ]
  },
  "/docs/integrations/tools/search_tools/": {
    "canonical": "/docs/integrations/tools/#search",
    "alternative": [
      "/v0.1/docs/integrations/tools/search_tools/"
    ]
  },
  "/docs/langsmith/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/langsmith/"
    ]
  },
  "/docs/langsmith/walkthrough/": {
    "canonical": "https://docs.smith.langchain.com/",
    "alternative": [
      "/v0.1/docs/langsmith/walkthrough/"
    ]
  },
  "/docs/modules/": {
    "canonical": "/docs/how_to/#components",
    "alternative": [
      "/v0.1/docs/modules/"
    ]
  },
  "/docs/modules/agents/": {
    "canonical": "/docs/how_to/#agents",
    "alternative": [
      "/v0.1/docs/modules/agents/"
    ]
  },
  "/docs/modules/agents/agent_types/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/"
    ]
  },
  "/docs/modules/agents/agent_types/json_agent/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/json_agent/"
    ]
  },
  "/docs/modules/agents/agent_types/openai_assistants/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/openai_assistants/"
    ]
  },
  "/docs/modules/agents/agent_types/openai_functions_agent/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/openai_functions_agent/"
    ]
  },
  "/docs/modules/agents/agent_types/openai_tools/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/openai_tools/"
    ]
  },
  "/docs/modules/agents/agent_types/react/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/react/"
    ]
  },
  "/docs/modules/agents/agent_types/self_ask_with_search/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/self_ask_with_search/"
    ]
  },
  "/docs/modules/agents/agent_types/structured_chat/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/structured_chat/"
    ]
  },
  "/docs/modules/agents/agent_types/tool_calling/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/tool_calling/"
    ]
  },
  "/docs/modules/agents/agent_types/xml_agent/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/agent_types/xml_agent/"
    ]
  },
  "/docs/modules/agents/concepts/": {
    "canonical": "https://langchain-ai.github.io/langgraph/concepts/",
    "alternative": [
      "/v0.1/docs/modules/agents/concepts/"
    ]
  },
  "/docs/modules/agents/how_to/agent_iter/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/agent_iter/"
    ]
  },
  "/docs/modules/agents/how_to/agent_structured/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/agent_structured/"
    ]
  },
  "/docs/modules/agents/how_to/custom_agent/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/custom_agent/"
    ]
  },
  "/docs/modules/agents/how_to/handle_parsing_errors/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/handle_parsing_errors/"
    ]
  },
  "/docs/modules/agents/how_to/intermediate_steps/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/intermediate_steps/"
    ]
  },
  "/docs/modules/agents/how_to/max_iterations/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/max_iterations/"
    ]
  },
  "/docs/modules/agents/how_to/max_time_limit/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/max_time_limit/"
    ]
  },
  "/docs/modules/agents/how_to/streaming/": {
    "canonical": "/docs/how_to/migrate_agent/",
    "alternative": [
      "/v0.1/docs/modules/agents/how_to/streaming/"
    ]
  },
  "/docs/modules/agents/quick_start/": {
    "canonical": "https://langchain-ai.github.io/langgraph/",
    "alternative": [
      "/v0.1/docs/modules/agents/quick_start/"
    ]
  },
  "/docs/modules/callbacks/": {
    "canonical": "/docs/how_to/#callbacks",
    "alternative": [
      "/v0.1/docs/modules/callbacks/"
    ]
  },
  "/docs/modules/callbacks/async_callbacks/": {
    "canonical": "/docs/how_to/callbacks_async/",
    "alternative": [
      "/v0.1/docs/modules/callbacks/async_callbacks/"
    ]
  },
  "/docs/modules/callbacks/custom_callbacks/": {
    "canonical": "/docs/how_to/custom_callbacks/",
    "alternative": [
      "/v0.1/docs/modules/callbacks/custom_callbacks/"
    ]
  },
  "/docs/modules/callbacks/filecallbackhandler/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/modules/callbacks/filecallbackhandler/"
    ]
  },
  "/docs/modules/callbacks/multiple_callbacks/": {
    "canonical": "/docs/how_to/#callbacks",
    "alternative": [
      "/v0.1/docs/modules/callbacks/multiple_callbacks/"
    ]
  },
  "/docs/modules/callbacks/tags/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/modules/callbacks/tags/"
    ]
  },
  "/docs/modules/callbacks/token_counting/": {
    "canonical": "/docs/how_to/chat_token_usage_tracking/",
    "alternative": [
      "/v0.1/docs/modules/callbacks/token_counting/"
    ]
  },
  "/docs/modules/chains/": {
    "canonical": "/docs/versions/migrating_chains/",
    "alternative": [
      "/v0.1/docs/modules/chains/"
    ]
  },
  "/docs/modules/composition/": {
    "canonical": "https://langchain-ai.github.io/langgraph/concepts/",
    "alternative": [
      "/v0.1/docs/modules/composition/"
    ]
  },
  "/docs/modules/data_connection/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/": {
    "canonical": "/docs/how_to/#document-loaders",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/csv/": {
    "canonical": "/docs/integrations/document_loaders/csv/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/csv/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/custom/": {
    "canonical": "/docs/how_to/document_loader_custom/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/custom/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/file_directory/": {
    "canonical": "/docs/how_to/document_loader_directory/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/file_directory/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/html/": {
    "canonical": "/docs/how_to/document_loader_html/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/html/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/json/": {
    "canonical": "/docs/how_to/document_loader_json/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/json/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/markdown/": {
    "canonical": "/docs/how_to/document_loader_markdown/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/markdown/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/office_file/": {
    "canonical": "/docs/how_to/document_loader_office_file/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/office_file/"
    ]
  },
  "/docs/modules/data_connection/document_loaders/pdf/": {
    "canonical": "/docs/how_to/document_loader_pdf/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_loaders/pdf/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/": {
    "canonical": "/docs/how_to/#text-splitters",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/character_text_splitter/": {
    "canonical": "/docs/how_to/character_text_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/code_splitter/": {
    "canonical": "/docs/how_to/code_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/code_splitter/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/HTML_header_metadata/": {
    "canonical": "/docs/how_to/HTML_header_metadata_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/HTML_header_metadata/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/HTML_section_aware_splitter/": {
    "canonical": "/docs/how_to/HTML_section_aware_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/HTML_section_aware_splitter/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/markdown_header_metadata/": {
    "canonical": "/docs/how_to/markdown_header_metadata_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/recursive_json_splitter/": {
    "canonical": "/docs/how_to/recursive_json_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/recursive_json_splitter/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/recursive_text_splitter/": {
    "canonical": "/docs/how_to/recursive_text_splitter/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/semantic-chunker/": {
    "canonical": "/docs/how_to/semantic-chunker/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/"
    ]
  },
  "/docs/modules/data_connection/document_transformers/split_by_token/": {
    "canonical": "/docs/how_to/split_by_token/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/document_transformers/split_by_token/"
    ]
  },
  "/docs/modules/data_connection/indexing/": {
    "canonical": "/docs/how_to/indexing/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/indexing/"
    ]
  },
  "/docs/modules/data_connection/retrievers/": {
    "canonical": "/docs/how_to/#retrievers",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/"
    ]
  },
  "/docs/modules/data_connection/retrievers/contextual_compression/": {
    "canonical": "/docs/how_to/contextual_compression/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/contextual_compression/"
    ]
  },
  "/docs/modules/data_connection/retrievers/custom_retriever/": {
    "canonical": "/docs/how_to/custom_retriever/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/custom_retriever/"
    ]
  },
  "/docs/modules/data_connection/retrievers/ensemble/": {
    "canonical": "/docs/how_to/ensemble_retriever/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/ensemble/"
    ]
  },
  "/docs/modules/data_connection/retrievers/long_context_reorder/": {
    "canonical": "/docs/how_to/long_context_reorder/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/long_context_reorder/"
    ]
  },
  "/docs/modules/data_connection/retrievers/multi_vector/": {
    "canonical": "/docs/how_to/multi_vector/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/multi_vector/"
    ]
  },
  "/docs/modules/data_connection/retrievers/MultiQueryRetriever/": {
    "canonical": "/docs/how_to/MultiQueryRetriever/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/"
    ]
  },
  "/docs/modules/data_connection/retrievers/parent_document_retriever/": {
    "canonical": "/docs/how_to/parent_document_retriever/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/"
    ]
  },
  "/docs/modules/data_connection/retrievers/self_query/": {
    "canonical": "/docs/how_to/self_query/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/self_query/"
    ]
  },
  "/docs/modules/data_connection/retrievers/time_weighted_vectorstore/": {
    "canonical": "/docs/how_to/time_weighted_vectorstore/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/time_weighted_vectorstore/"
    ]
  },
  "/docs/modules/data_connection/retrievers/vectorstore/": {
    "canonical": "/docs/how_to/vectorstore_retriever/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/retrievers/vectorstore/"
    ]
  },
  "/docs/modules/data_connection/text_embedding/": {
    "canonical": "/docs/how_to/embed_text/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/text_embedding/"
    ]
  },
  "/docs/modules/data_connection/text_embedding/caching_embeddings/": {
    "canonical": "/docs/how_to/caching_embeddings/",
    "alternative": [
      "/v0.1/docs/modules/data_connection/text_embedding/caching_embeddings/"
    ]
  },
  "/docs/modules/data_connection/vectorstores/": {
    "canonical": "/docs/how_to/#vector-stores",
    "alternative": [
      "/v0.1/docs/modules/data_connection/vectorstores/"
    ]
  },
  "/docs/modules/memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/"
    ]
  },
  "/docs/modules/memory/adding_memory_chain_multiple_inputs/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/adding_memory_chain_multiple_inputs/"
    ]
  },
  "/docs/modules/memory/adding_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/adding_memory/"
    ]
  },
  "/docs/modules/memory/agent_with_memory_in_db/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/agent_with_memory_in_db/"
    ]
  },
  "/docs/modules/memory/agent_with_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/agent_with_memory/"
    ]
  },
  "/docs/modules/memory/chat_messages/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/chat_messages/"
    ]
  },
  "/docs/modules/memory/conversational_customization/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/conversational_customization/"
    ]
  },
  "/docs/modules/memory/custom_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/custom_memory/"
    ]
  },
  "/docs/modules/memory/multiple_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/multiple_memory/"
    ]
  },
  "/docs/modules/memory/types/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/"
    ]
  },
  "/docs/modules/memory/types/buffer_window/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/buffer_window/"
    ]
  },
  "/docs/modules/memory/types/buffer/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/buffer/"
    ]
  },
  "/docs/modules/memory/types/entity_summary_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/entity_summary_memory/"
    ]
  },
  "/docs/modules/memory/types/kg/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/kg/"
    ]
  },
  "/docs/modules/memory/types/summary_buffer/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/summary_buffer/"
    ]
  },
  "/docs/modules/memory/types/summary/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/summary/"
    ]
  },
  "/docs/modules/memory/types/token_buffer/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/token_buffer/"
    ]
  },
  "/docs/modules/memory/types/vectorstore_retriever_memory/": {
    "canonical": "/docs/how_to/chatbots_memory/",
    "alternative": [
      "/v0.1/docs/modules/memory/types/vectorstore_retriever_memory/"
    ]
  },
  "/docs/modules/model_io/": {
    "canonical": "/docs/how_to/#chat-models",
    "alternative": [
      "/v0.1/docs/modules/model_io/"
    ]
  },
  "/docs/modules/model_io/chat/": {
    "canonical": "/docs/how_to/#chat-models",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/"
    ]
  },
  "/docs/modules/model_io/chat/chat_model_caching/": {
    "canonical": "/docs/how_to/chat_model_caching/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/chat_model_caching/"
    ]
  },
  "/docs/modules/model_io/chat/custom_chat_model/": {
    "canonical": "/docs/how_to/custom_chat_model/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/custom_chat_model/"
    ]
  },
  "/docs/modules/model_io/chat/function_calling/": {
    "canonical": "/docs/how_to/tool_calling/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/function_calling/"
    ]
  },
  "/docs/modules/model_io/chat/logprobs/": {
    "canonical": "/docs/how_to/logprobs/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/logprobs/"
    ]
  },
  "/docs/modules/model_io/chat/message_types/": {
    "canonical": "/docs/concepts/#messages",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/message_types/"
    ]
  },
  "/docs/modules/model_io/chat/quick_start/": {
    "canonical": "/docs/tutorials/llm_chain/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/quick_start/"
    ]
  },
  "/docs/modules/model_io/chat/response_metadata/": {
    "canonical": "/docs/how_to/response_metadata/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/response_metadata/"
    ]
  },
  "/docs/modules/model_io/chat/streaming/": {
    "canonical": "/docs/how_to/streaming/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/streaming/"
    ]
  },
  "/docs/modules/model_io/chat/structured_output/": {
    "canonical": "/docs/how_to/structured_output/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/structured_output/"
    ]
  },
  "/docs/modules/model_io/chat/token_usage_tracking/": {
    "canonical": "/docs/how_to/chat_token_usage_tracking/",
    "alternative": [
      "/v0.1/docs/modules/model_io/chat/token_usage_tracking/"
    ]
  },
  "/docs/modules/model_io/concepts/": {
    "canonical": "/docs/concepts/#chat-models",
    "alternative": [
      "/v0.1/docs/modules/model_io/concepts/"
    ]
  },
  "/docs/modules/model_io/llms/": {
    "canonical": "/docs/concepts/#llms",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/"
    ]
  },
  "/docs/modules/model_io/llms/custom_llm/": {
    "canonical": "/docs/how_to/custom_llm/",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/custom_llm/"
    ]
  },
  "/docs/modules/model_io/llms/llm_caching/": {
    "canonical": "/docs/how_to/llm_caching/",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/llm_caching/"
    ]
  },
  "/docs/modules/model_io/llms/quick_start/": {
    "canonical": "/docs/tutorials/llm_chain/",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/quick_start/"
    ]
  },
  "/docs/modules/model_io/llms/streaming_llm/": {
    "canonical": "/docs/how_to/streaming_llm/",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/streaming_llm/"
    ]
  },
  "/docs/modules/model_io/llms/token_usage_tracking/": {
    "canonical": "/docs/how_to/llm_token_usage_tracking/",
    "alternative": [
      "/v0.1/docs/modules/model_io/llms/token_usage_tracking/"
    ]
  },
  "/docs/modules/model_io/output_parsers/": {
    "canonical": "/docs/how_to/#output-parsers",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/"
    ]
  },
  "/docs/modules/model_io/output_parsers/custom/": {
    "canonical": "/docs/how_to/output_parser_custom/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/custom/"
    ]
  },
  "/docs/modules/model_io/output_parsers/quick_start/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/quick_start/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/csv/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/csv/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/datetime/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/datetime/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/enum/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/enum/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/json/": {
    "canonical": "/docs/how_to/output_parser_json/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/json/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/openai_functions/": {
    "canonical": "/docs/how_to/structured_output/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/openai_functions/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/openai_tools/": {
    "canonical": "/docs/how_to/tool_calling/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/openai_tools/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/output_fixing/": {
    "canonical": "/docs/how_to/output_parser_fixing/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/output_fixing/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/pandas_dataframe/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/pandas_dataframe/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/pydantic/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/pydantic/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/retry/": {
    "canonical": "/docs/how_to/output_parser_retry/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/retry/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/structured/": {
    "canonical": "/docs/how_to/output_parser_structured/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/structured/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/xml/": {
    "canonical": "/docs/how_to/output_parser_xml/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/xml/"
    ]
  },
  "/docs/modules/model_io/output_parsers/types/yaml/": {
    "canonical": "/docs/how_to/output_parser_yaml/",
    "alternative": [
      "/v0.1/docs/modules/model_io/output_parsers/types/yaml/"
    ]
  },
  "/docs/modules/model_io/prompts/": {
    "canonical": "/docs/how_to/#prompt-templates",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/"
    ]
  },
  "/docs/modules/model_io/prompts/composition/": {
    "canonical": "/docs/how_to/prompts_composition/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/composition/"
    ]
  },
  "/docs/modules/model_io/prompts/example_selectors/": {
    "canonical": "/docs/how_to/example_selectors/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/example_selectors/"
    ]
  },
  "/docs/modules/model_io/prompts/example_selectors/length_based/": {
    "canonical": "/docs/how_to/example_selectors_length_based/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/example_selectors/length_based/"
    ]
  },
  "/docs/modules/model_io/prompts/example_selectors/mmr/": {
    "canonical": "/docs/how_to/example_selectors_mmr/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/example_selectors/mmr/"
    ]
  },
  "/docs/modules/model_io/prompts/example_selectors/ngram_overlap/": {
    "canonical": "/docs/how_to/example_selectors_ngram/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/example_selectors/ngram_overlap/"
    ]
  },
  "/docs/modules/model_io/prompts/example_selectors/similarity/": {
    "canonical": "/docs/how_to/example_selectors_similarity/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/example_selectors/similarity/"
    ]
  },
  "/docs/modules/model_io/prompts/few_shot_examples_chat/": {
    "canonical": "/docs/how_to/few_shot_examples_chat/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/"
    ]
  },
  "/docs/modules/model_io/prompts/few_shot_examples/": {
    "canonical": "/docs/how_to/few_shot_examples/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/few_shot_examples/"
    ]
  },
  "/docs/modules/model_io/prompts/partial/": {
    "canonical": "/docs/how_to/prompts_partial/",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/partial/"
    ]
  },
  "/docs/modules/model_io/prompts/quick_start/": {
    "canonical": "/docs/how_to/#prompt-templates",
    "alternative": [
      "/v0.1/docs/modules/model_io/prompts/quick_start/"
    ]
  },
  "/docs/modules/model_io/quick_start/": {
    "canonical": "/docs/tutorials/llm_chain/",
    "alternative": [
      "/v0.1/docs/modules/model_io/quick_start/"
    ]
  },
  "/docs/modules/tools/": {
    "canonical": "/docs/how_to/#tools",
    "alternative": [
      "/v0.1/docs/modules/tools/"
    ]
  },
  "/docs/modules/tools/custom_tools/": {
    "canonical": "/docs/how_to/custom_tools/",
    "alternative": [
      "/v0.1/docs/modules/tools/custom_tools/"
    ]
  },
  "/docs/modules/tools/toolkits/": {
    "canonical": "/docs/how_to/#tools",
    "alternative": [
      "/v0.1/docs/modules/tools/toolkits/"
    ]
  },
  "/docs/modules/tools/tools_as_openai_functions/": {
    "canonical": "/docs/how_to/tool_calling/",
    "alternative": [
      "/v0.1/docs/modules/tools/tools_as_openai_functions/"
    ]
  },
  "/docs/packages/": {
    "canonical": "/docs/versions/release_policy/",
    "alternative": [
      "/v0.1/docs/packages/"
    ]
  },
  "/docs/templates/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/"
    ]
  },
  "/docs/templates/anthropic-iterative-search/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/anthropic-iterative-search/"
    ]
  },
  "/docs/templates/basic-critique-revise/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/basic-critique-revise/"
    ]
  },
  "/docs/templates/bedrock-jcvd/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/bedrock-jcvd/"
    ]
  },
  "/docs/templates/cassandra-entomology-rag/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/cassandra-entomology-rag/"
    ]
  },
  "/docs/templates/cassandra-synonym-caching/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/cassandra-synonym-caching/"
    ]
  },
  "/docs/templates/chain-of-note-wiki/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/chain-of-note-wiki/"
    ]
  },
  "/docs/templates/chat-bot-feedback/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/chat-bot-feedback/"
    ]
  },
  "/docs/templates/cohere-librarian/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/cohere-librarian/"
    ]
  },
  "/docs/templates/csv-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/csv-agent/"
    ]
  },
  "/docs/templates/elastic-query-generator/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/elastic-query-generator/"
    ]
  },
  "/docs/templates/extraction-anthropic-functions/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/extraction-anthropic-functions/"
    ]
  },
  "/docs/templates/extraction-openai-functions/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/extraction-openai-functions/"
    ]
  },
  "/docs/templates/gemini-functions-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/gemini-functions-agent/"
    ]
  },
  "/docs/templates/guardrails-output-parser/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/guardrails-output-parser/"
    ]
  },
  "/docs/templates/hybrid-search-weaviate/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/hybrid-search-weaviate/"
    ]
  },
  "/docs/templates/hyde/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/hyde/"
    ]
  },
  "/docs/templates/intel-rag-xeon/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/intel-rag-xeon/"
    ]
  },
  "/docs/templates/llama2-functions/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/llama2-functions/"
    ]
  },
  "/docs/templates/mongo-parent-document-retrieval/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/mongo-parent-document-retrieval/"
    ]
  },
  "/docs/templates/neo4j-advanced-rag/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-advanced-rag/"
    ]
  },
  "/docs/templates/neo4j-cypher-ft/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-cypher-ft/"
    ]
  },
  "/docs/templates/neo4j-cypher-memory/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-cypher-memory/"
    ]
  },
  "/docs/templates/neo4j-cypher/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-cypher/"
    ]
  },
  "/docs/templates/neo4j-generation/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-generation/"
    ]
  },
  "/docs/templates/neo4j-parent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-parent/"
    ]
  },
  "/docs/templates/neo4j-semantic-layer/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-semantic-layer/"
    ]
  },
  "/docs/templates/neo4j-semantic-ollama/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-semantic-ollama/"
    ]
  },
  "/docs/templates/neo4j-vector-memory/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/neo4j-vector-memory/"
    ]
  },
  "/docs/templates/nvidia-rag-canonical/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/nvidia-rag-canonical/"
    ]
  },
  "/docs/templates/openai-functions-agent-gmail/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/openai-functions-agent-gmail/"
    ]
  },
  "/docs/templates/openai-functions-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/openai-functions-agent/"
    ]
  },
  "/docs/templates/openai-functions-tool-retrieval-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/openai-functions-tool-retrieval-agent/"
    ]
  },
  "/docs/templates/pii-protected-chatbot/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/pii-protected-chatbot/"
    ]
  },
  "/docs/templates/pirate-speak-configurable/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/pirate-speak-configurable/"
    ]
  },
  "/docs/templates/pirate-speak/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/pirate-speak/"
    ]
  },
  "/docs/templates/plate-chain/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/plate-chain/"
    ]
  },
  "/docs/templates/propositional-retrieval/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/propositional-retrieval/"
    ]
  },
  "/docs/templates/python-lint/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/python-lint/"
    ]
  },
  "/docs/templates/rag-astradb/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-astradb/"
    ]
  },
  "/docs/templates/rag-aws-bedrock/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-aws-bedrock/"
    ]
  },
  "/docs/templates/rag-aws-kendra/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-aws-kendra/"
    ]
  },
  "/docs/templates/rag-azure-search/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-azure-search/"
    ]
  },
  "/docs/templates/rag-chroma-multi-modal-multi-vector/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-chroma-multi-modal-multi-vector/"
    ]
  },
  "/docs/templates/rag-chroma-multi-modal/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-chroma-multi-modal/"
    ]
  },
  "/docs/templates/rag-chroma-private/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-chroma-private/"
    ]
  },
  "/docs/templates/rag-chroma/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-chroma/"
    ]
  },
  "/docs/templates/rag-codellama-fireworks/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-codellama-fireworks/"
    ]
  },
  "/docs/templates/rag-conversation-zep/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-conversation-zep/"
    ]
  },
  "/docs/templates/rag-conversation/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-conversation/"
    ]
  },
  "/docs/templates/rag-elasticsearch/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-elasticsearch/"
    ]
  },
  "/docs/templates/rag-fusion/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-fusion/"
    ]
  },
  "/docs/templates/rag-gemini-multi-modal/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-gemini-multi-modal/"
    ]
  },
  "/docs/templates/rag-google-cloud-sensitive-data-protection/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-google-cloud-sensitive-data-protection/"
    ]
  },
  "/docs/templates/rag-google-cloud-vertexai-search/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-google-cloud-vertexai-search/"
    ]
  },
  "/docs/templates/rag-gpt-crawler/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-gpt-crawler/"
    ]
  },
  "/docs/templates/rag-jaguardb/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-jaguardb/"
    ]
  },
  "/docs/templates/rag-lancedb/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-lancedb/"
    ]
  },
  "/docs/templates/rag-lantern/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-lantern/"
    ]
  },
  "/docs/templates/rag-matching-engine/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-matching-engine/"
    ]
  },
  "/docs/templates/rag-momento-vector-index/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-momento-vector-index/"
    ]
  },
  "/docs/templates/rag-mongo/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-mongo/"
    ]
  },
  "/docs/templates/rag-multi-index-fusion/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-multi-index-fusion/"
    ]
  },
  "/docs/templates/rag-multi-index-router/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-multi-index-router/"
    ]
  },
  "/docs/templates/rag-multi-modal-local/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-multi-modal-local/"
    ]
  },
  "/docs/templates/rag-multi-modal-mv-local/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-multi-modal-mv-local/"
    ]
  },
  "/docs/templates/rag-ollama-multi-query/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-ollama-multi-query/"
    ]
  },
  "/docs/templates/rag-opensearch/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-opensearch/"
    ]
  },
  "/docs/templates/rag-pinecone-multi-query/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-pinecone-multi-query/"
    ]
  },
  "/docs/templates/rag-pinecone-rerank/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-pinecone-rerank/"
    ]
  },
  "/docs/templates/rag-pinecone/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-pinecone/"
    ]
  },
  "/docs/templates/rag-redis-multi-modal-multi-vector/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-redis-multi-modal-multi-vector/"
    ]
  },
  "/docs/templates/rag-redis/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-redis/"
    ]
  },
  "/docs/templates/rag-self-query/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-self-query/"
    ]
  },
  "/docs/templates/rag-semi-structured/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-semi-structured/"
    ]
  },
  "/docs/templates/rag-singlestoredb/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-singlestoredb/"
    ]
  },
  "/docs/templates/rag-supabase/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-supabase/"
    ]
  },
  "/docs/templates/rag-timescale-conversation/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-timescale-conversation/"
    ]
  },
  "/docs/templates/rag-timescale-hybrid-search-time/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-timescale-hybrid-search-time/"
    ]
  },
  "/docs/templates/rag-vectara-multiquery/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-vectara-multiquery/"
    ]
  },
  "/docs/templates/rag-vectara/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-vectara/"
    ]
  },
  "/docs/templates/rag-weaviate/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rag-weaviate/"
    ]
  },
  "/docs/templates/research-assistant/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/research-assistant/"
    ]
  },
  "/docs/templates/retrieval-agent-fireworks/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/retrieval-agent-fireworks/"
    ]
  },
  "/docs/templates/retrieval-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/retrieval-agent/"
    ]
  },
  "/docs/templates/rewrite-retrieve-read/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/rewrite-retrieve-read/"
    ]
  },
  "/docs/templates/robocorp-action-server/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/robocorp-action-server/"
    ]
  },
  "/docs/templates/self-query-qdrant/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/self-query-qdrant/"
    ]
  },
  "/docs/templates/self-query-supabase/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/self-query-supabase/"
    ]
  },
  "/docs/templates/shopping-assistant/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/shopping-assistant/"
    ]
  },
  "/docs/templates/skeleton-of-thought/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/skeleton-of-thought/"
    ]
  },
  "/docs/templates/solo-performance-prompting-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/solo-performance-prompting-agent/"
    ]
  },
  "/docs/templates/sql-llama2/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/sql-llama2/"
    ]
  },
  "/docs/templates/sql-llamacpp/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/sql-llamacpp/"
    ]
  },
  "/docs/templates/sql-ollama/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/sql-ollama/"
    ]
  },
  "/docs/templates/sql-pgvector/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/sql-pgvector/"
    ]
  },
  "/docs/templates/sql-research-assistant/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/sql-research-assistant/"
    ]
  },
  "/docs/templates/stepback-qa-prompting/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/stepback-qa-prompting/"
    ]
  },
  "/docs/templates/summarize-anthropic/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/summarize-anthropic/"
    ]
  },
  "/docs/templates/vertexai-chuck-norris/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/vertexai-chuck-norris/"
    ]
  },
  "/docs/templates/xml-agent/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/templates/xml-agent/"
    ]
  },
  "/docs/use_cases/": {
    "canonical": "/docs/tutorials/",
    "alternative": [
      "/v0.1/docs/use_cases/"
    ]
  },
  "/docs/use_cases/apis/": {
    "canonical": null,
    "alternative": [
      "/v0.1/docs/use_cases/apis/"
    ]
  },
  "/docs/use_cases/chatbots/": {
    "canonical": "/docs/tutorials/chatbot/",
    "alternative": [
      "/v0.1/docs/use_cases/chatbots/"
    ]
  },
  "/docs/use_cases/chatbots/memory_management/": {
    "canonical": "/docs/tutorials/chatbot/",
    "alternative": [
      "/v0.1/docs/use_cases/chatbots/memory_management/"
    ]
  },
  "/docs/use_cases/chatbots/quickstart/": {
    "canonical": "/docs/tutorials/chatbot/",
    "alternative": [
      "/v0.1/docs/use_cases/chatbots/quickstart/"
    ]
  },
  "/docs/use_cases/chatbots/retrieval/": {
    "canonical": "/docs/tutorials/chatbot/",
    "alternative": [
      "/v0.1/docs/use_cases/chatbots/retrieval/"
    ]
  },
  "/docs/use_cases/chatbots/tool_usage/": {
    "canonical": "/docs/tutorials/chatbot/",
    "alternative": [
      "/v0.1/docs/use_cases/chatbots/tool_usage/"
    ]
  },
  "/docs/use_cases/code_understanding/": {
    "canonical": "https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/",
    "alternative": [
      "/v0.1/docs/use_cases/code_understanding/"
    ]
  },
  "/docs/use_cases/data_generation/": {
    "canonical": "/docs/tutorials/data_generation/",
    "alternative": [
      "/v0.1/docs/use_cases/data_generation/"
    ]
  },
  "/docs/use_cases/extraction/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/"
    ]
  },
  "/docs/use_cases/extraction/guidelines/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/guidelines/"
    ]
  },
  "/docs/use_cases/extraction/how_to/examples/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/how_to/examples/"
    ]
  },
  "/docs/use_cases/extraction/how_to/handle_files/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/how_to/handle_files/"
    ]
  },
  "/docs/use_cases/extraction/how_to/handle_long_text/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/how_to/handle_long_text/"
    ]
  },
  "/docs/use_cases/extraction/how_to/parse/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/how_to/parse/"
    ]
  },
  "/docs/use_cases/extraction/quickstart/": {
    "canonical": "/docs/tutorials/extraction/",
    "alternative": [
      "/v0.1/docs/use_cases/extraction/quickstart/"
    ]
  },
  "/docs/use_cases/graph/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/"
    ]
  },
  "/docs/use_cases/graph/constructing/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/constructing/"
    ]
  },
  "/docs/use_cases/graph/mapping/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/mapping/"
    ]
  },
  "/docs/use_cases/graph/prompting/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/prompting/"
    ]
  },
  "/docs/use_cases/graph/quickstart/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/quickstart/"
    ]
  },
  "/docs/use_cases/graph/semantic/": {
    "canonical": "/docs/tutorials/graph/",
    "alternative": [
      "/v0.1/docs/use_cases/graph/semantic/"
    ]
  },
  "/docs/use_cases/query_analysis/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/constructing-filters/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/constructing-filters/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/few_shot/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/few_shot/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/high_cardinality/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/high_cardinality/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/multiple_queries/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/multiple_queries/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/multiple_retrievers/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/multiple_retrievers/"
    ]
  },
  "/docs/use_cases/query_analysis/how_to/no_queries/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/how_to/no_queries/"
    ]
  },
  "/docs/use_cases/query_analysis/quickstart/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/quickstart/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/decomposition/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/decomposition/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/expansion/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/expansion/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/hyde/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/hyde/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/routing/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/routing/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/step_back/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/step_back/"
    ]
  },
  "/docs/use_cases/query_analysis/techniques/structuring/": {
    "canonical": "/docs/tutorials/query_analysis/",
    "alternative": [
      "/v0.1/docs/use_cases/query_analysis/techniques/structuring/"
    ]
  },
  "/docs/use_cases/question_answering/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/"
    ]
  },
  "/docs/use_cases/question_answering/chat_history/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/chat_history/"
    ]
  },
  "/docs/use_cases/question_answering/citations/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/citations/"
    ]
  },
  "/docs/use_cases/question_answering/conversational_retrieval_agents/": {
    "canonical": "/docs/tutorials/qa_chat_history/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/"
    ]
  },
  "/docs/use_cases/question_answering/hybrid/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/hybrid/"
    ]
  },
  "/docs/use_cases/question_answering/local_retrieval_qa/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/local_retrieval_qa/"
    ]
  },
  "/docs/use_cases/question_answering/per_user/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/per_user/"
    ]
  },
  "/docs/use_cases/question_answering/quickstart/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/quickstart/"
    ]
  },
  "/docs/use_cases/question_answering/sources/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/sources/"
    ]
  },
  "/docs/use_cases/question_answering/streaming/": {
    "canonical": "/docs/tutorials/rag/",
    "alternative": [
      "/v0.1/docs/use_cases/question_answering/streaming/"
    ]
  },
  "/docs/use_cases/sql/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/"
    ]
  },
  "/docs/use_cases/sql/agents/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/agents/"
    ]
  },
  "/docs/use_cases/sql/csv/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/csv/"
    ]
  },
  "/docs/use_cases/sql/large_db/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/large_db/"
    ]
  },
  "/docs/use_cases/sql/prompting/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/prompting/"
    ]
  },
  "/docs/use_cases/sql/query_checking/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/query_checking/"
    ]
  },
  "/docs/use_cases/sql/quickstart/": {
    "canonical": "/docs/tutorials/sql_qa/",
    "alternative": [
      "/v0.1/docs/use_cases/sql/quickstart/"
    ]
  },
  "/docs/use_cases/summarization/": {
    "canonical": "/docs/tutorials/summarization/",
    "alternative": [
      "/v0.1/docs/use_cases/summarization/"
    ]
  },
  "/docs/use_cases/tagging/": {
    "canonical": "/docs/tutorials/classification/",
    "alternative": [
      "/v0.1/docs/use_cases/tagging/"
    ]
  },
  "/docs/use_cases/tool_use/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/"
    ]
  },
  "/docs/use_cases/tool_use/agents/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/agents/"
    ]
  },
  "/docs/use_cases/tool_use/human_in_the_loop/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/human_in_the_loop/"
    ]
  },
  "/docs/use_cases/tool_use/multiple_tools/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/multiple_tools/"
    ]
  },
  "/docs/use_cases/tool_use/parallel/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/parallel/"
    ]
  },
  "/docs/use_cases/tool_use/prompting/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/prompting/"
    ]
  },
  "/docs/use_cases/tool_use/quickstart/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/quickstart/"
    ]
  },
  "/docs/use_cases/tool_use/tool_error_handling/": {
    "canonical": "/docs/tutorials/agents/",
    "alternative": [
      "/v0.1/docs/use_cases/tool_use/tool_error_handling/"
    ]
  },
  "/docs/use_cases/web_scraping/": {
    "canonical": "https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/",
    "alternative": [
      "/v0.1/docs/use_cases/web_scraping/"
    ]
  },
  // below are new
  "/docs/modules/data_connection/document_transformers/text_splitters/": {"canonical": "/docs/how_to/#text-splitters", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter/": {"canonical": "/docs/how_to/character_text_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/code_splitter/": {"canonical": "/docs/how_to/code_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/code_splitter/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/HTML_header_metadata/": {"canonical": "/docs/how_to/HTML_header_metadata_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/HTML_header_metadata/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/HTML_section_aware_splitter/": {"canonical": "/docs/how_to/HTML_section_aware_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/HTML_section_aware_splitter/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata/": {"canonical": "/docs/how_to/markdown_header_metadata_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/recursive_json_splitter/": {"canonical": "/docs/how_to/recursive_json_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/recursive_json_splitter/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter/": {"canonical": "/docs/how_to/recursive_text_splitter/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/semantic-chunker/": {"canonical": "/docs/how_to/semantic-chunker/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/"]},
  "/docs/modules/data_connection/document_transformers/text_splitters/split_by_token/": {"canonical": "/docs/how_to/split_by_token/", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/split_by_token/"]},
  "/docs/modules/model_io/prompts/prompt_templates/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/"]},
  "/docs/modules/model_io/prompts/prompt_templates/composition/": {"canonical": "/docs/how_to/prompts_composition/", "alternative": ["/v0.1/docs/modules/model_io/prompts/composition/"]},
  "/docs/modules/model_io/prompts/prompt_templates/example_selectors/": {"canonical": "/docs/how_to/example_selectors/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/"]},
  "/docs/modules/model_io/prompts/prompt_templates/example_selectors/length_based/": {"canonical": "/docs/how_to/example_selectors_length_based/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/length_based/"]},
  "/docs/modules/model_io/prompts/prompt_templates/example_selectors/mmr/": {"canonical": "/docs/how_to/example_selectors_mmr/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/mmr/"]},
  "/docs/modules/model_io/prompts/prompt_templates/example_selectors/ngram_overlap/": {"canonical": "/docs/how_to/example_selectors_ngram/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/ngram_overlap/"]},
  "/docs/modules/model_io/prompts/prompt_templates/example_selectors/similarity/": {"canonical": "/docs/how_to/example_selectors_similarity/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/similarity/"]},
  "/docs/modules/model_io/prompts/prompt_templates/few_shot_examples_chat/": {"canonical": "/docs/how_to/few_shot_examples_chat/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/"]},
  "/docs/modules/model_io/prompts/prompt_templates/few_shot_examples/": {"canonical": "/docs/how_to/few_shot_examples/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples/"]},
  "/docs/modules/model_io/prompts/prompt_templates/partial/": {"canonical": "/docs/how_to/prompts_partial/", "alternative": ["/v0.1/docs/modules/model_io/prompts/partial/"]},
  "/docs/modules/model_io/prompts/prompt_templates/quick_start/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/quick_start/"]},
  "/docs/modules/model_io/models/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/"]},
  "/docs/modules/model_io/models/chat/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/chat/"]},
  "/docs/modules/model_io/models/chat/chat_model_caching/": {"canonical": "/docs/how_to/chat_model_caching/", "alternative": ["/v0.1/docs/modules/model_io/chat/chat_model_caching/"]},
  "/docs/modules/model_io/models/chat/custom_chat_model/": {"canonical": "/docs/how_to/custom_chat_model/", "alternative": ["/v0.1/docs/modules/model_io/chat/custom_chat_model/"]},
  "/docs/modules/model_io/models/chat/function_calling/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/model_io/chat/function_calling/"]},
  "/docs/modules/model_io/models/chat/logprobs/": {"canonical": "/docs/how_to/logprobs/", "alternative": ["/v0.1/docs/modules/model_io/chat/logprobs/"]},
  "/docs/modules/model_io/models/chat/message_types/": {"canonical": "/docs/concepts/#messages", "alternative": ["/v0.1/docs/modules/model_io/chat/message_types/"]},
  "/docs/modules/model_io/models/chat/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/chat/quick_start/"]},
  "/docs/modules/model_io/models/chat/response_metadata/": {"canonical": "/docs/how_to/response_metadata/", "alternative": ["/v0.1/docs/modules/model_io/chat/response_metadata/"]},
  "/docs/modules/model_io/models/chat/streaming/": {"canonical": "/docs/how_to/streaming/", "alternative": ["/v0.1/docs/modules/model_io/chat/streaming/"]},
  "/docs/modules/model_io/models/chat/structured_output/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/chat/structured_output/"]},
  "/docs/modules/model_io/models/chat/token_usage_tracking/": {"canonical": "/docs/how_to/chat_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/chat/token_usage_tracking/"]},
  "/docs/modules/model_io/models/concepts/": {"canonical": "/docs/concepts/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/concepts/"]},
  "/docs/modules/model_io/models/llms/": {"canonical": "/docs/concepts/#llms", "alternative": ["/v0.1/docs/modules/model_io/llms/"]},
  "/docs/modules/model_io/models/llms/custom_llm/": {"canonical": "/docs/how_to/custom_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/custom_llm/"]},
  "/docs/modules/model_io/models/llms/llm_caching/": {"canonical": "/docs/how_to/llm_caching/", "alternative": ["/v0.1/docs/modules/model_io/llms/llm_caching/"]},
  "/docs/modules/model_io/models/llms/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/llms/quick_start/"]},
  "/docs/modules/model_io/models/llms/streaming_llm/": {"canonical": "/docs/how_to/streaming_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/streaming_llm/"]},
  "/docs/modules/model_io/models/llms/token_usage_tracking/": {"canonical": "/docs/how_to/llm_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/llms/token_usage_tracking/"]},
  "/docs/modules/model_io/models/output_parsers/": {"canonical": "/docs/how_to/#output-parsers", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/"]},
  "/docs/modules/model_io/models/output_parsers/custom/": {"canonical": "/docs/how_to/output_parser_custom/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/custom/"]},
  "/docs/modules/model_io/models/output_parsers/quick_start/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/quick_start/"]},
  "/docs/modules/model_io/models/output_parsers/types/csv/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/csv/"]},
  "/docs/modules/model_io/models/output_parsers/types/datetime/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/datetime/"]},
  "/docs/modules/model_io/models/output_parsers/types/enum/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/enum/"]},
  "/docs/modules/model_io/models/output_parsers/types/json/": {"canonical": "/docs/how_to/output_parser_json/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/json/"]},
  "/docs/modules/model_io/models/output_parsers/types/openai_functions/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/openai_functions/"]},
  "/docs/modules/model_io/models/output_parsers/types/openai_tools/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/openai_tools/"]},
  "/docs/modules/model_io/models/output_parsers/types/output_fixing/": {"canonical": "/docs/how_to/output_parser_fixing/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/output_fixing/"]},
  "/docs/modules/model_io/models/output_parsers/types/pandas_dataframe/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/pandas_dataframe/"]},
  "/docs/modules/model_io/models/output_parsers/types/pydantic/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/pydantic/"]},
  "/docs/modules/model_io/models/output_parsers/types/retry/": {"canonical": "/docs/how_to/output_parser_retry/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/retry/"]},
  "/docs/modules/model_io/models/output_parsers/types/structured/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/structured/"]},
  "/docs/modules/model_io/models/output_parsers/types/xml/": {"canonical": "/docs/how_to/output_parser_xml/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/xml/"]},
  "/docs/modules/model_io/models/output_parsers/types/yaml/": {"canonical": "/docs/how_to/output_parser_yaml/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/yaml/"]},
  "/docs/modules/model_io/models/prompts/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/"]},
  "/docs/modules/model_io/models/prompts/composition/": {"canonical": "/docs/how_to/prompts_composition/", "alternative": ["/v0.1/docs/modules/model_io/prompts/composition/"]},
  "/docs/modules/model_io/models/prompts/example_selectors/": {"canonical": "/docs/how_to/example_selectors/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/"]},
  "/docs/modules/model_io/models/prompts/example_selectors/length_based/": {"canonical": "/docs/how_to/example_selectors_length_based/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/length_based/"]},
  "/docs/modules/model_io/models/prompts/example_selectors/mmr/": {"canonical": "/docs/how_to/example_selectors_mmr/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/mmr/"]},
  "/docs/modules/model_io/models/prompts/example_selectors/ngram_overlap/": {"canonical": "/docs/how_to/example_selectors_ngram/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/ngram_overlap/"]},
  "/docs/modules/model_io/models/prompts/example_selectors/similarity/": {"canonical": "/docs/how_to/example_selectors_similarity/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/similarity/"]},
  "/docs/modules/model_io/models/prompts/few_shot_examples_chat/": {"canonical": "/docs/how_to/few_shot_examples_chat/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/"]},
  "/docs/modules/model_io/models/prompts/few_shot_examples/": {"canonical": "/docs/how_to/few_shot_examples/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples/"]},
  "/docs/modules/model_io/models/prompts/partial/": {"canonical": "/docs/how_to/prompts_partial/", "alternative": ["/v0.1/docs/modules/model_io/prompts/partial/"]},
  "/docs/modules/model_io/models/prompts/quick_start/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/quick_start/"]},
  "/docs/modules/model_io/models/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/quick_start/"]},
  "/docs/use_cases/more/graph/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/"]},
  "/docs/use_cases/more/graph/constructing/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/constructing/"]},
  "/docs/use_cases/more/graph/mapping/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/mapping/"]},
  "/docs/use_cases/more/graph/prompting/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/prompting/"]},
  "/docs/use_cases/more/graph/quickstart/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/quickstart/"]},
  "/docs/use_cases/more/graph/semantic/": {"canonical": "/docs/tutorials/graph/", "alternative": ["/v0.1/docs/use_cases/graph/semantic/"]},
  "/docs/modules/model_io/chat/how_to/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/chat/"]},
  "/docs/modules/model_io/chat/how_to/chat_model_caching/": {"canonical": "/docs/how_to/chat_model_caching/", "alternative": ["/v0.1/docs/modules/model_io/chat/chat_model_caching/"]},
  "/docs/modules/model_io/chat/how_to/custom_chat_model/": {"canonical": "/docs/how_to/custom_chat_model/", "alternative": ["/v0.1/docs/modules/model_io/chat/custom_chat_model/"]},
  "/docs/modules/model_io/chat/how_to/function_calling/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/model_io/chat/function_calling/"]},
  "/docs/modules/model_io/chat/how_to/logprobs/": {"canonical": "/docs/how_to/logprobs/", "alternative": ["/v0.1/docs/modules/model_io/chat/logprobs/"]},
  "/docs/modules/model_io/chat/how_to/message_types/": {"canonical": "/docs/concepts/#messages", "alternative": ["/v0.1/docs/modules/model_io/chat/message_types/"]},
  "/docs/modules/model_io/chat/how_to/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/chat/quick_start/"]},
  "/docs/modules/model_io/chat/how_to/response_metadata/": {"canonical": "/docs/how_to/response_metadata/", "alternative": ["/v0.1/docs/modules/model_io/chat/response_metadata/"]},
  "/docs/modules/model_io/chat/how_to/streaming/": {"canonical": "/docs/how_to/streaming/", "alternative": ["/v0.1/docs/modules/model_io/chat/streaming/"]},
  "/docs/modules/model_io/chat/how_to/structured_output/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/chat/structured_output/"]},
  "/docs/modules/model_io/chat/how_to/token_usage_tracking/": {"canonical": "/docs/how_to/chat_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/chat/token_usage_tracking/"]},
  "/docs/modules/model_io/llms/how_to/": {"canonical": "/docs/concepts/#llms", "alternative": ["/v0.1/docs/modules/model_io/llms/"]},
  "/docs/modules/model_io/llms/how_to/custom_llm/": {"canonical": "/docs/how_to/custom_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/custom_llm/"]},
  "/docs/modules/model_io/llms/how_to/llm_caching/": {"canonical": "/docs/how_to/llm_caching/", "alternative": ["/v0.1/docs/modules/model_io/llms/llm_caching/"]},
  "/docs/modules/model_io/llms/how_to/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/llms/quick_start/"]},
  "/docs/modules/model_io/llms/how_to/streaming_llm/": {"canonical": "/docs/how_to/streaming_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/streaming_llm/"]},
  "/docs/modules/model_io/llms/how_to/token_usage_tracking/": {"canonical": "/docs/how_to/llm_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/llms/token_usage_tracking/"]},
  "/docs/modules/model_io/llms/integrations/llm_caching/": {"canonical": "/docs/how_to/llm_caching/", "alternative": ["/v0.1/docs/integrations/llms/llm_caching/"]},
  "/docs/modules/model_io/chat/integrations/ollama_functions/": {"canonical": "/docs/integrations/chat/ollama/", "alternative": ["/v0.1/docs/integrations/chat/ollama_functions/"]},
  "/en/latest/modules/models/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/"]},
  "/en/latest/modules/models/chat/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/chat/"]},
  "/en/latest/modules/models/chat/chat_model_caching/": {"canonical": "/docs/how_to/chat_model_caching/", "alternative": ["/v0.1/docs/modules/model_io/chat/chat_model_caching/"]},
  "/en/latest/modules/models/chat/custom_chat_model/": {"canonical": "/docs/how_to/custom_chat_model/", "alternative": ["/v0.1/docs/modules/model_io/chat/custom_chat_model/"]},
  "/en/latest/modules/models/chat/function_calling/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/model_io/chat/function_calling/"]},
  "/en/latest/modules/models/chat/logprobs/": {"canonical": "/docs/how_to/logprobs/", "alternative": ["/v0.1/docs/modules/model_io/chat/logprobs/"]},
  "/en/latest/modules/models/chat/message_types/": {"canonical": "/docs/concepts/#messages", "alternative": ["/v0.1/docs/modules/model_io/chat/message_types/"]},
  "/en/latest/modules/models/chat/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/chat/quick_start/"]},
  "/en/latest/modules/models/chat/response_metadata/": {"canonical": "/docs/how_to/response_metadata/", "alternative": ["/v0.1/docs/modules/model_io/chat/response_metadata/"]},
  "/en/latest/modules/models/chat/streaming/": {"canonical": "/docs/how_to/streaming/", "alternative": ["/v0.1/docs/modules/model_io/chat/streaming/"]},
  "/en/latest/modules/models/chat/structured_output/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/chat/structured_output/"]},
  "/en/latest/modules/models/chat/token_usage_tracking/": {"canonical": "/docs/how_to/chat_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/chat/token_usage_tracking/"]},
  "/en/latest/modules/models/concepts/": {"canonical": "/docs/concepts/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/concepts/"]},
  "/en/latest/modules/models/llms/": {"canonical": "/docs/concepts/#llms", "alternative": ["/v0.1/docs/modules/model_io/llms/"]},
  "/en/latest/modules/models/llms/custom_llm/": {"canonical": "/docs/how_to/custom_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/custom_llm/"]},
  "/en/latest/modules/models/llms/llm_caching/": {"canonical": "/docs/how_to/llm_caching/", "alternative": ["/v0.1/docs/modules/model_io/llms/llm_caching/"]},
  "/en/latest/modules/models/llms/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/llms/quick_start/"]},
  "/en/latest/modules/models/llms/streaming_llm/": {"canonical": "/docs/how_to/streaming_llm/", "alternative": ["/v0.1/docs/modules/model_io/llms/streaming_llm/"]},
  "/en/latest/modules/models/llms/token_usage_tracking/": {"canonical": "/docs/how_to/llm_token_usage_tracking/", "alternative": ["/v0.1/docs/modules/model_io/llms/token_usage_tracking/"]},
  "/en/latest/modules/models/output_parsers/": {"canonical": "/docs/how_to/#output-parsers", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/"]},
  "/en/latest/modules/models/output_parsers/custom/": {"canonical": "/docs/how_to/output_parser_custom/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/custom/"]},
  "/en/latest/modules/models/output_parsers/quick_start/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/quick_start/"]},
  "/en/latest/modules/models/output_parsers/types/csv/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/csv/"]},
  "/en/latest/modules/models/output_parsers/types/datetime/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/datetime/"]},
  "/en/latest/modules/models/output_parsers/types/enum/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/enum/"]},
  "/en/latest/modules/models/output_parsers/types/json/": {"canonical": "/docs/how_to/output_parser_json/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/json/"]},
  "/en/latest/modules/models/output_parsers/types/openai_functions/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/openai_functions/"]},
  "/en/latest/modules/models/output_parsers/types/openai_tools/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/openai_tools/"]},
  "/en/latest/modules/models/output_parsers/types/output_fixing/": {"canonical": "/docs/how_to/output_parser_fixing/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/output_fixing/"]},
  "/en/latest/modules/models/output_parsers/types/pandas_dataframe/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/pandas_dataframe/"]},
  "/en/latest/modules/models/output_parsers/types/pydantic/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/pydantic/"]},
  "/en/latest/modules/models/output_parsers/types/retry/": {"canonical": "/docs/how_to/output_parser_retry/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/retry/"]},
  "/en/latest/modules/models/output_parsers/types/structured/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/structured/"]},
  "/en/latest/modules/models/output_parsers/types/xml/": {"canonical": "/docs/how_to/output_parser_xml/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/xml/"]},
  "/en/latest/modules/models/output_parsers/types/yaml/": {"canonical": "/docs/how_to/output_parser_yaml/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/yaml/"]},
  "/en/latest/modules/models/prompts/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/"]},
  "/en/latest/modules/models/prompts/composition/": {"canonical": "/docs/how_to/prompts_composition/", "alternative": ["/v0.1/docs/modules/model_io/prompts/composition/"]},
  "/en/latest/modules/models/prompts/example_selectors/": {"canonical": "/docs/how_to/example_selectors/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/"]},
  "/en/latest/modules/models/prompts/example_selectors/length_based/": {"canonical": "/docs/how_to/example_selectors_length_based/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/length_based/"]},
  "/en/latest/modules/models/prompts/example_selectors/mmr/": {"canonical": "/docs/how_to/example_selectors_mmr/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/mmr/"]},
  "/en/latest/modules/models/prompts/example_selectors/ngram_overlap/": {"canonical": "/docs/how_to/example_selectors_ngram/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/ngram_overlap/"]},
  "/en/latest/modules/models/prompts/example_selectors/similarity/": {"canonical": "/docs/how_to/example_selectors_similarity/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/similarity/"]},
  "/en/latest/modules/models/prompts/few_shot_examples_chat/": {"canonical": "/docs/how_to/few_shot_examples_chat/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/"]},
  "/en/latest/modules/models/prompts/few_shot_examples/": {"canonical": "/docs/how_to/few_shot_examples/", "alternative": ["/v0.1/docs/modules/model_io/prompts/few_shot_examples/"]},
  "/en/latest/modules/models/prompts/partial/": {"canonical": "/docs/how_to/prompts_partial/", "alternative": ["/v0.1/docs/modules/model_io/prompts/partial/"]},
  "/en/latest/modules/models/prompts/quick_start/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/quick_start/"]},
  "/en/latest/modules/models/quick_start/": {"canonical": "/docs/tutorials/llm_chain/", "alternative": ["/v0.1/docs/modules/model_io/quick_start/"]},
  "/docs/modules/model_io/prompts/example_selector_types/": {"canonical": "/docs/how_to/example_selectors/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/"]},
  "/docs/modules/model_io/prompts/example_selector_types/length_based/": {"canonical": "/docs/how_to/example_selectors_length_based/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/length_based/"]},
  "/docs/modules/model_io/prompts/example_selector_types/mmr/": {"canonical": "/docs/how_to/example_selectors_mmr/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/mmr/"]},
  "/docs/modules/model_io/prompts/example_selector_types/ngram_overlap/": {"canonical": "/docs/how_to/example_selectors_ngram/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/ngram_overlap/"]},
  "/docs/modules/model_io/prompts/example_selector_types/similarity/": {"canonical": "/docs/how_to/example_selectors_similarity/", "alternative": ["/v0.1/docs/modules/model_io/prompts/example_selectors/similarity/"]},
  "/docs/modules/agents/tools/": {"canonical": "/docs/how_to/#tools", "alternative": ["/v0.1/docs/modules/tools/"]},
  "/docs/modules/agents/tools/custom_tools/": {"canonical": "/docs/how_to/custom_tools/", "alternative": ["/v0.1/docs/modules/tools/custom_tools/"]},
  "/docs/modules/agents/tools/toolkits/": {"canonical": "/docs/how_to/#tools", "alternative": ["/v0.1/docs/modules/tools/toolkits/"]},
  "/docs/modules/agents/tools/tools_as_openai_functions/": {"canonical": "/docs/how_to/tool_calling/", "alternative": ["/v0.1/docs/modules/tools/tools_as_openai_functions/"]},
  "/docs/guides/deployments/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/deployments/"]},
  "/docs/guides/deployments/template_repos/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/deployments/template_repos/"]},
  "/docs/guides/evaluation/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/"]},
  "/docs/guides/evaluation/comparison/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/comparison/"]},
  "/docs/guides/evaluation/comparison/custom/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/comparison/custom/"]},
  "/docs/guides/evaluation/comparison/pairwise_embedding_distance/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/comparison/pairwise_embedding_distance/"]},
  "/docs/guides/evaluation/comparison/pairwise_string/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/comparison/pairwise_string/"]},
  "/docs/guides/evaluation/examples/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/examples/"]},
  "/docs/guides/evaluation/examples/comparisons/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/examples/comparisons/"]},
  "/docs/guides/evaluation/string/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/"]},
  "/docs/guides/evaluation/string/criteria_eval_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/criteria_eval_chain/"]},
  "/docs/guides/evaluation/string/custom/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/custom/"]},
  "/docs/guides/evaluation/string/embedding_distance/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/embedding_distance/"]},
  "/docs/guides/evaluation/string/exact_match/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/exact_match/"]},
  "/docs/guides/evaluation/string/json/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/json/"]},
  "/docs/guides/evaluation/string/regex_match/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/regex_match/"]},
  "/docs/guides/evaluation/string/scoring_eval_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/scoring_eval_chain/"]},
  "/docs/guides/evaluation/string/string_distance/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/string/string_distance/"]},
  "/docs/guides/evaluation/trajectory/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/trajectory/"]},
  "/docs/guides/evaluation/trajectory/custom/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/trajectory/custom/"]},
  "/docs/guides/evaluation/trajectory/trajectory_eval/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/trajectory/trajectory_eval/"]},
  "/docs/guides/privacy/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/"]},
  "/docs/guides/privacy/amazon_comprehend_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/amazon_comprehend_chain/"]},
  "/docs/guides/privacy/constitutional_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/constitutional_chain/"]},
  "/docs/guides/privacy/hugging_face_prompt_injection/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/hugging_face_prompt_injection/"]},
  "/docs/guides/privacy/layerup_security/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/layerup_security/"]},
  "/docs/guides/privacy/logical_fallacy_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/logical_fallacy_chain/"]},
  "/docs/guides/privacy/moderation/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/moderation/"]},
  "/docs/guides/privacy/presidio_data_anonymization/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/"]},
  "/docs/guides/privacy/presidio_data_anonymization/multi_language/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/multi_language/"]},
  "/docs/guides/privacy/presidio_data_anonymization/qa_privacy_protection/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/qa_privacy_protection/"]},
  "/docs/guides/privacy/presidio_data_anonymization/reversible/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/reversible/"]},
  "/docs/guides/safety/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/"]},
  "/docs/guides/safety/amazon_comprehend_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/amazon_comprehend_chain/"]},
  "/docs/guides/safety/constitutional_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/constitutional_chain/"]},
  "/docs/guides/safety/hugging_face_prompt_injection/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/hugging_face_prompt_injection/"]},
  "/docs/guides/safety/layerup_security/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/layerup_security/"]},
  "/docs/guides/safety/logical_fallacy_chain/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/logical_fallacy_chain/"]},
  "/docs/guides/safety/moderation/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/moderation/"]},
  "/docs/guides/safety/presidio_data_anonymization/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/"]},
  "/docs/guides/safety/presidio_data_anonymization/multi_language/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/multi_language/"]},
  "/docs/guides/safety/presidio_data_anonymization/qa_privacy_protection/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/qa_privacy_protection/"]},
  "/docs/guides/safety/presidio_data_anonymization/reversible/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/reversible/"]},
  "/docs/integrations/llms/titan_takeoff_pro/": {"canonical": "/docs/integrations/llms/titan_takeoff/"},
  "/docs/integrations/providers/optimum_intel/": {"canonical": "/docs/integrations/providers/intel/"},
  "/docs/use_cases/graph/integrations/diffbot_graphtransformer/": {"canonical": "/docs/integrations/graphs/diffbot/"},
  "/docs/use_cases/graph/integrations/graph_arangodb_qa/": {"canonical": "/docs/integrations/graphs/arangodb/"},
  "/docs/use_cases/graph/integrations/graph_cypher_qa/": {"canonical": "/docs/integrations/graphs/neo4j_cypher/"},
  "/docs/use_cases/graph/integrations/graph_falkordb_qa/": {"canonical": "/docs/integrations/graphs/falkordb/"},
  "/docs/use_cases/graph/integrations/graph_gremlin_cosmosdb_qa/": {"canonical": "/docs/integrations/graphs/azure_cosmosdb_gremlin/"},
  "/docs/use_cases/graph/integrations/graph_hugegraph_qa/": {"canonical": "/docs/integrations/graphs/hugegraph/"},
  "/docs/use_cases/graph/integrations/graph_kuzu_qa/": {"canonical": "/docs/integrations/graphs/kuzu_db/"},
  "/docs/use_cases/graph/integrations/graph_memgraph_qa/": {"canonical": "/docs/integrations/graphs/memgraph/"},
  "/docs/use_cases/graph/integrations/graph_nebula_qa/": {"canonical": "/docs/integrations/graphs/nebula_graph/"},
  "/docs/use_cases/graph/integrations/graph_networkx_qa/": {"canonical": "/docs/integrations/graphs/networkx/"},
  "/docs/use_cases/graph/integrations/graph_ontotext_graphdb_qa/": {"canonical": "/docs/integrations/graphs/ontotext/"},
  "/docs/use_cases/graph/integrations/graph_sparql_qa/": {"canonical": "/docs/integrations/graphs/rdflib_sparql/"},
  "/docs/use_cases/graph/integrations/neptune_cypher_qa/": {"canonical": "/docs/integrations/graphs/amazon_neptune_open_cypher/"},
  "/docs/use_cases/graph/integrations/neptune_sparql_qa/": {"canonical": "/docs/integrations/graphs/amazon_neptune_sparql/"},
  "/docs/integrations/providers/facebook_chat/": {"canonical": "/docs/integrations/providers/facebook/"},
  "/docs/integrations/providers/facebook_faiss/": {"canonical": "/docs/integrations/providers/facebook/"},
  "/docs/integrations/memory/google_cloud_sql_mssql/": {"canonical": "/docs/integrations/memory/google_sql_mssql/"},
  "/docs/integrations/memory/google_cloud_sql_mysql/": {"canonical": "/docs/integrations/memory/google_sql_mysql/"},
  "/docs/integrations/memory/google_cloud_sql_pg/": {"canonical": "/docs/integrations/memory/google_sql_pg/"},
  "/docs/integrations/memory/google_datastore/": {"canonical": "/docs/integrations/memory/google_firestore_datastore/"},
  "/docs/integrations/llms/huggingface_textgen_inference/": {"canonical": "/docs/integrations/llms/huggingface_endpoint/"},
  "/docs/integrations/llms/huggingface_hub/": {"canonical": "/docs/integrations/llms/huggingface_endpoint/"},
  "/docs/integrations/llms/bigdl/": {"canonical": "/docs/integrations/llms/ipex_llm/"},
  "/docs/integrations/llms/watsonxllm/": {"canonical": "/docs/integrations/llms/ibm_watsonx/"},
  "/docs/integrations/llms/pai_eas_endpoint/": {"canonical": "/docs/integrations/llms/alibabacloud_pai_eas_endpoint/"},
  "/docs/integrations/vectorstores/hanavector/": {"canonical": "/docs/integrations/vectorstores/sap_hanavector/"},
  "/docs/use_cases/qa_structured/sql/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/"]},
  "/docs/contributing/packages/": {"canonical": "/docs/versions/release_policy/", "alternative": ["/v0.1/docs/packages/"]},
  "/docs/community/": {"canonical": "/docs/contributing/"},
  "/docs/modules/chains/(.+)/": {"canonical": "/docs/versions/migrating_chains/", "alternative": ["/v0.1/docs/modules/chains/"]},
  "/docs/modules/agents/how_to/custom_llm_agent/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/how_to/custom_agent/"]},
  "/docs/modules/agents/how_to/custom-functions-with-openai-functions-agent/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/how_to/custom_agent/"]},
  "/docs/modules/agents/how_to/custom_llm_chat_agent/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/how_to/custom_agent/"]},
  "/docs/modules/agents/how_to/custom_mrkl_agent/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/how_to/custom_agent/"]},
  "/docs/modules/agents/how_to/streaming_stdout_final_only/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/how_to/streaming/"]},
  "/docs/modules/model_io/prompts/prompts_pipelining/": {"canonical": "/docs/how_to/prompts_composition/", "alternative": ["/v0.1/docs/modules/model_io/prompts/composition/"]},
  "/docs/modules/model_io/output_parsers/enum/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/enum/"]},
  "/docs/modules/model_io/output_parsers/pandas_dataframe/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/pandas_dataframe/"]},
  "/docs/modules/model_io/output_parsers/structured/": {"canonical": "/docs/how_to/output_parser_structured/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/structured/"]},
  "/docs/modules/model_io/output_parsers/xml/": {"canonical": "/docs/how_to/output_parser_xml/", "alternative": ["/v0.1/docs/modules/model_io/output_parsers/types/xml/"]},
  "/docs/use_cases/question_answering/code_understanding/": {"canonical": "https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/", "alternative": ["/v0.1/docs/use_cases/code_understanding/"]},
  "/docs/use_cases/question_answering/document-context-aware-QA/": {"canonical": "/docs/how_to/#text-splitters", "alternative": ["/v0.1/docs/modules/data_connection/document_transformers/"]},
  "/docs/integrations/providers/alibabacloud_opensearch/": {"canonical": "/docs/integrations/providers/alibaba_cloud/"},
  "/docs/integrations/chat/pai_eas_chat_endpoint/": {"canonical": "/docs/integrations/chat/alibaba_cloud_pai_eas/"},
  "/docs/integrations/providers/tencentvectordb/": {"canonical": "/docs/integrations/providers/tencent/"},
  "/docs/integrations/chat/hunyuan/": {"canonical": "/docs/integrations/chat/tencent_hunyuan/"},
  "/docs/integrations/document_loaders/excel/": {"canonical": "/docs/integrations/document_loaders/microsoft_excel/"},
  "/docs/integrations/document_loaders/onenote/": {"canonical": "/docs/integrations/document_loaders/microsoft_onenote/"},
  "/docs/integrations/providers/aws_dynamodb/": {"canonical": "/docs/integrations/providers/aws/"},
  "/docs/integrations/providers/scann/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/toolkits/google_drive/": {"canonical": "/docs/integrations/tools/google_drive/"},
  "/docs/use_cases/question_answering/chat_vector_db/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/in_memory_question_answering/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/multi_retrieval_qa_router/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/multiple_retrieval/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/vector_db_qa/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/vector_db_text_generation/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/guides/langsmith/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/langsmith/"]},
  "/docs/guides/langsmith/walkthrough/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/langsmith/walkthrough/"]},
  "/docs/use_cases/qa_structured/integrations/sqlite/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/"]},
  "/docs/use_cases/more/data_generation/": {"canonical": "/docs/tutorials/data_generation/", "alternative": ["/v0.1/docs/use_cases/data_generation/"]},
  "/docs/use_cases/question_answering/how_to/chat_vector_db/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/how_to/conversational_retrieval_agents/": {"canonical": "/docs/tutorials/qa_chat_history/", "alternative": ["/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/"]},
  "/docs/use_cases/question_answering/question_answering/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/use_cases/question_answering/how_to/local_retrieval_qa/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/local_retrieval_qa/"]},
  "/docs/use_cases/question_answering/how_to/question_answering/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/modules/agents/agents/examples/mrkl_chat(.html?)/": {"canonical": "/docs/how_to/#agents", "alternative": ["/v0.1/docs/modules/agents/"]},
  "/docs/integrations/": {"canonical": "/docs/integrations/providers/"},
  "/docs/expression_language/cookbook/routing/": {"canonical": "/docs/how_to/routing/", "alternative": ["/v0.1/docs/expression_language/how_to/routing/"]},
  "/docs/guides/expression_language/": {"canonical": "/docs/how_to/#langchain-expression-language-lcel", "alternative": ["/v0.1/docs/expression_language/"]},
  "/docs/integrations/providers/amazon_api_gateway/": {"canonical": "/docs/integrations/providers/aws/"},
  "/docs/integrations/providers/huggingface/": {"canonical": "/docs/integrations/providers/huggingface/"},
  "/docs/integrations/providers/azure_blob_storage/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/google_vertexai_matchingengine/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/providers/aws_s3/": {"canonical": "/docs/integrations/providers/aws/"},
  "/docs/integrations/providers/azure_openai/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/azure_cognitive_search_/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/bedrock/": {"canonical": "/docs/integrations/providers/aws/"},
  "/docs/integrations/providers/google_bigquery/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/providers/google_cloud_storage/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/providers/google_drive/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/providers/google_search/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/providers/microsoft_onedrive/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/microsoft_powerpoint/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/microsoft_word/": {"canonical": "/docs/integrations/providers/microsoft/"},
  "/docs/integrations/providers/sagemaker_endpoint/": {"canonical": "/docs/integrations/providers/aws/"},
  "/docs/integrations/providers/sagemaker_tracking/": {"canonical": "/docs/integrations/callbacks/sagemaker_tracking/"},
  "/docs/integrations/providers/openai/": {"canonical": "/docs/integrations/providers/openai/"},
  "/docs/integrations/cassandra/": {"canonical": "/docs/integrations/providers/cassandra/"},
  "/docs/integrations/providers/providers/semadb/": {"canonical": "/docs/integrations/providers/semadb/"},
  "/docs/integrations/vectorstores/vectorstores/semadb/": {"canonical": "/docs/integrations/vectorstores/semadb/"},
  "/docs/integrations/vectorstores/async_faiss/": {"canonical": "/docs/integrations/vectorstores/faiss_async/"},
  "/docs/integrations/vectorstores/matchingengine/": {"canonical": "/docs/integrations/vectorstores/google_vertex_ai_vector_search/"},
  "/docs/integrations/tools/sqlite/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/"]},
  "/docs/integrations/document_loaders/pdf-amazonTextractPDFLoader/": {"canonical": "/docs/integrations/document_loaders/amazon_textract/"},
  "/docs/integrations/document_loaders/Etherscan/": {"canonical": "/docs/integrations/document_loaders/etherscan/"},
  "/docs/integrations/document_loaders/merge_doc_loader/": {"canonical": "/docs/integrations/document_loaders/merge_doc/"},
  "/docs/integrations/document_loaders/recursive_url_loader/": {"canonical": "/docs/integrations/document_loaders/recursive_url/"},
  "/docs/integrations/providers/google_document_ai/": {"canonical": "/docs/integrations/providers/google/"},
  "/docs/integrations/memory/motorhead_memory_managed/": {"canonical": "/docs/integrations/memory/motorhead_memory/"},
  "/docs/integrations/memory/dynamodb_chat_message_history/": {"canonical": "/docs/integrations/memory/aws_dynamodb/"},
  "/docs/integrations/memory/entity_memory_with_sqlite/": {"canonical": "/docs/integrations/memory/sqlite/"},
  "/docs/modules/model_io/chat/integrations/anthropic/": {"canonical": "/docs/integrations/chat/anthropic/"},
  "/docs/modules/model_io/chat/integrations/azure_chat_openai/": {"canonical": "/docs/integrations/chat/azure_chat_openai/"},
  "/docs/modules/model_io/chat/integrations/google_vertex_ai_palm/": {"canonical": "/docs/integrations/chat/google_vertex_ai_palm/"},
  "/docs/modules/model_io/chat/integrations/openai/": {"canonical": "/docs/integrations/chat/openai/"},
  "/docs/modules/model_io/chat/integrations/promptlayer_chatopenai/": {"canonical": "/docs/integrations/chat/promptlayer_chatopenai/"},
  "/docs/modules/model_io/llms/integrations/ai21/": {"canonical": "/docs/integrations/llms/ai21/"},
  "/docs/modules/model_io/llms/integrations/aleph_alpha/": {"canonical": "/docs/integrations/llms/aleph_alpha/"},
  "/docs/modules/model_io/llms/integrations/anyscale/": {"canonical": "/docs/integrations/llms/anyscale/"},
  "/docs/modules/model_io/llms/integrations/banana/": {"canonical": "/docs/integrations/llms/banana/"},
  "/docs/modules/model_io/llms/integrations/baseten/": {"canonical": "/docs/integrations/llms/baseten/"},
  "/docs/modules/model_io/llms/integrations/beam/": {"canonical": "/docs/integrations/llms/beam/"},
  "/docs/modules/model_io/llms/integrations/bedrock/": {"canonical": "/docs/integrations/llms/bedrock/"},
  "/docs/modules/model_io/llms/integrations/cohere/": {"canonical": "/docs/integrations/llms/cohere/"},
  "/docs/modules/model_io/llms/integrations/ctransformers/": {"canonical": "/docs/integrations/llms/ctransformers/"},
  "/docs/modules/model_io/llms/integrations/databricks/": {"canonical": "/docs/integrations/llms/databricks/"},
  "/docs/modules/model_io/llms/integrations/google_vertex_ai_palm/": {"canonical": "/docs/integrations/llms/google_vertex_ai_palm/"},
  "/docs/modules/model_io/llms/integrations/huggingface_pipelines/": {"canonical": "/docs/integrations/llms/huggingface_pipelines/"},
  "/docs/modules/model_io/llms/integrations/jsonformer_experimental/": {"canonical": "/docs/integrations/llms/jsonformer_experimental/"},
  "/docs/modules/model_io/llms/integrations/llamacpp/": {"canonical": "/docs/integrations/llms/llamacpp/"},
  "/docs/modules/model_io/llms/integrations/manifest/": {"canonical": "/docs/integrations/llms/manifest/"},
  "/docs/modules/model_io/llms/integrations/modal/": {"canonical": "/docs/integrations/llms/modal/"},
  "/docs/modules/model_io/llms/integrations/mosaicml/": {"canonical": "/docs/integrations/llms/mosaicml/"},
  "/docs/modules/model_io/llms/integrations/nlpcloud/": {"canonical": "/docs/integrations/llms/nlpcloud/"},
  "/docs/modules/model_io/llms/integrations/openai/": {"canonical": "/docs/integrations/llms/openai/"},
  "/docs/modules/model_io/llms/integrations/openlm/": {"canonical": "/docs/integrations/llms/openlm/"},
  "/docs/modules/model_io/llms/integrations/predictionguard/": {"canonical": "/docs/integrations/llms/predictionguard/"},
  "/docs/modules/model_io/llms/integrations/promptlayer_openai/": {"canonical": "/docs/integrations/llms/promptlayer_openai/"},
  "/docs/modules/model_io/llms/integrations/rellm_experimental/": {"canonical": "/docs/integrations/llms/rellm_experimental/"},
  "/docs/modules/model_io/llms/integrations/replicate/": {"canonical": "/docs/integrations/llms/replicate/"},
  "/docs/modules/model_io/llms/integrations/runhouse/": {"canonical": "/docs/integrations/llms/runhouse/"},
  "/docs/modules/model_io/llms/integrations/sagemaker/": {"canonical": "/docs/integrations/llms/sagemaker/"},
  "/docs/modules/model_io/llms/integrations/stochasticai/": {"canonical": "/docs/integrations/llms/stochasticai/"},
  "/docs/modules/model_io/llms/integrations/writer/": {"canonical": "/docs/integrations/llms/writer/"},
  "/en/latest/use_cases/apis.html/": {"canonical": null, "alternative": ["/v0.1/docs/use_cases/apis/"]},
  "/en/latest/use_cases/extraction.html/": {"canonical": "/docs/tutorials/extraction/", "alternative": ["/v0.1/docs/use_cases/extraction/"]},
  "/en/latest/use_cases/summarization.html/": {"canonical": "/docs/tutorials/summarization/", "alternative": ["/v0.1/docs/use_cases/summarization/"]},
  "/en/latest/use_cases/tabular.html/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/"]},
  "/en/latest/youtube.html/": {"canonical": "/docs/additional_resources/youtube/"},
  "/docs/": {"canonical": "/"},
  "/en/latest/": {"canonical": "/"},
  "/en/latest/index.html/": {"canonical": "/"},
  "/en/latest/modules/models.html/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/"]},
  "/docs/integrations/retrievers/google_cloud_enterprise_search/": {"canonical": "/docs/integrations/retrievers/google_vertex_ai_search/"},
  "/docs/integrations/tools/metaphor_search/": {"canonical": "/docs/integrations/tools/exa_search/"},
  "/docs/expression_language/how_to/fallbacks/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/fallbacks/"]},
  "/docs/expression_language/cookbook/retrieval/": {"canonical": "/docs/tutorials/rag/", "alternative": ["/v0.1/docs/use_cases/question_answering/"]},
  "/docs/expression_language/cookbook/agent/": {"canonical": "/docs/how_to/migrate_agent/", "alternative": ["/v0.1/docs/modules/agents/agent_types/xml_agent/"]},
  "/docs/modules/model_io/prompts/message_prompts/": {"canonical": "/docs/how_to/#prompt-templates", "alternative": ["/v0.1/docs/modules/model_io/prompts/quick_start/"]},
  "/docs/modules/model_io/prompts/pipeline/": {"canonical": "/docs/how_to/prompts_composition/", "alternative": ["/v0.1/docs/modules/model_io/prompts/composition/"]},
  "/docs/expression_language/cookbook/memory/": {"canonical": "/docs/how_to/chatbots_memory/", "alternative": ["/v0.1/docs/modules/memory/"]},
  "/docs/expression_language/cookbook/tools/": {"canonical": "/docs/tutorials/agents/", "alternative": ["/v0.1/docs/use_cases/tool_use/quickstart/"]},
  "/docs/expression_language/cookbook/sql_db/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/quickstart/"]},
  "/docs/expression_language/cookbook/moderation/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/safety/moderation/"]},
  "/docs/expression_language/cookbook/embedding_router/": {"canonical": "/docs/how_to/routing/", "alternative": ["/v0.1/docs/expression_language/how_to/routing/"]},
  "/docs/guides/structured_output/": {"canonical": "/docs/how_to/structured_output/", "alternative": ["/v0.1/docs/modules/model_io/chat/structured_output/"]},
  "/docs/modules/agents/how_to/structured_tools/": {"canonical": "/docs/how_to/#tools", "alternative": ["/v0.1/docs/modules/tools/"]},
  "/docs/use_cases/csv/": {"canonical": "/docs/tutorials/sql_qa/", "alternative": ["/v0.1/docs/use_cases/sql/csv/"]},
  "/docs/guides/debugging/": {"canonical": "/docs/how_to/debugging/", "alternative": ["/v0.1/docs/guides/development/debugging/"]},
  "/docs/guides/extending_langchain/": {"canonical": "/docs/how_to/#custom", "alternative": ["/v0.1/docs/guides/development/extending_langchain/"]},
  "/docs/guides/fallbacks/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/fallbacks/"]},
  "/docs/guides/model_laboratory/": {"canonical": "https://docs.smith.langchain.com/", "alternative": ["/v0.1/docs/guides/productionization/evaluation/"]},
  "/docs/guides/pydantic_compatibility/": {"canonical": "/docs/how_to/pydantic_compatibility/", "alternative": ["/v0.1/docs/guides/development/pydantic_compatibility/"]},
  "/docs/guides/local_llms/": {"canonical": "/docs/how_to/local_llms/", "alternative": ["/v0.1/docs/guides/development/local_llms/"]},
  "/docs/modules/model_io/quick_start/": {"canonical": "/docs/how_to/#chat-models", "alternative": ["/v0.1/docs/modules/model_io/"]},
  "/docs/expression_language/how_to/generators/": {"canonical": "/docs/how_to/functions/", "alternative": ["/v0.1/docs/expression_language/primitives/functions/"]},
  "/docs/expression_language/how_to/functions/": {"canonical": "/docs/how_to/functions/", "alternative": ["/v0.1/docs/expression_language/primitives/functions/"]},
  "/docs/expression_language/how_to/passthrough/": {"canonical": "/docs/how_to/passthrough/", "alternative": ["/v0.1/docs/expression_language/primitives/passthrough/"]},
  "/docs/expression_language/how_to/map/": {"canonical": "/docs/how_to/parallel/", "alternative": ["/v0.1/docs/expression_language/primitives/parallel/"]},
  "/docs/expression_language/how_to/binding/": {"canonical": "/docs/how_to/binding/", "alternative": ["/v0.1/docs/expression_language/primitives/binding/"]},
  "/docs/expression_language/how_to/configure/": {"canonical": "/docs/how_to/configure/", "alternative": ["/v0.1/docs/expression_language/primitives/configure/"]},
  "/docs/expression_language/cookbook/prompt_llm_parser/": {"canonical": "/docs/how_to/sequence/", "alternative": ["/v0.1/docs/expression_language/get_started/"]},
  "/docs/contributing/documentation/": {"canonical": "/docs/contributing/how_to/documentation/", "alternative": ["/v0.1/docs/contributing/documentation/technical_logistics/"]},
  "/docs/expression_language/cookbook/": {"canonical": "/docs/how_to/#langchain-expression-language-lcel", "alternative": ["/v0.1/docs/expression_language/"]},
  "/docs/integrations/text_embedding/solar/": {"canonical": "/docs/integrations/text_embedding/upstage/"},
  "/docs/integrations/chat/solar/": {"canonical": "/docs/integrations/chat/upstage/"},
  // custom ones

  "/docs/modules/model_io/chat/llm_chain/": {
    "canonical": "/docs/tutorials/llm_chain/"
  },

  "/docs/modules/agents/toolkits/": {
    "canonical": "/docs/integrations/tools/",
    "alternative": [
      "/v0.1/docs/integrations/toolkits/"
    ]
  }
}
