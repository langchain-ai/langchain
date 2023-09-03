"""Classification of logical fallacies in Natural Language Arguments \
from https://arxiv.org/pdf/2212.07425.pdf
"""

from typing import Dict

from langchain_experimental.fallacy_removal.models import LogicalFallacy

FALLACIES: Dict[str, LogicalFallacy] = {
    "adhominem": LogicalFallacy(
        name="adhominem",
        fallacy_critique_request="Identify any feasible ways in which \
        the assistant’s last response is attacking the character or \
        personal traits of the person making an argument rather than \
        addressing the actual argument and evidence.",
        fallacy_revision_request="Please rewrite the assistant response\
        to remove any attacking the character or personal traits of the\
        person making an argument rather than addressing the actual\
        argument and evidence.",
    ),
    "adpopulum": LogicalFallacy(
        name="adpopulum",
        fallacy_critique_request="Identify ways in which the assistant’s\
        last response may be asserting that something must be true or \
        correct simply because many people believe it or do it, without \
        actual facts or evidence to support the conclusion.",
        fallacy_revision_request="Please rewrite the assistant response \
        to remove any assertion that something must be true or correct \
        simply because many people believe it or do it, without actual \
        facts or evidence to support the conclusion.",
    ),
    "appealtoemotion": LogicalFallacy(
        name="appealtoemotion",
        fallacy_critique_request="Identify all ways in which the \
        assistant’s last response is an attempt to win support for an \
        argument by exploiting or manipulating people's emotions rather \
        than using facts and reason.",
        fallacy_revision_request="Please rewrite the assistant response \
        to remove any attempt to win support for an argument by \
        exploiting or manipulating people's emotions rather than using \
        facts and reason.",
    ),
    "fallacyofextension": LogicalFallacy(
        name="fallacyofextension",
        fallacy_critique_request="Identify any ways in which the \
        assitant's last response is making broad, sweeping generalizations\
        and extending the implications of an argument far beyond what the \
        initial premises support.",
        fallacy_revision_request="Rewrite the assistant response to remove\
         all broad, sweeping generalizations and extending the implications\
         of an argument far beyond what the initial premises support.",
    ),
    "intentionalfallacy": LogicalFallacy(
        name="intentionalfallacy",
        fallacy_critique_request="Identify any way in which the assistant’s\
        last response may be falsely supporting a conclusion by claiming to\
        understand an author or creator's subconscious intentions without \
        clear evidence.",
        fallacy_revision_request="Revise the assistant’s last response to \
        remove any false support of a conclusion by claiming to understand\
        an author or creator's subconscious intentions without clear \
        evidence.",
    ),
    "falsecausality": LogicalFallacy(
        name="falsecausality",
        fallacy_critique_request="Think carefully about whether the \
        assistant's last response is jumping to conclusions about causation\
        between events or circumstances without adequate evidence to infer \
        a causal relationship.",
        fallacy_revision_request="Please write a new version of the \
        assistant’s response that removes jumping to conclusions about\
        causation between events or circumstances without adequate \
        evidence to infer a causal relationship.",
    ),
    "falsedilemma": LogicalFallacy(
        name="falsedilemma",
        fallacy_critique_request="Identify any way in which the \
        assistant's last response may be presenting only two possible options\
        or sides to a situation when there are clearly other alternatives \
        that have not been considered or addressed.",
        fallacy_revision_request="Amend the assistant’s last response to \
        remove any presentation of only two possible options or sides to a \
        situation when there are clearly other alternatives that have not \
        been considered or addressed.",
    ),
    "hastygeneralization": LogicalFallacy(
        name="hastygeneralization",
        fallacy_critique_request="Identify any way in which the assistant’s\
        last response is making a broad inference or generalization to \
        situations, people, or circumstances that are not sufficiently \
        similar based on a specific example or limited evidence.",
        fallacy_revision_request="Please rewrite the assistant response to\
        remove a broad inference or generalization to situations, people, \
        or circumstances that are not sufficiently similar based on a \
        specific example or limited evidence.",
    ),
    "illogicalarrangement": LogicalFallacy(
        name="illogicalarrangement",
        fallacy_critique_request="Think carefully about any ways in which \
        the assistant's last response is constructing an argument in a \
        flawed, illogical way, so the premises do not connect to or lead\
        to the conclusion properly.",
        fallacy_revision_request="Please rewrite the assistant’s response\
        so as to remove any construction of an argument that is flawed and\
        illogical or if the premises do not connect to or lead to the \
        conclusion properly.",
    ),
    "fallacyofcredibility": LogicalFallacy(
        name="fallacyofcredibility",
        fallacy_critique_request="Discuss whether the assistant's last \
        response was dismissing or attacking the credibility of the person\
        making an argument rather than directly addressing the argument \
        itself.",
        fallacy_revision_request="Revise the assistant’s response so as \
        that it refrains from dismissing or attacking the credibility of\
        the person making an argument rather than directly addressing \
        the argument itself.",
    ),
    "circularreasoning": LogicalFallacy(
        name="circularreasoning",
        fallacy_critique_request="Discuss ways in which the assistant’s\
        last response may be supporting a premise by simply repeating \
        the premise as the conclusion without giving actual proof or \
        evidence.",
        fallacy_revision_request="Revise the assistant’s response if \
        possible so that it’s not supporting a premise by simply \
        repeating the premise as the conclusion without giving actual\
        proof or evidence.",
    ),
    "beggingthequestion": LogicalFallacy(
        name="beggingthequestion",
        fallacy_critique_request="Discuss ways in which the assistant's\
        last response is restating the conclusion of an argument as a \
        premise without providing actual support for the conclusion in \
        the first place.",
        fallacy_revision_request="Write a revision of the assistant’s \
        response that refrains from restating the conclusion of an \
        argument as a premise without providing actual support for the \
        conclusion in the first place.",
    ),
    "trickquestion": LogicalFallacy(
        name="trickquestion",
        fallacy_critique_request="Identify ways in which the \
        assistant’s last response is asking a question that \
        contains or assumes information that has not been proven or \
        substantiated.",
        fallacy_revision_request="Please write a new assistant \
        response so that it does not ask a question that contains \
        or assumes information that has not been proven or \
        substantiated.",
    ),
    "overapplier": LogicalFallacy(
        name="overapplier",
        fallacy_critique_request="Identify ways in which the assistant’s\
        last response is applying a general rule or generalization to a \
        specific case it was not meant to apply to.",
        fallacy_revision_request="Please write a new response that does\
        not apply a general rule or generalization to a specific case \
        it was not meant to apply to.",
    ),
    "equivocation": LogicalFallacy(
        name="equivocation",
        fallacy_critique_request="Read the assistant’s last response \
        carefully and identify if it is using the same word or phrase \
        in two different senses or contexts within an argument.",
        fallacy_revision_request="Rewrite the assistant response so \
        that it does not use the same word or phrase in two different \
        senses or contexts within an argument.",
    ),
    "amphiboly": LogicalFallacy(
        name="amphiboly",
        fallacy_critique_request="Critique the assistant’s last response\
        to see if it is constructing sentences such that the grammar \
        or structure is ambiguous, leading to multiple interpretations.",
        fallacy_revision_request="Please rewrite the assistant response\
        to remove any construction of sentences where the grammar or \
        structure is ambiguous or leading to multiple interpretations.",
    ),
    "accent": LogicalFallacy(
        name="accent",
        fallacy_critique_request="Discuss whether the assitant's response\
        is misrepresenting an argument by shifting the emphasis of a word\
        or phrase to give it a different meaning than intended.",
        fallacy_revision_request="Please rewrite the AI model's response\
        so that it is not misrepresenting an argument by shifting the \
        emphasis of a word or phrase to give it a different meaning than\
        intended.",
    ),
    "composition": LogicalFallacy(
        name="composition",
        fallacy_critique_request="Discuss whether the assistant's \
        response is erroneously inferring that something is true of \
        the whole based on the fact that it is true of some part or \
        parts.",
        fallacy_revision_request="Please rewrite the assitant's response\
        so that it is not erroneously inferring that something is true \
        of the whole based on the fact that it is true of some part or \
        parts.",
    ),
    "division": LogicalFallacy(
        name="division",
        fallacy_critique_request="Discuss whether the assistant's last \
        response is erroneously inferring that something is true of the \
        parts based on the fact that it is true of the whole.",
        fallacy_revision_request="Please rewrite the assitant's response\
        so that it is not erroneously inferring that something is true \
        of the parts based on the fact that it is true of the whole.",
    ),
}
