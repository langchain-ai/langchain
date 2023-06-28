# Causal program-aided language (CPAL) chain

## Motivation

The existing PAL chain, meant to reduce hallucination, hallucinates on a math problem with a nested chain of dependence. The CPAL chain includes causal structure to fix the hallucination.

For example, using the below word problem, PAL answers with 5, and CPAL answers with 13.

    "Tim buys the same number of pets as Cindy and Boris."
    "Cindy buys the same number of pets as Bill plus Bob."
    "Boris buys the same number of pets as Ben plus Beth."
    "Bill buys the same number of pets as Obama."
    "Bob buys the same number of pets as Obama."
    "Ben buys the same number of pets as Obama."
    "Beth buys the same number of pets as Obama."
    "If Obama buys one pet, how many pets total does everyone buy?"

CPAL translates the above narrative into this causal structure below, represented as a graph (DAG).

![complex-graph](https://github.com/hwchase17/langchain/assets/367522/d938db15-f941-493d-8605-536ad530f576)

.

The two major sections below are:

1. Technical overview
2. Future application

Also see [this jupyter notebook](https://github.com/borisdev/langchain/blob/master/docs/extras/modules/chains/additional/cpal.ipynb) doc.


## 1. Technical overview

### CPAL versus PAL

Like [PAL](https://arxiv.org/abs/2211.10435), CPAL intends to reduce large language model (LLM) hallucination. 

The CPAL chain is different from the PAL chain for a couple of reasons. 

* CPAL adds a causal structure (or DAG) to link entity actions (or math expressions).  
* The CPAL math expressions are modeling a chain of cause and effect relations, which can be intervened upon, whereas for the PAL chain math expressions are projected math identities. 

PAL's generated python code is wrong. It hallucinates when complexity increases. 

```python
def solution():
    """Tim buys the same number of pets as Cindy and Boris.Cindy buys the same number of pets as Bill plus Bob.Boris buys the same number of pets as Ben plus Beth.Bill buys the same number of pets as Obama.Bob buys the same number of pets as Obama.Ben buys the same number of pets as Obama.Beth buys the same number of pets as Obama.If Obama buys one pet, how many pets total does everyone buy?"""
    obama_pets = 1
    tim_pets = obama_pets
    cindy_pets = obama_pets + obama_pets
    boris_pets = obama_pets + obama_pets
    total_pets = tim_pets + cindy_pets + boris_pets
    result = total_pets
    return result
```

CPAL's generated python code is correct.

```python
story outcome data
    name                                   code  value      depends_on
0  obama                                   pass    1.0              []
1   bill               bill.value = obama.value    1.0         [obama]
2    bob                bob.value = obama.value    1.0         [obama]
3    ben                ben.value = obama.value    1.0         [obama]
4   beth               beth.value = obama.value    1.0         [obama]
5  cindy   cindy.value = bill.value + bob.value    2.0     [bill, bob]
6  boris   boris.value = ben.value + beth.value    2.0     [ben, beth]
7    tim  tim.value = cindy.value + boris.value    4.0  [cindy, boris]

query data
{
    "question": "how many pets total does everyone buy?",
    "expression": "SELECT SUM(value) FROM df",
    "llm_error_msg": ""
}
```

Based on the comments below, CPAL's intended location in the library is `experimental/chains/cpal` and PAL's location is`chains/pal`.

### CPAL vs Graph QA

Both the CPAL chain and the Graph QA chain extract entity-action-entity relations into a DAG.

The CPAL chain is different from the Graph QA chain for a few reasons.

* Graph QA does not connect entities to math expressions
* Graph QA does not associate actions in a sequence of dependence.
* Graph QA does not decompose the narrative into these three parts:
  1. Story plot or causal model
  4. Hypothetical question
  5. Hypothetical condition 

### Evaluation

Preliminary evaluation on simple math word problems shows that this CPAL chain generates less hallucination than the PAL chain on answering questions about a causal narrative. Two examples are in [this jupyter notebook](https://github.com/borisdev/langchain/blob/master/docs/extras/modules/chains/additional/cpal.ipynb) doc.

## 2. Future applications

### "Plan as Narrative, Test as Code" applications

The thesis here is that a proposed Plan as Narrative, Test as Code approach allows you to represent a project plan both as code and as a narrative, giving you the best of both worlds.

#### Why Plan as Narrative?

The narrative form is quick. At consensus building meeting, people use narratives to persuade others of their plan. The narrative form is scalable. You can share, version control and index a narrative as opposed to SaaS software work flow diagrams. 

#### Why Test as Code?

Though fast, narratives are problematic as their complexity increases. The problem is LLMs and humans are prone to hallucination when predicting the outcomes of a narrative. The cost of building a consensus around the validity of a narrative outcome grows as its narrative complexity increases. This is a culprit in the “tribal knowledge” problem and "highest-paid person in the room" problem. The Amazon-6-pager narrative meeting form attacks this problem. Likewise, the Plan as Code concept attacks this problem. Code does not require tribal knowledge or social power to validate. As narrative complexity increases, the value of representing a plan as code goes up. Code is testable, complex narratives are not.

Moreover, code is quickly composable, complex narratives are not. Composability means it can be integrated with other project plans and applications. The output of one or more plans can be the input into another team's plan or another computer system. For community voting and funding, composable plans can be integrated with a Dapp. For stochastic simulations, a composable plan can be integrated with the [DoWhy library](https://github.com/py-why/dowhy).

In summary, a code representation makes a project plan composable and testable.

An explanation of a future demo app is [here.](https://github.com/borisdev/cpal-llm-chain-demo)
