"""Prompt for the ARC dataset."""
# From https://arxiv.org/pdf/2203.11171.pdf

from langchain.prompt import Prompt


_PROMPT_TEMPLATE = """Q: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most
heat? (a) dry palms. (b) wet palms. (c) palms covered with oil. (d) palms covered with lotion.
A: Dry surfaces will more likely cause more friction via rubbing than other smoother surfaces, hence dry
palms will produce the most heat. The answer is (a).
Q: Which factor will most likely cause a person to develop a fever? (a) a leg muscle relaxing after exercise.
(b) a bacterial population in the bloodstream. (c) several viral particles on the skin. (d) carbohydrates being
digested in the stomach.
A: Option (b), bacterial population is the most likely cause for a person developing fever. The answer is (b).
Q: Which change in the state of water particles causes the particles to become arranged in a fixed position?
(a) boiling. (b) melting. (c) freezing. (d) evaporating.
A: When water is freezed, the particles are arranged in a fixed position; the particles are still moving for all
other options. The answer is (c).
Q: When a switch is used in an electrical circuit, the switch can (a) cause the charge to build. (b) increase
and decrease the voltage. (c) cause the current to change direction. (d) stop and start the flow of current.
A: The function of a switch is to start and stop the flow of a current. The answer is (d).
Q: {question} {choices}
A:"""

ARC_PROMPT = Prompt(
    input_variables=["question", "choices"],
    template=_PROMPT_TEMPLATE,
)