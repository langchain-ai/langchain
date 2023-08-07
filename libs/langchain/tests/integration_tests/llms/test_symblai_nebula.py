"""Test Nebula API wrapper."""

from langchain import LLMChain, PromptTemplate
from langchain.llms.symblai_nebula import Nebula


def test_symblai_nebula_call() -> None:
    """Test valid call to Nebula."""
    conversation = """Speaker 1: Thank you for calling ABC, company.Speaker 1: My name 
is Mary.Speaker 1: How may I help you?Speaker 2: Today?Speaker 1: All right, 
Madam.Speaker 1: I really apologize for this inconvenient.Speaker 1: I will be happy 
to assist you in this matter.Speaker 1: Could you please offer me Yuri your account 
number?Speaker 1: Alright Madam, thank you very much.Speaker 1: Let me check that 
for confirmation.Speaker 1: Did you say 534 00 365?Speaker 2: 48?Speaker 1: Very good 
man.Speaker 1: Now for verification purposes, can I please get your full?Speaker 
2: Name?Speaker 1: Alright, thank you.Speaker 1: Very much.Speaker 1: Madam.Speaker 
1: Can I, please get your birthdate now?Speaker 1: I am sorry madam.Speaker 1: I 
didn't make this clear is for verification.Speaker 1: Purposes is the company 
request.Speaker 1: The system requires me, your name, your complete name and your 
date of.Speaker 2: Birth.Speaker 2: Alright, thank you very much madam.Speaker 1: 
All right.Speaker 1: Thank you very much, Madam.Speaker 1: Thank you for that 
information.Speaker 1: Let me check what happens.Speaker 2: Here.Speaker 1: So 
according to our data space them, you did pay your last bill last August the 12, 
which was two days ago in one of our Affiliated payment centers.Speaker 1: So, at the 
moment you currently, We have zero balance.Speaker 1: So however, the bill that you 
received was generated a week before you made the pavement, this is reason why you 
already make this payment, have not been reflected yet.Speaker 1: So what we do in 
this case, you just simply disregard the amount indicated in the field and you 
continue to enjoy our service man.Speaker 1: Sure, Madam.Speaker 1: And I am sure 
you need your cell phone for everything for life, right?Speaker 1: So I really 
apologize for this inconvenience.Speaker 1: And let me tell you that delays in the 
bill is usually caused by delays in our Courier Service.Speaker 1: That is to say 
that it'''s a problem, not with the company, but with a courier service, For a more 
updated, feel of your account, you can visit our website and log into your account, 
and they'''re in the system.Speaker 1: On the website, you are going to have the 
possibility to pay the bill.Speaker 1: That is more.Speaker 2: Updated.Speaker 2: 
Of course, Madam I can definitely assist you with that.Speaker 2: Once you have, 
you want to see your bill updated, please go to www.hsn BC campus, any.com after 
that.Speaker 2: You will see in the tale.Speaker 1: All right corner.Speaker 1: So 
you're going to see a pay now button.Speaker 1: Please click on the pay now button 
and the serve.Speaker 1: The system is going to ask you for personal 
information.Speaker 1: Such as your first name, your ID account, your the number of 
your account, your email address, and your phone number once you complete this personal 
information."""
    llm = Nebula(
        conversation=conversation,
    )

    template = """Identify the {count} main objectives or goals mentioned in this 
context concisely in less points. Emphasize on key intents."""
    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(count="five")

    assert isinstance(output, str)
