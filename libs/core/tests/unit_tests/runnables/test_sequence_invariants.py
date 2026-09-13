def chain_transforms(*funcs):
    def runner(initial_input):
        res = initial_input
        for f in funcs:
            res = f(res)
        return res
    return runner

def test_chain_order_preservation():
    step1 = lambda x: f"[{x}]"
    step2 = lambda x: f"({x})"
    step3 = lambda x: f"{{{x}}}"
    
    pipeline = chain_transforms(step1, step2, step3)
    assert pipeline("data") == "{([data])}"
