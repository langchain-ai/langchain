def create_chat_router(
    program: Runnable,
) -> Callable[[Sequence[MessageLike]], Optional[WorkingMemoryProcessor]]:
    last_message = messages[-1] if messages else None
    if not last_message:
        return []

    new_messages = self.memory_processor.process(messages)

    match last_message:
        case AgentFinish():
            return []
        case _:
            return self.llm_program.invoke(new_messages, config=config)
