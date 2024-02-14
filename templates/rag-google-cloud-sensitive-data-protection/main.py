from rag_google_cloud_sensitive_data_protection.chain import chain

if __name__ == "__main__":
    query = {
        "question": "Good morning. My name is Captain Blackbeard. My phone number "
        "is 555-555-5555. And my email is lovely.pirate@gmail.com. Have a nice day.",
        "chat_history": [],
    }
    print(chain.invoke(query))  # noqa: T201
