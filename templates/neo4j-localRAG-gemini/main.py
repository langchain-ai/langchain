from app import App

if __name__ == "__main__":
    app = App()
    try:
        app.authenticate()

        # Simulating user input for prompt handling
        prompt = "Find all nodes"
        cypher_query, results = app.handle_prompt(prompt)

        if cypher_query:
            print("Generated Cypher Query:")
            print(cypher_query)
            print("Query Results:")
            for record in results:
                print(record)
            print("Check the local database for more")
    finally:
        app.close()
