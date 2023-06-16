"""O365 tool utils."""
from __future__ import annotations

import os
from O365 import Account


def clean_body(body: str) -> str:
    """Clean body of a message or event."""
    try:
        from bs4 import BeautifulSoup

        try:
            # Remove HTML
            soup = BeautifulSoup(str(body), "html.parser")
            body = soup.get_text()

            # Remove return characters
            body = "".join(body.splitlines())

            # Remove extra spaces
            body = " ".join(body.split())

            return str(body)
        except Exception as e:
            return str(body)
    except ImportError:
        return str(body)

def authenticate():
    """Authenticate using the Microsoft Grah API"""
    
    if "CLIENT_ID" in os.environ and "CLIENT_SECRET" in os.environ:
        client_id = os.environ["CLIENT_ID"]
        client_secret = os.environ["CLIENT_SECRET"]
        credentials = (client_id, client_secret)
    else:
        print("Error: The CLIENT_ID and CLIENT_SECRET environmental variables have not been set. "
              "Visit the following link on how to acquire these authorization tokens: "
              "https://learn.microsoft.com/en-us/graph/auth/")
        return None

    account = Account(credentials)

    if account.is_authenticated is False:

        if not account.authenticate(scopes=['https://graph.microsoft.com/Mail.ReadWrite', 
                                            'https://graph.microsoft.com/Mail.Send', 
                                            'https://graph.microsoft.com/Calendars.ReadWrite',
                                            'https://graph.microsoft.com/MailboxSettings.ReadWrite']):
            print("Error: Could not authenticate")
            return None
        else:
            return account
    else:
        return account
