# flake8: noqa
INFOBIP_SEND_SMS_PROMPT = """ 
    This tool is a wrapper around infobip-api-python-sdk's SMS API, useful when you need to send an SMS message. 
    The input to this tool is a dictionary specifying the fields of the SMS message, and will be passed into infobip-api-python-sdk's SMS `run` function.
    For example, to send an SMS message with text "test message" to number +1234567890, you would pass in the following dictionary: 
    {{"message": "test message", "to": "+1234567890"}}
    """

INFOBIP_SEND_EMAIL_PROMPT = """
    This tool is a wrapper around infobip-api-python-sdk's Email API, useful when you need to send an email message.
    The input to this tool is a dictionary specifying the fields of the email message, and will be passed into infobip-api-python-sdk's Email `run` function.
    For example, to send an email message with text "test message" to email address example@example.com with subject "test subject", you would pass in the following dictionary:
    {{"message": "test message", "to": "example@example.com", "subject": "test subject"}}
    """
