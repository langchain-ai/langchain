import string
from faker import Faker

fake = Faker()

# Listed  entities supported by Microsoft Presidio (for now, global and US only)
# Source: https://microsoft.github.io/presidio/supported_entities/
pseudoanonymizer_mapping = {
    # Global entities
    "PERSON": lambda: fake.name(),
    "EMAIL_ADDRESS": lambda: fake.email(),
    "PHONE_NUMBER": lambda: fake.phone_number(),
    "IBAN_CODE": lambda: fake.iban(),
    "CREDIT_CARD": lambda: fake.credit_card_number(),
    "CRYPTO": lambda: "bc1"
    + "".join(fake.random_choices(string.ascii_lowercase + \
              string.digits, length=26)),
    "IP_ADDRESS": lambda: fake.ipv4_public(),
    "LOCATION": lambda: fake.address(),
    "DATE_TIME": lambda: fake.iso8601(),
    "NRP": lambda: fake.random_number(digits=8, fix_len=True),
    "MEDICAL_LICENSE": lambda: fake.bothify(text="??######").upper(),
    "URL": lambda: fake.url(),
    # US-specific entities
    "US_BANK_NUMBER": lambda: fake.bban(),
    "US_DRIVER_LICENSE": lambda: fake.random_number(digits=9, fix_len=True),
    "US_ITIN": lambda: fake.bothify(text="9##-7#-####"),
    "US_PASSPORT": lambda: fake.bothify(text="#####??").upper(),
    "US_SSN": lambda: fake.ssn(),
}
