import string
from typing import Callable, Dict, Optional


def get_pseudoanonymizer_mapping(seed: Optional[int] = None) -> Dict[str, Callable]:
    try:
        from faker import Faker
    except ImportError as e:
        raise ImportError(
            "Could not import faker, please install it with `pip install Faker`."
        ) from e

    fake = Faker()
    fake.seed_instance(seed)

    # Listed  entities supported by Microsoft Presidio (for now, global and US only)
    # Source: https://microsoft.github.io/presidio/supported_entities/
    return {
        # Global entities
        "PERSON": lambda _: fake.name(),
        "EMAIL_ADDRESS": lambda _: fake.email(),
        "PHONE_NUMBER": lambda _: fake.phone_number(),
        "IBAN_CODE": lambda _: fake.iban(),
        "CREDIT_CARD": lambda _: fake.credit_card_number(),
        "CRYPTO": lambda _: "bc1"
        + "".join(
            fake.random_choices(string.ascii_lowercase + string.digits, length=26)
        ),
        "IP_ADDRESS": lambda _: fake.ipv4_public(),
        "LOCATION": lambda _: fake.city(),
        "DATE_TIME": lambda _: fake.date(),
        "NRP": lambda _: str(fake.random_number(digits=8, fix_len=True)),
        "MEDICAL_LICENSE": lambda _: fake.bothify(text="??######").upper(),
        "URL": lambda _: fake.url(),
        # US-specific entities
        "US_BANK_NUMBER": lambda _: fake.bban(),
        "US_DRIVER_LICENSE": lambda _: str(fake.random_number(digits=9, fix_len=True)),
        "US_ITIN": lambda _: fake.bothify(text="9##-7#-####"),
        "US_PASSPORT": lambda _: fake.bothify(text="#####??").upper(),
        "US_SSN": lambda _: fake.ssn(),
        # UK-specific entities
        "UK_NHS": lambda _: str(fake.random_number(digits=10, fix_len=True)),
        # Spain-specific entities
        "ES_NIF": lambda _: fake.bothify(text="########?").upper(),
        # Italy-specific entities
        "IT_FISCAL_CODE": lambda _: fake.bothify(text="??????##?##?###?").upper(),
        "IT_DRIVER_LICENSE": lambda _: fake.bothify(text="?A#######?").upper(),
        "IT_VAT_CODE": lambda _: fake.bothify(text="IT???????????"),
        "IT_PASSPORT": lambda _: str(fake.random_number(digits=9, fix_len=True)),
        "IT_IDENTITY_CARD": lambda _: lambda _: str(
            fake.random_number(digits=7, fix_len=True)
        ),
        # Singapore-specific entities
        "SG_NRIC_FIN": lambda _: fake.bothify(text="????####?").upper(),
        # Australia-specific entities
        "AU_ABN": lambda _: str(fake.random_number(digits=11, fix_len=True)),
        "AU_ACN": lambda _: str(fake.random_number(digits=9, fix_len=True)),
        "AU_TFN": lambda _: str(fake.random_number(digits=9, fix_len=True)),
        "AU_MEDICARE": lambda _: str(fake.random_number(digits=10, fix_len=True)),
    }
