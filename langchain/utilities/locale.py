
import os
import gettext

def localize(domain=None,lang=None,locales=None):
    """
        define translation function
        lang: default en_US
        locales: default ./locale
    """
    domain = domain or 'langchain'
    lang = lang or os.environ.get('LANG', 'en_US')
    locales = locales or os.environ.get('LOCALE_PATH', 'langchain/locale')

    # 加载语言包
    gettext.bindtextdomain(domain, locales)
    gettext.textdomain(domain)
    return gettext.gettext

_ = localize()