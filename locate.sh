cd langchain
find . -type f -name "*.py" -print0 | xargs -0 xgettext -o locale/en_US/LC_MESSAGES/langchain.pot --keyword=_

# msginit  --locale=zh_CN.UTF-8 -i locale/en_US/LC_MESSAGES/langchain.pot -o locale/zh_CN/LC_MESSAGES/langchain.po

msgmerge -U locale/zh_CN/LC_MESSAGES/langchain.po locale/en_US/LC_MESSAGES/langchain.pot

msgfmt locale/zh_CN/LC_MESSAGES/langchain.po -o locale/zh_CN/LC_MESSAGES/langchain.mo
