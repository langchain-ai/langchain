"""
    -- @Time    : 2023/4/25 9:31
    -- @Author  : yazhui Yu
    -- @email   : yuyazhui@bangdao-tech.com
    -- @File    : yuque
    -- @Software: Pycharm
"""
import logging
import re
from functools import partial
from typing import List, Tuple

from tqdm import tqdm

from .network import request
from .setting import API, USER_AGENT

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

requests = partial(request, method="GET")


class YuQueDocs:

    def __init__(self, custom_space: str = 'www', user_agent: str = '', user_token: str = '', **kwargs):
        user_agent = user_agent if user_agent != '' else USER_AGENT
        logging.warning(f"`user_agent` is default, then `user_agent={USER_AGENT}` instead!")
        if not len(user_token):
            raise ValueError(
                "`user_token` must be required"
            )
        self.custom_space = custom_space
        self.user_token = user_token
        self.kwargs = kwargs

        self.api = API.format(custom_space)
        self.headers = {
            "User-Agent": user_agent,
            "X-Auth-Token": user_token
        }

    async def __check_login(self) -> None:
        """check login"""
        res = await requests(url=self.api + '/user', headers=self.headers)
        if 200 != res.status_code:
            raise ValueError(f"Token {self.user_token} Error or "
                             f"Custom_space {self.custom_space} Error or"
                             f"Token not match Custom_space!")
        userJson = res.json()
        self.login_id = userJson['data']['login']
        self.uid = userJson['data']['id']
        self.username = userJson['data']['name']
        logging.info(f"{self.username} Login Success!")

    async def get_all_knowledge_base(self) -> List[dict]:
        """get all knowledge base"""
        reposRequest = await requests(url=self.api + '/users/' + self.login_id + '/repos', headers=self.headers)
        reposRequest = reposRequest.json()
        all_knowledge_base = []
        for item in reposRequest['data']:
            all_knowledge_base.append({"knowledge_base_id": item['id'], "knowledge_base_name": item['name']})
        return all_knowledge_base

    async def get_all_docs_slug(self, knowledge_base: dict) -> List[dict]:
        """get all docs slug"""
        knowledge_base_id = knowledge_base.get('knowledge_base_id', None)
        knowledge_base_name = knowledge_base.get('knowledge_base_name', "")

        if knowledge_base_id is None:
            raise ValueError(
                "knowledge_base does not have `knowledge_base_id`, "
                "please check knowledge_base."
            )

        if knowledge_base_name == "":
            logging.warning(
                f"knowledge_base does not have `knowledge_base_name`, "
                f"using `knowledge_base_name={knowledge_base_name}` instead."
            )

        listDocs = []
        docsRequest = await requests(url=self.api + '/repos/' + str(knowledge_base_id) + '/docs',
                                     headers=self.headers)
        docsRequest = docsRequest.json()
        for item in docsRequest['data']:
            listDocs.append(
                {
                    "knowledge_base_id": knowledge_base_id,
                    "knowledge_base_name": knowledge_base_name,
                    "title": item['title'],
                    "description": item['description'],
                    "slug": item['slug']
                }
            )
        return listDocs

    async def get_docs(self, item: dict) -> dict:
        knowledge_base_id = item.get('knowledge_base_id', None)
        knowledge_base_name = item.get('knowledge_base_name', None)
        title = item.get('title', None)
        description = item.get('description', None)
        slug = item.get('slug', "")

        if knowledge_base_id is None:
            raise ValueError(
                "knowledge_base does not have `knowledge_base_id`, "
                "please check knowledge_base."
            )
        docDetails = await requests(url=self.api + '/repos/' + str(knowledge_base_id) + '/docs/' + slug,
                                    headers=self.headers)
        docDetails = docDetails.json()

        docDetails_temp = re.sub(r'\\n', "\n", docDetails['data']['body'])
        data = re.sub(r'<a name="(.*)"></a>', "", docDetails_temp)

        return {
            "knowledge_base_id": knowledge_base_id,
            "knowledge_base_name": knowledge_base_name,
            "title": title,
            "description": description,
            "slug": slug,
            'data': data
        }

    async def get_all_docs(self, docs_slug: List[dict]) -> List[dict]:
        docs = []
        pbar = tqdm(docs_slug)
        for _, doc_slug in enumerate(pbar):
            doc = await self.get_docs(doc_slug)
            docs.append(doc)
            pbar.set_description("download")
            pbar.set_postfix({
                "knowledge_base_id": str(doc.get("knowledge_base_id", "")),
                "knowledge_base_name": doc.get("knowledge_base_name", ""),
                "title": doc.get("title", "")
            })
        return docs

    async def load_docs(self) -> List[dict]:
        work_type, work_knowledge_bases = await self.get_work_knowledge_bases()
        all_docs = []
        if work_type == "kb":
            for work_knowledge_base in work_knowledge_bases:
                docs_slug = await self.get_all_docs_slug(work_knowledge_base)
                docs = await self.get_all_docs(docs_slug)
                all_docs.extend(docs)
        elif work_type == "slug":
            docs_slug = work_knowledge_bases
            docs = await self.get_all_docs(docs_slug)
            all_docs.extend(docs)
        else:
            raise ValueError(
                "Work_type Error!"
            )

        return all_docs

    async def get_work_knowledge_bases(self) -> Tuple[str, List[dict]]:

        await self.__check_login()

        knowledge_base_id = self.kwargs.get('knowledge_base_id', [])
        knowledge_base_name = self.kwargs.get('knowledge_base_name', [])
        docs_slug = self.kwargs.get('docs_slug', [])
        docs_title = self.kwargs.get('docs_title', [])

        if isinstance(knowledge_base_id, str):
            knowledge_base_id = [knowledge_base_id]
        elif not isinstance(knowledge_base_id, list):
            raise TypeError(
                "Type of knowledge_base_id must be list"
            )

        if isinstance(knowledge_base_name, str):
            knowledge_base_name = [knowledge_base_name]
        elif not isinstance(knowledge_base_name, list):
            raise TypeError(
                "Type of knowledge_base_name must be list"
            )

        if isinstance(docs_slug, str):
            docs_slug = [docs_slug]
        elif not isinstance(docs_slug, list):
            raise TypeError(
                "Type of docs_slug must be list"
            )

        if isinstance(docs_title, str):
            docs_title = [docs_title]
        elif not isinstance(docs_title, list):
            raise TypeError(
                "Type of docs_title must be list"
            )

        knowledge_base_id = [str(i) for i in knowledge_base_id]
        knowledge_base_name = [str(i) for i in knowledge_base_name]
        docs_slug = [str(i) for i in docs_slug]
        docs_title = [str(i) for i in docs_title]

        all_knowledge_base = await self.get_all_knowledge_base()

        if knowledge_base_id or knowledge_base_name:
            work_knowledge_bases = [i for i in all_knowledge_base
                                    if str(i['knowledge_base_id']) in knowledge_base_id
                                    or str(i['knowledge_base_name']) in knowledge_base_name]
            if knowledge_base_id:
                not_find_kb_id = [i
                                  for i in knowledge_base_id
                                  if i not in [
                                      str(j['knowledge_base_id'])
                                      for j in work_knowledge_bases]
                                  ]
                if not_find_kb_id:
                    logging.warning(f"Can not find knowledge_base_id({'、'.join(not_find_kb_id)}) in {self.username}!")
            if knowledge_base_name:
                not_find_kb_names = [i
                                     for i in knowledge_base_name
                                     if i not in [
                                         str(j['knowledge_base_name'])
                                         for j in work_knowledge_bases]
                                     ]
                if not_find_kb_names:
                    logging.warning(
                        f"Can not find knowledge_base_name({'、'.join(not_find_kb_names)}) in {self.username}!")

        else:
            work_knowledge_bases = all_knowledge_base.copy()
            logging.info(f"knowledge_base_id and knowledge_base_name are both empty,"
                         f" try load all docs in {self.username}")

        if work_knowledge_bases:
            if docs_slug or docs_title:
                all_docs_slug = []
                for work_knowledge_base in work_knowledge_bases:
                    all_docs_slug.extend(await self.get_all_docs_slug(work_knowledge_base))
                work_doc_slug = [i for i in all_docs_slug if i["slug"] in docs_slug or i["title"] in docs_title]
                if docs_slug:
                    not_find_docs_slug = [i
                                          for i in docs_slug
                                          if i not in [
                                              str(j['slug'])
                                              for j in work_doc_slug]
                                          ]
                    if not_find_docs_slug:
                        logging.warning(f"Can not find docs_slug({'、'.join(not_find_docs_slug)}) in {self.username}!")
                if docs_title:
                    not_find_docs_title = [i
                                           for i in docs_title
                                           if i not in [
                                               str(j['title'])
                                               for j in work_doc_slug]
                                           ]
                    if not_find_docs_title:
                        logging.warning(f"Can not find docs_title({'、'.join(not_find_docs_title)}) in {self.username}!")
                if work_doc_slug:
                    return "slug", work_doc_slug
                else:
                    raise ValueError(
                        f"Check `knowledge_base_id`(`knowledge_base_name`) and"
                        f"`docs_slug`(`docs_title`), "
                        f"No `docs_slug`(`docs_title`) matching `knowledge_base_id`"
                        f"(`knowledge_base_name`) in `{self.username}` "
                    )
            else:
                return "kb", work_knowledge_bases

        else:
            raise ValueError(
                f"Check `knowledge_base_id` and `knowledge_base_name`, "
                f"No knowledge_base matching knowledge_base_id and "
                f"knowledge_base_name in `{self.username}` "
            )


if __name__ == "__main__":
    yq = YuQueDocs()
    yq.load_docs()
