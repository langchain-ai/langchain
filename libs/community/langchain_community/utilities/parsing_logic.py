from __future__ import absolute_import, print_function, unicode_literals
import logging
import os
import os.path
import shutil
import xml.etree.ElementTree as ET
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from lxml import etree
from langchain_core.pydantic_v1 import BaseModel, root_validator
from typing import Any, Dict
# from indra.literature import pubmed_client
# from indra.util import UnicodeXMLTreeBuilder as UTB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PubMed_Central_Parser(BaseModel):
    """
    A parser for handling and extracting data from PubMed Central (PMC) articles in XML format.

    This class provides functionality to interact with and extract information from PMC articles, 
    including retrieving the full-text XML, extracting plaintext from the XML, filtering articles 
    based on their availability in different formats, and downloading PMC files from AWS S3. 

    The methods in this class are designed to support various use cases such as:

    - **ID Lookup:** Convert between different article identifiers (PMID, PMCID, DOI) using the 
      PubMed ID mapping service.
    - **XML Retrieval:** Fetch and parse XML content for a given PMC ID, either directly from 
      PMC or from a file.
    - **Text Extraction:** Extract relevant paragraphs and titles from the article body, front 
      section, or subarticles, while excluding irrelevant content such as LaTeX formulas or certain 
      floating elements (e.g., tables and figures).
    - **Filtering:** Filter lists of PubMed IDs (PMIDs) to identify those that have full-text 
      availability in PMC, depending on the desired source type.
    - **S3 Integration:** One of the official ways to download PMC XML or text files is directly from the PMC Open Access 
      subset hosted on AWS S3. For more information, visit the https://www.ncbi.nlm.nih.gov/pmc/tools/pmcaws/

    """
    # Default values for the parameters
    # official ID Converter API for pmcid, can be found here: https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/
    pmid_convert_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    # Paths to resource files
    pmids_fulltext_path = os.path.join(os.path.dirname(__file__), "pmids_fulltext.txt")
    pmids_oa_xml_path = os.path.join(os.path.dirname(__file__), "pmids_oa_xml.txt")
    pmids_oa_txt_path = os.path.join(os.path.dirname(__file__), "pmids_oa_txt.txt")
    pmids_auth_xml_path = os.path.join(os.path.dirname(__file__), "pmids_auth_xml.txt")
    # Define global dict containing lists of PMIDs among mineable PMCs
    # to be lazily initialized
    pmids_fulltext_dict = {}

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import xmltodict
            import boto3
            from indra.literature import pubmed_client
            from indra.util import UnicodeXMLTreeBuilder as UTB
            values["parse"] = xmltodict.parse
            
        except ImportError as e:
            missing_package = str(e).split("No module named ")[-1]
            raise ImportError(
                f"Missing required package: '{missing_package}'. "
                "Please install it with `pip install {missing_package}`."
        )
        return values

    def id_lookup(self, paper_id, idtype=None):
        """Return PMID, DOI and PMCID based on an input ID.

        This function takes a Pubmed ID, Pubmed Central ID, or DOI
        and use the Pubmed ID mapping service and looks up all other IDs from one
        of these. The IDs are returned in a dictionary.

        Parameters
        ----------
        paper_id : str
            A PubMed ID, PubMed Central ID, or DOI.
        idtype : Optional[str]
            The type of the input ID. If not given, the function will try to
            determine the type from the input ID. If given, it must be one of
            'pmid', 'pmcid', or 'doi'.

        Returns
        -------
        dict
            A dictionary with keys 'pmid', 'pmcid', and 'doi' containing the
            corresponding IDs, or an empty dict if lookup fails.
        """
        if idtype is not None and idtype not in ("pmid", "pmcid", "doi"):
            raise ValueError(
                "Invalid idtype %s; must be 'pmid', 'pmcid', " "or 'doi'." % idtype
            )
        if paper_id.upper().startswith("PMC"):
            idtype = "pmcid"
        # Strip off any prefix
        if paper_id.upper().startswith("PMID"):
            paper_id = paper_id[4:]
        elif paper_id.upper().startswith("DOI"):
            paper_id = paper_id[3:]
        data = {"ids": paper_id}
        if idtype is not None:
            data["idtype"] = idtype
        try:
            tree = pubmed_client.send_request(self.pmid_convert_url, data)
        except Exception as e:
            logger.error("Error looking up PMID in PMC: %s" % e)
            return {}
        if tree is None:
            return {}
        record = tree.find("record")
        if record is None:
            return {}
        doi = record.attrib.get("doi")
        pmid = record.attrib.get("pmid")
        pmcid = record.attrib.get("pmcid")
        ids = {"doi": doi, "pmid": pmid, "pmcid": pmcid}
        return ids

    def get_ids(search_term, retmax=1000):
        return pubmed_client.get_ids(search_term, retmax=retmax, db='pmc')

    def extract_text(self, xml_string):
        """Get plaintext from the body of the given NLM XML string.
        This plaintext consists of all paragraphs returned by
        indra.literature.pmc_client.extract_paragraphs separated
        by newlines and then finally terminated by a newline.
        See the DocString of extract_paragraphs for more information.
        Parameters
        ----------
        xml_string : str
            String containing valid NLM XML.
        Returns
        -------
        str
            Extracted plaintext.
        """
        paragraphs = self.extract_paragraphs(xml_string)
        if paragraphs:
            return "\n".join(paragraphs) + "\n"
        else:
            return None

    def extract_paragraphs(self, xml_string):
        """Returns list of paragraphs in an NLM XML.
        This returns a list of the plaintexts for each paragraph and title in
        the input XML, excluding some paragraphs with text that should not
        be relevant to biomedical text processing.
        Relevant text includes titles, abstracts, and the contents of many body
        paragraphs. Within figures, tables, and floating elements, only captions
        are retained (One exception is that all paragraphs within floating
        boxed-text elements are retained. These elements often contain short
        summaries enriched with useful information.) Due to captions, nested
        paragraphs can appear in an NLM XML document. Occasionally there are
        multiple levels of nesting. If nested paragraphs appear in the input
        document their texts are returned in a pre-ordered traversal. The text
        within child paragraphs is not included in the output associated to the
        parent. Each parent appears in the output before its children. All children
        of an element appear before the elements following sibling.
        All tags are removed from each paragraph in the list that is returned.
        LaTeX surrounded by <tex-math> tags is removed entirely.
        Note: Some articles contain subarticles which are processed slightly
        differently from the article body. Only text from the body element
        of a subarticle is included, and all unwanted elements are excluded
        along with their captions. Boxed-text elements are excluded as well.
        Parameters
        ----------
        xml_string : str
            String containing valid NLM XML.
        Returns
        -------
        list of str
            List of extracted paragraphs from the input NLM XML
        """
        output = []
        tree = etree.fromstring(xml_string.encode("utf-8"))
        # Remove namespaces if any exist
        if tree.tag.startswith("{"):
            for element in tree.getiterator():
                # The following code will throw a ValueError for some
                # exceptional tags such as comments and processing instructions.
                # It's safe to just leave these tag names unchanged.
                try:
                    element.tag = etree.QName(element).localname
                except ValueError:
                    continue
            etree.cleanup_namespaces(tree)
        # Strip out latex
        self._remove_elements_by_tag(tree, "tex-math")
        # Strip out all content in unwanted elements except the captions
        self._replace_unwanted_elements_with_their_captions(tree)
        # First process front element. Titles alt-titles and abstracts
        # are pulled from here.
        front_elements = self._select_from_top_level(tree, "front")
        for element in front_elements:
            output.extend(self._extract_from_front(element))
        # All paragraphs except those in unwanted elements are extracted
        # from the article body
        body_elements = self._select_from_top_level(tree, "body")
        for element in body_elements:
            output.extend(self._extract_from_body(element))
        # Only the body sections of subarticles are processed. All
        # unwanted elements are removed entirely, including captions.
        # Even boxed-text elements are removed.
        subarticles = self._select_from_top_level(tree, "sub-article")
        for element in subarticles:
            output.extend(self._extract_from_subarticle(element))
        return output

    def _select_from_top_level(self, tree, tag):
        """Select direct children of the article element of a tree by tag.
        Different versions of NLM XML place the article element in different
        places. We cannot rely on a hard coded path to the article element.  This
        helper function helps select top level elements beneath article from their
        tag name. We use this to pull out the front, body, and sub-article elements
        of an article.
        An assumption is made that there is only one article element in the input
        XML tree. If this is not the case, only the firt article will be
        processed.
        Parameters
        ----------
        tree : :py:class:`lxml.etree._Element`
            lxml element for entire tree of a valid NLM XML
        tag : str
            Tag of top level elements to return
        Returns
        -------
        list
            List containing lxml Element objects of selected top level elements.
            Typically there is only one front and one body that are direct chilren
            of the article element, but there can be multiple subarticles.
        """
        if tree.tag == "article":
            article = tree
        else:
            article = tree.xpath(".//article")
            if not len(article):
                raise ValueError("Input XML contains no article element")
            # Assume there is only one article
            article = article[0]
        output = []
        xpath = "./%s" % tag
        for element in article.xpath(xpath):
            output.append(element)
        return output

    def _extract_from_front(self, front_element):
        """Return list of titles and paragraphs from front of NLM XML
        Parameters
        ----------
        front_element : :py:class:`lxml.etree._Element`
            etree element for front of a valid NLM XML
        Returns
        -------
        list of str
            List of relevant plain text titles and paragraphs taken from front
            section of NLM XML. These include the article title, alt title,
            and paragraphs within abstracts. Unwanted paragraphs such as
            author statements are excluded.
        """
        output = []
        title_xpath = "./article-meta/title-group/article-title"
        alt_title_xpath = "./article-meta/title-group/alt-title"
        abstracts_xpath = "./article-meta/abstract"
        for element in front_element.xpath(
            self._xpath_union(title_xpath, alt_title_xpath, abstracts_xpath)
        ):
            if element.tag == "abstract":
                # Extract paragraphs from abstracts
                output.extend(self._extract_paragraphs_from_tree(element))
            else:
                # No paragraphs in titles, Just strip tags
                output.append(" ".join(element.itertext()))
        return output

    def _extract_from_body(self, body_element):
        """Return list of paragraphs from main article body of NLM XML
        See DocString for extract_paragraphs for more info
        """
        return self._extract_paragraphs_from_tree(body_element)

    def _extract_from_subarticle(self, subarticle_element):
        """Return list of relevant paragraphs from a subarticle
        See DocString for extract_paragraphs for more info.
        """
        # Get only body element
        body = subarticle_element.xpath("./body")
        if not body:
            return []
        body = body[0]
        # Remove float elements. From observation these do not appear to
        # contain any meaningful information within sub-articles.
        for element in body.xpath(".//*[@position='float']"):
            element.getparent().remove(element)
        return self._extract_paragraphs_from_tree(body)

    def _remove_elements_by_tag(self, tree, *tags):
        """Remove elements with given tags
        Removes all element along with all of its content.
        Modifies input tree inplace
        Parameters
        ----------
        tree : :py:class:`lxml.etree._Element`
            etree element for valid NLM XML
        """
        bad_xpath = self._xpath_union(*[".//%s" % tag for tag in tags])
        for element in tree.xpath(bad_xpath):
            element.getparent().remove(element)

    def _replace_unwanted_elements_with_their_captions(self, tree):
        """Replace all unwanted elements with their captions
        Modifies input tree inplace.
        Parameters
        ----------
        tree : :py:class:`lxml.etree._Element`
            etree element for valid NLM XML
        """
        floats_xpath = "//*[@position='float']"
        figs_xpath = ".//fig"
        tables_xpath = ".//table-wrap"
        unwanted_xpath = self._xpath_union(floats_xpath, figs_xpath, tables_xpath)
        unwanted = tree.xpath(unwanted_xpath)
        # Iterating through xpath nodes in reverse leads to processing these
        # nodes from bottom up.
        for element in unwanted[::-1]:
            # Don't remove floats that are boxed-text elements. These often contain
            # useful information
            if element.tag == "boxed-text":
                continue
            captions = element.xpath("./caption")
            captions_element = etree.Element("captions")
            for caption in captions:
                captions_element.append(caption)
            element.getparent().replace(element, captions_element)

    def _retain_only_pars(self, tree):
        """Strip out all tags except title and p tags
        Function also changes title tags into p tags. This is a helpful
        preprocessing step that makes it easier to extract paragraphs in
        the order of a pre-ordered traversal.
        Modifies input tree inplace.
        Parameters
        ----------
        tree : :py:class:`lxml.etree._Element`
            etree element for valid NLM XML
        """
        for element in tree.xpath(".//*"):
            if element.tag == "title":
                element.tag = "p"
        for element in tree.xpath(".//*"):
            parent = element.getparent()
            if parent is not None and element.tag != "p":
                etree.strip_tags(element.getparent(), element.tag)

    def _pull_nested_paragraphs_to_top(self, tree):
        """Flatten nested paragraphs in pre-ordered traversal
        Requires _retain_only_pars to be run first.
        Modifies input tree inplace.
        Parameters
        ----------
        tree : :py:class:`lxml.etree._Element`
            etree element for valid NLM XML
        """
        # Since _retain_only_pars must be called first, input will contain only p
        # tags except for possibly the outer most tag. p elements directly beneath
        # the root will be called depth 1, those beneath depth 1 elements will be
        # called depth 2 and so on.  Proceed iteratively. At each step identify all
        # p elements with depth 2.  Cut all of the depth 2 p elements out of each
        # parent and append them in order as siblings following the parent (these
        # depth 2 elements may themselves be the parents of additional p elements).
        # The algorithm terminates when there are no depth 2 elements remaining.
        # Find depth 2 p elements
        nested_paragraphs = tree.xpath("./p/p")
        while nested_paragraphs:
            # This points to the location where the next depth 2 p element will
            # be appended
            last = None
            # Store parent of previously processed element to track when parent
            # changes.
            old_parent = None
            for p in nested_paragraphs:
                parent = p.getparent()
                # When the parent changes last must be set to the new parent
                # element. This ensures children will be appended in order
                # after there parents.
                if parent != old_parent:
                    last = parent
                # Remove child element from its parent
                parent.remove(p)
                # The parents text occuring after the current child p but before
                # p's following sibling is stored in p.tail. Append this text to
                # the parent's text and then clear out p.tail
                if not parent.text and p.tail:
                    parent.text = p.tail
                    p.tail = ""
                elif parent.text and p.tail:
                    parent.text += " " + p.tail
                    p.tail = ""
                # Place child in its new location
                last.addnext(p)
                last = p
            nested_paragraphs = tree.xpath("./p/p")

    def _extract_paragraphs_from_tree(self, tree):
        """Preprocess tree and return it's paragraphs."""
        self._retain_only_pars(tree)
        self._pull_nested_paragraphs_to_top(tree)
        paragraphs = []
        for element in tree.xpath("./p"):
            paragraph = "".join([x.strip() for x in element.itertext()])
            paragraphs.append(paragraph)
        return paragraphs

    def _xpath_union(self, *xpath_list):
        """Form union of xpath expressions"""
        return " | ".join(xpath_list)

    def get_title(self, pmcid):
        xml_string = self.get_xml(pmcid)
        if not xml_string:
            return
        tree = etree.fromstring(xml_string.encode("utf-8"))
        # Remove namespaces if any exist
        if tree.tag.startswith("{"):
            for element in tree.getiterator():
                # The following code will throw a ValueError for some
                # exceptional tags such as comments and processing instructions.
                # It's safe to just leave these tag names unchanged.
                try:
                    element.tag = etree.QName(element).localname
                except ValueError:
                    continue
            etree.cleanup_namespaces(tree)
        # Strip out latex
        self._remove_elements_by_tag(tree, "tex-math")
        # Strip out all content in unwanted elements except the captions
        self._replace_unwanted_elements_with_their_captions(tree)
        # First process front element. Titles alt-titles and abstracts
        # are pulled from here.
        front_elements = self._select_from_top_level(tree, "front")
        title_xpath = "./article-meta/title-group/article-title"
        for front_element in front_elements:
            for element in front_element.xpath(title_xpath):
                return " ".join(element.itertext())

    def download_pmc_s3(
        self,
        pmc_id,
        file_type="xml",
        output_dir="pmc",
        cache_dir="pmc",
        bucket_name="pmc-oa-opendata",
    ):
        """
        Download PMC files from AWS S3
        :param str pmc_id: PMC ID
        :param str file_type: File type (xml or txt). Default is 'xml'
        :param str output_dir: Output directory. Default is 'pmc'
        :param str cache_dir: Cache directory. Default is 'pmc'
        :param str bucket_name: S3 bucket name. Default is 'pmc-oa-opendata'
        :return: None
        >>> download_pmc_s3('PMC3898398')
        """

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{pmc_id}.{file_type}")
        cache_path = os.path.join(cache_dir, f"{pmc_id}.{file_type}")

        if not os.path.exists(output_path):
            if os.path.exists(cache_path):
                shutil.copy(cache_path, output_path)
            else:
                logger.info(
                    f"Attempting to download {pmc_id}.{file_type} to {output_path}"
                )

                try:
                    file_key = f"oa_comm/{file_type}/all/{pmc_id}.{file_type}"
                    s3.download_file(bucket_name, file_key, cache_path)
                    shutil.copy(cache_path, output_path)
                except Exception as e:
                    try:
                        file_key = f"oa_noncomm/{file_type}/all/{pmc_id}.{file_type}"
                        s3.download_file(bucket_name, file_key, cache_path)
                        shutil.copy(cache_path, output_path)
                    except Exception as e:
                        try:
                            file_key = f"author_manuscript/{file_type}/all/{pmc_id}.{file_type}"
                            s3.download_file(bucket_name, file_key, cache_path)
                            shutil.copy(cache_path, output_path)
                        except Exception as e:
                            if not os.path.exists(cache_path):
                                logger.error(e)

        if os.path.exists(cache_path):
            logger.info(f"DONE: File: {output_path}")
