Example Docs
------------

The sample docs directory contains the following files:

-  `example-10k.html` - A 10-K SEC filing in HTML format
-  `layout-parser-paper.pdf` - A PDF copy of the layout parser paper
-  `factbook.xml`/`factbook.xsl` - Example XML/XLS files that you
   can use to test stylesheets

These documents can be used to test out the parsers in the library. In
addition, here are instructions for pulling in some sample docs that are
too big to store in the repo.

XBRL 10-K
^^^^^^^^^

You can get an example 10-K in inline XBRL format using the following
`curl`. Note, you need to have the user agent set in the header or the
SEC site will reject your request.

.. code:: bash

   curl -O \
     -A '${organization} ${email}'
     https://www.sec.gov/Archives/edgar/data/311094/000117184321001344/0001171843-21-001344.txt

You can parse this document using the HTML parser.
