from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_core.documents import Document

html_string = '''
<!DOCTYPE html>
            <h1>Section 1</h1>
            <p>This section contains an important table and list that should not be split across chunks.</p>
            <div>
                <table>
                    <tr>
                        <th>Item</th>
                        <th>Quantity</th>
                        <th>Price</th>
                    </tr>
                    <tr>
                        <td>Apples</td>
                        <td>10</td>
                        <td>$1.00</td>
                    </tr>
                    <tr>
                        <td>Oranges</td>
                        <td>5</td>
                        <td>$0.50</td>
                    </tr>
                </table>
            </div>
            <p>Here is a detailed list:</p>
            <ul>
                <li>Item 1: Description of item 1, which is quite detailed and important.</li>
                <li>Item 2: Description of item 2, which also contains significant information.</li>
                <li>Item 3: Description of item 3, another item that we don't want to split across chunks.</li>
            </ul>
            <h2>Subsection 1.1</h2>
            <p>Additional text in subsection 1.1 that is separated from the table and list.</p>


'''

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2")
]

splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    max_chunk_size=50,
    elements_to_preserve=["table", "ul"]
)

documents = splitter.split_text(html_string)

print(documents)