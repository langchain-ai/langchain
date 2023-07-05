# Airtable

>[Airtable](https://en.wikipedia.org/wiki/Airtable) is a cloud collaboration service.
`Airtable` is a spreadsheet-database hybrid, with the features of a database but applied to a spreadsheet. 
> The fields in an Airtable table are similar to cells in a spreadsheet, but have types such as 'checkbox', 
> 'phone number', and 'drop-down list', and can reference file attachments like images.

>Users can create a database, set up column types, add records, link tables to one another, collaborate, sort records
> and publish views to external websites.

## Installation and Setup

```bash
pip install pyairtable
```

* Get your [API key](https://support.airtable.com/docs/creating-and-using-api-keys-and-access-tokens).
* Get the [ID of your base](https://airtable.com/developers/web/api/introduction).
* Get the [table ID from the table url](https://www.highviewapps.com/kb/where-can-i-find-the-airtable-base-id-and-table-id/#:~:text=Both%20the%20Airtable%20Base%20ID,URL%20that%20begins%20with%20tbl).

## Document Loader


```python
from langchain.document_loaders import AirtableLoader
```

See an [example](/docs/modules/data_connection/document_loaders/integrations/airtable.html).
