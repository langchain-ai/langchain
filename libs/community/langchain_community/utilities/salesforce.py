"""Salesforce wrapper around simple-salesforce."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from simple_salesforce import Salesforce

__all__ = ["SalesforceAPIWrapper"]


class SalesforceAPIWrapper:
    """Wrapper around Salesforce Simple-Salesforce API.

    To use, you should have the ``simple-salesforce`` python package installed.
    You can install it with ``pip install simple-salesforce``.
    """

    def __init__(self, salesforce_instance: "Salesforce") -> None:
        """Initialize the Salesforce wrapper.

        Args:
            salesforce_instance: An existing simple-salesforce instance.
        """
        try:
            from simple_salesforce import Salesforce
        except ImportError:
            # Allow instantiation if the provided instance has the required attributes (for testing purposes)
            if hasattr(salesforce_instance, "query") and hasattr(salesforce_instance, "describe"):
                pass
            else:
                raise ImportError("Please install simple-salesforce to use SalesforceAPIWrapper")
        self.sf = salesforce_instance

    def run(
        self, command: str, include_metadata: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Execute a SOQL query and return the results.

        Args:
            command: The SOQL query to execute.
            include_metadata: Whether to include Salesforce metadata in results.

        Returns:
            The query results either as a string or raw dict if include_metadata=True.
        """
        try:
            results = self.sf.query(command)
            if include_metadata:
                return results
            return self._format_results(results)
        except Exception as e:
            raise ValueError(f"Invalid SOQL query: {str(e)}")

    def run_no_throw(
        self, command: str, include_metadata: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Execute a SOQL query and return results, returning empty on failure.

        Args:
            command: The SOQL query to execute.
            include_metadata: Whether to include Salesforce metadata in results.

        Returns:
            The query results either as a string or raw dict if include_metadata=True.
            Returns empty string/dict on failure.
        """
        try:
            return self.run(command, include_metadata=include_metadata)
        except Exception as e:
            if include_metadata:
                return {}
            return f"Error: {str(e)}"

    def get_usable_object_names(self) -> List[str]:
        """Get names of queryable Salesforce objects."""
        try:
            objects = self.sf.describe()["sobjects"]
            return [obj["name"] for obj in objects if obj.get("queryable")]
        except Exception as e:
            raise ValueError(f"Error fetching Salesforce objects: {str(e)}")

    def get_object_info(self, object_names: Optional[List[str]] = None) -> str:
        """Get information about specified Salesforce objects.

        Args:
            object_names: List of object names to get info for. If None, gets all\
                queryable objects.

        Returns:
            Formatted string containing object information.
        """
        try:
            all_objects = self.get_usable_object_names()
            if object_names:
                invalid_objects = set(object_names) - set(all_objects)
                if invalid_objects:
                    raise ValueError(f"Invalid object names: {invalid_objects}")
                objects_to_describe = object_names
            else:
                objects_to_describe = all_objects

            output = []
            for obj_name in sorted(objects_to_describe):
                schema = getattr(self.sf, obj_name).describe()
                output.append(self._format_object_schema(schema))

            return "\n\n".join(output)
        except Exception as e:
            raise ValueError(f"Error getting object info: {str(e)}")

    def get_object_info_no_throw(self, object_names: Optional[List[str]] = None) -> str:
        """Get information about specified objects.

        Returns error message on failure.
        """
        try:
            return self.get_object_info(object_names)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_context(self) -> Dict[str, Any]:
        """Return context about the Salesforce instance for use in prompts."""
        object_names = self.get_usable_object_names()
        object_info = self.get_object_info_no_throw()
        return {
            "object_names": ", ".join(object_names),
            "object_info": object_info,
        }

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format query results into a readable string."""
        if not results.get("records"):
            return "No records found."

        records = results["records"]
        total_size = results.get("totalSize", len(records))

        output = f"Found {total_size} record(s):\n"
        for record in records:
            # Remove attributes dictionary that contains metadata
            record_copy = record.copy()
            record_copy.pop("attributes", None)
            output += f"\n{record_copy}"

        return output

    def _format_object_schema(self, schema: Dict[str, Any]) -> str:
        """Format object schema into a readable string."""
        output = [f"Object: {schema.get('name')} ({schema.get('label')})"]
        output.append("\nFields:")

        for field in schema.get("fields", []):
            output.extend(
                [
                    f"\n- {field['name']} ({field['type']})",
                    f"  Label: {field['label']}",
                    f"  Required: {not field['nillable']}",
                    f"  Description: {field.get('description', 'N/A')}",
                ]
            )

        return "\n".join(output)
