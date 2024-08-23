from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.type_util import SupportsStr


class ChildType(Enum):
    """Enumerator of the child type."""

    MARKDOWN = "MARKDOWN"
    EXCEPTION = "EXCEPTION"


class ChildRecord(NamedTuple):
    """Child record as a NamedTuple."""

    type: ChildType
    kwargs: Dict[str, Any]
    dg: DeltaGenerator


class MutableExpander:
    """Streamlit expander that can be renamed and dynamically expanded/collapsed."""

    def __init__(self, parent_container: DeltaGenerator, label: str, expanded: bool):
        """Create a new MutableExpander.

        Parameters
        ----------
        parent_container
            The `st.container` that the expander will be created inside.

            The expander transparently deletes and recreates its underlying
            `st.expander` instance when its label changes, and it uses
            `parent_container` to ensure it recreates this underlying expander in the
            same location onscreen.
        label
            The expander's initial label.
        expanded
            The expander's initial `expanded` value.
        """
        self._label = label
        self._expanded = expanded
        self._parent_cursor = parent_container.empty()
        self._container = self._parent_cursor.expander(label, expanded)
        self._child_records: List[ChildRecord] = []

    @property
    def label(self) -> str:
        """Expander's label string."""
        return self._label

    @property
    def expanded(self) -> bool:
        """True if the expander was created with `expanded=True`."""
        return self._expanded

    def clear(self) -> None:
        """Remove the container and its contents entirely. A cleared container can't
        be reused.
        """
        self._container = self._parent_cursor.empty()
        self._child_records.clear()

    def append_copy(self, other: MutableExpander) -> None:
        """Append a copy of another MutableExpander's children to this
        MutableExpander.
        """
        other_records = other._child_records.copy()
        for record in other_records:
            self._create_child(record.type, record.kwargs)

    def update(
        self, *, new_label: Optional[str] = None, new_expanded: Optional[bool] = None
    ) -> None:
        """Change the expander's label and expanded state"""
        if new_label is None:
            new_label = self._label
        if new_expanded is None:
            new_expanded = self._expanded

        if self._label == new_label and self._expanded == new_expanded:
            # No change!
            return

        self._label = new_label
        self._expanded = new_expanded
        self._container = self._parent_cursor.expander(new_label, new_expanded)

        prev_records = self._child_records
        self._child_records = []

        # Replay all children into the new container
        for record in prev_records:
            self._create_child(record.type, record.kwargs)

    def markdown(
        self,
        body: SupportsStr,
        unsafe_allow_html: bool = False,
        *,
        help: Optional[str] = None,
        index: Optional[int] = None,
    ) -> int:
        """Add a Markdown element to the container and return its index."""
        kwargs = {"body": body, "unsafe_allow_html": unsafe_allow_html, "help": help}
        new_dg = self._get_dg(index).markdown(**kwargs)
        record = ChildRecord(ChildType.MARKDOWN, kwargs, new_dg)
        return self._add_record(record, index)

    def exception(
        self, exception: BaseException, *, index: Optional[int] = None
    ) -> int:
        """Add an Exception element to the container and return its index."""
        kwargs = {"exception": exception}
        new_dg = self._get_dg(index).exception(**kwargs)
        record = ChildRecord(ChildType.EXCEPTION, kwargs, new_dg)
        return self._add_record(record, index)

    def _create_child(self, type: ChildType, kwargs: Dict[str, Any]) -> None:
        """Create a new child with the given params"""
        if type == ChildType.MARKDOWN:
            self.markdown(**kwargs)
        elif type == ChildType.EXCEPTION:
            self.exception(**kwargs)
        else:
            raise RuntimeError(f"Unexpected child type {type}")

    def _add_record(self, record: ChildRecord, index: Optional[int]) -> int:
        """Add a ChildRecord to self._children. If `index` is specified, replace
        the existing record at that index. Otherwise, append the record to the
        end of the list.

        Return the index of the added record.
        """
        if index is not None:
            # Replace existing child
            self._child_records[index] = record
            return index

        # Append new child
        self._child_records.append(record)
        return len(self._child_records) - 1

    def _get_dg(self, index: Optional[int]) -> DeltaGenerator:
        if index is not None:
            # Existing index: reuse child's DeltaGenerator
            assert 0 <= index < len(self._child_records), f"Bad index: {index}"
            return self._child_records[index].dg

        # No index: use container's DeltaGenerator
        return self._container
