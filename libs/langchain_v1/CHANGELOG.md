# Changelog

All notable changes to this package are documented here (see also [GitHub releases](https://github.com/langchain-ai/langchain/releases?q=tag%3A%22langchain%3D%3D1%22)).

## Unreleased

### Added

- `write_todos` / `TodoListMiddleware`: optional per-item `id` and `depends_on` (list of ids). When any item uses non-empty `depends_on`, every item in the update must have a unique `id`; dependencies must form a DAG; statuses must respect completion order (a task cannot be `in_progress` or `completed` until its dependencies are `completed`). Flat lists without dependencies behave as before and do not require `id`.
