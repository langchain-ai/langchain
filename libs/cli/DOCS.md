# `langchain`

**Usage**:

```console
$ langchain [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `hub`: Manage installable hub packages.
* `serve`: Manage LangServe application projects.
* `start`: Start the LangServe instance, whether it's...

## `langchain hub`

Manage installable hub packages.

**Usage**:

```console
$ langchain hub [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `new`: Creates a new hub package.
* `start`: Starts a demo LangServe instance for this...

### `langchain hub new`

Creates a new hub package.

**Usage**:

```console
$ langchain hub new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--with-poetry / --no-poetry`: Don't run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain hub start`

Starts a demo LangServe instance for this hub package.

**Usage**:

```console
$ langchain hub start [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.

## `langchain serve`

Manage LangServe application projects.

**Usage**:

```console
$ langchain serve [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add`: Adds the specified package to the current...
* `install`
* `list`: Lists all packages in the current...
* `new`: Create a new LangServe application.
* `remove`: Removes the specified package from the...
* `start`: Starts the LangServe instance.

### `langchain serve add`

Adds the specified package to the current LangServe instance.

e.g.:
langchain serve add extraction-openai-functions
langchain serve add git+ssh://git@github.com/efriis/simple-pirate.git

**Usage**:

```console
$ langchain serve add [OPTIONS] [DEPENDENCIES]...
```

**Arguments**:

* `[DEPENDENCIES]...`: The dependency to add

**Options**:

* `--api-path TEXT`: API paths to add
* `--project-dir PATH`: The project directory
* `--repo TEXT`: Install deps from a specific github repo instead
* `--branch TEXT`: Install deps from a specific branch
* `--with-poetry / --no-poetry`: Run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain serve install`

**Usage**:

```console
$ langchain serve install [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `langchain serve list`

Lists all packages in the current LangServe instance.

**Usage**:

```console
$ langchain serve list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `langchain serve new`

Create a new LangServe application.

**Usage**:

```console
$ langchain serve new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--package TEXT`: Packages to seed the project with
* `--with-poetry / --no-poetry`: Run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain serve remove`

Removes the specified package from the current LangServe instance.

**Usage**:

```console
$ langchain serve remove [OPTIONS] API_PATHS...
```

**Arguments**:

* `API_PATHS...`: The API paths to remove  [required]

**Options**:

* `--with_poetry / --no-poetry`: Don't run poetry remove  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain serve start`

Starts the LangServe instance.

**Usage**:

```console
$ langchain serve start [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--app TEXT`: The app to run
* `--help`: Show this message and exit.

## `langchain start`

Start the LangServe instance, whether it's a hub package or a serve project.

**Usage**:

```console
$ langchain start [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.
