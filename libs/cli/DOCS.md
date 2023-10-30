# `langchain`

**Usage**:

```console
$ langchain [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `app`: Manage LangServe application projects.
* `serve`: Start the LangServe instance, whether it's...
* `template`: Develop installable templates.

## `langchain app`

Manage LangServe application projects.

**Usage**:

```console
$ langchain app [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add`: Adds the specified package to the current...
* `install`
* `list`: Lists all packages in the current...
* `new`: Create a new LangServe application.
* `remove`: Removes the specified package from the...
* `serve`: Starts the LangServe instance.

### `langchain app add`

Adds the specified package to the current LangServe instance.

e.g.:
langchain serve add extraction-openai-functions
langchain serve add git+ssh://git@github.com/efriis/simple-pirate.git

**Usage**:

```console
$ langchain app add [OPTIONS] [DEPENDENCIES]...
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

### `langchain app install`

**Usage**:

```console
$ langchain app install [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `langchain app list`

Lists all packages in the current LangServe instance.

**Usage**:

```console
$ langchain app list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `langchain app new`

Create a new LangServe application.

**Usage**:

```console
$ langchain app new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--package TEXT`: Packages to seed the project with
* `--with-poetry / --no-poetry`: Run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain app remove`

Removes the specified package from the current LangServe instance.

**Usage**:

```console
$ langchain app remove [OPTIONS] API_PATHS...
```

**Arguments**:

* `API_PATHS...`: The API paths to remove  [required]

**Options**:

* `--with_poetry / --no-poetry`: Don't run poetry remove  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain app serve`

Starts the LangServe instance.

**Usage**:

```console
$ langchain app serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--app TEXT`: The app to run, e.g. `app.server:app`
* `--help`: Show this message and exit.

## `langchain serve`

Start the LangServe instance, whether it's a template package or an app.

**Usage**:

```console
$ langchain serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.

## `langchain template`

Develop installable templates.

**Usage**:

```console
$ langchain template [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `new`: Creates a new template package.
* `serve`: Starts a demo LangServe instance for this...

### `langchain template new`

Creates a new template package.

**Usage**:

```console
$ langchain template new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--with-poetry / --no-poetry`: Don't run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `langchain template serve`

Starts a demo LangServe instance for this template.

**Usage**:

```console
$ langchain template serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.
