# `langchain`

**Usage**:

```console
$ langchain [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.
* `-v, --version`: Print current CLI version.

**Commands**:

* `app`: Manage LangChain apps
* `serve`: Start the LangServe app, whether it's a...
* `template`: Develop installable templates.

## `langchain app`

Manage LangChain apps

**Usage**:

```console
$ gigachain app [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `add`: Adds the specified template to the current...
* `new`: Create a new LangServe application.
* `remove`: Removes the specified package from the...
* `serve`: Starts the LangServe app.

### `gigachain app add`

Adds the specified template to the current LangServe app.

e.g.:
gigachain app add extraction-openai-functions
gigachain app add git+ssh://git@github.com/efriis/simple-pirate.git

**Usage**:

```console
$ gigachain app add [OPTIONS] [DEPENDENCIES]...
```

**Arguments**:

* `[DEPENDENCIES]...`: The dependency to add

**Options**:

* `--api-path TEXT`: API paths to add
* `--project-dir PATH`: The project directory
* `--repo TEXT`: Install templates from a specific github repo instead
* `--branch TEXT`: Install templates from a specific branch
* `--help`: Show this message and exit.

### `gigachain app new`

Create a new LangServe application.

**Usage**:

```console
$ gigachain app new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--package TEXT`: Packages to seed the project with
* `--help`: Show this message and exit.

### `gigachain app remove`

Removes the specified package from the current LangServe app.

**Usage**:

```console
$ gigachain app remove [OPTIONS] API_PATHS...
```

**Arguments**:

* `API_PATHS...`: The API paths to remove  [required]

**Options**:

* `--help`: Show this message and exit.

### `gigachain app serve`

Starts the LangServe app.

**Usage**:

```console
$ gigachain app serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--app TEXT`: The app to run, e.g. `app.server:app`
* `--help`: Show this message and exit.

## `gigachain serve`

Start the LangServe app, whether it's a template or an app.

**Usage**:

```console
$ gigachain serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.

## `gigachain template`

Develop installable templates.

**Usage**:

```console
$ gigachain template [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `new`: Creates a new template package.
* `serve`: Starts a demo app for this template.

### `gigachain template new`

Creates a new template package.

**Usage**:

```console
$ gigachain template new [OPTIONS] NAME
```

**Arguments**:

* `NAME`: The name of the folder to create  [required]

**Options**:

* `--with-poetry / --no-poetry`: Don't run poetry install  [default: no-poetry]
* `--help`: Show this message and exit.

### `gigachain template serve`

Starts a demo app for this template.

**Usage**:

```console
$ gigachain template serve [OPTIONS]
```

**Options**:

* `--port INTEGER`: The port to run the server on
* `--host TEXT`: The host to run the server on
* `--help`: Show this message and exit.
