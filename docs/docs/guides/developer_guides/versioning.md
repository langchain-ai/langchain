# Versioning

Here we discuss how the various `langchain` packages are versioned.

## `langchain-core`

`langchain-core` is currently on version `0.1.x`. 

As `langchain-core` contains the base abstractions and runtime for the whole LangChain ecosystem, we will communicate any breaking changes with advance notice and version bumps. The exception for this is anything in `langchain_core.beta`. The reason for `langchain_core.beta` is that given the rate of change of the field, being able to move quickly is still a priority, and this module is our attempt to do so.

Minor version increases will occur for:

- Breaking changes for any public interfaces NOT in `langchain_core.beta`

Patch version increases will occur for:

- Bug fixes
- New features
- Any changes to private interfaces
- Any changes to `langchain_core.beta`

## `langchain`

`langchain` is currently on version `0.0.x`

All changes will be accompanied by a patch version increase. Any changes to public interfaces are nearly always done in a backwards compatible way.

We are targeting January 2024 for a release of `langchain` v0.1, at which point `langchain` will adopt the same versioning policy as `langchain-core`.

## `langchain-community`

`langchain-community` is currently on version `0.0.x`

All changes will be accompanied by a patch version increase.

## `langchain-experimental`

`langchain-experimental` is currently on version `0.0.x`

All changes will be accompanied by a patch version increase.

## `langchain-{integration}`

Starting December 2023, we will split out several integrations into standalone packages. Those packages will likely follow [semantic versioning](https://semver.org/).