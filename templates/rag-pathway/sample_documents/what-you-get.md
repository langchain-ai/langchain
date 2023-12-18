---
title: What you get
description: 'What to expect when using Pathway: experience and performance'
---

# What you get with Pathway

## 1. Python + Rust: the best of both worlds

Pathway provides a Python interface and experience created with data developers in mind. You can easily build pipelines by manipulating Pathway tables and rely on the vast resources and libraries of the Python ecosystem. Also, Pathway can seamlessly be integrated into your CI/CD chain as it is inherently compatible with popular tools such as mypy or pytest.
Your Pathway pipelines can be automatically tested, built, and deployed, like any other Python workflow.

Pathway also relies on a powerful Rust engine to ensure high performance for your pipelines, no matter if you are dealing with batch or streaming data.
Pathway engine makes the utmost of Rust speed and memory safety to provide efficient parallel and distributed processing without being limited by Python's [GIL](https://en.wikipedia.org/w/index.php?title=Global_interpreter_lock&oldid=1144836295).

Pathway takes the best of both worlds and efficiently associates the convenience of Python with the power of Rust.


## 2. Incremental computation

Pathway's engine incrementally processes data updates. This means that the minimum work needed by any algorithm or transformation is performed to refresh its results when fresh data arrives.



## 3. An ML-friendly code life cycle

### Writing your code

As a Pathway user, you write code in Python, with Pathway imported as a Python module.
Pathway provides a cookiecutter template in https://github.com/pathwaycom/cookiecutter-pathway to help start Pathway projects.

Data manipulation syntax is built around a Table API, which closely resembles the DataFrame API of PySpark which in turn resembles DataFrames in pandas.

The same code developed with the Pathway module can be used for batch and streaming use cases, depending only on data connector settings and method of launching.
For many operations, Pathway returns exactly the same results when run in batch and streaming modes.
Exceptions are out-of-order data for which streaming mode may optionally ignore and functions that explicitly depend on processing time.

### Running code and prototyping

During rapid prototyping, Pathway code can be written and tested interactively, without waiting for compilation or deployment, with the computation graph being built in the background.
This is helpful for instance during data science work in Jupyter notebooks.

The developed code prototype can then run on streaming data sources.
The computation graph is handed down to the runtime engine when executing the line `pathway.run()`.

In terms of the interplay between interactive usability and launching compute graphs, Pathway takes direct inspiration from TensorFlow and PyTorch: just like TensorFlow, we explicitly represent computations as graphs that are executed with a `run()` command, however, similarly to PyTorch, we aim to offer the best in class interactive work environment for hands-on, data-driven algorithm design.\
\
(Footnote: Our advisor and Business Angel, Lukasz Kaiser, was a co-author of TensorFlow, and our CTO, Jan Chorowski, was a contributor to Theano).

Pathway calls into the same Rust runtime in any deployment - on either streaming data sources or in interactive mode.

## 4. Consistency of results

Pathway computes consistent results: each produced output is computed based on data contained in prefixes of the input data streams.
For most of the operations, exactly the same outputs are produced:

1. If the input data was sent to the engine all at once in batch mode.
2. If the inputs were sent in several smaller batches, or in streaming mode.

This equivalence of stream and batch processing facilitates easy development of data processing logic: one can reason in terms of steady-state and consistent results and not worry about all intermediate and wrong states through which an eventually consistent system may pass.

Consider a Kafka topic containing messages, each holding a list (pack) of several events that co-occurred together in a banking system.
Events indicate that an account was debited or credited money.
Clearly, a transaction touches two accounts and consists of two events that must be processed atomically (either both or none at all), or some outputs, e.g., account balances, will be inconsistent.
Kafka is atomic with respect to individual messages. Thus, this event-pack design ensures atomic delivery of all events that form a transaction.
However, processing event-packs is complex, and logic is much easier if the event-pack stream is unpacked into a single message stream (e.g., we can then group events by account and compute the balances).

Since Kafka only guarantees the atomic processing of single messages, consistency is lost once the event-packs are flattened into a single-even stream.
On the other hand, Pathway guarantees consistency.
If a Pathway computation unpacks the event-pack into individual events, all messages that form a transaction will be consistently processed, ensuring that every produced output depends on either all or no events that were grouped together into a single event-pack.

## 5. Containerized deployments

Pathway is meant to be deployed in a containerized manner.

Single-machine deployments can easily be achieved using Docker.
The deployment can run concurrently on multiple cores using multiple processes or threads.

We provide a pathway spawn command to aid in launching multi-process and multi-threaded jobs.

The choice between threads and multiple processes depends on the nature of the computation.
While communication between threads is faster, Python-heavy workloads may require multiprocess parallelism to bypass the GIL.
We are eagerly awaiting the GIL-less Python era!

For using Pathway on large workloads beyond a single machine, see Distributed deployment.

## 6. Easy testing and CI/CD

Pathway tests on offline data snippets can be run locally in any CI/CD pipeline with Python.
Tests can cover the handling of temporal (late, out of order) aspects of data by comparing results on multiple revisions.
Pathway supports several session-replay mechanisms, such as the demo API.
These allow recreating streaming scenarios predictably within standard CI/CD pipelines (Jenkins, GitHub Actions, etc.)
