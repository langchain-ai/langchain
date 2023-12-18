Pathway is an open framework for high-throughput and low-latency real-time data processing. It is used to create Python code which seamlessly combines batch processing, streaming, and real-time API's for LLM apps. Pathway's distributed runtime (🦀-🐍) provides fresh results of your data pipelines whenever new inputs and requests are received.

In the first place, Pathway was designed to be a life-saver (or at least a time-saver) for Python developers and ML/AI engineers faced with live data sources, where you need to react quickly to fresh data. Still, Pathway is a powerful tool that can be used for a lot of things. If you want to do streaming in Python, build an AI data pipeline, or if you are looking for your next Python data processing framework, keep reading.

Pathway provides a high-level programming interface in Python for defining data transformations, aggregations, and other operations on data streams.
With Pathway, you can effortlessly design and deploy sophisticated data workflows that efficiently handle high volumes of data in real time.

Pathway is interoperable with various data sources and sinks such as Kafka, CSV files, SQL/noSQL databases, and REST API's, allowing you to connect and process data from different storage systems.

Typical use-cases of Pathway include realtime data processing, ETL (Extract, Transform, Load) pipelines, data analytics, monitoring, anomaly detection, and recommendation. Pathway can also independently provide the backbone of a light LLMops stack for real-time LLM applications.

In Pathway, data is represented in the form of Tables. Live data streams are also treated as Tables. The library provides a rich set of operations like filtering, joining, grouping, and windowing.

For any questions, you will find the community and team behind the project on Discord.

Screencast animation of converting batch code to streaming by changing one keyword argument in the script.

## Getting started


### Installation

Pathway requires Python 3.10 or above.

You can install the current release of Pathway using `pip`:

```
$ pip install -U pathway
```

⚠️ Pathway is available on MacOS and Linux. Users of other systems should run Pathway on a Virtual Machine.

### Running Pathway locally

To use Pathway, you only need to import it:

```python
import pathway as pw
```

Now, you can easily create your processing pipeline, and let Pathway handle the updates. Once your pipeline is created, you can launch the computation on streaming data with a one-line command:

```python
pw.run()
```

You can then run your Pathway project (say, `main.py`) just like a normal Python script: `$ python main.py`. Alternatively, use the pathway'ish version:

```
$ pathway spawn python main.py
```

Pathway natively supports multithreading.
To launch your application with 3 threads, you can do as follows:
```
$ pathway spawn --threads 3 python main.py
```

To jumpstart a Pathway project, you can use our cookiecutter template


### Example

```python
import pathway as pw

# Using the `demo` module to create a data stream
table = pw.demo.range_stream(nb_rows=50)
# Storing the stream into a CSV file
pw.io.csv.write(table, "output_table.csv")

# Summing all the values in a new table
sum_table = table.reduce(sum=pw.reducers.sum(pw.this.value))
# Storing the sum (which is a stream) in another CSV file
pw.io.csv.write(sum_table, "sum_table.csv")

# Now that the pipeline is built, the computation is started
pw.run()
```

Run this example in Google Colab

## Deployment

Do you feel limited by a local run?
If you want to scale your Pathway application, you may be interested in our Pathway for Enterprise.
Pathway for Enterprise is specially tailored towards end-to-end data processing and real time intelligent analytics.
It scales using distributed computing on the cloud and supports Kubernetes deployment.

You can learn more about the features of Pathway for Enterprise on our website.

If you are interested, don't hesitate to contact us to learn more.

## Monitoring Pathway

Pathway comes with a monitoring dashboard that allows you to keep track of the number of messages sent by each connector and the latency of the system. The dashboard also includes log messages. 

This dashboard is enabled by default; you can disable it by passing `monitoring_level = pathway.MonitoringLevel.NONE` to `pathway.run()`.

In addition to Pathway's built-in dashboard, you can use Prometheus to monitor your Pathway application.

## Resources

See also: Pathway Documentation (https://pathway.com/developers/) webpage (including API Docs).

### Videos about Pathway<a id="videos-about-pathway"></a>
[▶️ Building an LLM Application without a vector database](https://www.youtube.com/watch?v=kcrJSk00duw) - by [Jan Chorowski](https://scholar.google.com/citations?user=Yc94070AAAAJ) (7min 56s)

[▶️ Linear regression on a Kafka Stream](https://vimeo.com/805069039) - by [Richard Pelgrim](https://twitter.com/richardpelgrim) (7min 53s)

[▶️ Introduction to reactive data processing](https://pathway.com/developers/user-guide/concepts/welcome) - by [Adrian Kosowski](https://scholar.google.com/citations?user=om8De_0AAAAJ) (27min 54s)

If you would like to share with us some Pathway-related content, please give an admin a shout on Discord https://discord.gg/pathway.

### Manul conventions

Manuls (aka Pallas's Cats) [are creatures with fascinating habits](https://www.youtube.com/watch?v=rlSTBvViflc). As a tribute to them, we usually read `pw`, one of the most frequent tokens in Pathway code, as: `"paw"`. 

## Performance

Pathway is made to outperform state-of-the-art technologies designed for streaming and batch data processing tasks, including: Flink, Spark, and Kafka Streaming. It also makes it possible to implement a lot of algorithms/UDF's in streaming mode which are not readily supported by other streaming frameworks (especially: temporal joins, iterative graph algorithms, machine learning routines).

If you are curious, here are some benchmarks to play with https://github.com/pathwaycom/pathway-benchmarks.

If you try your own benchmarks, please don't hesitate to let us know. We investigate situations in which Pathway is underperforming on par with bugs (i.e., to our knowledge, they shouldn't happen...).

## Coming soon

Here are some features we plan to incorporate in the near future:

- Enhanced monitoring, observability, and data drift detection (integrates with Grafana visualization and other dashboarding tools).
- New connectors: interoperability with Delta Lake and Snowflake data sources.
- Easier connection setup for MongoDB.
- More performant garbage collection.


## Dependencies

Pathway is made to run in a "clean" Linux/MacOS + Python environment. When installing the pathway package with `pip` (from a wheel), you are likely to encounter a small number of Python package dependencies, such as sqlglot (used in the SQL API) and python-sat (useful for resolving dependencies during compilation). All necessary Rust crates are pre-built; the Rust compiler is not required to install Pathway, unless building from sources. A modified version of Timely/Differential Dataflow (which provides a dataflow assembly layer) is part of this repo. 

## License

Pathway is distributed on a BSL 1.1 License https://github.com/pathwaycom/pathway/blob/main/LICENSE.txt which allows for unlimited non-commercial use, as well as use of the Pathway package for most commercial purposes https://pathway.com/license/, free of charge. Code in this repository automatically converts to Open Source (Apache 2.0 License) after 4 years. Some public repos https://github.com/pathwaycom which are complementary to this one (examples, libraries, connectors, etc.) are licensed as Open Source, under the MIT license.


## Contribution guidelines

If you develop a library or connector which you would like to integrate with this repo, we suggest releasing it first as a separate repo on a MIT/Apache 2.0 license. 

For all concerns regarding core Pathway functionalities, Issues are encouraged. For further information, don't hesitate to engage with Pathway's Discord https://discord.gg/pathway.

## Get Help

If you have any questions, issues, or just want to chat about Pathway, we're here to help! Feel free to:
- Check out the documentation in https://pathway.com/developers/ for detailed information.
- Reach out to us via email at contact@pathway.com.

Our team is always happy to help you and ensure that you get the most out of Pathway.
If you would like to better understand how best to use Pathway in your project, please don't hesitate to reach out to us.
