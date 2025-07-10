"""
Setup script for Ticket Broker Optimization System
"""
from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("ticket_broker/README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("ticket_broker/requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ticket-broker-pro",
    version="1.0.0",
    author="Ticket Broker Development Team",
    author_email="support@ticketbrokerpro.com",
    description="Intelligent Event Analysis & Investment Platform for Ticket Brokers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ticket-broker-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ticket-broker=ticket_broker.main:main",
            "ticket-analyzer=ticket_broker.main:run_example_analysis",
        ],
    },
    include_package_data=True,
    package_data={
        "ticket_broker": [
            "*.md",
            "*.txt",
            ".env.example",
            "config/*.py",
            "models/*.py",
            "data_collectors/*.py",
            "utils/*.py",
        ],
    },
    zip_safe=False,
    keywords="ticket broker investment analysis profit optimization events concerts",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ticket-broker-pro/issues",
        "Source": "https://github.com/yourusername/ticket-broker-pro",
        "Documentation": "https://ticket-broker-pro.readthedocs.io/",
    },
)