from setuptools import setup, find_packages

setup(
    name="medgemma-council",
    version="0.1.0",
    description="Multi-Agent Clinical Decision Support System using MedGemma 1.5",
    author="MedGemma Council Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=1.0.0",
        "langchain-core>=1.0.0",
        "streamlit>=1.30.0",
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "accelerate>=0.25.0",
        "llama-index>=0.10.0",
        "chromadb>=0.4.0",
        "biopython>=1.83",
        "requests>=2.31.0",
        "bitsandbytes>=0.41.0",
        "pydantic>=2.0.0",
        # 'llama-cpp-python' is installed separately with hardware-specific flags
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-mock>=3.12.0",
            "black",
            "isort",
            "flake8",
        ]
    },
)
