from setuptools import find_packages, setup


def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    setup(
        name="smat-exo",
        version="1.0.0",
        description="SMAT: Staged Multi-Agent Training for co-adaptive hip exoskeleton control (built on MyoAssist)",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/Fitz-Yuan/SMAT-Exo",
        license="Apache 2.0",
        packages=find_packages(include=("myosuite*", "myoassist_utils*", "rl_train*")),
        include_package_data=True,
        python_requires=">=3.11",
        install_requires=fetch_requirements(),
    )
