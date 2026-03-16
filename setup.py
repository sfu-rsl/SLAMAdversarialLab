"""Setup script for SLAMAdverserialLab."""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
PACKAGE_NAME = "slamadverseriallab"

version_ns = {}
exec((SRC_ROOT / "__version__.py").read_text(encoding="utf-8"), version_ns)


def _discover_package_layout(src_root: Path, top_package: str):
    """Map src/* packages to slamadverseriallab.* package names."""
    packages = [top_package]
    package_dir = {top_package: str(src_root)}

    for init_file in sorted(src_root.rglob("__init__.py")):
        pkg_dir = init_file.parent
        if pkg_dir == src_root:
            continue

        rel = pkg_dir.relative_to(src_root)
        pkg_name = top_package + "." + ".".join(rel.parts)
        packages.append(pkg_name)
        package_dir[pkg_name] = str(pkg_dir)

    return packages, package_dir


packages, package_dir = _discover_package_layout(SRC_ROOT, PACKAGE_NAME)

with open(ROOT / "requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(ROOT / "README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    version=version_ns["__version__"],
    author="SLAMAdverserialLab Team",
    description="An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohhef/SLAMAdverserialLab",
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="slam, computer vision, perturbation, benchmarking, robustness",
    project_urls={
        "Bug Reports": "https://github.com/mohhef/SLAMAdverserialLab/issues",
        "Source": "https://github.com/mohhef/SLAMAdverserialLab",
    },
)
