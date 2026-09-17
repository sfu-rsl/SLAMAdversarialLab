"""Setup script for SLAMAdversarialLab."""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
PACKAGE_NAME = "slamadversariallab"

version_ns = {}
exec((SRC_ROOT / "__version__.py").read_text(encoding="utf-8"), version_ns)


def _discover_package_layout(src_root: Path, top_package: str):
    """Map src/* packages to slamadversariallab.* package names."""
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

def _read_requirement_groups(path):
    """Partition requirements.txt on its `# --- name ---` section markers.

    Comments and blank lines are dropped here rather than handed to setuptools.
    They were tolerated only because `parse_requirements` happens to skip them,
    which is not a contract worth resting an install on.
    """
    groups, current = {}, "core"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("# ---") and line.endswith("---"):
                current = line.strip("# -").strip()
                groups.setdefault(current, [])
                continue
            if not line or line.startswith("#"):
                continue
            groups.setdefault(current, []).append(line)
    return groups


_groups = _read_requirement_groups(ROOT / "requirements.txt")
# torch ships with core: the SLAM wrappers and depth perturbations both import
# it. The generative stack does not, so it is an extra.
requirements = _groups.get("core", []) + _groups.get("torch", [])
extras = {name: reqs for name, reqs in _groups.items() if name not in ("core", "torch")}
extras["all"] = [r for reqs in _groups.values() for r in reqs]

with open(ROOT / "README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    version=version_ns["__version__"],
    author="SLAMAdversarialLab Team",
    description="An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfu-rsl/SLAMAdversarialLab",
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=extras,
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
        "Bug Reports": "https://github.com/sfu-rsl/SLAMAdversarialLab/issues",
        "Source": "https://github.com/sfu-rsl/SLAMAdversarialLab",
    },
)
