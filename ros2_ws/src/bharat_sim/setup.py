import os
from glob import glob

from setuptools import find_packages, setup    

package_name = "bharat_sim"


def tree(src):
    """Install a directory tree under share/<pkg>/<src>, keeping sub-folders."""
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join("share", package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ] + tree("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bharat-AV team",
    maintainer_email="bharat-av@users.noreply.github.com",
    description="Adaptive path planning on unstructured Indian roads: Gazebo Sim 8 digital twin",
    license="MIT",
    entry_points={
        "console_scripts": [
            "traffic_manager = bharat_sim.traffic_manager:main",
            "autonomy = bharat_sim.autonomy:main",
            "gen_world = bharat_sim.gen_world:main",
        ],
    },
)
