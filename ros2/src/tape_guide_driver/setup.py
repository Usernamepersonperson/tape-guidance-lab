from setuptools import setup

package_name = "tape_guide_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VB-Builds",
    maintainer_email="noreply@example.com",
    description="ROS 2 driver for the open tape-guide sensor CANopen profile",
    license="MIT",
    entry_points={"console_scripts": ["guide_driver = tape_guide_driver.node:main"]},
)
