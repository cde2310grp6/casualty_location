from setuptools import find_packages, setup

package_name = 'casualty_location'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kaijie',
    maintainer_email='kaijie.chong@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'casualty_location = casualty_location.casualty_location:main',
            'casualty_saver = casualty_location.casualty_saver:main',
            'ir_pub = casualty_location.ir_node:main',
        ],
    },
)
