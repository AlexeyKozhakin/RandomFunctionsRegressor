from setuptools import setup, find_packages

setup(
    name='RandomFunctionsRegressor',
    version='0.1.0',
    description='A Python library for random function regression',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AlexeyKozhakin/RandomFunctionsRegressor',
    author='Alexey Kozhakin',
    author_email='alexeykozhakin@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'scikit-learn>=1.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
