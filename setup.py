from setuptools import setup, find_packages

setup(
    name='RandomFunctionsRegressor',
    version='0.1.0',
    description='A Python library for random function regression',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/username/RandomFunctionsRegressor',
    author='Your Name',
    author_email='your.email@example.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        # укажите зависимости здесь
        # Например: 'numpy', 'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)