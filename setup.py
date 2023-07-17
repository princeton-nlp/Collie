import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name="collie-bench",
    version="0.1.0",
    packages=['collie'],
    description='Official Implementation of "COLLIE: Systematic Construction of Constrained Text Generation Tasks"',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'nltk>=3.8',
        'openai',
        'rich',
        'dill',
        'datasets',
        'tqdm',
        'apache_beam',
        'tenacity',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
