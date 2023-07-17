import setuptools

setuptools.setup(
    name="collie-bench",
    version="0.1.0",
    packages=['collie'],
    install_requires=[
        'nltk>=3.8',
        'openai',
        'rich',
        'dill',
        'datasets',
        'tqdm',
        'apache_beam',
        'tenacity',
        'google-generativeai'
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
