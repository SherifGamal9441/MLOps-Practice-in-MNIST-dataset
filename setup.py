from setuptools import setup, find_packages

setup(
    name='mnist_pipeline',
    version='1.0',
    packages=find_packages(),  # will find 'scripts' if it has __init__.py
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'mlflow',
        'dvc',
        'fastapi',
        'uvicorn',
        'python-multipart',
        'Pillow'
    ],
    entry_points={
        'console_scripts': [
            'mnist-train=scripts.train:main',
            'mnist-infer=scripts.inference:main',
            'mnist-eval=scripts.evaluate:main'
        ]
    }
)
