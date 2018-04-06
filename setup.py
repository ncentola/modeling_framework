from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='modeling_framework',
    version='0.1',
    description='iLoan Fraud Model',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/ncentola/modeling_framework',
    author='Nick Centola',
    author_email='centola.nick@gmail.com',
    packages=['modeling_framework'],
    install_requires=[
        'pypandoc>=1.4',
        'numpy>=1.14.0'
    ],
)
