from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='deciplot',
    version='0.0.1',
    description='A handy tool that helps you better visualize your data by plotting 2D decision boundaries, using any sklearn classifier models.',
    py_modules=['deciplot'],
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy >= 1.15',
        'sklearn',
        'matplotlib >= 3.0',
    ],
    url='https://github.com/hxt365/Deciplot',
    author='Truong Hoang',
    author_email='hxt365@gmail.com',
)
