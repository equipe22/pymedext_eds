from setuptools import setup
#from distutils.core import setup

setup(
    name='pymedext_eds',
    version='0.1dev',
    packages=['pymedext_eds','pymedext_eds.extract','pymedext_eds.pyromedi'],
    license='MIT',
    package_data={'demo': ['data/demo/*.txt'],'romedi': ['data/romedi/*.p']}
    #long_description=open('README.txt').read(),
)