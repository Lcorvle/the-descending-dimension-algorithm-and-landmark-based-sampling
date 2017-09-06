import os
from setuptools.command.install import install
from setuptools import setup


class MyInstall(install):
    def run(self):
        if not os.path.exists('release'):
            os.makedirs('release')
        else:
            os.system('rd /s/q release')
            os.makedirs('release')

        os.chdir('release/')
        return_val = os.system('cmake -DCMAKE_C_COMPILER="C:/MinGW/bin/gcc.exe" -DCMAKE_CXX_COMPILER="C:/MinGW/bin/g++.exe" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=RELEASE ..')

        if return_val != 0:
            print('cannot find cmake')
            exit(-1)

        os.system('make VERBOSE=1')
        os.chdir('..')

        install.run(self)
        os.system(
            'copy C:\\Users\\xsx19\\laboratory\\the-descending-dimension-algorithm-and-landmark-based-sampling'
            '\\release\\libLandmarkBasedSamplingAndDescendingDimension.dll C:\\Users\\xsx19\\AppData\\Local\\'
            'Programs\\Python\\Python35-32\\Lib\\site-packages\\LandmarkBasedSamplingAndDescendingDimension\\'
            'libLandmarkBasedSamplingAndDescendingDimension.dll')
        os.system(
            'copy C:\\Users\\xsx19\\laboratory\\the-descending-dimension-algorithm-and-landmark-based-sampling'
            '\\script\\__init__.py C:\\Users\\xsx19\\AppData\\Local\\Programs\\Python\\Python35-32\\Lib\\'
            'site-packages\\LandmarkBasedSamplingAndDescendingDimension\\__init__.py')

setup(
    name="LandmarkBasedSamplingAndDescendingDimension",
    version="1.0",
    description='Landmark based sampling and descending dimension algorithm: t-SNE',
    author="Shouxing Xiang (based on Dmitry Ulyanov's code)",
    author_email='xsx1996@163.com',
    url='https://github.com/Lcorvle/the-descending-dimension-algorithm-and-landmark-based-sampling',
    install_requires=[
        'numpy',
        'psutil',
        'cffi'
    ],

    packages=['LandmarkBasedSamplingAndDescendingDimension'],
    package_dir={'LandmarkBasedSamplingAndDescendingDimension': 'script'},
    package_data={'LandmarkBasedSamplingAndDescendingDimension': ['LandmarkBasedSamplingAndDescendingDimension.dll']},
    include_package_data=True,

    cmdclass={"install": MyInstall},
)

