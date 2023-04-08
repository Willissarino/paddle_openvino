from setuptools import find_packages, setup

setup(
    name='PaddleOCR_OpenVINO',
    packages=find_packages(),
    include_package_data=True,
    version='1',
    description='PaddleOCR with OpenVINO libary for number plate recognition',
    author='Willi',
    license='MIT',
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        'openvino-dev==2022.3.0',
        'openvino-telemetry==2022.3.0',
        
        'nncf==2.4.0',
        
        'paddlepaddle>=2.4.0',
        'paddle2onnx>=0.6',
        'paddlenlp>=2.0.8',
        
        'transformers>=4.21.1',
        'monai>=0.9.1',

        'pytube>=12.1.0',
        'librosa>=0.8.1',
        'shapely>=1.7.1',
        'pyclipper>=1.2.1',
        'gdown',
        'yaspin',
        ],
)