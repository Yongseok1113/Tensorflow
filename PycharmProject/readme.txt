설치환경
    OS : Windows 10
    Graphic card : NVIDIA Geforce 940MX
    CUDA Toolkit version : 10.0
    cuDNN : v7.5.0 (Feb 21, 2019), for CUDA 10.0

설치 가이드 : https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows

환경변수 설정
    CUDA Toolkit 설치경로 + \bin , \extras\CUPTI\libx64 , \include  3개 경로 따로 추가

텐서플로우 설치
    Anaconda3 설치
    Anaconda Prompt 관리자 권한 실행

    가상환경 생성 : conda create -n 환경이름 pip python=3.6
    환경 활성화 : activate 환경이름

    pip 업그레이드 : python -m pip install --upgrade pip
    텐서플로우 gpu 버전 설치 : pip install --ignore-installed --upgrade tensorflow-gpu

파이참 연동
    프로젝트 생성 -> 프로젝트 인터프리터 -> Existing interpreter -> add.. -> System interpreter ->
    텐서플로우 설치한 환경이름 디렉터리 내 파이썬.exe 파일 설정 -> 생성완료

    ex) C:\Users\사용자이름\Anaconda3\envs\Tensorflow\python.exe
