
<img src="magenta-logo-bg.png" height="75">

[![Build Status](https://travis-ci.org/tensorflow/magenta.svg?branch=master)](https://travis-ci.org/tensorflow/magenta)
 [![PyPI version](https://badge.fury.io/py/magenta.svg)](https://badge.fury.io/py/magenta)

**Magenta**는 기계학습의 예술과 음악을 창조하는 과정의 역할을 
탐구하는 연구 프로젝트입니다.
주로 이것은 새로운 노래, 이미지, 도면 및 기타 자료를 생성하기 위한 
알고리즘의 심층 학습과 보강 학습을 포함합니다.
그러나 또한 스마트 도구와 인터페이스의 구축에 관한 연구들은
아티스트와 뮤지션들에게 이 모델들을 사용하여 그들의 프로세스를 확장하게합니다(교체하지 않는다!).
Magenta는 Google 브레인 [team](https://research.google.com/teams/brain/)의
몇몇 연구원들과 엔지니어들에 의해 시작되었습니다.
그러나 다른 많은 사람들이 이 프로젝트에 크게 기여했다.
우리는 [TensorFlow](https://www.tensorflow.org)를 사용하고 방문하여 모델과 도구를
이 GitHub의 오픈 소스에 공개합니다.
Magenta에 대해서 더 배우고 싶다면 우리 기술적 세부사항을 게시할 수 있는 웹사이트 
[blog](https://magenta.tensorflow.org)를 확인해보십시오.
당신은 또한 우리의 [discussion group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss)
에 참여할 수 있습니다.


## Getting Started

* [Installation](#installation)
* [Using Magenta](#using-magenta)
* [Playing a MIDI Instrument](#playing-a-midi-instrument)
* [Development Environment (Advanced)](#development-environment)

## Installation

Magenta는 쉬운 설치를 위한 [pip package](https://pypi.python.org/pypi/magenta)를 유지합니다.
우리는 Anaconda를 사용하여 설치할 것을 권장하지만,
아무런 표준 파이썬 환경에서 자동 가능합니다.
우리는 Python 2 (>= 2.7)와 Python 3 (>= 3.5) 모두를 지원합니다.
이 지침들은 당신이 Anaconda를 사용하고 있다고 가정할 것입니다.

GPU 지원을 사용하려면 아래의 [GPU Installation](#gpu-installation) 지침을 따르십시오.

### Automated Install (w/ Anaconda)

Mac OS X 또는 Ubuntu를 실행 중인 경우 자동화된 제품을 설치 스크립트를 사용해 보십시오. 다음 명령을 단말기에 붙여넣기만 하면 됩니다.

```bash
curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```

//스크립트가 완료된 후 새 터미널 창을 열어 환경 변수의 변을 설정하십시오.

Magenta 라이브러리는 이제 Python 프로그램과 Jupyter notebooks 내에서 사용할 수 있으며,
그리고 마젠타 스크립트가 당신의 path에 설치되었습니다.

Magenta를 사용하려면 새로운 터미널 창을 열 때
`source activate magenta`를 실행해야 함을 유념하십시오.

### Manual Install (w/o Anaconda)

어떤 이유로든 자동 스크립트가 실패하거나 다음을 통해 설치하려는 경우 다음 단계를 수행하십시오.

Magenta pip package를 설치하십시오:

```bash
pip install magenta
```

**NOTE**: 우리가 의존하는 `rtmidi` 패키지를 설치하려면 사운드 라이브러리를 위한 헤더를 설치해야 할 수도 있습니다. 
Linux에서 이 명령은 필요한 패키지를 설치해야 합니다:

```bash
sudo apt-get install build-essential libasound2-dev libjack-dev
```

Magenta 라이브러리는 이제 Python 프로그램, Jupyter notebooks 내에서 사용할 수 있으며,
마젠타 스크립트가 당신의 path에 설치되었습니다.

### GPU Installation

GPU가 설치되어 있고 Magenta를 사용하기 원하는 경우
[수동 설치](#수동 설치) 지침을 따르되, 몇 가지 수정 지침을 따르십시오.

먼저, 당신의 시스템이[requirements to run tensorflow with GPU support](
https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support)충족하는지 보십시오.

그런다음, [Manual Install](#manual-install)지침을 따르시고, 
`magenta` 패키지 대신 `magenta-gpu` 패키지를 설치하십시오:

```bash
pip install magenta-gpu
```

유일한 두 패키지의 차이점은 `magenta-gpu` 가 `tensorflow` 대신 `tensorflow-gpu`에 의존한다는 것입니다.

이제 Magenta가 GPU에 액세스할 수 있어야 합니다.

## Using Magenta

You can now train our various models and use them to generate music, audio, and images. You can
find instructions for each of the models by exploring the [models directory](magenta/models).

To get started, create your own melodies with TensorFlow using one of the various configurations of our [Melody RNN](magenta/models/melody_rnn) model; a recurrent neural network for predicting melodies.

## Playing a MIDI Instrument

After you've trained one of the models above, you can use our [MIDI interface](magenta/interfaces/midi) to play with it interactively.

We also have created several [demos](https://github.com/tensorflow/magenta-demos) that provide a UI for this interface, making it easier to use (e.g., the browser-based [AI Jam](https://github.com/tensorflow/magenta-demos/tree/master/ai-jam-js)).

## Development Environment
Magenta를 발전시키기를 원하면, 모든 개발 환경을 설정해야 합니다.

먼저, 이 repository를 clone하십시오:

```bash
git clone https://github.com/tensorflow/magenta.git
```

그런 다음 기본 디렉터리로 변경하고 설정 명령을 실행하여 종속된 파일을 설치하십시오:

```bash
pip install -e .
```

이제 파일을 편집하고 Python을 평소처럼 호출하여 스크립트를 실행할 수 있습니다.
예를 들어 기본 디렉터리에서 `melody_rnn_generate` 스크립트를 실행하는 방법은 다음과 같습니다:

```bash
python magenta/models/melody_rnn/melody_rnn_generate --config=...
```

당신은(잠재적으로 수정될) 패키지를 설치하십시오:

```bash
pip install .
```

pull request를 생성하기 전에 변경사항도 테스트해보십시오:

```bash
pip install pytest-pylint
pytest
```

## PIP Release

pip에 대한 새 버전을 만들려면 버전을 충돌시킨 다음 실행하십시오:

```bash
python setup.py test
python setup.py bdist_wheel --universal
python setup.py bdist_wheel --universal --gpu
twine upload dist/magenta-N.N.N-py2.py3-none-any.whl
twine upload dist/magenta_gpu-N.N.N-py2.py3-none-any.whl
```
