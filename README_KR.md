
<img src="magenta-logo-bg.png" height="75">

[![Build Status](https://travis-ci.org/tensorflow/magenta.svg?branch=master)](https://travis-ci.org/tensorflow/magenta)
 [![PyPI version](https://badge.fury.io/py/magenta.svg)](https://badge.fury.io/py/magenta)

**Magenta**는 기계학습의 역할을 탐구하는 연구 프로젝트
예술과 음악을 창조하는 과정에서 주로 이것
새로운 심층 학습과 보강 학습을 포함합니다.
노래, 이미지, 도면 및 기타 자료를 생성하기 위한 알고리즘. 그러나 역시 그렇다.
스마트 도구와 인터페이스의 구축에 관한 연구
아티스트와 뮤지션들은 그들의 프로세스를 확장한다(교체하지 않는다!)
이 모델들 마젠타는 몇몇 연구원들과 엔지니어들에 의해 시작되었다.
Google 브레인 [team](https://research.google.com/teams/brain/),에서
그러나 다른 많은 사람들이 이 프로젝트에 크게 기여했다. 우리는 사용한다
[TensorFlow](https://www.tensorflow.org)을 방문하여 모델을 출시하십시오.
이 GitHub의 오픈 소스에 있는 공구 더 배우고 싶다면
Magenta에 대해서, 우리 웹사이트 [blog](https://magenta.tensorflow.org),를 확인해봐.
기술적 세부사항을 게시할 수 있는 곳. 당신은 또한 우리의 [토론]에 참여할 수 있다.
[group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

## Getting Started

* [Installation](#installation)
* [Using Magenta](#using-magenta)
* [Playing a MIDI Instrument](#playing-a-midi-instrument)
* [Development Environment (Advanced)](#development-environment)

## Installation

Magenta는 쉽게 [pip package](https://pypi.python.org/pypi/magenta)을 유지한다.
설치 아나콘다를 사용하여 설치할 것을 권장하지만, 아무데서나 사용할 수 있다.
표준 파이썬 환경 우리는 Python 2 (>= 2.7)와 Python 3 (>= 3.5) 모두를 지원한다.
이 지침들은 당신이 아나콘다를 사용하고 있다고 가정할 것이다.

GPU 지원을 사용하려면 아래의 [GPU 설치](#gpu-installation) 지침을 따르십시오.
