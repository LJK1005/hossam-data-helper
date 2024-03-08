# hossam-data-util

이광호 강사 데이터 분석 유틸리티 


![Generic badge](https://img.shields.io/badge/version-0.0.1-critical.svg?style=flat-square&logo=appveyor) &nbsp;
[![The MIT License](https://img.shields.io/badge/license-MIT-orange.svg?style=flat-square&logo=appveyor)](http://opensource.org/licenses/MIT) &nbsp;
![Badge](https://img.shields.io/badge/Author-Lee%20KwangHo-blue.svg?style=flat-square&logo=appveyor) &nbsp;
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=appveyor) &nbsp;
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=appveyor) &nbsp;
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=appveyor) &nbsp;
![Scikit-learn](https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=appveyor)


## Installation

### [1] Remote Repository

```shell
pip install --upgrade git+ssh://git@github.com:leekh4232/hossam-data-helper.git
```

or

```shell
pip install --upgrade git+https://github.com/leekh4232/hossam-data-helper.git
```


### [2] Local Repository

```shell
pip install --upgrade git+file///path/to/your/git/project/
```

## Uninstallation

```shell
pip uninstall -y Hossam-Data-Util
```

## How to use

수업 중 적용되던 패키지 참조 코드가 아래와 같이 변경됩니다.

### 변경전

```Python
import sys
import os
work_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(work_path)

from helper.util import *
from helper.plot import *
from helper.analysis import *
from helper.classification import *
```

### 변경후

```Python
from hossam.util import *
from hossam.plot import *
from hossam.analysis import *
from hossam.classification import *
```


## Documentation

[Documentation](https://leekh4232.github.io/hossam-data-helper/hossam)