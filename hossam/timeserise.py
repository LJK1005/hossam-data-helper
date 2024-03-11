from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller

from .util import my_pretty_table


def my_diff(data: DataFrame, yname: str, max_diff: int = None) -> None:
    """데이터의 정상성을 확인하고, 정상성을 충족하지 않을 경우 차분을 수행하여 정상성을 만족시킨다.
    반드시 데이터 프레임의 인덱스가 타임시리즈 데이터여야 한다.

    Args:
        data (DataFrame): 데이터 프레임
        yname (str): 차분을 수행할 데이터 컬럼명
        max_diff (int, optional): 최대 차분 횟수. 지정되지 않을 경우 최대 반복. Defaults to None.
    """
    df = data.copy()

    # 데이터 정상성 여부
    stationarity = False

    # 반복 수행 횟수
    count = 0

    # 데이터가 정상성을 충족하지 않는 동안 반복
    while not stationarity:
        if count == 0:
            print("=========== 원본 데이터 ===========")
        else:
            print("=========== %d차 차분 데이터 ===========" % count)

        # ADF Test
        ar = adfuller(df[yname])

        ardict = {
            "검정통계량(ADF Statistic)": [ar[0]],
            "유의수준(p-value)": [ar[1]],
            "최적차수(num of lags)": [ar[2]],
            "관측치 개수(num of observations)": [ar[3]],
        }

        for key, value in ar[4].items():
            ardict["기각값(Critical Values) %s" % key] = value

        stationarity = ar[1] <= 0.05
        ardict["데이터 정상성 여부"] = "정상" if stationarity else "비정상"

        ardf = DataFrame(ardict, index=["ADF Test"]).T
        my_pretty_table(ardf)

        # 반복회차 1 증가
        count += 1

        # 최대 차분 횟수가 지정되어 있고, 반복회차가 최대 차분 횟수에 도달하면 종료
        if max_diff and count == max_diff:
            break
