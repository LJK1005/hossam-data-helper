from pandas import DataFrame, Series
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
from .util import my_pretty_table
from .plot import my_lineplot


def my_diff(
    data: DataFrame,
    yname: str,
    plot: bool = True,
    max_diff: int = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> None:
    """데이터의 정상성을 확인하고, 정상성을 충족하지 않을 경우 차분을 수행하여 정상성을 만족시킨다.
    반드시 데이터 프레임의 인덱스가 타임시리즈 데이터여야 한다.

    Args:
        data (DataFrame): 데이터 프레임
        yname (str): 차분을 수행할 데이터 컬럼명
        plot (bool, optional): 차분 결과를 그래프로 표시할지 여부. Defaults to True.
        max_diff (int, optional): 최대 차분 횟수. 지정되지 않을 경우 최대 반복. Defaults to None.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
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

        if count > 0:
            # 차분 수행
            df = df.diff().dropna()

        if plot:
            my_lineplot(df=df, yname=yname, xname=df.index, figsize=figsize, dpi=dpi)

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

    return df


def my_rolling(
    data: Series,
    window: int,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> Series:
    """이동평균을 계산한다.

    Args:
        data (Series): 시리즈 데이터
        window (int): 이동평균 계산 기간
        plot (bool, optional): 이동평균 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.

    Returns:
        Series: 이동평균 데이터
    """
    rolling = data.rolling(window=window).mean()
    df = DataFrame(
        {
            rolling.name: rolling,
        },
        index=rolling.index,
    )

    if plot:
        my_lineplot(
            df=df,
            yname=rolling.name,
            xname=df.index,
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title(f"Rolling (window={window})"),
        )

    return rolling


def my_ewm(
    data: Series, span: int, plot: bool = True, figsize: tuple = (10, 5), dpi: int = 100
) -> Series:
    """지수가중이동평균을 계산한다.

    Args:
        data (Series): 시리즈 데이터
        span (int): 지수가중이동평균 계산 기간
        plot (bool, optional): 지수가중이동평균 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.

    Returns:
        Series: 지수가중이동평균 데이터
    """
    ewm = data.ewm(span=span).mean()

    df = DataFrame(
        {
            ewm.name: ewm,
        },
        index=ewm.index,
    )

    if plot:
        my_lineplot(
            df=df,
            yname=ewm.name,
            xname=df.index,
            figsize=figsize,
            dpi=dpi,
        )

    return ewm


def my_seasonal_decompose(
    data: Series,
    model: str = "additive",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
):
    """시계열 데이터를 계절적, 추세적, 불규칙적 성분으로 분해한다.

    Args:
        data (Series): 시리즈 데이터
        model (str, optional): 분해 모델. "additive" 또는 "multiplicative". Defaults to "additive".
        plot (bool, optional): 분해 결과 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    sd = seasonal_decompose(data, model=model)

    sd_df = DataFrame(
        {
            "original": sd.observed,
            "trend": sd.trend,
            "seasonal": sd.seasonal,
            "resid": sd.resid,
        },
        index=data.index,
    )

    if plot:
        figure = sd.plot()
        figure.set_size_inches((figsize[0], figsize[1] * 4))
        figure.set_dpi(dpi)

        fig, ax1, ax2, ax3, ax4 = figure.get_children()

        ax1.set_ylabel("Original")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.show()
        plt.close()

    return sd_df
