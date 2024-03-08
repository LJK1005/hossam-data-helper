import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from pandas import DataFrame, Series
from scipy.spatial import ConvexHull
from statannotations.Annotator import Annotator
from scipy.stats import zscore, probplot
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

import sys

plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["axes.unicode_minus"] = False

def my_boxplot(df: DataFrame, xname: str = None, yname: str = None, orient: str = 'v', palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 상자그림을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str, optional): x축에 사용할 컬럼명. Defaults to None.
        yname (str, optional): y축에 사용할 컬럼명. Defaults to None.
        orient (str, optional): 그래프의 방향. 'v' 또는 'h'. Defaults to 'v'.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    if xname != None and yname != None:
        sb.boxplot(data=df, x=xname, y=yname, orient=orient, palette=palette, ax=ax)
    else:
        sb.boxplot(data=df, orient=orient, palette=palette, ax=ax)
        
    ax.grid()
    
    if callback:
        callback(ax)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def my_kdeplot(df: DataFrame, xname: str = None, yname: str = None, hue: str = None, palette: str = None, fill: bool = False, fill_alpha: float = 0.3, linewidth: float = 1, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 커널밀도추정을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str, optional): x축에 사용할 컬럼명. Defaults to None.
        yname (str, optional): y축에 사용할 컬럼명. Defaults to None.
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        fill (bool, optional): 채우기 여부. Defaults to False.
        fill_alpha (float, optional): 채우기의 투명도. Defaults to 0.3.
        linewidth (float, optional): 선의 두께. Defaults to 0.5.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    if fill:
        sb.kdeplot(data=df, x=xname, y=yname, hue=hue, palette=palette, fill=fill, alpha=fill_alpha, linewidth=linewidth, ax=ax)
    else:
        sb.kdeplot(data=df, x=xname, y=yname, hue=hue, palette=palette, fill=fill, linewidth=linewidth, ax=ax)
    
    #plt.grid()
    ax.grid()
    
    if callback:
        callback(ax)

    plt.tight_layout()
    plt.show()
    plt.close()


def my_histplot(df: DataFrame, xname: str, hue=None, bins = None, kde: bool = True, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 히스토그램을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        bins (optional): 히스토그램의 구간 수 혹은 리스트. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    if bins:
        sb.histplot(data=df, x=xname, hue=hue, kde=kde, bins=bins, palette=palette, ax=ax)
    else:
        sb.histplot(data=df, x=xname, hue=hue, kde=kde, palette=palette, ax=ax)
        
    #plt.grid()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_stackplot(df: DataFrame, xname: str, hue: str, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """hue로 구분되는 막대 그래프를 비율로 표시한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        multiple (str, optional): hue가 있을 경우 전체의 비율을 어떻게 표시할지 여부. Deafults to layer
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    df2 = df[[xname, hue]]
    df2[xname] = df2[xname].astype(str)
    
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.histplot(data=df2, x=xname, hue=hue, palette=palette, 
                linewidth=0.5,
                stat='probability',     # 전체에서의 비율로 그리기
                multiple='fill',        # 전체를 100%로 그리기
                shrink=0.8,             # 막대의 폭
                ax=ax)
    
    # 그래프의 x축 항목 수 만큼 반복
    for p in ax.patches:
        # 각 막대의 위치, 넓이, 높이
        left, bottom, width, height = p.get_bbox().bounds
        # 막대의 중앙에 글자 표시하기
        ax.annotate("%0.1f%%" % (height * 100), xy=(left+width/2, bottom+height/2), ha='center', va='center')
        
    #plt.grid()
    
    if str(df[xname].dtype) in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
        xticks = list(df[xname].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_scatterplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 두 컬럼에 대해 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    sb.scatterplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_regplot(df: DataFrame, xname: str, yname: str, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.
    색상 팔레트는 지원하지 않는다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    sb.regplot(data=df, x=xname, y=yname, color=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_lmplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    g = sb.lmplot(data=df, x=xname, y=yname, hue=hue, palette=palette)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.grid()
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_pairplot(df: DataFrame, diag_kind: str = "hist", hue=None, palette: str = None, figsize: tuple=(3.5, 2.7), dpi: int=100) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 쌍별 관계를 시각화한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    w = figsize[0] * len(df.columns)
    h = figsize[1] * len(df.columns)
    
    g = sb.pairplot(df, hue=hue, diag_kind=diag_kind, palette=palette)
    g.fig.set_size_inches(w, h)
    g.fig.set_dpi(dpi)
    g.map_lower(sb.kdeplot, fill=True, alpha=0.3)
    g.map_upper(sb.scatterplot)
    plt.tight_layout()
    plt.show()
    plt.close()


def my_countplot(df: DataFrame, xname: str, hue=None, palette: str = None, order: int = 1, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 카운트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        order (str, optional): 정렬 방식. '1=값의 이름순' 또는 '2=수량 많은 순'. Defaults to "1".
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    sort = None
    if str(df[xname].dtype) in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
        if order == 1:
            sort = sorted(list(df[xname].unique()))
        else:
            sort = sorted(list(df[xname].value_counts().index))
    
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    sb.countplot(data=df, x=xname, hue=hue, palette=palette, order=sort, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_barplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 바플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    sb.barplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_boxenplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 박슨플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    sb.boxenplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_violinplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 바이올린플롯(상자그림+커널밀도)을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.violinplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_pointplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 포인트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.pointplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_jointplot(df: DataFrame, xname: str, yname: str, hue=None, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 산점도와 히스토그램을 함께 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    g = sb.jointplot(data=df, x=xname, y=yname, hue=hue, palette=palette)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.tight_layout()
    plt.show()
    plt.close()
    

def my_heatmap(data: DataFrame, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 히트맵을 그려서 관계를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        palette (str, optional): 칼라맵. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.heatmap(data, annot=True, cmap=palette, fmt='.2f', ax=ax)

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_convex_hull(data: DataFrame, xname: str, yname: str, hue: str, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None):
    """데이터프레임 내의 컬럼에 대해 외곽선을 그려서 군집을 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str): 색상을 구분할 기준이 되는 컬럼명
        palette (str, optional): 칼라맵. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    # 군집별 값의 종류별로 반복 수행
    for c in data[hue].unique():
        # 한 종류만 필터링한 결과에서 두 변수만 선택
        df_c = data.loc[data[hue] == c, [xname, yname]]
        
        # 외각선 좌표 계산
        hull = ConvexHull(df_c)
        
        # 마지막 좌표 이후에 첫 번째 좌표를 연결
        points = np.append(hull.vertices, hull.vertices[0])
        
        ax.plot(df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":")
        ax.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)
        
    sb.scatterplot(data=data, x=xname, y=yname, hue=hue, palette=palette, ax=ax)
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_kde_confidence_interval(data: DataFrame, clevel=0.95, figsize: tuple=(10, 5), dpi: int = 200, callback: any = None) -> None:
    """커널밀도추정을 이용하여 신뢰구간을 그려서 분포를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        clevel (float, optional): 신뢰수준. Defaults to 0.95.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # 데이터 프레임의 컬럼이 여러 개인 경우 처리
    for c in data.columns:
        column = data[c].dropna()
        #print(column)
        max = column.max()      # 최대값
        dof = len(column) - 1   # 자유도
        sample_mean = column.mean()  # 표본평균
        sample_std = column.std(ddof=1) # 표본표준편차
        sample_std_error = sample_std / sqrt(len(column)) # 표본표준오차
        #print(max, dof, sample_mean, sample_std, sample_std_error)
        
        # 신뢰구간
        cmin, cmax = t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)
        #print(cmin, cmax)

        # 현재 컬럼에 대한 커널밀도추정
        sb.kdeplot(data=column, ax=ax)

        # 그래프 축의 범위
        xmin, xmax, ymin, ymax = plt.axis()

        # 신뢰구간 그리기
        ax.plot([cmin, cmin], [ymin, ymax], linestyle=':')
        ax.plot([cmax, cmax], [ymin, ymax], linestyle=':')
        #print([cmin, cmax])
        ax.fill_between([cmin, cmax], y1=ymin, y2=ymax, alpha=0.1)

        # 평균 그리기
        ax.plot([sample_mean, sample_mean], [0, ymax], linestyle='--', linewidth=2)

        ax.text(x=(cmax-cmin)/2+cmin,
                y=ymax,
                s="[%s] %0.1f ~ %0.1f" % (column.name, cmin, cmax),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontdict={"size": 10, "color": "red"})

    ax.set_ylim([ymin, ymax*1.1])
    ax.grid()

    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    

def my_pvalue1_anotation(data : DataFrame, target: str, hue: str, pairs: list, test: str = "t-test_ind", text_format: str = "star", loc: str = "outside", figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """데이터프레임 내의 컬럼에 대해 상자그림을 그리고 p-value를 함께 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        target (str): 종속변수에 대한 컬럼명
        hue (str): 명목형 변수에 대한 컬럼명
        pairs (list, optional): 비교할 그룹의 목록. 명목형 변수에 포함된 값 중에서 비교 대상을 [("A","B")] 형태로 선정한다.
        test (str, optional): 검정방법. Defaults to "t-test_ind".
            - t-test_ind(독립,등분산), t-test_welch(독립,이분산)
            - t-test_paired(대응,등분산), Mann-Whitney(대응,이분산), Mann-Whitney-gt, Mann-Whitney-ls
            - Levene(분산분석), Wilcoxon, Kruskal
        text_format (str, optional): 출력형식(full, simple, star). Defaults to "star".
        loc (str, optional): 출력위치(inside, outside). Defaults to "outside".
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.boxplot(data=data, x=hue, y=target, ax=ax)

    annotator = Annotator(ax, data=data, x=hue, y=target, pairs=pairs)
    annotator.configure(test=test, text_format=text_format, loc=loc)
    annotator.apply_and_annotate()

    sb.despine()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()


def my_resid_histplot(y: np.ndarray, y_pred: np.ndarray, bins = None, kde: bool = True, palette: str = None, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """예측값과 잔차를 히스토그램으로 출력한다.

    Args:
        y (np.ndarray): 종속변수에 대한 관측치
        y_pred (np.ndarray): 종속변수에 대한 예측치
        bins (_type_, optional): 히스토그램의 구간 수 혹은 리스트. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    resid = y - y_pred
    resid_df = DataFrame({"resid": resid}).reset_index(drop=True)
    my_histplot(resid_df, xname="resid", bins=bins, figsize=figsize, dpi=dpi, palette=palette)  
    

def my_residplot(y, y_pred, lowess: bool = False, mse: bool = False, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """예측값과 잔차를 그래프로 출력한다.

    Args:
        y (_type_): 종속변수에 대한 관측치
        y_pred (_type_): 종속변수에 대한 예측치
        lowess (bool, optional): 로우에스티메이션을 사용할지 여부. Defaults to False.
        mse (bool, optional): MSE를 통한 잔차의 분포를 함께 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    resid = y - y_pred
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.residplot(x=y_pred, y=resid, lowess=lowess, line_kws={'color': 'red', 'linewidth': 1}, scatter_kws = {'edgecolor':"white", "alpha":0.7}, ax=ax)
    
    if mse:
        mse = mean_squared_error(y, y_pred)
        mse_sq = np.sqrt(mse)

        r1 = resid[ (resid > -mse_sq) & (resid < mse_sq)].count() / resid.count() * 100
        r2 = resid[ (resid > -2*mse_sq) & (resid < 2*mse_sq)].count() / resid.count() * 100
        r3 = resid[ (resid > -3*mse_sq) & (resid < 3*mse_sq)].count() / resid.count() * 100

        mse_r = [r1, r2, r3]
        
        for i, c in enumerate(['red', 'green', 'black']):
            ax.axhline(mse_sq * (i+1), color=c, linestyle='--', linewidth=0.5)
            ax.axhline(mse_sq * (-(i+1)), color=c, linestyle='--', linewidth=0.5)

        # 현재 표시되는 그래프의 x축 범위를 가져온다.
        xmin, xmax = ax.get_xlim()

        target = [68,95,99]
        for i, c in enumerate(['red', 'green', 'black']):
            ax.text(s=f"{i+1}"r'${}\sqrt{MSE}$ = %.2f%% (%.2f%%)' % (mse_r[i], mse_r[i]-target[i]), x=xmax+0.2, y=(i+1)*mse_sq, color=c)
            ax.text(s=f"-{i+1}"r'${}\sqrt{MSE}$ = %.2f%% (%.2f%%)' % (mse_r[i], mse_r[i]-target[i]), x=xmax+0.2, y=-(i+1)*mse_sq, color=c)
    else:
        ax.grid()
        
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_qqplot(y_pred, figsize: tuple=(10, 5), dpi: int=100, callback: any = None) -> None:
    """QQ플롯을 출력한다.

    Args:
        y_pred (Series): 종속변수에 대한 예측치
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    (x, y), _ = probplot(zscore(y_pred))
    k = (max(x)+0.5).round()

    sb.scatterplot(x=x, y=y, ax=ax)
    sb.lineplot(x=[-k, k], y=[-k, k], color='red', linestyle='--', ax=ax)
    ax.grid()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    

def my_learing_curve(estimator: any, data: DataFrame, yname: str='target', scalling: bool = False, cv: int=10, train_sizes: np.ndarray=np.linspace(0.01, 1.0, 10), scoring: str = None, figsize: tuple=(10, 5), dpi: int=100, random_state: int = 123, callback: any = None) -> None:
    """학습곡선을 출력한다.

    Args:
        estimator (any): 학습모델 객체
        data (DataFrame): 독립변수
        yname (Series): 종속변수
        scaling (bool, optional): 스케일링 여부. Defaults to False.
        cv (int, optional): 교차검증의 수. Defaults to 10.
        train_sizes (np.ndarray, optional): 훈련 데이터의 비율. Defaults to np.linspace(0.1, 1.0, 10).
        scoring (str, optional): 평가지표. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        random_state (int, optional): 난수 시드값. Defaults to 123.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    if yname not in data.columns:
        raise Exception(f"\x1b[31m종속변수 {yname}가 존재하지 않습니다.\x1b[0m")
    
    x = data.drop(yname, axis=1)
    y = data[yname]
    w = 1

    if scalling:
        scaler = StandardScaler()
        x = DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)
        
    # 평가지표가 없는 경우
    if scoring == None:
        train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, random_state=random_state)
        
        ylabel = 'Score'
        
    # 평가지표가 있는 경우
    else:
        scoring = scoring.lower()
        
        if scoring == 'rmse':
            scoring ='neg_root_mean_squared_error'
            w = -1
        elif scoring == 'mse':
            scoring ='neg_mean_squared_error'
            w = -1
        elif scoring == 'r2':
            scoring ='r2'
            
        scoring_list = ["r2", 
                        "max_error", 
                        "matthews_corrcoef", 
                        "neg_median_absolute_error", 
                        "neg_mean_absolute_error", 
                        "neg_mean_absolute_percentage_error", 
                        "neg_mean_squared_error", 
                        "neg_mean_squared_log_error", 
                        "neg_root_mean_squared_error", 
                        "neg_root_mean_squared_log_error", 
                        "neg_mean_poisson_deviance", 
                        "neg_mean_gamma_deviance", 
                        "accuracy", 
                        "top_k_accuracy", 
                        "roc_auc", 
                        "roc_auc_ovr", 
                        "roc_auc_ovo", 
                        "roc_auc_ovr_weighted", 
                        "roc_auc_ovo_weighted", 
                        "balanced_accuracy", 
                        "average_precision", 
                        "neg_log_loss", 
                        "neg_brier_score", 
                        "positive_likelihood_ratio", 
                        "neg_negative_likelihood_ratio", 
                        "adjusted_rand_score", 
                        "rand_score", 
                        "homogeneity_score", 
                        "completeness_score", 
                        "v_measure_score", 
                        "mutual_info_score", 
                        "adjusted_mutual_info_score", 
                        "normalized_mutual_info_score", 
                        "fowlkes_mallows_score"]
        
        if scoring not in scoring_list:
            raise Exception(f"\x1b[31m평가지표 {scoring}가 존재하지 않습니다.\x1b[0m")
        
        ylabel = scoring.upper()
            
        train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring=scoring, random_state=random_state)

    train_mean = w * np.mean(train_scores, axis=1)
    train_std = w * np.std(train_scores, axis=1)
    test_mean = w * np.mean(test_scores, axis=1) 
    test_std = w * np.std(test_scores, axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # 훈련 데이터 수에 따른 훈련 데이터의 score 평균
    sb.lineplot(x=train_sizes, y=train_mean,  marker='o', markersize=5, label='훈련 데이터', color='#ff2200', ax=ax)
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='#ff2200')

    # 검증 데이터 수에 따른 검증 데이터의 score 평균
    sb.lineplot(x=train_sizes, y=test_mean, linestyle='--', marker='s', markersize=5, label='검증 데이터', color='#0066ff', ax=ax)
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='#0066ff')

    ax.grid()
    ax.set_xlabel('훈련 셋트 크기')
    ax.set_ylabel(ylabel)
    ax.legend()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_confusion_matrix(y: np.ndarray, y_pred: np.ndarray, cmap: str = None, figsize: tuple=(4,3), dpi: int=100, callback: any = None) -> None:
    """혼동행렬을 출력한다.

    Args:
        y_true (np.ndarray): 실제값
        y_pred (np.ndarray): 예측값
        cmap (str, optional): 칼라맵. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # 이진분류인지 다항분류인지 구분
    labels = sorted(list(y.unique()))
    is_binary = len(labels) == 2
    
    if is_binary:
        labels = ["Negative", "Positive"]
    else:
        labels = None
    
    # 다중 로지스틱을 살펴볼 때 함수 파라미터 설정이 변경될 수 있다.
    ConfusionMatrixDisplay.from_predictions(
        y,              # 관측치
        y_pred,         # 예측치
        display_labels=labels,
        cmap=cmap,
        text_kw={'fontsize': 24, 'weight': 'bold'},
        ax=ax
    )
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_roc_curve(y: Series, y_proba: Series, figsize: tuple = (8, 6), dpi: int = 200, callback: any = None) -> None:
    """ROC곡선을 출력한다.

    Args:
        y (Series): 실제값
        y_proba (Series): 예측확률
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    labels = sorted(list(y.unique()))
    is_binary = len(labels) == 2

    # 두 번째 파라미터가 판정결과가 아닌 1로 판정할 확률값
    fpr, tpr, thresholds = roc_curve(y, y_proba)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.lineplot(x=fpr, y=tpr, color='red', linewidth=1, label='ROC Curve', ax=ax)
    ax.fill_between(fpr,tpr, facecolor='blue',alpha=0.1)
    
    sb.lineplot(x=[0,1], y=[0,1], color='black', linestyle='--', linewidth=0.7, ax=ax)
    ax.set_xlabel('Fase Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xticks(np.round(np.arange(0, 1.1, 0.1),2))
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.text(0.95, 0.05, 'AUC=%0.3f' % roc_auc_score(y, y_proba), fontsize=16, ha='right', va='bottom')
    ax.grid()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_pr_curve(y: Series, y_proba: Series, figsize: tuple = (8, 6), dpi: int = 200, callback: any = None) -> None:
    """Precision-Recall 곡선을 출력한다.

    Args:
        y (Series): 실제값
        y_proba (Series): 예측확률
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    labels = sorted(list(y.unique()))
    is_binary = len(labels) == 2

    precision, recall, thresholds = precision_recall_curve(y_true=y, probas_pred=y_proba)
    y_test_mean = y.mean()

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    sb.lineplot(x=recall, y=precision, label='Precision / Recall Curve', color='blue', linewidth=1, ax=ax)
    sb.lineplot(x=[0,1], y=[y_test_mean,y_test_mean], color='black', linewidth=0.7, linestyle='--', ax=ax)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xticks(np.round(np.arange(0, 1.1, 0.1),2))
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([y_test_mean-0.05, 1.01])
    ax.grid()
    
    if callback:
        callback(ax)
        
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
def my_roc_pr_curve(y: Series, y_proba: Series, figsize: tuple = (16, 6), dpi: int = 200, callback: any = None) -> None:
    """ROC와 Precision-Recall 곡선을 출력한다.

    Args:
        y (Series): 실제값
        y_proba (Series): 예측확률
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    sb.lineplot(x=fpr, y=tpr, color='red', linewidth=1, label='ROC Curve', ax=ax[0])
    ax[0].fill_between(fpr, tpr, facecolor='blue',alpha=0.1)
    sb.lineplot(x=[0, 1], y=[0, 1], color='black', linestyle='--', linewidth=0.7, ax=ax[0])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
    ax[0].set_xlim([-0.01, 1.01])
    ax[0].set_ylim([-0.01, 1.01])
    ax[0].text(0.95, 0.05, 'AUC=%0.3f' % roc_auc_score(y, y_proba), fontsize=16, ha='right', va='bottom')
    ax[0].legend()
    ax[0].grid()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    y_mean = y.mean()
    
    sb.lineplot(x=recall, y=precision, label='Precision / Recall Curve', color='blue', linewidth=1, ax=ax[1])
    sb.lineplot(x=[0,1], y=[y_mean,y_mean], color='black', linewidth=0.7, linestyle='--', ax=ax[1])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
    ax[1].set_xlim([-0.01, 1.01])
    ax[1].set_ylim([y_mean-0.05, 1.01])
    ax[1].legend()
    ax[1].grid()

    if callback:
        callback(ax[0], ax[1])
        
    plt.tight_layout()
    plt.show()
    plt.close()