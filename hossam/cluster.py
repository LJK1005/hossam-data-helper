from pandas import DataFrame
from typing import Literal
import concurrent.futures as futures

from kneed import KneeLocator

from sklearn.cluster import KMeans

from hossam.plot import my_lineplot


def __kmeans(
    data: DataFrame,
    n_clusters: int,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=0,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
) -> float:
    """KMmeans 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int): 클러스터 개수
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): 알고리즘. Defaults to "lloyd".

    Returns:
        float
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm=algorithm,
    )
    kmeans.fit(data)
    kmeans.n_clusters = n_clusters
    return kmeans


def my_kmeans(
    data: DataFrame,
    max_clusters: int | list = 10,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=0,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    scoring: Literal["inertia", "silhouette"] = "inertia",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DataFrame:
    """클러스터 개수에 따른 이너셔 값을 계산한다.

    Args:
        data (DataFrame): 원본 데이터
        max_clusters (int | list, optional): 최대 클러스터 개수. 정수로 전달할 경우 `2`부터 주어진 개수까지 반복 수행한다. Defaults to 10.
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): _description_. Defaults to "lloyd".

    Returns:
        DataFrame: _description_
    """
    with futures.ThreadPoolExecutor() as executor:
        results = []
        for n_clusters in range(2, max_clusters + 1):
            results.append(
                executor.submit(
                    __kmeans,
                    data,
                    n_clusters,
                    init,
                    max_iter,
                    random_state,
                    algorithm,
                )
            )

        kmeans_list = [r.result() for r in futures.as_completed(results)]

        if scoring == "inertia":
            inertia = [k.inertia_ for k in kmeans_list]
            n_clusters = [k.n_clusters for k in kmeans_list]

            kn = KneeLocator(
                x=n_clusters,
                y=inertia,
                curve="convex",
                direction="decreasing",
            )

            best_k = kn.elbow
            best_y = kn.elbow_y
            best_model = next(filter(lambda x: x.n_clusters == best_k, kmeans_list))

            if plot:

                def hvline(ax):
                    ax.set_ylabel("inertia")
                    ax.set_xlabel("cluster count")
                    ax.set_title("Elbow Method")
                    ax.axhline(best_y, color="red", linestyle="--", linewidth=0.7)
                    ax.axvline(best_k, color="red", linestyle="--", linewidth=0.7)
                    ax.text(
                        best_k + 0.2,
                        best_y + 0.2,
                        f"k={best_k}",
                        fontsize=20,
                        color="red",
                        va="bottom",
                        ha="left",
                    )

                my_lineplot(
                    df=None,
                    xname=n_clusters,
                    yname=inertia,
                    marker="o",
                    linewidth=2,
                    figsize=figsize,
                    dpi=dpi,
                    callback=hvline,
                )

            return best_model
        else:
            return sorted(kmeans_list, key=lambda x: x.n_clusters)
