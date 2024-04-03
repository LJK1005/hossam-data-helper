def startup(package: bool = False, mecab: bool = False) -> None:
    # 경고 메시지 off
    import warnings, os, sys

    warnings.filterwarnings(action="ignore")

    # Google Colab 환경인지 확인
    try:
        import google.colab

        IN_COLAB = True
    except:
        IN_COLAB = False

    # Google Colab 환경이 아닌 경우
    if not IN_COLAB:
        if sys.platform == "win32":
            from sklearnex import patch_sklearn

            patch_sklearn()

    else:
        print("Initializing...")
        print("-" * 100)

        print("nanum font installation start!!!")
        os.system(command="sudo apt-get install -y fonts-nanum")
        os.system(command="sudo fc-cache -fv")
        os.system(command="rm ~/.cache/matplotlib -rf")
        print("nanum font installation success!!!")

        if package:
            print("-" * 100)
            print("package installation start!!!")

            # Google Colab 환경인 경우 필요한 패키지 설치
            addon_packages = ["pca", "pingouin", "statannotations"]
            l = len(addon_packages)

            for i, v in enumerate(iterable=addon_packages):
                os.system(command=f"pip3 install --upgrade {v}")
                print(f"[{i+1}/{l}] {v} package install success")

            print("package installation success!!!")

        if mecab:
            print("-" * 100)
            print("mecab installation start!!!")
            print("[1/2] Clone Mecab-ko-for-Google-Colab from github")
            os.system(
                command="git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git"
            )
            print("[2/2] Install Mecab-ko-for-Google-Colab")
            os.system(
                command="bash Mecab-ko-for-Google-Colab/install_mecab-ko_on_colab_light_220429.sh"
            )
            print("mecab installation success!!!")

        print("-" * 100)
        print("Initialization complete")

    return IN_COLAB


if __name__ == "__main__":
    IN_COLAB = startup()
