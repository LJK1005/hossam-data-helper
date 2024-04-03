HELLOWORLD = "Hello World!"

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
    os.system(command="sudo apt-get install -y fonts-nanum")
    os.system(command="sudo fc-cache -fv")
    os.system(command="!rm ~/.cache/matplotlib -rf")"