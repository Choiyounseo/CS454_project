git clone 받으신후 GD 폴더안에 들어와서 가상환경 만들고 가상환경안에 들어가서 pip install 받아야합니다
```
git clone https://github.com/Choiyounseo/CS454_project.git
cd GD
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
실행하기전에 main.py 에 있는 fgsm_image_path, model_weight_path 설정해주면 끝
```
python main.py
```
