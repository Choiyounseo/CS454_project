git clone 받으신후 GD 폴더안에 들어와서 가상환경 만들고 가상환경안에 들어가서 pip install 받아야합니다
```
git clone https://github.com/Choiyounseo/CS454_project.git
cd GD
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
실행
```
python main.py
```
defensegan 안에 observation_change 라는 거 추가했습니다.

observation_step 만큼 진행될때마다 rr(default=10) 개의 랜덤이미지의 변화를 보실수 있습니다.
