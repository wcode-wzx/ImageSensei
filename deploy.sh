# 【vscode plug-in required】
- Python

# 【miniconda install】
# - https://blog.csdn.net/baidu_41805096/article/details/108501099
# - https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py38_4.10.3-Windows-x86_64.exe

# 【environment creation】
# - (base) 环境下执行
conda create -n opencv python=3.8

conda activate opencv

# pip 安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 启动
python app.py

# 访问主页
http://127.0.0.1:5001/admin/index
