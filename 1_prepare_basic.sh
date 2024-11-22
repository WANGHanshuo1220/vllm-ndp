cd $HOME
ssh-keygen -t rsa -b 4096 -C "hanshuo_wang@163.com"
cat ~/.ssh/id_rsa.pub
echo "============================="

read -p "Have you saved key to github? (yes/no): " user_input

if [ "$user_input" == "yes" ]; then
    echo "OK! Great"
else
    echo "What are you doing!? Exiting."
    exit 1
fi

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ###下载miniconda
chmod +x Miniconda3-latest-Linux-x86_64.sh  
./Miniconda3-latest-Linux-x86_64.sh   ###安装

conda create -n mp python=3.10 -y
conda activate mp

git clone git@github.com:WANGHanshuo1220/vllm-ndp.git
