# How to set up EC2 servers to run benchmarks

- Create instance
    - Use ubuntu version with python version that works with old scikit-learn
    - 24 GiB volume is too small, trying 100
- Run the following commands

```bash
sudo apt update

# Install dependencies for pyenv to work
# TODO: Not sure if all is needed
sudo apt install -y build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
   libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
   libffi-dev liblzma-dev python3-openssl git

# Install pyenv
curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11.11
pyenv global 3.11.11

# Go to secure-eval/benchmarks
python -m venv .venv 
source .venv/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```