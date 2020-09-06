# Installation Notes

```bash
# install the library and create gpt2 encoded binaries
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e . 
```

jsonnet is installed easier if using conda
```
conda install jsonnet -c conda-forge
```

A dev docker is provided
```bash
make dev-docker
```