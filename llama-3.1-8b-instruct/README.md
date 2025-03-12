```bash
uv pip install --index-strategy unsafe-best-match -r requirements.txt
bentoml serve

# to produce docker image
bentoml build --containerize
```
