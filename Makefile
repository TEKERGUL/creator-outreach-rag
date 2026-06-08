.PHONY: install build-index demo test clean

install:
	pip install -r requirements.txt

build-index:
	python app.py build-index --data data/creators.csv

demo:
	@echo ""
	@echo "=== Semantic Search Demo ==="
	python app.py search "beauty creators who do skincare tutorials" --top-k 5
	@echo ""
	@echo "=== Filtered Search Demo ==="
	python app.py search "high engagement fitness content" --top-k 5 --platform tiktok --min-engagement 9.0
	@echo ""
	@echo "=== Food Creator Search ==="
	python app.py search "healthy meal prep on a budget" --top-k 5 --niche food

test:
	pytest tests/ -v

clean:
	rm -rf index/*.npy __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
