.PHONY: setup run docker-build docker-run health-check clean

# Initialize environment & install pinned dependencies
setup:
	@echo "🔧 Creating venv & installing dependencies..."
	python -m venv venv_gpu
	venv_gpu/bin/pip install -r requirements.txt
	@echo "✅ Environment ready. Activate with: source venv_gpu/bin/activate"

# Run Streamlit dashboard locally
run:
	@echo "🚀 Launching Dashboard on port 8501..."
	streamlit run dashboard/app.py --server.port 8501

# Build Docker image
docker-build:
	@echo "📦 Building container..."
	docker compose build --no-cache

# Run containerized app
docker-run: docker-build
	@echo "🐳 Starting container..."
	docker compose up -d
	@echo "🌐 Dashboard live: http://localhost:8501"

# Pre-deployment diagnostics
health-check:
	@echo "🩺 Running system diagnostics..."
	python utils/health_check.py

# Clean cache & logs
clean:
	@echo "🧹 Cleaning artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "✅ Done cleaning."
