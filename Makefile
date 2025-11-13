# Load environment variables from .env file
include .env
export

.PHONY: run-local
run-local:
	@echo "Running application locally..."
	uv run streamlit run app.py

.PHONY: lint
lint:
	@echo "Running linting and formatting..."
	uvx ruff check --fix
	uvx ruff format
	@echo "Linting completed!"

.PHONY: deploy
deploy: lint
	@echo "Deploying $(SERVICE_NAME) to production..."
	gcloud builds submit \
		--project=$(PROJECT_ID) \
		--config cloudbuild.yaml \
		--substitutions=SHORT_SHA=$(SHORT_SHA),_SERVICE_NAME=$(SERVICE_NAME),_REGION=$(REGION) \
		.
	@echo "Deployment completed!"