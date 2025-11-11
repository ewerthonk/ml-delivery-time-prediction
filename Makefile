# Variables
PROJECT_ID := ewerthonk-data
SERVICE_NAME := delivery-time-prediction
REGION := southamerica-east1
SHORT_SHA := latest

.PHONY: run-local
run-local:
	@echo "Running application locally..."
	uv run streamlit run app.py

.PHONY: deploy
deploy:
	@echo "Deploying $(SERVICE_NAME) to production..."
	gcloud builds submit \
		--project=$(PROJECT_ID) \
		--config cloudbuild.yaml \
		--substitutions=SHORT_SHA=$(SHORT_SHA) \
		.
	@echo "Deployment completed!"