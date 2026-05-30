.PHONY: help build up down logs shell restart

help:
	@echo "DerMind API Docker Commands"
	@echo "============================"
	@echo "make build          - Build Docker image"
	@echo "make up             - Start containers"
	@echo "make down           - Stop containers"
	@echo "make logs           - View container logs"
	@echo "make shell          - Access container shell"
	@echo "make restart        - Restart containers"
	@echo "make build-prod     - Build production image"
	@echo "make up-prod        - Start production containers"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec dermind-api sh

restart:
	docker-compose restart

build-prod:
	docker-compose -f docker-compose.prod.yml build

up-prod:
	docker-compose -f docker-compose.prod.yml up -d

down-prod:
	docker-compose -f docker-compose.prod.yml down
