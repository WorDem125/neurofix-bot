.PHONY: setup start stop restart logs status clean

setup:
	@echo "Загрузка моделей..."
	@bash scripts/download_models.sh
	@echo ""
	@echo "Готово. Запустите: make start"

start:
	@bash scripts/download_models.sh
	docker compose up -d --build
	@echo ""
	@echo "NeuroFix запущен. Статус: make status"

stop:
	docker compose down

restart:
	docker compose restart

status:
	docker compose ps

logs:
	docker compose logs -f

logs-bot:
	docker compose logs -f bot

logs-worker:
	docker compose logs -f worker

logs-enhancer:
	docker compose logs -f enhancer

logs-colorizer:
	docker compose logs -f colorizer

logs-classifier:
	docker compose logs -f classifier

logs-emotion:
	docker compose logs -f emotion

clean:
	docker compose down -v
	@echo "Контейнеры и volumes удалены"
