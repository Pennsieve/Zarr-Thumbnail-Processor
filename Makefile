.PHONY: build run clean local

build:
	docker compose build

run: build
	docker compose up

clean:
	docker compose down --rmi local
	rm -rf test_outputs/*

local:
	INPUT_DIR=./test_inputs OUTPUT_DIR=./test_outputs python -m processor.main
