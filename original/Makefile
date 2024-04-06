DATA_ROOT ?= ./data
AHI_FILE_PATH=${DATA_ROOT}/physionet.org/AHI.csv

.PHONY: print-data-root
print-data-root:
	@echo "Using DATA_ROOT=${DATA_ROOT}"

.PHONY: ensure-ahi
ensure-ahi: print-data-root
	./scripts/ensure_ahi.sh "${DATA_ROOT}" "${AHI_FILE_PATH}"
	
.PHONY: dataload
dataload: ensure-ahi
	./scripts/preprocess.sh ${DATA_ROOT}
