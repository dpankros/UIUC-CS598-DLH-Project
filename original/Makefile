DATA_ROOT ?= ./data
AHI_FILE_PATH=${DATA_ROOT}/physionet.org/AHI.csv

.PHONY: print-data-root
print-data-root:
	@echo "Using DATA_ROOT=${DATA_ROOT}"

.PHONY: ensure-ahi
ensure-ahi: print-data-root
	@echo "----------------------------"
	@echo "ensuring AHI.csv file exists"
	@echo "----------------------------"
	./scripts/ensure_ahi.sh "${DATA_ROOT}" "${AHI_FILE_PATH}"

.PHONY: purge-ahi
purge-ahi:
	@echo "--------------------"
	@echo "Removing AHI file ${AHI_FILE_PATH}"
	@echo "--------------------"
	rm ${AHI_FILE_PATH}

.PHONY: dataload
dataload: ensure-ahi
	@echo "--------------------"
	@echo "starting dataloading"
	@echo "--------------------"
	./scripts/preprocess.sh ${DATA_ROOT}

.PHONY: purge-dataload
purge-dataload:
	@echo "--------------------"
	@echo "purging dataload and training data files"
	@echo "--------------------"
	# remove preprocessed data files
	@rm -rf "${DATA_ROOT}/physionet.org/nch_30x64"
	# remove final training data file
	@rm -rf "${DATA_ROOT}/physionet.org/nch_30x64.npz"
	

.PHONY: train
train: dataload
	@echo "-----------------------------------"
	@echo "dataloading done. starting training"
	@echo "-----------------------------------"
	DLHPROJ_DATA_ROOT=${DATA_ROOT}/physionet.org python main_nch.py

.PHONY: eval
eval:
	@echo "-----------------------------------"
	@echo "evaluating"
	@echo "-----------------------------------"
	DLHPROJ_DATA_ROOT=${DATA_ROOT}/physionet.org python results/results_to_csv.py
