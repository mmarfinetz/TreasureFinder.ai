.PHONY: convert-dofa verify-dofa

# Absolute paths (edit if repo moved)
REPO_ROOT := /Users/mitch/Desktop/Organized/Compare_Satellite_scripts
WEIGHTS_DIR := $(REPO_ROOT)/weights
STATE_DICT := $(WEIGHTS_DIR)/DOFA_ViT_base_e100.pth
OUT_MODULE := $(WEIGHTS_DIR)/dofa.pth

convert-dofa:
	@echo "==> Converting DOFA HF state_dict to serialized Module"
	python3 $(REPO_ROOT)/tools/convert_dofa_state_to_module.py \
	  --hub-repo zhu-xlab/DOFA \
	  --state $(STATE_DICT) \
	  --out $(OUT_MODULE) \
	  --backbone vit_base_dofa \
	  --in-channels 8
	@echo "==> Saved: $(OUT_MODULE)"

verify-dofa:
	@echo "==> Verifying DOFA production load"
	USE_DOFA=true PRODUCTION_MODE=true DOFA_LOCAL_WEIGHTS=$(OUT_MODULE) \
	python3 -c 'import os, numpy as np; os.environ.setdefault("USE_DOFA","true"); os.environ.setdefault("PRODUCTION_MODE","true"); import treasure_hunter_module as thm; m=thm.load_dofa_segmenter(); print("Loaded:", type(m)); img=np.zeros((thm.NUM_CHANNELS,64,64), dtype=np.float32); p,mask=thm.run_dofa_inference(img, return_mask=True); print("Smoke OK: p=", round(float(p),4), "mask=", mask.shape)'
	@echo "==> DOFA verify complete"


