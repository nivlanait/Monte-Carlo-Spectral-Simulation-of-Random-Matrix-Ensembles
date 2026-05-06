# ===== CONFIG =====
PYTHON = python3
SCRIPT = main.py
INPUT = test_matrix.csv

# ===== DEFAULT RUN =====
run:
	$(PYTHON) $(SCRIPT)

# ===== MONTE CARLO ENSEMBLES =====
goe:
	$(PYTHON) $(SCRIPT) --ensemble goe --n 150 --trials 60

gue:
	$(PYTHON) $(SCRIPT) --ensemble gue --n 150 --trials 60

poisson:
	$(PYTHON) $(SCRIPT) --ensemble poisson --n 150 --trials 60

# ===== COMPARISON MODE =====
compare:
	$(PYTHON) $(SCRIPT) --input $(INPUT) --compare-simulated --n 100 --trials 100 --bins 60

# ===== INPUT MATRIX ONLY =====
input:
	$(PYTHON) $(SCRIPT) --input $(INPUT) --trim-fraction 0

# ===== FAST TEST (NO PLOT) =====
test:
	$(PYTHON) $(SCRIPT) --ensemble goe --n 80 --trials 10 --no-plot
	$(PYTHON) $(SCRIPT) --ensemble gue --n 80 --trials 10 --no-plot
	$(PYTHON) $(SCRIPT) --ensemble poisson --n 80 --trials 10 --no-plot

# ===== INSTALL DEPENDENCIES =====
install:
	pip install numpy matplotlib

# ===== CLEAN =====
clean:
	rm -rf __pycache__ *.pyc