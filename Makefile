build:
	python setup.py bdist_wheel
install:
	pip install `ls dist/*.whl`

run:
	streamlit run Homepage.py