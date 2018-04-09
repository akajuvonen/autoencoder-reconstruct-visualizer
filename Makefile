init:
	pip install -r requirements.txt

gif:
	convert -delay 50 figures/*.png animation.gif

run:
	python autoencoder_visualizer.py

.PHONY: init gif
