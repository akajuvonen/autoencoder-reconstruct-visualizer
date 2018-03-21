init:
	pip install -r requirements.txt

gif:
	convert -delay 100 figures/*.png animation.gif

.PHONY: init gif
